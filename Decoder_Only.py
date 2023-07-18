# %%
import torch
import einops
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import einops 
from fancy_einsum import einsum
import math
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
intransitive_verbs = ['runs', 'sleeps', 'laughs', 'cries', 'talks', 'jumps', 'dances', 'sings']
transitive_verbs = ['eats', 'sees', 'hugs', 'paints', 'kicks', 'throws', 'compliments']
female_names = ['Alice', 'Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte', 'Amelia', 'Harper', 'Evelyn', 'Abigail', 'Emily', 'Elizabeth', 'Mila']
male_names = ['Bob', 'John', 'Noah', 'Oliver', 'Elijah', 'William', 'James', 'Benjamin', 'Lucas', 'Henry', 'Michael']

names = female_names+male_names
verbs = intransitive_verbs+transitive_verbs

corpus = []
for name in names:
    for verb in intransitive_verbs:
        new_sentence = " ".join([name, verb])
        parsed = " ".join([verb.upper(), name.upper()])
        seq = "<SOS> " + new_sentence + " <UNK> "+parsed+" <EOS>"
        corpus.append(seq)

for subj in names:
    for verb in transitive_verbs:
        if subj in female_names:
            new_sentence = " ".join([subj, verb, "herself"])
        else:
            new_sentence = " ".join([subj, verb, "himself"])
        parsed = " ".join([verb.upper(), subj.upper(), subj.upper()])
        seq = "<SOS> " + new_sentence + " <UNK> "+parsed+" <EOS>"
        corpus.append(seq)
        for obj in names:
            new_sentence = " ".join([subj, verb, obj])
            parsed = " ".join([verb.upper(), subj.upper(), obj.upper()])
            seq = "<SOS> " + new_sentence + " <UNK> "+parsed+" <EOS>"
            corpus.append(seq)

print(f"{len(corpus)} total examples.") 
print(f"Example: {corpus[-2]}") 
print(f"Example: {corpus[-1]}")

# %%
def create_train_corpus(corpus, excluded_females=0, exclude_men=False):
    excluded_names = female_names[:excluded_females]
    if exclude_men:
        excluded_names += male_names
    corpus_test = []
    for name in excluded_names:
        corpus_test += [sentence for sentence in corpus if  name in sentence and ("self" in sentence)]
    
    corpus_train = [sentence for sentence in corpus if not sentence in corpus_test]
    corpus_test = [sentence for sentence in corpus_test if not "himself" in sentence]

    return corpus_train, corpus_test

train, test = create_train_corpus(corpus)
print(f"Excluding Nothing: {len(train)} train, {len(test)} test")

train, test = create_train_corpus(corpus, excluded_females=1)
print(f"Excluding Alice: {len(train)} train, {len(test)} test")

train, test = create_train_corpus(corpus, excluded_females=2)
print(f"Excluding Alice, Emma: {len(train)} train, {len(test)} test")

train, test = create_train_corpus(corpus, excluded_females=0, exclude_men=True)
print(f"Excluding Men: {len(train)} train, {len(test)} test")
# %%
special_vocab = ["<PAD>", "<SOS>", "<UNK>", "<EOS>"]

vocab = special_vocab+sorted(list(set(verbs+names+["himself", "herself"]+[verb.upper() for verb in verbs]+[name.upper() for name in names])))
def ids_from_chars(str):
    return torch.tensor([vocab.index(word) for word in str.split()])

def chars_from_ids(tensor):
    return " ".join([vocab[idx] for idx in tensor])

example = corpus[-2]
print("recoverd example:", chars_from_ids(ids_from_chars(example)))

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, corpus):
        super().__init__()        
        self.corpus = corpus

    def __len__(self):        
        return len(self.corpus)

    def __getitem__(self, idx):
        return ids_from_chars(self.corpus[idx])
    
ds = TextDataset(corpus)

# train_size = int(0.8*len(ds))
# test_size = len(ds)-train_size
# ds_train, ds_test = torch.utils.data.random_split(ds, [train_size, test_size])



# %%
class LayerNorm(nn.Module):
    """Taken from https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb#scrollTo=kWpfPKHs9tHI"""
    def __init__(self, d_model, layer_norm_eps=1e-5):
        super().__init__()
        self.layer_norm_eps = layer_norm_eps

        self.w = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, residual):
        # residual: [batch, position, d_model]
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + self.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        return normalized

class SelfAttention(nn.Module):
    """Adopted from https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb#scrollTo=kWpfPKHs9tHI"""
    def __init__(self, n_heads, d_head, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        
        self.init_range = 1

        self.W_Q = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        nn.init.normal_(self.W_Q, std=self.init_range)
        self.b_Q = nn.Parameter(torch.zeros((n_heads, d_head)))
        self.W_K = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        nn.init.normal_(self.W_K, std=self.init_range)
        self.b_K = nn.Parameter(torch.zeros((n_heads, d_head)))
        self.W_V = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        nn.init.normal_(self.W_V, std=self.init_range)
        self.b_V = nn.Parameter(torch.zeros((n_heads, d_head)))
        
        self.W_O = nn.Parameter(torch.empty((n_heads, d_head, d_model)))
        nn.init.normal_(self.W_O, std=self.init_range)
        self.b_O = nn.Parameter(torch.zeros((d_model)))

    def apply_causal_mask(self, attn_scores):
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, -1e5)
        return attn_scores
    
    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]

        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        
        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model", z, self.W_O) + self.b_O
        
        return attn_out, pattern

class AttnBlock(nn.Module):
    def __init__(self, n_heads, d_head, d_model, attn_only=False):
        super().__init__()
        self.attn_only = attn_only

        self.ln1 = LayerNorm(d_model=d_model)
        self.attn = SelfAttention(n_heads=n_heads, d_model=d_model, d_head=d_head)

        self.ln1 = LayerNorm(d_model=d_model)
        self.attn = SelfAttention(n_heads=n_heads, d_model=d_model, d_head=d_head)

        if not attn_only:
            self.linear_expand = nn.Linear(d_model, 4*d_model)
            self.linear_contract = nn.Linear(4*d_model, d_model)
            self.ln2 = LayerNorm(d_model=d_model)
    
    def forward(self, resid_pre):
        normalized_resid_pre = self.ln1(resid_pre)

        attn_out, attn_pattern = self.attn(normalized_resid_pre)
        resid_mid = attn_out+normalized_resid_pre
        
        if self.attn_only:
            return resid_mid
    
        normalized_resid_mid = self.ln2(resid_mid)

        linear_out = self.linear_contract(self.linear_expand(normalized_resid_mid))
        resid_post = linear_out+resid_mid

        return resid_mid


class AttnOnly_Transformer(nn.Module):
    def __init__(self, vocab_size, n_heads, d_model, d_head, n_layers, attn_only=False, ctx_length=9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(ctx_length, d_model)

        self.blocks = nn.ModuleList([])
        self.blocks.extend([AttnBlock(n_heads=n_heads, d_model=d_model, d_head=d_head, attn_only=attn_only) for i in range(n_layers)])

        self.unembedding = nn.Linear(d_model, vocab_size)

    def embed(self, tensor):
        return self.embedding(tensor)+self.pos_embedding(einops.repeat(torch.arange(tensor.size(0)), "n -> n b", b=tensor.size(1)).to(torch.int64).to(device))
    
    def forward(self, seq):
        # seq is of shape BATCH X POS, where entries are integers
        residual_stream = einops.rearrange(self.embed(seq), "s b d -> b s d")
        for block in self.blocks:
            residual_stream = block(residual_stream)
        model_out = self.unembedding(residual_stream) # shape is BATCH X POS X VOCAB

        return model_out
    

# %%
def train_model(model, ds_train, ds_test, num_epochs, print_every=5, batch_size=32):
    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=lambda batch: pad_sequence(batch, batch_first=False, padding_value=3))
    dl_test = DataLoader(ds_test, batch_size=32, shuffle=True, collate_fn=lambda batch: pad_sequence(batch, batch_first=False, padding_value=3))

    optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        avg_loss = 0
        for batch in dl_train:
            loss = 0
            optimizer.zero_grad()

            batch = batch.to(device)
            model_out = model(batch) # Model_Out shape is SEQ X BATCH X VOCAB
            batch = einops.rearrange(batch, 'S B -> B S')
            predictions = 0
            for i, example in enumerate(batch):
                break_point = example.tolist().index(2)+1
                loss += criterion(model_out[i, break_point-1:-1], example[break_point:])
                predictions += example[break_point:].size(0)
            loss /= predictions
            avg_loss += loss

            loss.backward()
            optimizer.step()
        avg_loss /= len(dl_train)


        if epoch+1 % print_every == 0:
            total = 0
            correct = 0
            for batch in dl_test:
                total += batch.size(1)

                loss = 0
                batch = batch.to(device)
                model_out = model(batch)
                batch = einops.rearrange(batch, 'S B -> B S')
                model_out = model_out.topk(1)[1].squeeze(-1)
                for index in range(model_out.size(0)):
                    break_point = batch[index].tolist().index(2)+1
                    if torch.all(model_out[index, break_point-1:-1].eq(batch[index, break_point:])):
                        correct += 1
            print(f"Epoch {epoch+1} Train Loss {avg_loss:.5f} Test Accuracy {correct/total:.3f}")
    
    return model
# %%
def test_index(model, ds, index, max_length=10, verbose=True):
    if verbose: print(f"===== TESTING INDEX {index} =====")
    example = ds[index].to(device)
    break_point = example.tolist().index(2)+1
    input = example[:break_point]
    output = example[break_point:]
    if verbose: print(f"input: {chars_from_ids(input)}")
    if verbose: print(f"goal: {chars_from_ids(output)}")

    input = input.view(-1, 1)
    while len(input) < max_length and input[-1][0] != 3:
        model_out = model(input)
        _, new_out = model_out[0][-1].topk(1)
        input = torch.cat([input, torch.tensor([new_out]).to(device).view(1, 1)])
    if verbose: print(f"model out: {chars_from_ids(input.flatten()[break_point:])}")
    
    return chars_from_ids(output) == chars_from_ids(input.flatten()[break_point:])

# %%
from tqdm import tqdm

results = dict()
experiments = [1, 2, 4, 6, 8]
for n in experiments:
    for i in range(10):
        print(f"training model {i+1}/10... in experiment {n}")
        corpus_train, corpus_test = create_train_corpus(corpus, excluded_females=n, exclude_men=False)
        ds_train, ds_test = TextDataset(corpus_train), TextDataset(corpus_test)
        model = AttnOnly_Transformer(vocab_size=len(vocab), n_heads=4, d_model=32, d_head=8, n_layers=3, attn_only=False).to(device)
        model = train_model(model, ds_train, ds_train, num_epochs=100, print_every=1e10)

        test_performance = []
        for idx in range(len(ds_test)):
            test_performance.append(test_index(model, ds=ds_test, index=idx, max_length=10, verbose=False))
        test_performance = sum(test_performance)/len(test_performance)
        print(f"accuracy on test set: {test_performance}")

        key = f"excluding {n} females, including all males"
        if key in results.keys():
            results[key].append(test_performance)
        else:
            results[key] = [test_performance]
    print(f"EXPERIMENT {n} RESULTS: {results[key]}")

# train_size = int(0.98*len(ds))
# test_size = len(ds)-train_size
# ds_train, ds_test = torch.utils.data.random_split(ds, [train_size, test_size])




# %%


reflexive_idxs = [corpus.index(reflexive_example) for reflexive_example in [example for example in corpus if "self" in example]]
total = 0
correct = 0
for idx in reflexive_idxs:
    total += 1
    if test_index(model, index = idx, ds=ds, verbose=False):
        correct += 1
    else:
       test_index(model, index = idx, ds=ds, verbose=True) 
print(correct/total)


# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x = []
y = []
cellText = [["Number of Excluded Names", "Mean", "STD"]]
for n in experiments:
    new_points = results[f"excluding {n} females, including all males"]
    cellText += [[str(n), f"{torch.mean(torch.tensor(new_points)).item():.3f}", f"{torch.std(torch.tensor(new_points)).item():.3f}"]]
    x += [n for i in range(len(new_points))]
    y += new_points
  
ax = sns.pointplot(x, y, errorbar="sd")
plt.title('Effect of Removing Female Names on ED_Transformer \n Generalization (\"himself\" examples NOT included)')
# Set x-axis label
plt.xlabel('Excluded Female Names (of 15)')
# Set y-axis label
plt.ylabel('Accuracy on Excluded Names')

plt.table(cellText=cellText, loc='bottom', bbox = [0, -0.7, 1, 0.5])

plt.show()


# %%

for verb in transitive_verbs:
    partial = f"Emma {verb} herself"
    full = [sent for sent in corpus if partial in sent]
    index = corpus.index(full[0])
    test_index(model, index = index, ds=ds)

# %%
for i in range(len(ds_test)):
    test_index(model, index=i, ds=ds_test)
