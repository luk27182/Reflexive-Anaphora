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
dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=lambda batch: pad_sequence(batch, batch_first=False, padding_value=3))
for batch in dl:
    break
print("Batch has shape length by batch:", batch.shape)
# %%
class SelfAttention(nn.Module):
    """Shamelessly borrowed from https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb#scrollTo=kWpfPKHs9tHI"""

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
    
class AttnOnly_Transformer(nn.Module):
    def __init__(self, vocab_size, n_heads, d_model, d_head, ctx_length=9):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(ctx_length, d_model)

        self.attn = SelfAttention(n_heads=n_heads, d_model=d_model, d_head=d_head)
        self.unembedding = nn.Linear(d_model, vocab_size)

    def embed(self, tensor):
        return self.embedding(tensor)+self.pos_embedding(einops.repeat(torch.arange(tensor.size(0)), "n -> n b", b=tensor.size(1)).to(torch.int64).to(device))
    
    def forward(self, seq):
        # seq is of shape BATCH X POS, where entries are integers
        embed = einops.rearrange(self.embed(seq), "s b d -> b s d")
        attn_out, pattern = self.attn(embed)
        model_out = self.unembedding(attn_out+embed) # shape is BATCH X POS X VOCAB

        return model_out, pattern
    

# %%
model = AttnOnly_Transformer(vocab_size=len(vocab), n_heads=3, d_model=32, d_head=8).to(device)
optimizer = torch.optim.Adam(params=model.parameters())
criterion = torch.nn.CrossEntropyLoss()
num_epochs=50

for epoch in range(num_epochs):
    avg_loss = 0
    for batch in dl:
        loss = 0
        optimizer.zero_grad()

        batch = batch.to(device)
        model_out, pattern = model(batch) # Model_Out shape is SEQ X BATCH X VOCAB
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
    avg_loss /= len(dl)
    print(f"epoch {epoch}, loss {avg_loss}")
# %%
def test_index(model, index, max_length=10):
    print(f"===== TESTING INDEX {index} =====")
    example = ds[index].to(device)
    break_point = example.tolist().index(2)+1
    input = example[:break_point]
    output = example[break_point:]
    print(f"input: {chars_from_ids(input)}")
    print(f"goal: {chars_from_ids(output)}")

    input = input.view(-1, 1)
    while len(input) < max_length and input[-1][0] != 3:
        model_out, pattern = model(input)
        _, new_out = model_out[0][-1].topk(1)
        input = torch.cat([input, torch.tensor([new_out]).to(device).view(1, 1)])
    print(f"model out: {chars_from_ids(input.flatten()[break_point:])}")
    print()

test_index(model, index = 208)
test_index(model, index = 300)
test_index(model, index = 10)