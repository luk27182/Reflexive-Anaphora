# %%
import torch
import einops
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
intransitive_verbs = ['runs', 'sleeps', 'laughs', 'cries', 'talks', 'jumps', 'dances', 'sings']
transitive_verbs = ['eats', 'sees', 'hugs', 'paints', 'kicks', 'throws', 'compliments']
female_names = ['Alice', 'Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte', 'Amelia', 'Harper', 'Evelyn', 'Abigail', 'Emily', 'Elizabeth', 'Mila']
male_names = ['Bob', 'John', 'Noah', 'Oliver', 'Elijah', 'William', 'James', 'Benjamin', 'Lucas', 'Henry', 'Michael']

names = female_names+male_names

corpus = []
for name in names:
    for verb in intransitive_verbs:
        new_sentence = " ".join([name, verb])
        parsed = " ".join([verb.upper(), name.upper()])
        corpus.append((new_sentence, parsed))

for subj in names:
    for verb in transitive_verbs:
        if subj in female_names:
            new_sentence = " ".join([subj, verb, "herself"])
        else:
            new_sentence = " ".join([subj, verb, "himself"])
        parsed = " ".join([verb.upper(), subj.upper(), subj.upper()])
        corpus.append((new_sentence, parsed))
        for obj in names:
            new_sentence = " ".join([subj, verb, obj])
            parsed = " ".join([verb.upper(), subj.upper(), obj.upper()])
            corpus.append((new_sentence, parsed))

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
        corpus_test += [pair for pair in corpus if  name in pair[0] and ("herself" in pair[0] or "himself" in pair[0])]
    
    corpus_train = [pair for pair in corpus if not pair in corpus_test]
    corpus_test = [pair for pair in corpus_test if not "himself" in pair[0]]

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

eng_vocab = special_vocab+sorted(list(set(intransitive_verbs+transitive_verbs+names+["himself", "herself"])))
print("English vocab:", eng_vocab)
print("length of English vocab:", len(eng_vocab))

parsed_vocab = special_vocab+sorted(list(set([word.upper() for word in intransitive_verbs+transitive_verbs+names])))
print("parsed vocab:", parsed_vocab)
print("length of parsed vocab (- him/herself):", len(parsed_vocab))

# %%
def ids_from_chars(str, lang):
    if lang == "eng":
        vocab = eng_vocab
    else:
        vocab = parsed_vocab
    return torch.tensor([vocab.index(word) for word in str.split()])

def chars_from_ids(tensor, lang):
    if lang == "eng":
        vocab = eng_vocab
    else:
        vocab = parsed_vocab
    return " ".join([vocab[idx] for idx in tensor])

example = corpus[-2][0]
print("recoverd english example:", chars_from_ids(ids_from_chars(example, lang="eng"), lang="eng"))

example = corpus[-2][1]
print("recoverd parsed example:", chars_from_ids(ids_from_chars(example, lang="parsed"), lang="parsed"))

# %%
# Next, we define a Dataset class in PyTorch
class Autoregressive_TextDataset(torch.utils.data.Dataset):
    def __init__(self, corpus):
        super().__init__()        
        self.corpus = corpus

    def __len__(self):        
        return len(self.corpus)

    def __getitem__(self, idx):
        raw = ids_from_chars(self.corpus[idx][0], lang="eng")
        parsed = ids_from_chars(self.corpus[idx][1], lang="parsed")
        return raw, parsed
    
ds = Autoregressive_TextDataset(corpus)

def add_padding(batch):
    original_raw = [pair[0] for pair in batch]
    original_parsed = [pair[1] for pair in batch]

    padded_raw = pad_sequence(original_raw, padding_value=0)
    padded_parsed = pad_sequence(original_parsed, padding_value=3)
    return (padded_raw, padded_parsed)

dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, collate_fn=add_padding)
for batch in dl:
    break
print("Batch has shape length by batch:", batch[0].shape)
# %%
from torch import nn
from torch.nn import functional as F

# %%
class GRU_Encoder(nn.Module):
    def __init__(self, input_size, d_model):
        super(GRU_Encoder, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(input_size, d_model)
        self.gru = nn.GRU(d_model, d_model)


    def forward(self, input):
        embedded = self.embedding(input)
        _, gru_out = self.gru(embedded)
        return gru_out

class GRU_Decoder(nn.Module):
    def __init__(self, d_model, output_size):
        super(GRU_Decoder, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(output_size, d_model)
        self.gru = nn.GRU(d_model, d_model)
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        _, gru_out = self.gru(embedded, hidden)
        linear_out = self.linear(gru_out)

        return gru_out, linear_out
    
class GRU_Full(nn.Module):
    def __init__(self, input_size, output_size, d_model):
        super(GRU_Full, self).__init__()
        self.encoder = GRU_Encoder(input_size=input_size, d_model=d_model)
        self.decoder = GRU_Decoder(d_model=d_model, output_size=output_size)
    
    def forward(self, src, tgt, tf_ratio=0.5):
        if not model.training:
            tf_ratio = 0
        target_length = tgt.size(0)+1

        

        encoder_outputs = self.encoder(src)
        decoder_hidden = encoder_outputs
        decoder_input = tgt[0]

        use_teacher_forcing = True if random.random() < tf_ratio else False
        model_out = None
        for index in range(target_length-1):
            decoder_hidden, decoder_output = self.decoder(input=decoder_input.view(1, -1), hidden=decoder_hidden)
            
            if model_out is None:
                model_out = decoder_output
            else:
                model_out = torch.cat([model_out, decoder_output], dim=0)
            
            if not target_length == model_out.size(0)+1:
                if use_teacher_forcing:
                    decoder_input = tgt[index+1]
                else:
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

        return model_out
# %%
class ED_Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead=1, ctx_length=8, preserve_encoding_embedding_space=False):
        super(ED_Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(input_size, d_model)
        self.preserve_encoding_embedding_space = preserve_encoding_embedding_space

        self.decoder_embedding = nn.Embedding(output_size, d_model)

        self.encoder_pos_embedding = nn.Embedding(ctx_length, d_model)
        self.decoder_pos_embedding = nn.Embedding(ctx_length, d_model)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4*d_model)

        self.linear_out = nn.Linear(d_model, output_size)

    def embed(self, tensor, embedding, pos_embedding):
        embed = embedding(tensor)+pos_embedding(einops.repeat(torch.arange(tensor.size(0)), "n -> n b", b=tensor.size(1)).to(torch.int64).to(device))
        return embed
    
    def forward(self, src, tgt):
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0), device=device)

        src_embed = self.embed(src, self.encoder_embedding, self.encoder_pos_embedding)
        if self.preserve_encoding_embedding_space:
            src_embed[:, :, -1] = 0

        tgt_embed = self.embed(tgt, self.decoder_embedding, self.decoder_pos_embedding)

        transformer_out = self.transformer(src=src_embed, tgt=tgt_embed, tgt_mask=tgt_mask)
        out = self.linear_out(transformer_out) # Output is SEQ_LEN X BATCH X OUTPUT_VOCAB_SIZE

        return out

# %%
import random
import copy

SOS_token = parsed_vocab.index("<SOS>") # 1
EOS_token = parsed_vocab.index("<EOS>") # 3

def train(dl_train, model, optimizer, criterion, epochs, verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train().to(device)

    lowest_loss = 1e6

    for epoch in range(epochs):
        avg_loss = 0
        for src, tgt in dl_train:
            src = torch.cat([SOS_token*torch.ones(1, tgt.size(1)), src]).to(torch.int64).to(device)
            decoder_inputs = torch.cat([SOS_token*torch.ones(1, tgt.size(1)), tgt]).to(torch.int64).to(device)
            tgt = torch.cat([tgt, EOS_token*torch.ones(1, tgt.size(1))]).to(torch.int64).to(device)

            optimizer.zero_grad()
            loss = 0

            model_out = model(src, decoder_inputs)
            for i in range(model_out.size(0)):
                loss += criterion(model_out[i], tgt[i])/model_out.size(0)
            loss.backward()
            optimizer.step()
            avg_loss += loss
        
        avg_loss /= len(dl.dataset)
        if avg_loss < lowest_loss:
            best_state_dict = copy.deepcopy(model.state_dict())
            lowest_loss = avg_loss 
        if verbose:
            print(f"Epoch {epoch} Loss: {avg_loss:.9f}")
    model.load_state_dict(best_state_dict)
    return model

# %%
corpus_train, corpus_test = create_train_corpus(corpus=corpus, excluded_females=15, exclude_men=True)
ds_train = Autoregressive_TextDataset(corpus=corpus_train)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=add_padding)
for i in range(5):
    model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32, preserve_encoding_embedding_space=True)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model = train(dl_train, model, optimizer, criterion, epochs=100, verbose=True)
    torch.save(obj=model.state_dict(), f=f"./Models/transformer_1head_32hidden_100epochs_noReflexives.pth")


# %%
corpus_train, corpus_test = create_train_corpus(corpus=corpus, excluded_females=15, exclude_men=True) # Remove ALL reflexives
ds_train = Autoregressive_TextDataset(corpus=corpus_train)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=add_padding)

model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32, preserve_encoding_embedding_space=False)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss(reduction="sum")
model = train(dl_train, model, optimizer, criterion, epochs=100, verbose=True)
torch.save(obj=model.state_dict(), f='./Models/transformer_1head_32hidden_100epochs_noReflexives.pth')

# %%
def manipulate_embeddings(state_dict):
    # First we clear out the final dimension of the embedding
    state_dict['encoder_embedding.weight'][:, -1] = 0
    state_dict['decoder_embedding.weight'][:, -1] = 0

    def embed_reflexive(tensor, id):
        tensor[id, :-1] = 0
        tensor[id, -1] = 1

    # Then we add in our embeddings for him/herself
    himself_id = ids_from_chars("himself", lang="eng")
    embed_reflexive(tensor=state_dict['encoder_embedding.weight'], id=ids_from_chars("himself", lang="eng"))
    embed_reflexive(tensor=state_dict['encoder_embedding.weight'], id=ids_from_chars("herself", lang="eng"))

    return state_dict

model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32, preserve_encoding_embedding_space=True).to(device)
model.load_state_dict(state_dict=torch.load(f='./Models/transformer_1head_32hidden_100epochs_noReflexives.pth'))
out = manipulate_embeddings(model.state_dict())
torch.save(obj=model.state_dict(), f='./Models/transformer_1head_32hidden_100epochs_noReflexives_manipulated.pth')

# %%

corpus_train, corpus_test = create_train_corpus(corpus=corpus, excluded_females=0, exclude_men=False) # Include all reflexives
ds_train = Autoregressive_TextDataset(corpus=corpus_train)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=add_padding)

criterion = torch.nn.CrossEntropyLoss(reduction="sum")

model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32, preserve_encoding_embedding_space=True).to(device)
model.load_state_dict(state_dict=torch.load(f='./Models/transformer_1head_32hidden_100epochs_noReflexives_manipulated.pth'))
for param in model.encoder_embedding.parameters():
    param.requires_grad = False
for param in model.decoder_embedding.parameters():
    param.requires_grad = False

for param in model.encoder_pos_embedding.parameters():
    param.requires_grad = False
for param in model.decoder_pos_embedding.parameters():
    param.requires_grad = False

for param in model.transformer.encoder.parameters():
    param.requires_grad = False
# for param in model.linear_out.parameters():
#     param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

# %%
model = train(dl_train, model, optimizer, criterion, epochs=20, verbose=True)
torch.save(obj=model.state_dict(), f='./Models/transformer_1head_32hidden_100epochs_OnlyDecoderTransformerFree.pth')


# %%
 
# %%


# %%

# %%
model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32)
model.load_state_dict(state_dict=torch.load('Models/transformer_1head_32hidden_100epochs_minus5females_model0.pth'))
model = model.to(device)

# %%
def test_example(model, index, max_gen_len=10):
    model.eval()
    src, _ = ds[index]
    src = torch.cat([SOS_token*torch.ones(1), src]).to(torch.int64).unsqueeze(dim=-1).to(device)
    tgt = SOS_token*torch.ones(1, 1).to(torch.int64).to(device)

    while tgt.size(0) < max_gen_len and not (tgt[-1][0] == EOS_token):
        model_out = model(src, tgt)
        _, pred = model_out[-1].topk(1)
        pred = pred.flatten()
        
        tgt = torch.cat([tgt, pred.view(1,1)])
    
    print(f"Input: {corpus[index][0]}, Model Output: {chars_from_ids(tgt.flatten(), lang='parsed')}")

reflexives = [[pair[0] for pair in corpus].index(example) for example in [pair[0] for pair in corpus if "self" in pair[0]]]
for index in reflexives[:20]:
    test_example(model, index=index)
# %%
interesting_indexes = [0, 1, 200, 208, 964, 829, 300, 919, 663]

print("GRU MODEL PERFORMANCE:")
model = GRU_Full(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32).to(device)
model.load_state_dict(torch.load("./Models/GRU_100epochs_fulldata.pth"))
for index in interesting_indexes:
    test_example(model, index=index)
# %%
print("TRANSFORMER MODEL PERFORMANCE:")
model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32).to(device)
model.load_state_dict(torch.load("./Models/Transformer_100epochs_fulldata.pth"))

for index in interesting_indexes:
    test_example(model, index=index)

# %%
print("percent of 2 word examples:", len([pair for pair in corpus if len(pair[0].split()) < 3])/len(corpus))


# %%
print("GRU MODEL PERFORMANCE ON LIMITTED DATASET:")
model = GRU_Full(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32).to(device)
model.load_state_dict(torch.load("./Models/GRU_1000epochs_firstAndLast10Examples.pth"))
limitted_indexes = list(range(10))+list(range(len(corpus)))[-11:-2]
for index in limitted_indexes:
    test_example(model, index)

# %%
print("TRANSFORMER MODEL PERFORMANCE ON LIMITTED DATASET:")
model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32).to(device)
model.load_state_dict(torch.load("./Models/Transformer_1000epochs_firstAndLast10Examples.pth"))
limitted_indexes = list(range(10))+list(range(len(corpus)))[-11:-2]
for index in limitted_indexes:
    test_example(model, index)

# %%
def transformer_arithmetic(model, pos_examples, neg_examples, max_gen_len=10):
    def get_encoder_memory(model, index):
        src, _  = ds[index]
        src = torch.cat([SOS_token*torch.ones(1), src]).to(torch.int64).unsqueeze(dim=-1).to(device)
        src_encoded = model.embed(src, model.encoder_embedding, model.encoder_pos_embedding)

        encoder_memory = model.transformer.encoder(src_encoded)
        return encoder_memory
    
    model.eval()
    
    composite_memory = None
    for index in pos_examples:
        if composite_memory is None:
            composite_memory = get_encoder_memory(model, index)
        else:
            composite_memory += get_encoder_memory(model, index)

    for index in neg_examples:
        composite_memory -= get_encoder_memory(model, index)

    tgt = SOS_token*torch.ones(1, 1).to(torch.int64).to(device)
    while tgt.size(0) < max_gen_len and not (tgt[-1][0] == EOS_token):
        tgt_encoded = model.embed(tgt, model.decoder_embedding, model.decoder_pos_embedding)
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.size(0), device=device)

        decoder_out = model.transformer.decoder(tgt_encoded, composite_memory, tgt_mask)
        model_out = model.linear_out(decoder_out)[-1]
        pred = model_out.topk(1)[1].view(1,1)
        tgt = torch.cat([tgt, pred], dim=0)

    return chars_from_ids(tgt.flatten(), lang='parsed')


model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32).to(device)
model.load_state_dict(torch.load(f"./Models/transformer_1head_32hidden_100epochs_ForcedEncoderSolver.pth"))

# %%
model.load_state_dict(torch.load(f"./Models/transformer_1head_32hidden_100epochs_ForcedEncoderSolver.pth"))

sentences = [pair[0] for pair in corpus]
pos1 = sentences.index("Alice eats herself")
neg1 = sentences.index("Alice eats Alice")
pos2 = sentences.index("Emma sees Alice")

transformer_arithmetic(model, pos_examples=[pos1, pos2], neg_examples=[neg1])

# %%
from tqdm import tqdm
def determine_solver(model, sentences):
    encoder_solved = []
    decoder_solved = []
    neither_solved = []

    for name1 in tqdm(female_names):
        for name2 in female_names:
            for verb1 in transitive_verbs[0:1]:
                for verb2 in transitive_verbs[0:1]:
                    for name3 in female_names:
                        if not name1 == name3 :
                            pos1 = sentences.index(name1+" "+verb1+" herself")
                            neg1 = sentences.index(name1+" "+verb2+" "+name2)
                            pos2 = sentences.index(name3+" "+verb2+" "+name2)
                            output = transformer_arithmetic(model, pos_examples=[pos1, pos2], neg_examples=[neg1])
                            if output == "<SOS> "+verb1.upper()+" "+name3.upper()+" "+name3.upper()+" <EOS>":
                                decoder_solved.append((name1, name2, name3, verb1, verb2, output))
                            elif output == "<SOS> "+verb1.upper()+" "+name3.upper()+" "+name1.upper()+" <EOS>":
                                encoder_solved.append((name1, name2, name3, verb1, verb2, output))
                            else:
                                neither_solved.append((name1, name2, name3, verb1, verb2, output))
    return encoder_solved, decoder_solved, neither_solved

# %%
sentences = [pair[0] for pair in corpus]
model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32).to(device)
model.load_state_dict(torch.load(f"./Models/transformer_1head_32hidden_100epochs_OnlyDecoderTransformerFree.pth"))

# %%
encoder_solved, decoder_solved, neither_solved = determine_solver(model=model, sentences=sentences)
# %%
torch.save(obj={"encoder":encoder_solved, "decoder":decoder_solved, "neither":neither_solved}, f=f"./Experiments/071123_resultsOnlyEncoderTransformerFree.pth") #FIX 071023_resultsDecoderForced.pth
# %%
results = torch.load(f="./Experiments/071123_resultsOnlyEncoderTransformerFree.pth") #TODO: Fix this for the DecoderForced
plot_piecewise_breakdowns(results, "OnlyEncoderTransformerFree")
# %%
# THIS CODE FOR EVALUATING IF THE ENCODER/DECODER SOLVED THE TASK!
# 
# sentences = [pair[0] for pair in corpus]
# for i in range(2,5):
#     model = ED_Transformer(input_size=len(eng_vocab), output_size=len(parsed_vocab), d_model=32).to(device)
#     print(F"loading ./Models/transformer_1head_32hidden_100epochs_minus5females_model{i}.pth")
#     model.load_state_dict(torch.load(f"./Models/transformer_1head_32hidden_100epochs_minus5females_model{i}.pth"))
#     encoder_solved, decoder_solved, neither_solved = determine_solver(model=model, sentences=sentences)
#     torch.save(obj={"encoder":encoder_solved, "decoder":decoder_solved, "neither":neither_solved}, f=f"./Experiments/070423_resultsmodel{i}.pth")
# %%
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

results = torch.load(f='./Experiments/070423_resultsmodel2.pth')
data_encoder = [list(out[:-1])+["encoder"] for out in results["encoder"]]
data_decoder = [list(out[:-1])+["decoder"] for out in results["decoder"]]
data_neither = [list(out[:-1])+["neither"] for out in results["neither"]]
data = data_encoder+data_decoder+data_neither
df = pd.DataFrame(data=data, columns=["name1", "name2", "name3", "verb1", "verb2", "type"])

g = sns.histplot(data=df, x="verb2", hue="type", multiple="dodge", shrink=0.5)
g.set_xticklabels(labels=female_names, rotation=45, horizontalalignment='right')

plt.show()

# %%

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

verbs = transitive_verbs+intransitive_verbs


def plot_piecewise_breakdowns(results, model_number):
    data_encoder = [list(out[:-1])+["encoder"] for out in results["encoder"]]
    data_decoder = [list(out[:-1])+["decoder"] for out in results["decoder"]]
    data_neither = [list(out[:-1])+["neither"] for out in results["neither"]]
    data = data_encoder+data_decoder+data_neither
    df = pd.DataFrame(data=data, columns=["name1", "name2", "name3", "verb1", "verb2", "type"])

    fig, axes = plt.subplots(3,2)
    name1_plot = sns.histplot(data=df, x="name1", hue="type", multiple="dodge", shrink=0.5, ax=axes[0,0], legend=False)
    name1_plot.set_xticklabels(labels=female_names, rotation=45, horizontalalignment='right')

    name2_plot = sns.histplot(data=df, x="name2", hue="type", multiple="dodge", shrink=0.5, ax=axes[1, 0], legend=False)
    name2_plot.set_xticklabels(labels=female_names, rotation=45, horizontalalignment='right')

    name3_plot = sns.histplot(data=df, x="name3", hue="type", multiple="dodge", shrink=0.5, ax=axes[2, 0], legend=False)
    name3_plot.set_xticklabels(labels=female_names, rotation=45, horizontalalignment='right')

    verb1_plot = sns.histplot(data=df, x="verb1", hue="type", multiple="dodge", shrink=0.5, ax=axes[0,1], legend=False)
    verb1_plot.set_xticklabels(labels=verbs, rotation=45, horizontalalignment='right')

    verb2_plot = sns.histplot(data=df, x="verb2", hue="type", multiple="dodge", shrink=0.5, ax=axes[1,1])
    verb2_plot.set_xticklabels(labels=verbs, rotation=45, horizontalalignment='right')

    sns.set(rc={'figure.figsize':(75,20)})
    sns.move_legend(verb2_plot, "upper right", bbox_to_anchor=(1.16, 1), fontsize=50)


    plt.suptitle(f'Model{model_number}: Number of female examples solved by encoder/decoder,\n split up by name1/name2/name3.', size=30)
    plt.tight_layout()
    fig.savefig(f'./Figures/Experiment_Results_071023_model_{model_number}_breakdown', bbox_inches='tight')
# %%
results = torch.load(f=f'./Experiments/070323_resultsmodel0.pth')
plot_piecewise_breakdowns(results, model_number=0)
for i in range(1,5):
    results = torch.load(f=f'./Experiments/070423_resultsmodel{i}.pth')
    plot_piecewise_breakdowns(results, model_number=i)

# %%
clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(X,y)
tree.plot_tree(clf)

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
# TODO:

# Run the encoder/decoder forced 5 times each. Freeze all but the transformer encoder/decoder



# Run the fmale name tests for the male arithmetics, see if they distributions are similar

# Run GRU decoder only, see if it generalized (compare to transformer decoder only)


# Train a transformer on limitted training set, and 
# See if the encoder/decoder is getting the credit on test examples
# See what of name1,name2,name3,verb1,verb2 has the biggest effect on errors in transformer arithmetic also biggest effect on encoder/decoder. USE DECISION TREE
# CAREFUL- There are two reflexive sentences at play
# Rerun generalization experiments with positional encoding

# Try training a decoder/encoder WITHOUT HIM/HERSELF. Then add in the him/herself examples, but only train the encoder/decoder, keeping the alternate part the same/fixed parameters

# Implement decoder only


# Increase the number of excluded female names
# Compare cosine similarities of himself/herself encoder embeddings
# Experiment with hidden dimension size
# Including male names/not in training
# Implement encoder/decoder transformer, run same code
# Implement decoder only transformer
# Try GPT2
# %%

# Reach out to Michael Wilson about Github issues (michael.a.wilson@yale.edu)