# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

intransitive_verbs = ['runs', 'sleeps', 'laughs', 'cries', 'talks', 'jumps', 'dances', 'sings']
transitive_verbs = ['eats', 'sees', 'hugs', 'paints', 'kicks', 'throws', 'compliments']
female_names = ['Alice', 'Emma', 'Olivia', 'Ava', 'Isabella', 'Sophia', 'Mia', 'Charlotte', 'Amelia', 'Harper', 'Evelyn', 'Abigail', 'Emily', 'Elizabeth', 'Mila']
male_names = ['Bob', 'John', 'Noah', 'Oliver', 'Elijah', 'William', 'James', 'Benjamin', 'Lucas', 'Henry', 'Michael']
# male_names = []

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
    
ds = Autoregressive_TextDataset(corpus_train)

def add_padding(batch):
    original_raw = [pair[0] for pair in batch]
    original_parsed = [pair[1] for pair in batch]

    padded_raw = pad_sequence(original_raw, padding_value=0)
    padded_parsed = pad_sequence(original_parsed, padding_value=0)
    return (padded_raw, padded_parsed)

dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, collate_fn=add_padding)
for batch in dl:
    break
print("Batch has shape length by batch:", batch[0].shape)
# %%
from torch import nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)


    def forward(self, input):
        embedded = self.embedding(input)
        _, gru_out = self.gru(embedded)
        return gru_out
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        _, gru_out = self.gru(embedded, hidden)
        linear_out = self.linear(gru_out)

        return gru_out, linear_out


# %%
import random
SOS_token = parsed_vocab.index("<SOS>") # 1
EOS_token = parsed_vocab.index("<EOS>") # 3

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=0.5):
    
    target_length = target_tensor.size(0)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    encoder_outputs = encoder(input_tensor)
    decoder_hidden = encoder_outputs

    decoder_input = SOS_token * torch.ones(1, encoder_outputs.shape[1]).to(torch.int32)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for index in range(target_length):
        decoder_hidden, decoder_output = decoder(input=decoder_input.view(1, -1), hidden=decoder_hidden)
        loss += criterion(decoder_output.squeeze(0), target_tensor[index])

        if use_teacher_forcing:
            decoder_input = target_tensor[index]
        else:
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# %%
from torch import optim
from tqdm import tqdm

def test_example(input_tensor, encoder, decoder):
    generated_output = []

    encoder_outputs = encoder(input_tensor)
    decoder_hidden = encoder_outputs

    decoder_input = SOS_token * torch.ones(1, encoder_outputs.shape[1]).to(torch.int32)

    for index in range(3):
        decoder_hidden, decoder_output = decoder(input=decoder_input.view(1, -1), hidden=decoder_hidden)

        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        generated_output.append(decoder_input)

    return generated_output


def train_on_corpus(corpus_train, corpus_test, num_epochs=15, verbose=False, n=20, hidden_size = 32):
    assert len(corpus_test) != 0

    ds = Autoregressive_TextDataset(corpus_train)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, collate_fn=add_padding)

    test_accuracies = []
    for i in range(n):
        encoder = EncoderRNN(input_size=len(eng_vocab), hidden_size=hidden_size)
        decoder = DecoderRNN(hidden_size=hidden_size, output_size=len(parsed_vocab))

        encoder_optimizer = optim.Adam(encoder.parameters())
        decoder_optimizer = optim.Adam(decoder.parameters())

        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            for batch in dl:
                input_tensor = batch[0]
                target_tensor = batch[1]
                loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            if verbose:
                print(f"epoch: {epoch+1} loss: {loss:.4f}")

        scores = []
        for example_pair in corpus_test:
            input_tensor = ids_from_chars(example_pair[0], lang="eng").view(-1, 1)
            predicted_target = test_example(input_tensor, encoder, decoder)
            if example_pair[1] == chars_from_ids(predicted_target, lang='parsed'):
                scores.append(1)
            else:
                scores.append(0)
            
        test_accuracies.append(sum(scores)/len(scores))

        if verbose:
            for i in range(10):
                example_pair = random.choice(corpus_test)
                input_tensor = ids_from_chars(example_pair[0], lang="eng").view(-1, 1)
                predicted_target = test_example(input_tensor, encoder, decoder)
                print(f"{example_pair[0]} <-> {chars_from_ids(predicted_target, lang='parsed')}") 
                

    return test_accuracies

results = dict()
experiments = [1, 5, 10, 11, 12, 13, 14]
for n in experiments:
    corpus_train, corpus_test = create_train_corpus(corpus, excluded_females=1)
    new_results = train_on_corpus(corpus_train, corpus_test)
    print(f"Excluding {n} FEMALE names, test accuracy was {new_results}")
    results[f"excluding {n} females, including all males"] = new_results


# %%

# TODO:
# Increase the number of excluded female names
# Compare cosine similarities of himself/herself encoder embeddings
# Experiment with hidden dimension size
# Including male names/not in training
# Implement encoder/decoder transformer, run same code
# Implement decoder only transformer
# Try GPT2

# %%
