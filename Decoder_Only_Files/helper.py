# %%
import torch
import torch
import numpy as np
import einops
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

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


special_vocab = ["<PAD>", "<SOS>", "<UNK>", "<EOS>"]

vocab = special_vocab+sorted(list(set(verbs+names+["himself", "herself"]+[verb.upper() for verb in verbs]+[name.upper() for name in names])))
def ids_from_chars(str):
    return torch.tensor([vocab.index(word) for word in str.split()])

def chars_from_ids(tensor):
    return " ".join([vocab[idx] for idx in tensor])

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, corpus):
        super().__init__()        
        self.corpus = corpus

    def __len__(self):        
        return len(self.corpus)

    def __getitem__(self, idx):
        return ids_from_chars(self.corpus[idx])

# %%
def train_model(model, ds_train, ds_test, num_epochs, print_every=5, batch_size=32):
    device = next(model.parameters()).device

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


        if True: #epoch+1 % print_every == 0:
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

def test_index(model, example, max_length=10, verbose=True, be_safe=False):
    device = next(model.parameters()).device
    model.eval()

    if verbose: print(f"===== TESTING NEW EXAMPLE =====")
    
    if be_safe:
        break_point = example.tolist().index(2)+1

        input = example[:break_point]
        output = example[break_point:]
        if verbose: print(f"input: {chars_from_ids(input)}")
        if verbose: print(f"goal: {chars_from_ids(output)}")

        input = input.view(-1, 1)
        while len(input) < max_length and input[-1][0] != 3:
            model_out  = model(input)
            _, new_out = model_out[0][-1].topk(1)
            input = torch.cat([input, torch.tensor([new_out]).to(device).view(1, 1)])
        
        if verbose: print(f"model out: {chars_from_ids(input.flatten()[break_point:])}")
    
        return chars_from_ids(output) == chars_from_ids(input.flatten()[break_point:])
    else:
        break_point = example.tolist().index(2)
        model_out = model(example[:-1].view(-1, 1))
        _, pred = model_out[0].topk(1)
        target = example[break_point+1:]
        if verbose:
            print(f"input: {chars_from_ids(example[:-1].view(-1, 1))}")
            print(f"model out: {chars_from_ids(pred.flatten()[break_point:])}")

        return torch.all(pred.flatten()[break_point:] == target.flatten()).item()
    

def plot_pattern(model, example):
    test_index(model, example=example, verbose=False, be_safe=True)
    fig, axs = plt.subplots(model.n_layers, model.n_heads)
    for i in range(model.n_layers):
        for j in range(model.n_heads):
            axs[i, j].imshow(model.cache[f'l{i}h{j}_pattern'][0].to('cpu'), cmap='hot')
            axs[i, j].set_xticks(range(len(example)), chars_from_ids(example).split(" "), rotation=90)
            axs[i, j].set_yticks(range(len(example)), chars_from_ids(example).split(" "))
            axs[i, j].set_title(f'l{i}h{j} pattern')
            # axs[i, j].colorbar()
    fig.set_size_inches(25, 20)
    plt.show()


def read_probes(model, examples, probe_location, lex_loc="o"):
    assert lex_loc in ["s", "v", "o"], "invalid lexical location. Must be s, v, or o."
    results = []
    for example in examples:
        assert len(example) == 9, "invalid example, must be intransitive example."

        if lex_loc == "s":
            split_point = 1
        elif lex_loc == "v":
            split_point = 2
        else: # lex_loc == o
            split_point = 3
        
        test_index(model, example, verbose=False)

        if probe_location >= 0:
            results.append(np.array(model.cache[f"resid_postBlock_{probe_location}"][0][split_point].to("cpu")))
        else:
            results.append(np.array(model.cache[f"resid_initial"][0][split_point].to("cpu")))
    return np.array(results)

def run_PCA_analysis_by_name(model, num_names, probe_location, lex_loc):
    corpi = dict()
    all_results = None
    labels = None

    for n in range(num_names):
        name = female_names[n]
        examples = [ids_from_chars(example) for example in corpus if example.split()[1] == name and example.split()[3] in ["herself"]]
        results = read_probes(model, examples, probe_location=probe_location, lex_loc=lex_loc)
        if not all_results is None:
            all_results = np.concatenate([all_results, results])
            labels += [n for i in examples]
        else:
            all_results = results
            labels = [n for i in examples]


    for n in range(num_names):
        name = female_names[n]
        examples = [ids_from_chars(example) for example in corpus if example.split()[1] == name and example.split()[3] in ["herself"]][0:1]
        result = read_probes(model, examples, probe_location=probe_location, lex_loc='s')
        all_results = np.concatenate([all_results, result])
        labels += [n]

    pca = PCA()
    pca.fit(all_results)
    pca_results = pca.transform(all_results)

    plt.scatter(pca_results[:-num_names, 0], pca_results[:-num_names, 1], c = labels[:-num_names], alpha=0.1)
    plt.scatter(pca_results[-num_names:, 0], pca_results[-num_names:, 1], c = labels[-num_names:],  marker="^")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

def run_PCA_analysis_by_verb(model, num_verbs, probe_location, lex_loc):
    corpi = dict()
    all_results = None
    labels = None

    for n in range(num_verbs):
        verb = transitive_verbs[n]
        examples = [ids_from_chars(example) for example in corpus if example.split()[2] == verb and example.split()[3] in ["herself"]]
        results = read_probes(model, examples, probe_location=probe_location, lex_loc=lex_loc)
        if not all_results is None:
            all_results = np.concatenate([all_results, results])
            labels += [n for i in examples]
        else:
            all_results = results
            labels = [n for i in examples]


    for n in range(num_verbs):
        verb = transitive_verbs[n]
        examples = [ids_from_chars(example) for example in corpus if example.split()[2] == verb and example.split()[3] in ["herself"]][0:1]
        result = read_probes(model, examples, probe_location=probe_location, lex_loc='v')
        all_results = np.concatenate([all_results, result])
        labels += [n]

    pca = PCA()
    pca.fit(all_results)
    pca_results = pca.transform(all_results)

    plt.scatter(pca_results[:-num_verbs, 0], pca_results[:-num_verbs, 1], c = labels[:-num_verbs], alpha=0.1)
    plt.scatter(pca_results[-num_verbs:, 0], pca_results[-num_verbs:, 1], c = labels[-num_verbs:],  marker="^")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

# %%
