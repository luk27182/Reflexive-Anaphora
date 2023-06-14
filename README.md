# Reflexive-Anaphora

## The Dataset

In this repository, we explore how neural net architectures are able to learn the parsing task relating to reflexive anaphora proposed by Jackson Petty and Robert Frank in their [2020 paper](https://arxiv.org/abs/2011.00682).

The dataset consists of 5122 basic english sentences, all of the form "SUBJ VERB" or "SUBJ VERB OBJ". The sentences are generated exhaustively from a list of 11 male names, 15 female names, 7 transitive verbs, and 8 transitive verbs. Importantly, the dataset also includes all possible sentenses where the object of a transitive verb is "himself" or "herself" (these two words are called "reflexive anaphora"). The goal of the model is to produce parsings of these english phrases. For example, the model should parse the English phrase "Alice sees Bob" as "SEES(ALICE, BOB)." Sentences with reflexive anaphora such as "Alice sees herself" are parsed "SEES(Alice, Alice)." For simplicity, we have removed parantheses from the dataset. Below are some examples from the dataset.
| MODEL INPUTS       | MODEL OUTPUTS    |
|--------------------|------------------|
| Alice kicks        | KICKS ALICE      |
| Alice kicks Bob    | KICKS ALICE BOB   |
| Alice kicks herself | KICKS ALICE ALICE |

It is trivially easy for most model architectures to solve the task if trained on the entire training set. What we are interested in is rather the generalization capabilities of the model. For example, one would hope that even if all sentences of the form "ALICE VERB herself" were removed from the training set, the model should still be able to learn to parse such sentences correctly. If this occured, it would be evidence that the model is not just memorizing the training data but rather understanding that "herself" refers to a copying operation. Indeed we (as well as Petty) find that the model is able to generalize in this way.

## GRU Model

### Experiments Run 06-13-23 and 06-14-23
**ARCHITECTURE:** All the following experiments used the same GRU encoder-decoder architecture. The encoder consisted of an embedding layer (embedding dimension 32) and a single [PyTorch GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) (hidden dimension 32). The final output of the GRU was the output of the encoder. The decoder consisted of an embedding layer (embedding dimension 32), followed by a single GRU layer (hidden dimension 32), followed by a linear layer to project the GRU output to a distribution the size of the parsed english vocabulary (in this case, a projection from 32 dimensions to 45 dimensions). The output of the decoder was thus the logits for the next likely word in the parsed output.

**TRAINING:** The initial input to the decoder was always special "\<START\>" character. For all future inputs to the decoder, we adopted a teacher forcing training method with a ratio of 0.5. We used [cross entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) for the loss function, which allowed us to use the logit outputs of the model directly. The optimizer for both the encoder and the decoder was [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with default settings. For all models trained in this experiment, we trained over the shuffled data for 15 epochs with a batch size of 32. For the results below, we trained each model in this way 20 times to get accurate confidence intervals.

**EXPERIMENTS** The first experiment we ran was letting the training set be all sentences except those of the form "WITHHELD-NAME VERB herself" for all WITHHELD-NAMEs from specific sets of female names. We then tested the accuracy of the model on these withheld sentences. The graph to the left below shows the accuracy on the test set for excluding 1, 5, 10, 11, 12, 13, and 14 female names in this way. For the second experiment, we did the same exact test except we removed all sentenes of the form "MALE-NAME VERB himself" from the dataset (both train and test.) Before running the experiment, we hypothesise that:
1. In both experiments, we will see a negative correlation between the number of withheld female names in the training set and the test accuracy on these withheld names
2. We will see generally lower accuracies (and perhaps a sharper decline) in the second experiment. This is because we predict that the model is able to learn that "himself" and "herself" mean basically the same thing, so learning about "himself" helps the model learn about "himself."
<p align="center">
    <img height="500" src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_061423-Removing_Female_Names.png?raw=true" alt="Experiment Results">
    <img height="500" src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_061423-Removing_Female_Names-WITHOUT_HIMSELF.png?raw=true" alt="Experiment Results">
</p>

<img src="https://github.com/favicon.ico" width="48">`
