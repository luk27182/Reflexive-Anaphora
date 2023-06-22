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
Experiments: 061423


## Encoder-Decoder Transformer
Experiments: 062023