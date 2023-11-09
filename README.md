# Reflexive-Anaphora

*This research was conducted by Luke Reynolds under Dr. Robert Frank through funding from Yale University.*

**DESCRIPTION:** In recent years, large language models (LLMs) have seen incredible gains in capabilities. Despite the success of models such as GPT4, however, modern LLMs remain almost entirely black boxes, raising concerns about their safety and robustness. One emerging approach to understand these models is the field of Mechanistic Interpretability (MI), which aims to deconstruct the specific ``circuits" (i.e. learned algorithms) which trained transformer models use to complete a certain task. In this repo, we aim to uncover the circuits that allow decoder-only transformer models to complete linguistic tasks related to reflexive anaphora. We will first consider a small toy model which we will train ourselves on a dataset specific to the task. After analyzing the circuits within this toy model, we will move on to a more realistic model, GPT-small, and adjust the task to make it more comparable to naturally occurring text. Though a small step, our research will bring us closer to understanding how LLMs such as GPT4 reason, which will aide the future development of transformer research.

In this repository, we explore how neural net architectures are able to learn the parsing task relating to reflexive anaphora proposed by Jackson Petty and Robert Frank in their [2020 paper](https://arxiv.org/abs/2011.00682). The dataset consists of 5122 basic english sentences, all of the form "SUBJ VERB" or "SUBJ VERB OBJ". The sentences are generated exhaustively from a list of 11 male names, 15 female names, 7 transitive verbs, and 8 transitive verbs. Importantly, the dataset also includes all possible sentenses where the object of a transitive verb is "himself" or "herself" (these two words are called "reflexive anaphora"). The goal of the model is to produce parsings of these english phrases. For example, the model should parse the English phrase "Alice sees Bob" as "SEES(ALICE, BOB)." Sentences with reflexive anaphora such as "Alice sees herself" are parsed "SEES(Alice, Alice)." For simplicity, we have removed parantheses from the dataset.

## Encoder-Decoder Models
### Generalization of GRU Encoder-Decoder models
To confirm we are running experiments comparable to their work, we began be re-running their experiments on the generalization of GRU encoder-decoder models. The encoder consisted of an embedding layer (embedding dimension 32) and a single PyTorch GRU (hidden dimension 32). The final output of the GRU was the output of the encoder. The decoder consisted of an embedding layer (embedding dimension 32), followed by a single GRU layer (hidden dimension 32), followed by a linear layer to project the GRU output to a distribution the size of the parsed English vocabulary (in this case, a projection from 32 dimensions to 45 dimensions). The output of the decoder was thus the logits for the next likely word in the parsed output. The initial input to the decoder was always special "$START$" character. For all future inputs to the decoder, we adopted a teacher forcing training method with a ratio of 0.5. We used cross entropy for the loss function, which allowed us to use the logit outputs of the model directly. The optimizer for both the encoder and the decoder was Adam with default settings. For all models trained in this experiment, we trained over the shuffled data for 15 epochs with a batch size of 32. For the results below, we trained each model in this way 20 times to get accurate confidence intervals.

The first experiment we ran was letting the training set be all sentences except those of the form "WITHHELD-NAME VERB herself" for all WITHHELD-NAMEs from specific sets of female names. We then tested the accuracy of the model on these withheld sentences. The graph to the left below shows the accuracy on the test set for excluding 1, 5, 10, 11, 12, 13, and 14 female names in this way. For the second experiment, we did the same exact test except we removed all sentences of the form "MALE-NAME VERB himself" from the dataset (both train and test.) Before running the experiment, we hypothesize that:
- In both experiments, we will see a negative correlation between the number of withheld female names in the training set and the test accuracy on these withheld names
- We will see generally lower accuracies (and perhaps a sharper decline) in the second experiment. This is because we predict that the model is able to learn that "himself" and "herself" mean basically the same thing, so learning about "himself" helps the model learn about "himself."

<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/GRU_combined.png" alt="Experiment Results">
</p>
Looking at these graphs, we see that the first of our hypotheses was indeed correct- the more female names we exclude, the harder it is for the model to generalize. The second hypothesis is less obviously true. Excluding the male name examples only decreased test generalization very slightly. This gives some evidence that the GRU model learns a different circuit for dealing with male and female reflexive anaphora, which is surprising.

We note that these graphs are essentially in agreement with past work.

### Generalization of Transformer Encoder-Decoder Models
We continued by running the same experiments as before, but now using a transformer encoder-decoder model rather than the GRU. This extends previous work in a new direction. The encoder consisted of an embedding layer (embedding dimension 32) and a Pytorch Transformer module (2 heads, a single layer, and a linear layer with width 128). The decoder consisted of another embedding layer (embedding dimension 32) along with a final linear layer to project the decoder output to the size of the parsed english vocabulary (in this case, a projection from 32 dimensions to 45 dimensions). The output of the decoder was thus the logits for the next likely word in the parsed output. The initial input to the decoder was always special "$START$" character. For all future inputs to the decoder, we adopted a teacher forcing training method with a ratio of 0.5. We used cross entropy for the loss function, which allowed us to use the logit outputs of the model directly. The optimizer for both the encoder and the decoder was Adam with default settings. For all models trained in this experiment, we trained over the shuffled data for 5 epochs with a batch size of 32. For the results below, we trained each model in this way 20 times to get accurate confidence intervals.

Before running the experiment, we hypothesize that the results will be essentially identical to the previous experiments on the GRU architecture. As we increase the number of excluded female names, the test accuracy will decrease. Excluding male names will have no detectable effect.

<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/Transformer_Combined.jpg" alt="Experiment Results">
</p>

Looking at the results, they are essentially what we predicted. We note that the transformer model was able to generalize much better than the GRU model, especially when the training set is more restricted. Another thing to note is that the transformer encoder-decoder has roughly twice as many parameters as the GRU encoder-decoder (34253 versus 17101).

### "Encoder" vs "Decoder" Based Transformer Solutions
When training these encoder-decoder models, the natural question is to ask whether it is the encoder or the decoder which "solved the task" of reflexive anaphora. That is, we might wonder if the embedding of the sentence "Alice sees herself" is encoded as "Alice sees herself" or as "Alice sees Alice". In the former case, it is clear that the encoder does not understand what "herself" means, functionally. Hence it is the decoder that is "solving the task". In the latter case, the encoder is "solving the task." For a given model, we can do vector arithmetic on the embedding of an example sentence to investigate whether the encoder or decoder of a model is solving the task as follows:

Begin with the embedding for a sentence of the form "[name1] [verb1] 'herself'." Then, subtract the embedding for a sentence of the form "[name1] [verb2] [name2]." Finally, add the embedding for a sentence of the form "[name3] [verb2] [name2]." After performing this arithmetic, we get a new embedding vector which can be fed into the decoder of the model. 

If the reconstructed sentence corresponds to "[name3] [verb1] [name3]", then it follows that the encoding of "herself" does not understand that herself=[name1] in the original sentence "[name1] [verb1] herself." Hence, in this case, the model acted as a decoder-solving model.

If instead the reconstructed sentence corresponds to "[name3] [verb1] [name1]", then it follows that the encoding of "herself" \textit{does} understand that herself=[name1] in the original sentence "[name1] [verb1] herself." Hence, in this case, the model acted as a encoder-solving model.

This type of analogy was investigated previously in the case of encoder-decoder RNN and GRU models. To expand the work to encoder-decoder transformers, we had to ensure that we were applying the arithmetic to the encoding at every time-step (i.e., each token). This required going into the internals of the PyTorch transformer architecture, but was otherwise fairly routine.

### Encoder vs Decoder Transformers
To begin this experiment, we trained five encoder-decoder transformer models. All these models had a single layer and a single attention head, with a hidden dimension size of 32. There was an embedding layer to project the one-hot encoding of the input sequence to the hidden dimension size, followed by an nn.Transformer layer, followed by a final linear projection to the output vocabulary size. All the models were trained for 150 epochs on the same training dataset. This training dataset consisted of nearly all corpus examples, but with all examples of the form "[Exclude Female Name] [Transitive Verb] herself" excluded, where there were exactly five female names excluded in this manner.

For all five of these models trained previously, we ran the arithmetic described in 4.2 for all possible [name1], [name2], [name3] in the female names list (15 names) as well as all possible transitive verbs (7 verbs). HOWEVER- we did not include examples when name1 == name3, as in these cases a decoder-solver and an encoder-solver would produce the same arithmetic result. In the histograms below, we show the distributions of how many examples are solved by the encoder/decoder/neither conditional on specific parts of the sentence. The top left chart in the figures below show the distribution conditional on what name1 was in the analogies. Below this is the distribution conditional on name2, and below this the distribution conditional on name3. The top right chart in the figures below show the distribution conditional on what verb1 was in the analogies. Below this is the distribution conditional on verb2.

<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/bars1.png">
</p>
Remarkably, this first model seemed to solve every example using the decoder. We would hope
that all models would be this clear cut, but alas.

<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/bars2.png">
</p>
<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/bars3.png">
</p>
In the second two models cases, we see that the arithmetic does not work as ideally. Sometimes
the model seems to be using the encoder to solve reflexive cases, and other times the model seems to
use the decoder. In these models, it is clear that the decoder is predominantly used. We also see that the
distributions are quite different when we condition on name1 and name3, but are essentially unchanging
when we change name2, verb1, and verb2. This makes some sense since the parsed model output is
presumably one of ”[verb1] [name3] [name3]” or ”[verb1] [name3] [name1]” and these only differ by a
change in name1 and name3. Hence it would be very strange if the distributions changed depending on
name2, verb1, or verb2. It is still surprising to me that the name involved changes the distribution, as
one would hope that the model is learning something algorithmic rather than anything specific to certain
names

<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/bars4.png">
</p>
<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/bars5.png">
</p>

These results are essentially the same as the previous example. One interesting difference is that
these two models preferred to use the decoder to solve the arithmetic examples rather than the encoder.
Another interesting result here is that in the case of model 4, it seems that name2 also had a large effect
on what was used to solve the arithmetic. This is surprising. I attribute it to most likely being a quirk of
the model being used in this way (it was not trained to do arithmetic).

### Manually Creating Encoder/Decoder Solving Transformers

Prior to running the experiment 4.2.1, we hoped that each model would consistently function as either a “encoder solving” model or a “decoder solving” model. We instead found that almost none of the models which we trained were “consistent” in this way. Each model tended to function as a “encoder solving” model in some cases and as a “decoder solving” model in others. This was troubling from our goals of understanding how the models function, as it implied that the model used radically different algorithms given different sentences to parse.\\

To confirm the validity of these sentence level analogy results, we developed our own method of creating “decoder solving” and “encoder solving” models. This was accomplished in 4 steps:

First, we train an encoder-decoder transformer model for 50 epochs, with the essentially the same hyper parameter settings as in experiment 4.2.1. There was two key differences in the training of these new models. Firstly, in these models, the training dataset consisting of (and only of) the examples in the training corpus which did not include a reflexive. Thus, these models had no idea how to solve reflexive examples. Secondly, we forced the encoder embedding for these models to leave the final dimension of the encoding space to be "0". After training, we manually changed the encoder embeddings for "himself" and "herself" to be a one-hot encoding which was all 0s except for a 1 in the final dimension spot. This allowed the model to embed reflexive examples in a dimension completely orthogonal to everything else it encodes. We saved the state dictionaries of these model weights in for later use, calling it the "base model."

From the base model, we were able to create a new model which was guaranteed to use the encoder to solve the reflexive as follows. First, we loaded in the base model created in step 1. We then froze all weights of the model except for those in the transformer module's encoder (NOTE: the weights for the encoder embedding WERE frozen. We found this to be a necessary step to get the model to actually use the encoder to solve the reflexive examples.) We them trained these models on the entire corpus (including reflexives) for 25 epochs each. The state dictionaries of these models were saved and called the "encoder forced" models.

We then did essentially the same thing to create decoder forced models. First we load in the base models' state dictionaries from step 1. Then we freeze all weights except those in the transformer's decoder (NOTE: We also froze the final linear projection after the transformer decoder layers.) We trained these model on the entire corpus as well for 25 epochs each. The state dictionaries of these models were saved and called "decoder forced models."

We now have 5 models which we know use the encoder to solve reflexive cases, and 5 models which we know use the decoder to solve reflexive cases. Hence, we can test whether our transformer-arithmetic approach proposed in 4.2.1 were actually valid. To do this, we ran (almost) the same experiments as in 4.2.1 on these new models. The one change which we made was that we only looked at training examples where (to use notation from that experiment) verb1=verb2="eats", because we found in 4.2.1 that verb1 and verb2 seemed to have essentially no affect on how the model classified the examples.

Below are the results (NOTE: percents are based off of 3150 example analogies. This is 15\*\*3 combinations for name1,name2,name3 minus 15\*\*2 combinations of name1,name2,name3 when name1==name3):

<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/forced_table.png">
</p>
The results are shockingly clear cut! This supports that our idea of using transformer arithmetic
to figure out which part of the model was solving the task.

## Decoder Only Models
Following the disheartening results from section 4.2 (disheartening from a standpoint of interpretability), we decided to move on to analyzing decoder-only transformer models. which we hoped would be easier to interpret. This is the type of architecture used by models such as ChatGPT.

To extend the task to decoder only transformers, we ask the model to predict the next token (word) in sentences such "Alice saw herself $UNK$ SAW ALICE ALICE $EOS$." We then only compute the loss on tokens after the $UNK$ token. Besides this formatting change, the dataset is essentially the same as before.

### Decoding a Successful Model

We implemented an architecture similar to GPT2 (though much smaller) and obtained successful generalization to an excluded female names (perfect test accuracy). Following this, we attempted to perform mechanistic interpretability on this model.

####  Investigating Attention Patterns
First, we looked at the attention patterns for sample inputs.

<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/allpatterns.png">
</p>

We noticed that the attention patterns were looked mostly equivalent on any given layer, so we include only one head of each layer above.

<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/specific_patterns.png" alt="Experiment Results">
</p>

The interpretation of the bright 0.c square above indicates that this component (layer 0 head 0) is paying attention to the word “Alice” when it is analyzing the word “herself”. Hence, it seems possible that this component of the model understands that “herself” corresponds to whatever the word “Alice” means! Most other heads in this layer have similar attention patterns.

The uniform triangle in the center image above shows that this head is paying the same attention to all tokens. Hence, this head is essentially doing nothing. Surprisingly, every head in this layer was the same- they did not contribute to the computation.

The bright 2.c square in the right image above shows that the model is looking at what the word “herself” encodes to determine what the direct object of the sentence is. Combined with the pattern of layer 0 head 0 described earlier, we propose that layer 0 essentially transforms the vector representation corresponding to the reflexive anaphora into their antecedent, allowing later layers to simply read off the results. This is the type of human-understandable algorithm we hoped to find! Indeed, this "reading off the result" is confirmed even more by the bright 2.a and 2.b squares.

#### PCA clustering after layer 0

To provide more evidence for the theory described in 5.1.1 (namely, that layer 0 essentially transforms the vector representation corresponding to the reflexive anaphora into their antecedent), we ran the model over all examples of the form “[name] [verb] herself” for some small selection of names, caching the internal states of the model for each example. We then used PCA to project all the internal states corresponding to the reflexive anaphora after the first layer of computation into a 2D plot.
<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/pca2.png">
</p>
We see that the residual stream at the timestep of the reflexive after the first layer results in the points being "mapped to" the noun the correspond to, with a relatively constant vector added on. Thus, we have supported the theory that the first layer is what solves the reflexive anaphora problem.

For reference, we plot the same PCA plot where we instead probe the initial embeddings:
<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/README_figures/pca1.png">
</p>
As one would expect, all the reflexives are overlapping and the other nouns are spread out on the other side of the plot.
