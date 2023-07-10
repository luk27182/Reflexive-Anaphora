# Experiment Description
First, we trained five encoder-decoder transformer models. All these models had a single layer and a single attention head, with a hidden dimension size of 32. There was an embedding layer to project the one-hot encoding of the input sequence to the hidden dimension size, followed by an nn.Transformer layer, followed by a final linear projection to the output vocabulary size. All the models were trained for 150 epochs on the same training dataset. This training dataset consisted of nearly all corpus examples, but with all examples of the form "[Excluded_Female_Name] [Transitive_Verb] herself" excluded, where there were exactly five female names excluded in this manner.

To each of these models, we ran a series of transformer arithmetic experiments. Specifically, we added/subtracted the encoder outputs (including intermediate outputs) for three examples of the following form:
(+) [name1] [verb1] "herself"
(-) [name1] [verb2] [name2]
(+) [name3] [verb2] [name2]
The goal of such arithmeitc is as follows: If we naively inspect the arithmetic and see what cancels out, we get the encoding "[name3] [verb1] herself". Importantly, however, the model might encode "herself" as simply [name1], as this is the semantic interpretation. Therefore, we might also expect to get the encoding for "[name3] [verb1] [name1]." Indeed, it seems intuitively clear (though I hope to run an experiment to confirm this) that if the *encoder* is doing the work for interpretting the meaning of "herself", then we should expect the encoding for "[name3] [verb1] [name1]". If the encoder is just copying the information directly and instead the *decoder* is doing the work for interpretting the meaning of "herself", then we should expect the encoding for "[name3] [verb1] herself". We We can directly test to see what this transformer arithmetic encoding represents by plugging the result into the decoder.

For all five of the models trained previously, we ran the above arithmetic for all possible [name1], [name2], [name3] in the female_names list (15 names) as well as all possible transitive verbs (7 verbs). HOWEVER- we did not include examples when name1 == name3, as in these cases a decoder-solver and an encoder-solver would produce the same arithmetic result. This gave us five large dictionaries, where the key was "encoder", "decoder", or "neither", and the value was a list of tuples, each tuple of the form ([name1], [name2], [name3], [verb1], [verb2]). This sorted all examples in a way that made it clear how that example was decoded. This made it easy to see how many examples were solved by the encoder/decoder or neither. Additionally, it allowed us to further refine our chart to, for example, see the split for how many examples were solved by the encoder/decoder/neither when we only focus on examples which include the name "Alice". In the histograms below, we show the distributions of how many examples are solved by the encoder/decoder/neither conditional on specific parts of the sentence (for example, the top right chart in all of the below figures show the distributions conditional on a specific name1 being used in the sentence.)

# Results for model 0:
<p align="center">
    <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_070423_model0_breakdown.png">
</p>
Remarkably, this  first model seemd to solve *every* example using the decoder. We would hope that all models would be this clear cut, but alas.

# Results for model 1, model 2:
<p align="center">
  <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_070423_model1_breakdown.png">
  <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_070423_model2_breakdown.png">
</p>
In these cases, we see that the arithmetic does not work as ideally. Sometimes the model seems to be using the encoder to solve reflexive cases, and other times the model seems to use the decoder. In these models, it is clear that the decoder is predominantly used. We also see that the distributions are quite different when we condition on name1 and name3, but are essentially unchanging when we change name2, verb1, and verb2. This makes some sense since the parsed model output is presumably one of "[verb1] [name3] [name3]" or "[verb1] [name3] [name1]" and these only differ by a change in name1 and name3. Hence it would be very strange if the distributions changed depending on name2, verb1, or verb3. It is still surprising to me that the name involved changes the distribution, as one would hope that the model is learning something algorithmic rather than anything specific to certain names.

# Results for model 3, model 4:
<p align="center">
  <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_070423_model3_breakdown.png">
  <img src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_070423_model4_breakdown.png">
</p>
These results are essentially the same as the previous example. One interesting difference is that these two models prefered to use the decoder to solve the arithmetic examples rather than the encoder. Another interesting result here is that in the case of model 4, it seems that name2 also had a large effect on what was used to solve the arithmetic. This is surprising. I attribute it to most likely being a quirk of the model being used in this way (it was not trained to do arithmetic).
