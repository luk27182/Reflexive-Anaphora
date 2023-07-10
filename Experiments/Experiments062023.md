### Experiments Run 06-19-23 and 06-20-23
**ARCHITECTURE:** All the following experiments used the same transformer encoder-decoder architecture. The encoder consisted of an embedding layer (embedding dimension 32) and a [Pytorch Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) module (2 heads, a single layer, and a linear layer with width 128). The decoder consisted of another embedding layer (embedding dimension 32) along with a final linear layer to project the decoder output to the size of the parsed english vocabulary (in this case, a projection from 32 dimensions to 45 dimensions). The output of the decoder was thus the logits for the next likely word in the parsed output.

**TRAINING:** The initial input to the decoder was always special "\<START\>" character. For all future inputs to the decoder, we adopted a teacher forcing training method with a ratio of 0.5. We used [cross entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) for the loss function, which allowed us to use the logit outputs of the model directly. The optimizer for both the encoder and the decoder was [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with default settings. For all models trained in this experiment, we trained over the shuffled data for 5 epochs with a batch size of 32. For the results below, we trained each model in this way 20 times to get accurate confidence intervals.

**EXPERIMENTS** We ran the same experiments as on 06-13-23 and 06-14-23, but now on the new model architecture. Before running the experiment, we hypothesise that:
1. The results will be essentially identical to the previous experiments on the GRU architecture. As we increase the number of excluded female names, the test accuracy will decrease. Excluding male names will have no detectible effect

NOTE- I forgot to change the code for the title when creating the first graph, but both of the below graphs are for the transformer encoder-decoder model. The result of the information/labels in the graphs is correct
<p align="center">
    <img height="500" src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_061923-Removing_Female_Names.png?raw=true" alt="Experiment Results">
    <img height="500" src="https://github.com/luk27182/Reflexive-Anaphora/blob/main/Figures/Experiment_Results_062023-Removing_Female_Names_WITHOUT_HIMSELF.png?raw=true" alt="Experiment Results">
</p>
Looking at the results, they are essentially what we predicted. We note that the transformer model was able to generalize much better than the GRU model. Another thing to note is that the transformer encoder-decoder has roughly twice as many parameters as the GRU encoder-decoder (34253 versus 17101).
