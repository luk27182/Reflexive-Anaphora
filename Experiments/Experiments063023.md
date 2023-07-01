### Experiments Run 06-30-23

**ARCHITECTURE:** For all of these experiments, I used a very simple decoder-only attention only transformer. There was a single embedding layer, followed by some number of masked self attention layers adding to the residual stream via skip connections, followed by a final linear projection. In all experiments, the hidden dimension was 32, and there were 4 heads which had a head dimension of 8 each. It was essentially the GPT architecture, but without the linear layers or layer norms. This was done in the hopes that it might be easier to interpret.

**TRAINING:** We used the Adam optimizers with the default settings for 150 epochs for all of the experiments. To use the decoder-only model, we needed to refactor the dataset: Instead of having the input be "Alice hugs herself" and the output be "HUGS ALICE ALICE", we have the model guess the next word in the sequence "<SOS> Alice hugs herself <UNK> HUGS ALICE ALICE <EOS>". Of course, we only apply the loss function to the "HUGS ALICE ALICE <EOS>" part of the sequence in this case.

**EXPERIMENTS:**

When I tested the model's generalization by holding out some female names, I found that the model was essentially unable to generalize at all. Even if we only held out a single female name! This was surprising. Here was the training results for training a single layer model:

Epoch 1 Train Loss 4.85942 Test Accuracy 0.143
Epoch 6 Train Loss 0.31909 Test Accuracy 0.000
Epoch 11 Train Loss 0.12191 Test Accuracy 0.000
Epoch 16 Train Loss 0.07835 Test Accuracy 0.000
Epoch 21 Train Loss 0.06511 Test Accuracy 0.000
Epoch 26 Train Loss 0.06138 Test Accuracy 0.000
Epoch 31 Train Loss 0.05881 Test Accuracy 0.000
Epoch 36 Train Loss 0.05703 Test Accuracy 0.000
Epoch 41 Train Loss 0.05509 Test Accuracy 0.000
Epoch 46 Train Loss 0.05503 Test Accuracy 0.000
Epoch 51 Train Loss 0.05529 Test Accuracy 0.000
Epoch 56 Train Loss 0.05325 Test Accuracy 0.000
Epoch 61 Train Loss 0.05319 Test Accuracy 0.000
Epoch 66 Train Loss 0.05201 Test Accuracy 0.000
Epoch 71 Train Loss 0.05277 Test Accuracy 0.000
Epoch 76 Train Loss 0.05080 Test Accuracy 0.000
Epoch 81 Train Loss 0.04941 Test Accuracy 0.000
Epoch 86 Train Loss 0.04899 Test Accuracy 0.000
Epoch 91 Train Loss 0.04870 Test Accuracy 0.000
Epoch 96 Train Loss 0.04841 Test Accuracy 0.000
Epoch 101 Train Loss 0.04821 Test Accuracy 0.000
Epoch 106 Train Loss 0.04795 Test Accuracy 0.000
Epoch 111 Train Loss 0.04778 Test Accuracy 0.000
Epoch 116 Train Loss 0.04925 Test Accuracy 0.000
Epoch 121 Train Loss 0.04765 Test Accuracy 0.000
Epoch 126 Train Loss 0.04751 Test Accuracy 0.000
Epoch 131 Train Loss 0.04752 Test Accuracy 0.000
Epoch 136 Train Loss 0.04747 Test Accuracy 0.000
Epoch 141 Train Loss 0.04739 Test Accuracy 0.000
Epoch 146 Train Loss 0.04731 Test Accuracy 0.000

Here are the test outputs of the model. The model consistantly replaces "herself" with "evelyn":

===== TESTING INDEX 0 =====
input: <SOS> Alice eats herself <UNK>
goal: EATS ALICE ALICE <EOS>
model out: COMPLIMENTS ALICE EVELYN <EOS>

===== TESTING INDEX 1 =====
input: <SOS> Alice sees herself <UNK>
goal: SEES ALICE ALICE <EOS>
model out: COMPLIMENTS ALICE EVELYN <EOS>

===== TESTING INDEX 2 =====
input: <SOS> Alice hugs herself <UNK>
goal: HUGS ALICE ALICE <EOS>
model out: HUGS ALICE EVELYN <EOS>

===== TESTING INDEX 3 =====
input: <SOS> Alice paints herself <UNK>
goal: PAINTS ALICE ALICE <EOS>
model out: COMPLIMENTS ALICE EVELYN <EOS>

===== TESTING INDEX 4 =====
input: <SOS> Alice kicks herself <UNK>
goal: KICKS ALICE ALICE <EOS>
model out: KICKS ALICE EVELYN <EOS>

===== TESTING INDEX 5 =====
input: <SOS> Alice throws herself <UNK>
goal: THROWS ALICE ALICE <EOS>
model out: THROWS ALICE EVELYN <EOS>

===== TESTING INDEX 6 =====
input: <SOS> Alice compliments herself <UNK>
goal: COMPLIMENTS ALICE ALICE <EOS>
model out: COMPLIMENTS ALICE EVELYN <EOS>

To sanity check, I ran some experiments on basic generalization where I did a rand 98% 2% train-test split. A model with a single layer was able to achieve 99% accuracy on the held out test data. The training was as follows:

Epoch 1 Train Loss 5.39430 Test Accuracy 0.000
Epoch 6 Train Loss 0.36769 Test Accuracy 0.126
Epoch 11 Train Loss 0.23620 Test Accuracy 0.398
Epoch 16 Train Loss 0.07340 Test Accuracy 0.854
Epoch 21 Train Loss 0.02905 Test Accuracy 0.932
Epoch 26 Train Loss 0.01724 Test Accuracy 0.971
Epoch 31 Train Loss 0.01096 Test Accuracy 0.961
Epoch 36 Train Loss 0.00731 Test Accuracy 0.971
Epoch 41 Train Loss 0.00487 Test Accuracy 0.981
Epoch 46 Train Loss 0.00378 Test Accuracy 0.990
Epoch 51 Train Loss 0.00365 Test Accuracy 0.961
Epoch 56 Train Loss 0.00309 Test Accuracy 0.981
Epoch 61 Train Loss 0.00203 Test Accuracy 0.981
Epoch 66 Train Loss 0.00236 Test Accuracy 0.981
Epoch 71 Train Loss 0.00131 Test Accuracy 1.000
Epoch 76 Train Loss 0.00153 Test Accuracy 0.990
Epoch 81 Train Loss 0.00148 Test Accuracy 0.990
Epoch 86 Train Loss 0.00061 Test Accuracy 0.981
Epoch 91 Train Loss 0.00055 Test Accuracy 0.990
Epoch 96 Train Loss 0.00081 Test Accuracy 1.000
Epoch 101 Train Loss 0.00031 Test Accuracy 0.990
Epoch 106 Train Loss 0.00039 Test Accuracy 1.000
Epoch 111 Train Loss 0.00040 Test Accuracy 1.000
Epoch 116 Train Loss 0.00061 Test Accuracy 1.000
Epoch 121 Train Loss 0.00037 Test Accuracy 0.990
Epoch 126 Train Loss 0.00027 Test Accuracy 0.990
Epoch 131 Train Loss 0.00031 Test Accuracy 0.990
Epoch 136 Train Loss 0.00022 Test Accuracy 0.990
Epoch 141 Train Loss 0.00012 Test Accuracy 1.000
Epoch 146 Train Loss 0.00165 Test Accuracy 0.990

Here are some test examples (one reflexive, one regular transitive, and one intransitive):

===== TESTING INDEX 38 =====

input: <SOS> Amelia runs <UNK>
goal: RUNS AMELIA <EOS>
model out: RUNS AMELIA <EOS>

===== TESTING INDEX 55 =====
input: <SOS> Elizabeth throws Elijah <UNK>
goal: THROWS ELIZABETH ELIJAH <EOS>
model out: THROWS ELIZABETH ELIJAH <EOS>

===== TESTING INDEX 56 =====
input: <SOS> Henry eats himself <UNK>
goal: EATS HENRY HENRY <EOS>
model out: EATS HENRY HENRY <EOS>

I also saved the state dictionaries of both of these models in the ./Models folders.