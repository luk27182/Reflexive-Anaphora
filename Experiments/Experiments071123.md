1. Train a model for 100 epochs with NO REFLEXIVE EXAMPLES. For this model, I hard coded it so that the model will replace the final dimension of the hidden space with "0" in the encoder. This is saved at Research-2023\Models\transformer_1head_32hidden_100epochs_noReflexives.pth

2. I manipulated the encoder embedding weights so that the encoder always encodes himself/herself as [0,0,0,0,1]. Everything else the same. I saved this to Research-2023\Models\transformer_1head_32hidden_100epochs_noReflexives_manipulated.pth

3. I froze the weights of the encoder embedding/transformer parts, and trained the model on the entire training corpus for 100 more epochs. I saves this at Research-2023\Models\transformer_1head_32hidden_100epochs_ForcedDecoderSolver.pth

4. Same thing but freeze the decoder