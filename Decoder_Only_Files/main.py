# %%
from helper import *
from model_classes import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

# %%
# Create train/test dataset objects
corpus_train, corpus_test = create_train_corpus(corpus, excluded_females=1, exclude_men=False)
ds_train, ds_test = TextDataset(corpus_train), TextDataset(corpus_test)

# Create model
model = AttnOnly_Transformer(vocab_size=len(vocab), n_heads=4, d_model=32, d_head=8, n_layers=3, attn_only=False).to(device)

# %%
# Here we load in a pretrained model which generalized to the name "Alice."
model.load_state_dict(state_dict=torch.load('./Decoder_Only_Transformers/dec_only_100epochs_1fexcluded_GENERALIZING.pth', map_location=device))

for index in range(len(ds_test)):
    example = ds_test[index]
    test_index(model, example=example, verbose=True, be_safe=True)

# %%
plot_pattern(model, ds_test[0])

# %%
# We plot the initial embedding (location -1) of the reflexive
run_PCA_analysis_by_name(model, num_names=15, probe_location=-1, lex_loc="o")

# We plot the residual stream of the reflexive after the first layer
run_PCA_analysis_by_name(model, num_names=15, probe_location=1, lex_loc="o")
# %%
