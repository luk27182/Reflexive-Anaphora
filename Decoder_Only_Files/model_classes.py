# %%
import torch
from torch import nn
import einops
from fancy_einsum import einsum
import math

# %%
class LayerNorm(nn.Module):
    """Taken from https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb#scrollTo=kWpfPKHs9tHI"""
    def __init__(self, d_model, layer_norm_eps=1e-5):
        super().__init__()
        self.layer_norm_eps = layer_norm_eps

        self.w = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, residual):
        # residual: [batch, position, d_model]
        residual = residual - einops.reduce(residual, "batch position d_model -> batch position 1", "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "batch position d_model -> batch position 1", "mean") + self.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        return normalized

class SelfAttention(nn.Module):
    """Adopted from https://colab.research.google.com/github/neelnanda-io/Easy-Transformer/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb#scrollTo=kWpfPKHs9tHI"""
    def __init__(self, n_heads, d_head, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        
        self.init_range = 1

        self.W_Q = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        nn.init.normal_(self.W_Q, std=self.init_range)
        self.b_Q = nn.Parameter(torch.zeros((n_heads, d_head)))
        self.W_K = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        nn.init.normal_(self.W_K, std=self.init_range)
        self.b_K = nn.Parameter(torch.zeros((n_heads, d_head)))
        self.W_V = nn.Parameter(torch.empty((n_heads, d_model, d_head)))
        nn.init.normal_(self.W_V, std=self.init_range)
        self.b_V = nn.Parameter(torch.zeros((n_heads, d_head)))

        self.dropout1 = nn.Dropout(p=0.2)
        
        self.W_O = nn.Parameter(torch.empty((n_heads, d_head, d_model)))
        nn.init.normal_(self.W_O, std=self.init_range)
        self.b_O = nn.Parameter(torch.zeros((d_model)))
        self.dropout2 = nn.Dropout(p=0.2)

    def apply_causal_mask(self, attn_scores):
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, -1e5)
        return attn_scores
    
    def forward(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]

        q = einsum("batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        
        attn_scores = einsum("batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]
        if self.training:
            pattern = self.dropout1(pattern)

        v = einsum("batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_heads query_pos key_pos, batch key_pos n_heads d_head -> batch query_pos n_heads d_head", pattern, v)

        attn_out_by_head = einsum("batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos n_heads d_model", z, self.W_O)
        attn_out = torch.sum(attn_out_by_head, dim=2) + self.b_O

        if self.training:
            attn_out = self.dropout2(attn_out)
        
        return attn_out, pattern, attn_out_by_head

class AttnBlock(nn.Module):
    def __init__(self, n_heads, d_head, d_model, attn_only=False):
        super().__init__()
        self.attn_only = attn_only

        self.ln1 = LayerNorm(d_model=d_model)
        self.attn = SelfAttention(n_heads=n_heads, d_model=d_model, d_head=d_head)

        if not attn_only:
            self.linear_expand = nn.Linear(d_model, 4*d_model)
            self.linear_contract = nn.Linear(4*d_model, d_model)
            self.ln2 = LayerNorm(d_model=d_model)
            self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, resid_pre):
        normalized_resid_pre = self.ln1(resid_pre)

        attn_out, attn_pattern, attn_out_by_head = self.attn(normalized_resid_pre)

        resid_mid = attn_out+resid_pre
        
        if self.attn_only:
            return resid_mid, attn_pattern, attn_out_by_head
    
        normalized_resid_mid = self.ln2(resid_mid)

        linear_out = self.linear_contract(nn.GELU()(self.linear_expand(normalized_resid_mid)))
        if self.training:
            linear_out = self.dropout(linear_out)
        
        resid_post = linear_out+resid_mid

        return resid_post, attn_pattern, attn_out_by_head


class AttnOnly_Transformer(nn.Module):
    def __init__(self, vocab_size, n_heads, d_model, d_head, n_layers, attn_only=False, ctx_length=9):
        super().__init__()
        self.use_pos_embedding = True
        self.use_lex_embedding = True

        self.n_layers = n_layers
        self.n_heads = n_heads

        self.cache = dict()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(ctx_length, d_model)
        self.dropout = nn.Dropout(p=0.2)

        self.blocks = nn.ModuleList([])
        self.blocks.extend([AttnBlock(n_heads=n_heads, d_model=d_model, d_head=d_head, attn_only=attn_only) for i in range(n_layers)])

        self.final_layer_norm = LayerNorm(d_model=32)
        self.unembedding = nn.Linear(d_model, vocab_size)

    def embed(self, tensor):
        embedding = torch.zeros(self.embedding(tensor).shape).to(next(self.parameters()).device)
        if self.use_pos_embedding:
            embedding += self.pos_embedding(einops.repeat(torch.arange(tensor.size(0)), "n -> n b", b=tensor.size(1)).to(torch.int64).to(next(self.parameters()).device))
        if self.use_lex_embedding:
            embedding += self.embedding(tensor)
        return embedding # of size [SEQ BATCH EMBED]
        
    def forward(self, seq):
        # seq is of shape BATCH X POS, where entries are integers]
        residual_stream = einops.rearrange(self.embed(seq), "s b d -> b s d")
        self.cache["resid_initial"] = residual_stream.detach()
        if self.training:
            residual_stream = self.dropout(residual_stream)
        for n, block in enumerate(self.blocks):
            residual_stream, attn_pattern, attn_out_by_head = block(residual_stream) # attn_pattern is of shape BATCH x n_head x SEQ x SEQ
            self.cache[f"resid_postBlock_{n}"] = residual_stream.detach()

            for head in range(attn_pattern.size(1)):
                self.cache[f"l{n}h{head}_attnOut"] = attn_out_by_head[:, :, head].detach()
                self.cache[f"l{n}h{head}_pattern"] = attn_pattern[:, head].detach()

        residual_stream = self.final_layer_norm(residual_stream)
        model_out = self.unembedding(residual_stream) # shape is BATCH X POS X VOCAB

        return model_out

# %%
