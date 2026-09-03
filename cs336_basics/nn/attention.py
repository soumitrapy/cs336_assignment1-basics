
from jaxtyping import Float, Int
from einops import einsum, rearrange

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from .linear import Linear
from .functional import scaled_dot_product_attention, softmax
from .embedding import RotaryPositionalEmbedding, Embedding
from .normalization import RMSNorm
from .activation import SwiGLU

class MultiheadSelfAttention(Module):
    def __init__(self, d_model: int, 
                 num_heads: int,
                 theta: float = 10000.0,
                 max_seq_len: int | None = None,
                 ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.qkv_proj = Linear(d_model, (self.d_k + self.d_k + self.d_v)*self.num_heads)
        self.out_proj = Linear(self.num_heads * self.d_v, d_model)
        self.rope = None
        if max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(base=theta, dim=self.d_k, max_seq_len=max_seq_len)

    def forward(self, 
                x: Float[Tensor, "... seq_len d_model"],
                token_positions: Int[Tensor, "... seq_len"] | None = None
                ) -> Float[Tensor, "... seq_len d_model"]:
        proj = self.qkv_proj(x)
        q, k, v = torch.split(proj, [self.num_heads*self.d_k, self.num_heads*self.d_k, self.num_heads*self.d_v], dim=-1) # shape (..., num_heads, seq_len, d_k)
        q = rearrange(q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        if self.rope is not None:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions) # apply rotary positional embeddings
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=0) # shape (seq_len, seq_len)
        attn_out = scaled_dot_product_attention(q, k, v, mask) # shape (..., num_heads, seq_len, d_v)
        attn_out = rearrange(attn_out, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        out = self.out_proj(attn_out)
        return out

class TransformerBlock(Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 theta: float = 10000.0, 
                 max_seq_len: int | None = None,
                 ) -> None:
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(in_features=d_model, hidden_features=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        
    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]:
        attn_out = self.attn(self.ln1(x), token_positions)
        x = x + attn_out
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x


class TransformerLM(Module):
    def __init__(self, 
                 vocab_size: int,
                 context_len: int,
                 num_layers: int,
                 d_model: int = 2**13,
                 num_heads: int = 16,
                 d_ff: int = 2**15,
                 rope_theta: float = 10000.0,
                 ) -> None:
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.num_layers = num_layers
        self.d_model = d_model
        self.layers = ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_len) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)
        self.output_proj = Linear(d_model, vocab_size)

    def forward(self, x: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len vocab_size"]:
        x = self.embedding(x)                   # shape (..., seq_len, d_model)
        for mha in self.layers:
            x = mha(x, token_positions=None)    # shape (..., seq_len, d_model)
        x = self.final_norm(x)                  # shape (..., seq_len, d_model)
        logits = self.output_proj(x)            # shape (..., seq_len, vocab_size)
        #logits = softmax(logits, dim=-1)
        return logits