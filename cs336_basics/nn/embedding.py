import math
from typing import Optional
from jaxtyping import Int, Float
from einops import einsum, rearrange

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .init import trunc_normal_


class Embedding(Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.w = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.w, mean=0.0, std=1.0)

    def forward(self,
                token_ids: Int[Tensor,"..."]
                ) -> Float[Tensor,"... embedding_dim"]:
        y = self.w[token_ids]
        return y


class RotaryPositionalEmbedding(Module):
    def __init__(self,
                 base: float, # Capital Theta
                 dim: int,
                 max_seq_len: int,
                 device: torch.device | None = None,
                 ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for rotary positional embeddings, got {dim}")
        self.base = base
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.rope_init()

    def rope_init(self) -> None:
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=self.device) / self.dim)) # shape: (dim/2,)
        self.register_buffer("theta", theta, persistent=False)

        # Precompute the sin and cos values for all positions and dimensions
        idx = torch.arange(self.max_seq_len, device=self.theta.device, dtype=self.theta.dtype) # shape: (max_seq_len,)
        idx_theta = einsum(idx, theta, "max_seq_len, half_dim ->  max_seq_len half_dim")
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1) # shape: (max_seq_len, dim/2, 2)
        self.register_buffer("cache", cache, persistent=False)
    
    def forward(self,
                x: Float[Tensor,"... seq_len d_k"],
                token_positions: Optional[Int[Tensor,"... seq_len"]] = None,
                ) -> Float[Tensor,"... seq_len d_k"]:

        seq_len = x.shape[-2]
        rope_cache = self.cache[token_positions] if token_positions is not None else self.cache[:seq_len] # shape: (seq_len, dim/2, 2)
        x_half = rearrange(x, '... (h w) -> ... h w', w=2) # shape: (..., seq_len, dim/2, 2)
        x_rotated = torch.stack([x_half[..., 0] * rope_cache[..., 0] - x_half[..., 1] * rope_cache[..., 1],
                                 x_half[..., 0] * rope_cache[..., 1] + x_half[..., 1] * rope_cache[..., 0]], dim=-1) # shape: (..., seq_len, dim/2, 2)
        x_rotated = rearrange(x_rotated, '... h w -> ... (h w)', w=2)
        return x_rotated