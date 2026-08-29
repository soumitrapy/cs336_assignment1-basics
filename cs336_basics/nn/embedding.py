import math
from jaxtyping import Int, Float

import torch
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
                token_ids: Int[torch.Tensor,"..."]
                ) -> Float[torch.Tensor,"... embedding_dim"]:
        y = self.w[token_ids]
        return y