from einops import einsum
from jaxtyping import Float

import torch
from torch import Tensor
from torch.nn import Module

from .linear import Linear




class SiLU(Module):
    def forward(self,
                x: Float[Tensor,"... d_model"]
                ) -> Float[Tensor,"... d_model"]:
        return einsum(x, torch.sigmoid(x), "... d_model, ... d_model-> ... d_model")

class SwiGLU(Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int | None = None,
                 ) -> None:
        super().__init__()
        self.in_features = in_features
        if hidden_features is None:
            self.d_ff = max(round(in_features/(3*8)), 1)*64 # d_ff = 8/3 * d_model nearest multiple of 64
        else:
            self.d_ff = hidden_features
        self.linear1 = Linear(self.in_features, self.d_ff)
        self.linear3 = Linear(self.in_features, self.d_ff)
        self.linear2 = Linear(self.d_ff, self.in_features)
        self.silu = SiLU()

    def forward(self,
                x: Float[Tensor,"... d_model"]
                ) -> Float[Tensor,"... d_model"]:
        x1 = self.silu(self.linear1(x))
        x3 = self.linear3(x)
        x2 = self.linear2(einsum(x1, x3, "... d_ff, ... d_ff -> ... d_ff"))
        return x2
    
