import math
from jaxtyping import Float
from einops import einsum

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .init import trunc_normal_


class RMSNorm(Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.eps = eps
        self.g = Parameter(torch.empty((d_model,), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.g)

    def forward(self,
                x: Float[Tensor,"... d_model"]
                ) -> Float[Tensor,"... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        y = einsum(x/rms, self.g, "... d_model, d_model -> ... d_model")
        y = y.to(in_dtype)
        return y        





