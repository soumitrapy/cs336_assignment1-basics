import math
from einops import einsum
from jaxtyping import Float

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .init import trunc_normal_


class Linear(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.w = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.b = Parameter(torch.empty((out_features,), **factory_kwargs))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.w)
        if self.b is not None:
            trunc_normal_(self.b)


    def forward(self,
                x: Float[Tensor,"... in_features"]
                ) -> Float[Tensor,"... out_features"]:
        y = einsum(x, self.w, "... in_features, out_features in_features -> ... out_features")
        if self.b is not None:
            y = y + self.b
        return y