import math
from einops import einsum
from jaxtyping import Float

import torch
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
        #sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        # with no_grad(): # automatically done by init functions
        trunc_normal_(self.w, mean=0.0, std=math.sqrt(2.0 / (self.in_features + self.out_features)))
        if self.b is not None:
            trunc_normal_(self.b, mean=0.0, std=math.sqrt(2.0 / self.out_features))


    def forward(self,
                x: Float[torch.Tensor,"... in_features"]
                ) -> Float[torch.Tensor,"... out_features"]:
        y = einsum(x, self.w, "... in_features, out_features in_features -> ... out_features")
        if self.b is not None:
            y = y + self.b
        return y