import math
from einops import einsum
from jaxtyping import Float

import torch
from torch.nn import Module, Parameter


class Linear(Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.w = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        # with no_grad(): # automatically done by init functions
        torch.nn.init.trunc_normal_(self.w, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)
    

    def forward(self,
                x: Float[torch.Tensor,"... in_features"]
                ) -> Float[torch.Tensor,"... out_features"]:
        y = einsum(x, self.w, "... in_features, out_features in_features -> ... out_features")
        return y