import math
import torch
from torch import Tensor


def trunc_normal_(tensor: Tensor,
                  mean: float = 0.0,
                  std: float | None = None) -> Tensor:
    if std is None:
        std = math.sqrt(2.0/sum(tensor.shape))
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=-3*std, b=3*std)