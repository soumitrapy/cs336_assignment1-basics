from collections.abc import Iterable
import torch
from torch import Tensor

def gradient_clipping(parameters: Iterable[torch.nn.Parameter],
                      max_norm: float,
                      eps: float = 1e-6) -> Tensor:

    with torch.no_grad():
        total_norm = sum(
            p.grad.detach().pow(2).sum() for p in parameters if p.grad is not None
        ).sqrt()
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + eps)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
    
    return total_norm