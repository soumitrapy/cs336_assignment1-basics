from torch import Tensor
from torch.nn import Module
from .functional import cross_entropy_loss


class CrossEntropyLoss(Module):
    def forward(self,
                input: Tensor,
                target: Tensor,
                reduction: str = "mean"
                ) -> Tensor:
        return cross_entropy_loss(input, target, reduction=reduction)