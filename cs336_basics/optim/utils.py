from typing import List
from torch import Tensor
def gradient_clipping(gradients: List[Tensor], max_norm: float, eps: float = 1e-6):
    """
    Clips the gradients to have a maximum norm of `max_norm`.

    Args:
        gradients: A list of gradients to be clipped.
        max_norm: The maximum allowed norm for the gradients.

    Returns:
        A list of clipped gradients.
    """
    total_norm = sum((g ** 2).sum() for g in gradients) ** 0.5
    if total_norm> max_norm:
        clip_coef = max_norm / (total_norm + eps)
        return [g * clip_coef for g in gradients]
    return gradients