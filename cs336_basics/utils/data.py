import numpy as np
import torch
from torch import LongTensor, Tensor
from typing import Iterator
import numpy.typing as npt

from einops import rearrange

# def get_batch(x: np.ndarray,
#               batch_size: int,
#               seq_len: int,
#               device: str = "cpu") -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
#     n = len(x)
#     for i in range(0, n - seq_len, batch_size * seq_len):
#         if i + batch_size * seq_len < n:
#             k = batch_size
#         else:
#             k = (n - i - 1) // seq_len
#         if k > 0:
#             x_batch = x[i:i+k*seq_len].reshape(k, seq_len)
#             y_batch = x[i+1:i+1+k*seq_len].reshape(k, seq_len)
#             yield LongTensor(x_batch, device=device), LongTensor(y_batch, device=device)

def get_batch(x: npt.NDArray,
              batch_size: int,
              seq_len: int,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.random.randint(0, len(x) - seq_len, size=batch_size)
    x_batch = np.stack([x[i:i+seq_len] for i in indices])
    y_batch = np.stack([x[i+1:i+1+seq_len] for i in indices])
    return LongTensor(x_batch, device=device), LongTensor(y_batch, device=device)


def get_data_loader(x: npt.NDArray,
                    batch_size: int,
                    seq_len: int,
                    device: str = "cpu") -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    n = len(x)
    for i in range(0, n - seq_len, batch_size * seq_len):
        if i + batch_size * seq_len < n:
            k = batch_size
        else:
            k = (n - i - 1) // seq_len
        if k > 0:
            x_batch = x[i:i+k*seq_len].reshape(k, seq_len)
            y_batch = x[i+1:i+1+k*seq_len].reshape(k, seq_len)
            yield LongTensor(x_batch, device=device), LongTensor(y_batch, device=device)
        

        