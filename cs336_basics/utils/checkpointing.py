import os
import json
import typing

import torch
from torch.nn import Module


def save_checkpoint(model: Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                    ) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "iteration": iteration
    }, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: Module,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                    ) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return checkpoint["iteration"]

def save_vocab_and_merges(vocab: dict[int, bytes],
                          merges: list[tuple[bytes, bytes]], 
                          vocab_path: str, 
                          merges_path: str) -> None:
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    os.makedirs(os.path.dirname(merges_path), exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v.hex() for k, v in vocab.items()}, f, indent=2)
    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.hex()} {right.hex()}\n")

def load_vocab_and_merges(vocab_path: str,
                          merges_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = {int(k): bytes.fromhex(v) for k, v in json.load(f).items()}
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            left_hex, right_hex = line.strip().split()
            merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
    return vocab, merges