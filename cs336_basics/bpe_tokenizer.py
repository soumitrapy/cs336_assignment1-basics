from typing import Iterable

from .utils.tokenization_utils import load_vocab_and_merges

class BPETokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [tok.encode("utf-8") for tok in special_tokens] if special_tokens is not None else []
        self.pretokenization_pattern = pretokenization_pattern

    @classmethod
    def from_file(cls,
                  vocab_filepath: str,
                  merges_filepath: str,
                  special_tokens: list[str] | None = None) -> "BPETokenizer":
        vocab, merges = load_vocab_and_merges(vocab_path=vocab_filepath, merges_path=merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        pass
