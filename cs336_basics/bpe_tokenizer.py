import pickle
import re
from typing import Iterable

from .utils.tokenization_utils import find_chunk_boundaries

class BPETokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None, 
                 pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [tok.encode("utf-8") for tok in special_tokens] if special_tokens is not None else []
        self.pretokenization_pattern = pretokenization_pattern

    @classmethod
    def from_file(cls, file_path: str, special_tokens: list[str] | None = None):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            vocab, merges, pretokenization_pattern = data["vocab"], data["merges"], data["pretokenization_pattern"]
        return cls(vocab, merges, special_tokens=special_tokens, pretokenization_pattern=pretokenization_pattern)

    def encode(self, text: str) -> list[bytes]:
        # pretokenization
        text = text.encode("utf-8")
        
        special_tokens_pattern = b"(" + b"|".join(re.escape(tok) for tok in self.special_tokens) + b")"
        pretokenization_pattern = self.pretokenization_pattern.encode("utf-8")
        pieces = re.split(special_tokens_pattern, text)
        tokens = []
        for piece in pieces:
            if piece == b"":
                continue
            elif piece in self.special_tokens:
                tokens.append(piece)
            else:
                tokens.extend(
                    match.group(0) for match in re.finditer(pretokenization_pattern, piece)
                )
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[bytes]:
        special_tokens_pattern = b"(" + b"|".join(re.escape(tok) for tok in self.special_tokens) + b")"
        pretokenization_pattern = self.pretokenization_pattern.encode("utf-8")
        for text in iterable:
            text = text.encode("utf-8")
            for piece in re.split(special_tokens_pattern, text):
                if piece == b"":
                    continue
                elif piece in self.special_tokens:
                    yield piece
                else:
                    for match in re.finditer(pretokenization_pattern, piece):
                        yield match.group(0)

    def decode(self, tokens):
        # Implement the decoding logic here
        pass
