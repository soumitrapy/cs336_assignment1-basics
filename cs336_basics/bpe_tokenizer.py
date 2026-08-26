import regex as re
from typing import Iterable, Iterator

from .utils.tokenization_utils import load_vocab_and_merges

class BPETokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else None
        self.special_tokens_pattern = re.compile("|".join(re.escape(token) for token in self.special_tokens)) if self.special_tokens is not None else None
        self.pretokenization_pattern = re.compile(pretokenization_pattern)
        self.bytestoid = {v: k for k, v in vocab.items()}
        #self.merges_ids = [(self.bytestoid[a], self.bytestoid[b]) for a, b in merges]

    @classmethod
    def from_file(cls,
                  vocab_filepath: str,
                  merges_filepath: str,
                  special_tokens: list[str] | None = None,
                  pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                ) -> "BPETokenizer":
        vocab, merges = load_vocab_and_merges(vocab_path=vocab_filepath, merges_path=merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens, pretokenization_pattern=pretokenization_pattern)

    
    def _bpe(self, pretoken: str) -> Iterator[int]:
        pretoken_bytes = [bytes([b]) for b in pretoken.encode("utf-8")]
        for pair in self.merges:
            new_bytes = []
            i = 0
            while i < len(pretoken_bytes):
                if i < len(pretoken_bytes) - 1 and (pretoken_bytes[i], pretoken_bytes[i + 1]) == pair:
                    new_bytes.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_bytes.append(pretoken_bytes[i])
                    i += 1
            pretoken_bytes = new_bytes
        for b in pretoken_bytes:
            yield self.bytestoid[b]

    def _encode_normal_text(self, text: str) -> Iterator[int]:
        for match in self.pretokenization_pattern.finditer(text):
            pretoken = match.group(0)
            yield from self._bpe(pretoken)

    def _encode(self, text: str) -> Iterator[int]:
        if self.special_tokens_pattern is None:
            yield from self._encode_normal_text(text)
        else:
            last_end = 0
            for match in self.special_tokens_pattern.finditer(text):
                start, end = match.span()
                normal_text = text[last_end:start]
                if normal_text:
                    yield from self._encode_normal_text(normal_text)
                special_token = match.group(0)
                yield self.bytestoid[special_token.encode("utf-8")]
                last_end = end
            remaining_text = text[last_end:]
            if remaining_text:
                yield from self._encode_normal_text(remaining_text)

    def encode(self, text: str) -> list[int]:
        return list(self._encode(text))

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self._encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[i] for i in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")
