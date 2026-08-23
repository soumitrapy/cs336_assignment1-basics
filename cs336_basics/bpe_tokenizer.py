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
        self.special_tokens = special_tokens
        self.pretokenization_pattern = pretokenization_pattern
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

    def pretokenize(self, text: str) -> Iterator[str]:
        if self.special_tokens:
            special_tokens_pattern = "(" + "|".join(re.escape(token) for token in sorted(self.special_tokens, reverse=True)) + ")"
            for part in re.split(special_tokens_pattern, text):
                if not part:
                    continue
                if part in self.special_tokens:
                    yield part
                else:

                    for match in re.finditer(self.pretokenization_pattern, part):
                        yield match.group(0)
        else:

            for match in re.finditer(self.pretokenization_pattern, text):
                yield match.group(0)


    def merge(self, pretoken: str) -> list[int]:
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
        return [self.bytestoid[b] for b in pretoken_bytes]

    def encode(self, text: str) -> list[int]:
        



        return list(self.encode_generator(text))

    def encode_generator(self, text: str) -> Iterator[int]:
        for pretoken in self.pretokenize(text):
            if self.special_tokens and pretoken in self.special_tokens:

                yield self.bytestoid[pretoken.encode("utf-8")]
            else:


                yield from self.merge(pretoken)

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:

            yield from self.encode_generator(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[i] for i in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

