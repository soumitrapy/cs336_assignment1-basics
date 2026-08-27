import regex as re
from typing import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor

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
        self.bytestoid = {v: k for k, v in vocab.items()}
        self.merge_ids = {(self.bytestoid[pair[0]], self.bytestoid[pair[1]]): self.bytestoid[pair[0]+pair[1]] for pair in self.merges}
        self.merge_ranks = {(self.bytestoid[pair[0]], self.bytestoid[pair[1]]): i for i, pair in enumerate(self.merges)}
        #
        self.pretokenization_pattern = pretokenization_pattern
        self.special_tokens = None
        self.special_pattern = None
        self.sp2id = {}
        self.id2sp = {}
        if special_tokens is not None and len(special_tokens) > 0:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            count = len(self.vocab)
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes in self.bytestoid:
                    self.sp2id[token] = self.bytestoid[token_bytes]
                    self.id2sp[self.bytestoid[token_bytes]] = token
                else:
                    self.sp2id[token] = count
                    self.id2sp[count] = token
                    count += 1
            self.special_pattern = "|".join(re.escape(token) for token in self.special_tokens)
    
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
        if not pretoken:
            return
        ids = [self.bytestoid[bytes([b])] for b in pretoken.encode("utf-8")]
        while len(ids) > 1:
            # Find the best pair to merge
            k = None
            best_rank = float("inf")
            for i in range(len(ids) - 1):
                rank = self.merge_ranks.get((ids[i], ids[i + 1]), float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    #best_pair = (ids[i], ids[i + 1])
                    k = i
            if best_rank == float("inf"):
                break
            # Merge the best pair
            new_id = self.merge_ids[(ids[k], ids[k + 1])]
            new_ids = []
            i = 0
            while i < len(ids):
                if i == k:
                    new_ids.append(new_id)
                    i += 2  # Skip the next one since it's merged
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        yield from ids

    def _encode_normal_text(self, text: str) -> Iterator[int]:
        for match in re.finditer(self.pretokenization_pattern, text):
            pretoken = match.group(0)
            yield from self._bpe(pretoken)

    def _encode(self, text: str) -> Iterator[int]:
        if self.special_tokens is None:
            yield from self._encode_normal_text(text)
        else:
            last_end = 0
            for match in re.finditer(self.special_pattern, text):
                start, end = match.span()
                normal_text = text[last_end:start]
                if normal_text:
                    yield from self._encode_normal_text(normal_text)
                special_token = match.group(0)
                yield self.sp2id[special_token]
                last_end = end
            remaining_text = text[last_end:]
            if remaining_text:
                yield from self._encode_normal_text(remaining_text)

    def encode(self, text: str) -> list[int]:
        return list(self._encode(text))

    def encode_iterable(self, texts: Iterable[str], num_workers: int = 4) -> Iterator[int]:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for ids in executor.map(self.encode, texts):
                yield from ids
            

    def decode(self, ids: list[int]) -> str:
        tokens =  []
        buffer: bytes = b""
        for token_id in ids:
            if token_id in self.id2sp:
                if buffer:
                    tokens.append(buffer.decode("utf-8", errors="replace"))
                    buffer = b""
                tokens.append(self.id2sp[token_id])
            elif token_id in self.vocab:
                buffer += self.vocab[token_id]
            else:
                raise ValueError(f"ID {token_id} not found in vocabulary or special tokens.")
        if buffer:
            tokens.append(buffer.decode("utf-8", errors="replace"))
        return "".join(tokens)