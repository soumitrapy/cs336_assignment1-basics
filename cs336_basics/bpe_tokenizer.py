import regex as re
from typing import Iterable, Iterator
#from concurrent.futures import ProcessPoolExecutor
import warnings
from functools import lru_cache
from multiprocessing import get_context

from .utils.tokenization_utils import load_vocab_and_merges, create_batch

_tokenizer = None
def init_worker(config: dict):
    global _tokenizer
    _tokenizer = BPETokenizer(**config)

def worker_encode(batch: tuple[str, ...]) -> tuple[int, ...]:
    ids = []
    for text in batch:
        ids.extend(_tokenizer._encode(text))
    return tuple(ids)



class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        cache_size: int = 0,
        ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.bytes2id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {(self.bytes2id[pair[0]], self.bytes2id[pair[1]]): (i, self.bytes2id[pair[0] + pair[1]]) for i, pair in enumerate(self.merges)}
        #
        self.pretokenization_pattern = re.compile(pretokenization_pattern)
        self.special_tokens = None
        self.special_pattern = None
        self.sp2id = {}
        self.id2sp = {}
        if special_tokens is not None and len(special_tokens) > 0:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            count = len(self.vocab)
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes in self.bytes2id:
                    self.sp2id[token] = self.bytes2id[token_bytes]
                    self.id2sp[self.bytes2id[token_bytes]] = token
                else:
                    self.sp2id[token] = count
                    self.id2sp[count] = token
                    count += 1
            self.special_pattern = re.compile("|".join(re.escape(token) for token in self.special_tokens))
        self._cached_bpe = lru_cache(maxsize=cache_size)(self._cached_bpe) if cache_size > 0 else None
        self.cache_size = cache_size
    
    @classmethod
    def from_file(cls,
                  vocab_filepath: str,
                  merges_filepath: str,
                  special_tokens: list[str] | None = None,
                  #pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                  **kwargs
                ) -> "BPETokenizer":
        vocab, merges = load_vocab_and_merges(vocab_path=vocab_filepath, merges_path=merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens, **kwargs)


    def _bpe(self, pretoken: str) -> Iterator[int]:
        if not pretoken:
            return
        ids = [self.bytes2id[bytes([b])] for b in pretoken.encode("utf-8")]
        pairs = set([(ids[i], ids[i + 1]) for i in range(len(ids) - 1)])
        ranks = {pair: self.merge_ranks.get(pair, (float('inf'), None)) for pair in pairs} # locally store for faster access

        while len(ids) > 1:
            best_pair = min(pairs, key=lambda pair: ranks[pair][0], default=None)
            if best_pair is None or ranks[best_pair][1] is None:
                break
            newids = []
            _, merged_id = ranks[best_pair]
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    newids.append(merged_id)
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1
            pairs.clear()
            for i in range(1, len(newids)):
                pair = (newids[i - 1], newids[i])
                pairs.add(pair)
                if pair not in ranks:
                    ranks[pair] = self.merge_ranks.get(pair, (float('inf'), None))

            ids = newids
        yield from ids

    #@lru_cache(maxsize=4096)
    def _cached_bpe(self, pretoken: str) -> tuple[int, ...]:
        return tuple(self._bpe(pretoken))

    def _encode_normal_text(self, text: str) -> Iterator[int]:
        for match in self.pretokenization_pattern.finditer(text):
            pretoken = match.group(0)
            #yield from self._bpe(pretoken)
            if self._cached_bpe is not None:
                yield from self._cached_bpe(pretoken)
            else:
                yield from self._bpe(pretoken)

    def _encode(self, text: str) -> Iterator[int]:
        if self.special_tokens is None:
            yield from self._encode_normal_text(text)
        else:
            last_end = 0
            for match in self.special_pattern.finditer(text):
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


    def encode_iterable(self, texts: Iterable[str], num_workers: int = 4, batch_size: int = 1024*1024, cache_size=1024*4) -> Iterator[int]:
        # sequential processing if num_workers <= 1
        if num_workers <= 1:
            for text in texts:
                yield from self._encode(text)
        else:
            batches = create_batch(texts, batch_size=batch_size)
            try:
                ctx = get_context("fork")
            except:
                ctx = get_context("spawn")
            config = {
                "vocab": self.vocab,
                "merges": self.merges,
                "special_tokens": self.special_tokens,
                "pretokenization_pattern": self.pretokenization_pattern.pattern,
                "cache_size": self.cache_size if cache_size is None else cache_size
            }
            with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(config,)) as pool:
                for ids in pool.imap(worker_encode, batches):
                    yield from ids

            

    def decode(self, ids: list[int]) -> str:
        tokens =  []
        buffer: list[bytes] = []
        for token_id in ids:
            if token_id in self.id2sp:
                if buffer:
                    tokens.append(b"".join(buffer).decode("utf-8", errors="replace"))
                    buffer = []
                tokens.append(self.id2sp[token_id])
            elif token_id in self.vocab:
                buffer.append(self.vocab[token_id])
            else:
                warnings.warn(f"ID {token_id} not found in vocabulary or special tokens. Skipping this ID.")
        if buffer:
            tokens.append(b"".join(buffer).decode("utf-8", errors="replace"))
        return "".join(tokens)