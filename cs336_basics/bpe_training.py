import regex as re
from typing import BinaryIO
from multiprocessing import get_context
from tqdm import tqdm
import heapq

from .utils.tokenization_utils import find_chunk_boundaries, save_vocab_and_merges

class Pretoken:
    def __init__(self, value: tuple[int, ...], freq: int):
        self.value = value
        self.freq = freq

    def _search_pair(self, pair: tuple[int, int]) -> list[int]:
        idx = []
        prev = -3
        for i in range(len(self.value) - 1):
            if (self.value[i], self.value[i + 1]) == pair and i-1 != prev:
                idx.append(i)
                prev = i
        return idx

    def _merge(self, pair: tuple[int, int], merged_token: int) -> bool:
        idx = self._search_pair(pair)
        if len(idx) == 0:
            return False
        j = 0
        newvalue: list[int] = []
        while j < len(self.value):
            if j in idx:
                newvalue.append(merged_token)
                j += 2
            else:
                newvalue.append(self.value[j])
                j += 1
        self.value = tuple(newvalue)
        return True

def initialize_vocab(special_tokens: list[bytes]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        if token not in vocab.values():
            vocab[len(vocab)] = token
    return vocab

def pretokenization_chunk(
    chunk: bytes,
    special_tokens_pattern: str,
    pretokenization_pattern: str,
) -> dict[tuple[int, ...], int]:

    docs = re.split(special_tokens_pattern, chunk)

    # Pretokenization
    pretokens = {}
    for doc in docs:
        for x in re.finditer(pretokenization_pattern.encode("utf-8"), doc):
            pretoken = tuple(x.group(0))
            pretokens[pretoken] = pretokens.get(pretoken, 0) + 1
    return pretokens

def pretokenization(
    file: BinaryIO,
    num_chunks: int = 1,
    special_tokens: list[bytes] = [b"<|endoftext|>", b"<|unknown|>"],
    split_token: bytes = b"<|endoftext|>",
    pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> list[Pretoken]:
    boundaries = find_chunk_boundaries(file=file, num_chunks=num_chunks, split_token=split_token)
    chunks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        file.seek(start)
        chunk = file.read(end - start)
        chunks.append(chunk)
    special_tokens_pattern = b"|".join(re.escape(tok) for tok in special_tokens)
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_chunks) as pool:
        results = pool.starmap(pretokenization_chunk, [(chunk, special_tokens_pattern, pretokenization_pattern) for chunk in chunks])
    pretokens = []
    for result in results:
        for pretoken, freq in result.items():
            pretokens.append(Pretoken(value=pretoken, freq=freq))
    return pretokens


class Pair:
    __slots__ = ["pair", "freq", "byte_vals"]
    def __init__(self, pair: tuple[int, int], freq: int, byte_vals: tuple[bytes, bytes]):
        self.pair = pair
        self.freq = freq
        self.byte_vals = byte_vals

    def __lt__(self, other: "Pair") -> bool:
        if self.freq == other.freq:
            return self.byte_vals > other.byte_vals  # For deterministic behavior
        return self.freq > other.freq  # For max-heap behavior


def single_merge(
    pretokens: list[Pretoken],
    heap: list[Pair],
    pair_counts: dict[tuple[int, int], int],
    pair2ids: dict[tuple[int, int], set[int]],
    id2pretoken: dict[int, Pretoken],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    topPair = None
    while heap:
        candidate = heapq.heappop(heap)
        if pair_counts.get(candidate.pair, 0) == candidate.freq:
            topPair = candidate
            break
    if topPair is None:
        print("All pairs have been merged. Total vocabulary size {}.".format(len(vocab)))
        return vocab, merges

    top = topPair.pair
    byte_vals = topPair.byte_vals
    new_token = len(vocab)
    merges.append(byte_vals)
    vocab[new_token] = byte_vals[0] + byte_vals[1]

    # Updates
    #pair_counts.pop(top)
    #ids = pair2ids[top].copy()  # Copy to avoid modification during iteration
    #ids = pair2ids.pop(top)
    ids = pair2ids[top].copy()
    #print(top, len(ids))
    #new_pairs = set()
    updated_pairs = set()
    for id in ids:
        new_pairs = {}
        prev_pairs = {}
        pretoken = id2pretoken[id]
        for i in range(len(pretoken.value) - 1):
            pair = (pretoken.value[i], pretoken.value[i + 1])
            prev_pairs[pair] = prev_pairs.get(pair, 0) + pretoken.freq
            # if pair == top:
            #     continue  # Skip the pair that is being merged
            # pair_counts[pair] -= pretoken.freq
            # pair2ids[pair].discard(id)
            # if pair_counts[pair] <= 0:
            #     pair_counts.pop(pair)
                #pair2ids.pop(pair)
            
        pretoken._merge(pair=top, merged_token=new_token)
        for i in range(len(pretoken.value) - 1):
            pair = (pretoken.value[i], pretoken.value[i + 1])
            new_pairs[pair] = new_pairs.get(pair, 0) + pretoken.freq
            # pair_counts[pair] = pair_counts.get(pair, 0) + pretoken.freq
            # if pair[0]==new_token or pair[1]==new_token:
            #     new_pairs.add(pair)
            # if pair not in pair2ids:
            #     pair2ids[pair] = set()
            # pair2ids[pair].add(id)
        all_pairs = set(new_pairs.keys()) | set(prev_pairs.keys())
        for pair in all_pairs:
            delta = new_pairs.get(pair, 0) - prev_pairs.get(pair, 0)
            if delta != 0:
                pair_counts[pair] = pair_counts.get(pair, 0) + delta
                if pair not in pair2ids:
                    pair2ids[pair] = set()
                if new_pairs.get(pair, 0) > 0:
                    pair2ids[pair].add(id)
                else:
                    pair2ids[pair].discard(id)
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair)
                    pair2ids.pop(pair)
                updated_pairs.add(pair)
        pretokens[id] = pretoken
    #print(new_pairs)
    # Pushing newly created pairs to the heap
    assert top not in pair_counts, f"pair_counts[top]={pair_counts[top]}"
    for pair in updated_pairs:
        if pair_counts.get(pair, 0) > 0:
            heapq.heappush(heap, Pair(pair, pair_counts[pair], (vocab[pair[0]], vocab[pair[1]])))
    return vocab, merges


def merging(pretokens: list[Pretoken],
            vocab: dict[int, bytes], 
            vocab_size: int,
            merges: list[tuple[bytes, bytes]] | None = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges = [] if merges is None else merges
    pair_counts: dict[tuple[int, int], int] = {}
    pair2ids: dict[tuple[int, int], set[int]] = {}
    id2pretoken = {i: pretoken for i, pretoken in enumerate(pretokens)}
    # initialization
    for id, pretoken in id2pretoken.items():
        k = len(pretoken.value)
        for i in range(k - 1):
            pair = (pretoken.value[i], pretoken.value[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + pretoken.freq
            if pair not in pair2ids:
                pair2ids[pair] = set()
            pair2ids[pair].add(id)
    
    heap = [Pair(pair, freq, (vocab[pair[0]], vocab[pair[1]])) for pair, freq in pair_counts.items()]
    heapq.heapify(heap)
    with tqdm(total=vocab_size - len(vocab), desc="Merging pairs...") as pbar:
        while len(vocab) < vocab_size:
            if len(pair_counts) == 0:
                print(f"All pairs have been merged. Total vocabulary size {len(vocab)}.")
                break
            vocab, merges = single_merge(pretokens, heap, pair_counts, pair2ids, id2pretoken, vocab, merges)
            if len(heap)>4*len(pair_counts):
                heap = [Pair(pair, freq, (vocab[pair[0]], vocab[pair[1]])) for pair, freq in pair_counts.items()]
                heapq.heapify(heap)
            pbar.update(1)
    return vocab, merges

def train_bpe(
        input_path: str,
        vocab_size: int = 10000,
        special_tokens: list[str] = ["<|endoftext|>", "<|unknown|>"],
        split_token: str = "<|endoftext|>",
        num_chunks: int = 12,
        pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        vocab_path: str = None,
        merges_path: str = None,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocabulary with all bytes and special tokens
    special_tokens = [tok.encode("utf-8") for tok in special_tokens]
    split_token = split_token.encode("utf-8")
    vocab = initialize_vocab(special_tokens)

    # pretokenization
    print(f"Pretokenizing with {num_chunks} chunk(s)...")
    with open(input_path, "rb") as file:
        pretokens = pretokenization(
            file=file,
            num_chunks=num_chunks,
            special_tokens=special_tokens,
            split_token=split_token,
            pretokenization_pattern=pretokenization_pattern,
        )

    # Merging
    vocab, merges = merging(pretokens=pretokens, vocab=vocab, vocab_size=vocab_size, merges=None)
    #merges = [(vocab[left], vocab[right]) for left, right in merges]
    if vocab_path and merges_path:
        save_vocab_and_merges(vocab=vocab, merges=merges, vocab_path=vocab_path, merges_path=merges_path)

    return vocab, merges


def main(**kwargs):
    from .utils.config_utils import load_config
    config = load_config("configs/bpe_training_config.yaml", kwargs)
    print(f"Training on: file: {config['input_path']}, vocab size: {config['vocab_size']}, chunks: {config['num_chunks']}")
    vocab, merges = train_bpe(**config)
    #print(merges[-10:])
    #print(len(vocab), len(merges))

if __name__ == "__main__":
    import fire
    import time
    import tracemalloc
    import cProfile
    import pstats

    #tracemalloc.start()
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    fire.Fire(main)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats(30)  # Print the top 10 functions by cumulative time
    end_time = time.time()
    #current, peak = tracemalloc.get_traced_memory()
    print(f"Time taken: {end_time - start_time} seconds")
    #print(f"Current memory usage: {current / 10**6} MB; Peak: {peak / 10**6} MB")
