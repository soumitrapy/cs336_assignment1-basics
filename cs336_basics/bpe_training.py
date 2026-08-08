# from .pretokenization_example import find_chunk_boundaries
from .utils.tokenization_utils import find_chunk_boundaries
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
import pickle
from tqdm import tqdm
import heapq
from dataclasses import dataclass

def initialize_vocab(special_tokens: list[bytes]) -> dict[int, bytes]:
    vocab = {}
    count = 0
    # Adding all bytes to the vocabulary
    for i in range(256):
        vocab[count] = bytes([i])
        count += 1

    # Adding special tokens to the vocabulary
    for token in special_tokens:
        vocab[count] = token
        count += 1
    return vocab


def pretokenization_chunk(
    chunk: bytes,
    special_tokens: list[bytes],
    pretokenization_pattern: str,
) -> dict[tuple[bytes, ...], int]:

    special_tokens_pattern = b"|".join(re.escape(tok) for tok in special_tokens)
    docs = re.split(special_tokens_pattern, chunk)

    # Pretokenization
    pretokens = {}
    for doc in docs:
        for x in re.finditer(pretokenization_pattern.encode("utf-8"), doc):
            pretoken = tuple([bytes([b]) for b in x.group(0)])
            pretokens[pretoken] = pretokens.get(pretoken, 0) + 1
    return pretokens

@dataclass
class Pretoken:
    value: tuple[bytes, ...]
    freq: int
    def _search_pair(self, pair: tuple[bytes, bytes]) -> int:
        for i in range(len(self.value) - 1):
            if (self.value[i], self.value[i + 1]) == pair:
                return i
        return -1
    def _merge(self, pair: tuple[bytes, bytes]) -> bool:
        idx = self._search_pair(pair)
        if idx < 0:
            return False
        merged = pair[0] + pair[1]
        self.value = self.value[:idx] + (merged,) + self.value[idx + 2 :]
        return True

def pretokenization(
    file: BinaryIO,
    num_chunks: int = 1,
    special_tokens: list[bytes] = [b"<|endoftext|>", b"<|unknown|>"],
    split_special_token: bytes = b"<|endoftext|>",
    pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> dict[tuple[bytes, ...], int]:
    boundaries = find_chunk_boundaries(file=file, desired_num_chunks=num_chunks, split_special_token=split_special_token)
    chunks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        file.seek(start)
        chunk = file.read(end - start)
        chunks.append(chunk)
    with Pool(processes=num_chunks) as pool:
        results = pool.starmap(pretokenization_chunk, [(chunk, special_tokens, pretokenization_pattern) for chunk in chunks])
    pretokens = []
    for result in results:
        for pretoken, freq in result.items():
            pretokens.append(Pretoken(value=pretoken, freq=freq))
    return pretokens

@dataclass(frozen=True)
class Pair:
    pair: tuple[bytes, bytes]
    freq: int

    def __lt__(self, other: "Pair") -> bool:
        if self.freq == other.freq:
            return self.pair > other.pair  # For deterministic behavior
        return self.freq > other.freq  # For max-heap behavior


def merge_pairs(
    pretokens: list[Pretoken],
    heap: list[Pair],
    pair_counts: dict[tuple[bytes, bytes], int],
    inverted_index: dict[tuple[bytes, bytes], set[int]],
    id2pretoken: dict[int, Pretoken],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    mfpair = heapq.heappop(heap).pair
    # lazy invalidation: if most_frequent_pair is a stale pair ignore it and pop the next one
    while mfpair not in pair_counts:
        mfpair = heapq.heappop(heap).pair

    new_token = mfpair[0] + mfpair[1]
    merges.append(mfpair)
    vocab[len(vocab)] = new_token

    # Updates
    all_newly_created_pairs = set()
    pair_counts.pop(mfpair)
    for id in inverted_index[mfpair]:
        pretoken = id2pretoken[id]
        idx = pretoken._search_pair(mfpair)
        while idx != -1:
            prev_pairs = []
            new_pairs = []
            if idx > 0:
                prev_pairs.append((pretoken.value[idx - 1], pretoken.value[idx]))
                new_pairs.append((pretoken.value[idx - 1], new_token))
            if idx < len(pretoken.value) - 2:
                prev_pairs.append((pretoken.value[idx + 1], pretoken.value[idx + 2]))
                new_pairs.append((new_token, pretoken.value[idx + 2]))
            all_newly_created_pairs.update(new_pairs)
            for prev_pair, new_pair in zip(prev_pairs, new_pairs):
                pair_counts[prev_pair] -= pretoken.freq
                inverted_index[prev_pair].remove(id)
                if pair_counts[prev_pair] <= 0:
                    pair_counts.pop(prev_pair)
                    inverted_index.pop(prev_pair)

                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + pretoken.freq
                if new_pair not in inverted_index:
                    inverted_index[new_pair] = set()
                inverted_index[new_pair].add(id)
                
            pretoken._merge(mfpair)
            idx = pretoken._search_pair(mfpair)
        pretokens[id] = pretoken
    inverted_index.pop(mfpair)
    
    for new_pair in all_newly_created_pairs:
        heapq.heappush(heap, Pair(new_pair, pair_counts[new_pair]))
    return vocab, merges


def merging(pretokens: list[Pretoken],
            vocab: dict[int, bytes], 
            vocab_size: int,
            merges = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if merges is None:
        merges = []
    pair_counts = {}    # count of pairs across all pretokens
    inverted_index = {} # a mapping from pairs to the pretokens that contain them
    id2pretoken = {i: pretoken for i, pretoken in enumerate(pretokens)}
    # initialization
    for id, pretoken in id2pretoken.items():
        k = len(pretoken.value)
        for i in range(k - 1):
            pair = (pretoken.value[i], pretoken.value[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + pretoken.freq
            if pair not in inverted_index:
                inverted_index[pair] = set()
            inverted_index[pair].add(id)
    
    heap = [Pair(pair, freq) for pair, freq in pair_counts.items()]
    heapq.heapify(heap)
    
    with tqdm(total=vocab_size - len(vocab), desc="Merging pairs...") as pbar:
        while len(vocab) < vocab_size:
            vocab, merges = merge_pairs(pretokens, heap, pair_counts, inverted_index, id2pretoken, vocab, merges)
            if len(pair_counts) == 0:
                print(f"All pairs have been merged. Total vocabulary size {len(vocab)}.")
                break
            if pbar.n % 100 == 0:
                pbar.update(1)
    return vocab, merges

def train_bpe(
        input_path: str = "../data/TinyStoriesV2-GPT4-train-10k.txt",
        vocab_size: int = 1000,
        special_tokens: list[str] = ["<|endoftext|>", "<|unknown|>"],
        split_special_token: str = "<|endoftext|>",
        num_chunks: int = 1,
        pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        save_path: str = None,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocabulary with all bytes and special tokens
    special_tokens = [tok.encode("utf-8") for tok in special_tokens]
    split_special_token = split_special_token.encode("utf-8")
    vocab = initialize_vocab(special_tokens)

    # pretokenization
    print(f"Pretokenizing with {num_chunks} chunk(s)...")
    with open(input_path, "rb") as file:
        pretokens = pretokenization(
            file=file,
            num_chunks=num_chunks,
            special_tokens=special_tokens,
            split_special_token=split_special_token,
            pretokenization_pattern=pretokenization_pattern,
        )

    # Merging
    vocab, merges = merging(pretokens=pretokens, vocab=vocab, vocab_size=vocab_size, merges=None)

    if save_path:
        print(f"Saving to {save_path}...")
        with open(save_path, "wb") as f:
            pickle.dump({"vocab": vocab, 
                         "merges": merges, 
                         "pretokenization_pattern": pretokenization_pattern
                        }, f)
    
    return vocab, merges


def main(**kwargs):
    from .utils.config_utils import load_config
    config = load_config("configs/bpe_training_config.yaml", kwargs)
    print(f"Training on: file: {config['input_path']}, vocab size: {config['vocab_size']}, chunks: {config['num_chunks']}")
    vocab, merges = train_bpe(**config)
    print(merges[-10:])
    print(len(vocab), len(merges))

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
    stats.print_stats(30)  # Print the top 10 functions by cumulative time
    end_time = time.time()
    #current, peak = tracemalloc.get_traced_memory()
    print(f"Time taken: {end_time - start_time} seconds")
    #print(f"Current memory usage: {current / 10**6} MB; Peak: {peak / 10**6} MB")
