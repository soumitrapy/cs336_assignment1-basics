from .pretokenization_example import find_chunk_boundaries
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
import pickle
from tqdm import tqdm

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


def pretokenization(
    file: BinaryIO,
    desired_num_chunks: int = 1,
    special_tokens: list[bytes] = [b"<|endoftext|>", b"<|unknown|>"],
    split_special_token: bytes = b"<|endoftext|>",
    pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> dict[tuple[bytes, ...], int]:
    boundaries = find_chunk_boundaries(file=file, desired_num_chunks=desired_num_chunks, split_special_token=split_special_token)
    chunks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        file.seek(start)
        chunk = file.read(end - start)
        chunks.append(chunk)
    with Pool(processes=desired_num_chunks) as pool:
        results = pool.starmap(pretokenization_chunk, [(chunk, special_tokens, pretokenization_pattern) for chunk in chunks])
    pretokens = {}
    for result in results:
        for pretoken, freq in result.items():
            pretokens[pretoken] = pretokens.get(pretoken, 0) + freq
    return pretokens

def merge_pairs(
    pairs: dict[tuple[bytes, bytes], int],
    pretokens: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> list[tuple[bytes, bytes]]:
    most_frequent_pair = max(pairs, key=lambda pair: (pairs[pair], pair))
    new_token = most_frequent_pair[0] + most_frequent_pair[1]
    pairs.pop(most_frequent_pair)
    merges.append(most_frequent_pair)
    vocab[len(vocab)] = new_token

    # Update pretokens with the new token
    new_pretokens = {}
    for pretoken, freq in pretokens.items():
        new_pretoken = []
        i = 0
        k = len(pretoken)
        while i < k:
            if i < k - 1 and (pretoken[i], pretoken[i+1]) == most_frequent_pair:
                new_pretoken.append(new_token)
                prev_pairs, new_pairs = [], []
                if i>0:
                    prev_pairs.append((pretoken[i-1], pretoken[i]))
                    new_pairs.append((pretoken[i-1], new_token))
                if i < k - 2:
                    prev_pairs.append((pretoken[i+1], pretoken[i+2]))
                    new_pairs.append((new_token, pretoken[i+2]))
                
                for prev_pair, new_pair in zip(prev_pairs, new_pairs):
                    pairs[new_pair] = pairs.get(new_pair, 0) + freq
                    if prev_pair in pairs:
                        pairs[prev_pair] -= freq
                        if pairs[prev_pair] <= 0:
                            pairs.pop(prev_pair)

                i += 2
            else:
                new_pretoken.append(pretoken[i])
                i += 1
        new_pretokens[tuple(new_pretoken)] = new_pretokens.get(tuple(new_pretoken), 0) + freq
    pretokens = new_pretokens
    return pairs, pretokens, vocab, merges


def train_bpe(
        input_path: str = "../data/TinyStoriesV2-GPT4-train-10k.txt",
        vocab_size: int = 1000,
        special_tokens: list[bytes] = [b"<|endoftext|>", b"<|unknown|>"],
        split_special_token: bytes = b"<|endoftext|>",
        desired_num_chunks: int = 1,
        pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        save_path: str = None,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocabulary with all bytes and special tokens
    vocab = initialize_vocab(special_tokens)

    # pretokenization
    print(f"Pretokenizing with {desired_num_chunks} chunk(s)...")
    with open(input_path, "rb") as file:
        pretokens = pretokenization(
            file=file,
            desired_num_chunks=desired_num_chunks,
            special_tokens=special_tokens,
            split_special_token=split_special_token,
            pretokenization_pattern=pretokenization_pattern,
        )

    # Initialize pairs and merges
    merges = []
    pairs = {}
    for pretoken, freq in pretokens.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i+1])
            pairs[pair] = pairs.get(pair, 0) + freq

    # Merging
    with tqdm(total=vocab_size - len(vocab), desc="Merging pairs...") as pbar:
        while len(vocab) < vocab_size:
            pairs, pretokens, vocab, merges = merge_pairs(pairs, pretokens, vocab, merges)
            if not pairs:
                break
            pbar.update(1)
    # saving
    if save_path:
        print(f"Saving to {save_path}...")
        with open(save_path, "wb") as f:
            pickle.dump({"vocab": vocab, 
                         "merges": merges, 
                         "pretokenization_pattern": pretokenization_pattern
                        }, f)
    
    return vocab, merges


def main(**kwargs):
    from .utils import load_config
    config = load_config("configs/bpe_training_config.yaml", kwargs)
    print(f"Training on: file: {config['input_path']}, vocab size: {config['vocab_size']}, chunks: {config['desired_num_chunks']}")
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
