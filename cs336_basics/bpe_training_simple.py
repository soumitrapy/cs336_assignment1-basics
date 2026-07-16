from .pretokenization_example import find_chunk_boundaries
import regex as re


def train_bpe(
        input_path: str = "../data/TinyStoriesV2-GPT4-train-10k.txt",
        vocab_size: int = 1000,
        special_tokens: list[bytes] = [b"<|endoftext|>", b"<|unknown|>"],
        split_special_token: bytes = b"<|endoftext|>",
        desired_num_chunks: int = 1,
        pretokenization_pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    file = open(input_path, "rb")
    count = 0
    vocab = {}
    merges = []

    # Adding all bytes to the vocabulary
    for i in range(256):
        vocab[count] = bytes([i])
        count += 1
        
    # Adding special tokens to the vocabulary
    for token in special_tokens:
        vocab[count] = token
        count += 1

    # Parallelizing pre-tokenization
    boundaries = find_chunk_boundaries(file=file, desired_num_chunks=desired_num_chunks, split_special_token=split_special_token)
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        file.seek(start)
        chunk = file.read(end - start)

        # Removing special tokens before pre-tokenization
        pattern = b"|".join(re.escape(tok) for tok in special_tokens)
        docs = re.split(pattern, chunk)

        # Pretokenization
        pretokens = {}
        for doc in docs:
            for x in re.finditer(pretokenization_pattern.encode("utf-8"), doc):
                pretoken = tuple([bytes([b]) for b in x.group(0)])
                pretokens[pretoken] = pretokens.get(pretoken, 0) + 1

        # Merging
        while len(vocab) < vocab_size:
            # Find the most frequent pair of tokens
            pairs = {}
            for pretoken, freq in pretokens.items():
                for i in range(len(pretoken) - 1):
                    pair = (pretoken[i], pretoken[i+1])
                    pairs[pair] = pairs.get(pair, 0) + freq

            if not pairs:
                break

            most_frequent_pair = max(pairs, key=lambda pair: (pairs[pair], pair))
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            # Add the new token to the vocabulary
            vocab[count] = new_token
            count += 1
            merges.append(most_frequent_pair)

            # Update pretokens with the new token
            new_pretokens = {}
            for pretoken, freq in pretokens.items():
                new_pretoken = []
                i = 0
                while i < len(pretoken):
                    if i < len(pretoken) - 1 and (pretoken[i], pretoken[i+1]) == most_frequent_pair:
                        new_pretoken.append(new_token)
                        i += 2
                    else:
                        new_pretoken.append(pretoken[i])
                        i += 1
                new_pretokens[tuple(new_pretoken)] = new_pretokens.get(tuple(new_pretoken), 0) + freq
            pretokens = new_pretokens
    return vocab, merges


if __name__ == "__main__":
    import time
    start_time = time.time()
    # input_path = "data/sample.txt"
    input_path = "data/TinyStoriesV2-GPT4-train-1k.txt"
    # input_path = "data/TinyStoriesV2-GPT4-train-10k.txt"
    #input_path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 8000
    special_tokens = [b"<|endoftext|>", b"<|unknown|>"]
    split_special_token = b"<|endoftext|>"
    desired_num_chunks = 10
    pretokenization_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        split_special_token=split_special_token,
        desired_num_chunks=desired_num_chunks,
        pretokenization_pattern=pretokenization_pattern,
    )
    end_time = time.time()
    print(
        f"chunks: {desired_num_chunks}, Time: {end_time - start_time} seconds, file: {input_path}, vocab size: {vocab_size}"
    )
    print(merges[-10:])
    print(len(vocab), len(merges))
