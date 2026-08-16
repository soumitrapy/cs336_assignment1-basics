import os
from typing import BinaryIO
import json


def find_chunk_boundaries(
    file: BinaryIO,
    split_token: bytes,
    num_chunks: int | None = None,
    chunk_size: int | None = None,
) -> list[int]:
    """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        ## Usage
        with open(..., "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
    """

    assert isinstance(
        split_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if chunk_size is None:
        assert num_chunks is not None, "Must specify either chunk_size or num_chunks"
        chunk_size = file_size // num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        chunk_boundaries[-1] = file_size
    else:
        assert num_chunks is None, "Must specify either chunk_size or num_chunks but not both"
        chunk_boundaries = []
        size = 0
        while size<file_size:
            chunk_boundaries.append(size)
            size += chunk_size
        if file_size - chunk_boundaries[-1]< 4096:
            chunk_boundaries[-1] = file_size
        else:
            chunk_boundaries.append(file_size)

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    chunk_boundaries = sorted(set(chunk_boundaries))
    boundaries = [(chunk_boundaries[i], chunk_boundaries[i + 1]) for i in range(len(chunk_boundaries) - 1)]
    return boundaries

def save_vocab_and_merges(vocab: dict[int, bytes],
                          merges: list[tuple[bytes, bytes]], 
                          vocab_path: str, 
                          merges_path: str) -> None:
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    os.makedirs(os.path.dirname(merges_path), exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v.hex() for k, v in vocab.items()}, f, indent=2)
    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.hex()} {right.hex()}\n")

def load_vocab_and_merges(vocab_path: str,
                          merges_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = {int(k): bytes.fromhex(v) for k, v in json.load(f).items()}
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            left_hex, right_hex = line.strip().split()
            merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))
    return vocab, merges