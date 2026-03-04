# %%
from collections import defaultdict
from typing import BinaryIO
from tqdm import tqdm
from multiprocessing import Pool, RLock
from dataclasses import dataclass
from itertools import repeat, count
from heapq import heapify, heappop, heappush, heapreplace
import pickle
import os
import sys

import regex as re

text = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

@dataclass(frozen=True)
class RevLex:
    pair: tuple[bytes, bytes]
    def __lt__(self, other: "RevLex") -> bool:
        return self.pair > other.pair

def add_cnt(heap, cnt, pair, count):
    cnt[pair] += count
    heappush(
        heap,
        (
            -cnt[pair],
            RevLex(pair),
            pair
        )
    )

def get_max(heap, cnt):
    while heap:
        c, _, p = heappop(heap)
        if cnt.get(p, 0) == -c:
            return p

    assert False

def merge_pair(pair_heap, words, word_counts, pair_to_words, pair_count, pair):
    pair_words_idx = pair_to_words[pair]
    merged_pair = pair[0]+pair[1]
    for w_idx in pair_words_idx:
        word = words[w_idx]
        word_count = word_counts[w_idx]
        idx = 0
        while idx < len(word)-1:
            curr_pair = (word[idx], word[idx+1])
            if curr_pair == pair:
                if idx > 0:
                    prev_pair = (word[idx-1], word[idx])
                    add_cnt(pair_heap, pair_count, prev_pair, -word_count)
                    if pair_count[prev_pair] == 0:
                        pair_to_words[prev_pair].remove(w_idx)

                    new_pair = (word[idx-1], merged_pair)
                    add_cnt(pair_heap, pair_count, new_pair, word_count)
                    pair_to_words[new_pair].add(w_idx)
                
                if idx < len(word) - 2:
                    prev_pair = (word[idx+1], word[idx+2])
                    add_cnt(pair_heap, pair_count, prev_pair, -word_count)
                    if pair_count[prev_pair] == 0:
                        pair_to_words[prev_pair].remove(w_idx)

                    new_pair = (merged_pair, word[idx+2])
                    add_cnt(pair_heap, pair_count, new_pair, word_count)
                    pair_to_words[new_pair].add(w_idx)
                
                word = word[:idx] + (merged_pair,) + word[idx+2:]
                idx += 1
            idx += 1

        words[w_idx] = word

    del pair_count[pair]
    del pair_to_words[pair]


def tokenize(word_counts, vocab_size, special_tokens):
    tuple_word_counts = {tuple(bytes([p]) for p in w.encode()):c for w, c in word_counts.items()}
    words = list(tuple_word_counts.keys())
    word_to_idx = {w:i for i, w in enumerate(words)}
    word_counts = [tuple_word_counts[w] for w in words]

    pair_to_words = defaultdict(set)
    pair_count = defaultdict(int)
    pair_heap = []

    for word in tqdm(words, "initial word count"):
        w_idx = word_to_idx[word]
        for a, b in zip(word[:-1], word[1:]):
            add_cnt(pair_heap, pair_count, (a, b), word_counts[w_idx])
            pair_to_words[(a, b)].add(w_idx)

    merged_pairs = []

    vocab = [s.encode() for s in special_tokens] + [bytes([b]) for b in range(256)]
    with tqdm(total=vocab_size, initial=len(vocab), desc="training bpe") as bar:
        while len(vocab) < vocab_size:
            max_pair = get_max(pair_heap, pair_count)
            merged_pairs.append(max_pair)
            vocab.append(max_pair[0] + max_pair[1])
            merge_pair(pair_heap, words, word_counts, pair_to_words, pair_count, max_pair)
            bar.update(1)
    
    return {i:v for i, v in enumerate(vocab)}, merged_pairs

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

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
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def worker_pools(chunk, special_tokens, idx):
    parts = [chunk]
    for token in special_tokens:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(token))
        parts = new_parts
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    c_pat = re.compile(PAT)
    
    word_counts = defaultdict(int)
    for part in tqdm(parts, position=idx, desc=f"pre-tokenization thread {idx}"):
        pre_tokens = re.finditer(c_pat, part)
        for w in pre_tokens:
            word_counts[w.group()] += 1
    
    return word_counts

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict, list]:
    num_processes = 8
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

    tqdm.set_lock(RLock())
    with Pool(processes=num_processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        word_count_per_pool = pool.starmap(worker_pools, zip(chunks, repeat(special_tokens), count()))

    del chunks

    word_counts = defaultdict(int)
    for wc in tqdm(word_count_per_pool, "merging counts"):
        for k, v in wc.items():
            word_counts[k] += v

    del word_count_per_pool

    return tokenize(word_counts, vocab_size, special_tokens)

def train_bpe_tinystories():
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    train_data = "./data/TinyStoriesV2-GPT4-train.txt"

    vocab, merges = train_bpe(train_data, vocab_size, special_tokens)

    with open("./cs336_basics/output/bpe_tinystories.pkl", "w+b") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

def train_bpe_openwebtext():
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    train_data = "./data/owt_train.txt"

    vocab, merges = train_bpe(train_data, vocab_size, special_tokens)

    with open("./cs336_basics/output/bpe_owt.pkl", "w+b") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
# %%
if __name__ == "__main__":
    train_bpe_tinystories()