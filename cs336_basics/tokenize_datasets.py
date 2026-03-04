from typing import Iterator
import pickle
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


def read_file_chunks(path, chunksize=2**20) -> Iterator[str]:
    with open(path, "r") as f:
        while True:
            data = f.read(chunksize)
            if not data:
                break
            yield data


def estimate_num_chunks(path: str, chunksize: int) -> int:
    file_size = os.path.getsize(path)
    return (file_size + chunksize - 1) // chunksize


def chunks_with_progress(path: str, chunksize: int, pbar: tqdm) -> Iterator[str]:
    for chunk in read_file_chunks(path, chunksize=chunksize):
        pbar.update(1)
        yield chunk


def encode_file(
    path, save_path, tokenizer: Tokenizer, flush_size=2**20, chunksize=2**20
):
    buf = []
    total_chunks = estimate_num_chunks(path, chunksize)
    tokens_written = 0
    with open(save_path, "wb") as f_out:
        desc = f"Tokenizing {Path(path).name}"
        with tqdm(total=total_chunks, desc=desc, unit="chunk", mininterval=0.2) as pbar:
            chunks = chunks_with_progress(path, chunksize, pbar)
            token_chunks = tokenizer.encode_iterable(chunks)
            for t in token_chunks:
                buf.append(t)
                if len(buf) >= flush_size:
                    np.asarray(buf, dtype=np.uint16).tofile(f_out)
                    tokens_written += len(buf)
                    buf.clear()
                    pbar.set_postfix_str(f"tokens={tokens_written:,}")

        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f_out)


if __name__ == "__main__":
    with open("cs336_basics/output/bpe_owt.pkl", "rb") as f:
        s = pickle.load(f)
    token = Tokenizer(s["vocab"], s["merges"], ["<|endoftext|>"])
    encode_file("data/owt_train.txt", "data/tokenized/owt_train.npy", token)
