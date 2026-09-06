"""Take a representative slice of a packed token file.

Stage-3 replay consumes ~6.9M general tokens per epoch, while
general_train.bin holds 2.94B - 426x more than one epoch needs. Uploading
5.9 GB to a rented box to use 0.2% of it is a poor trade when the upload runs
over a home connection.

A prefix would be wrong: the file is four parquet shards concatenated, so the
first N tokens come from shard 0 only. Evenly spaced chunks cover the whole
corpus while keeping each chunk contiguous, which matters because the loader
reads windows of `block_size` consecutive tokens - a shuffled slice would
splice unrelated documents mid-sentence at every boundary.

    py train/slice_bin.py data/packed/general_train.bin \
        data/packed/general_slice.bin --tokens 100000000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def chunk_starts(total: int, tokens: int, chunks: int) -> np.ndarray:
    """Evenly spaced, non-overlapping, int64.

    int64 is not cosmetic: numpy defaults to int32 on Windows and this file
    holds 2.94e9 tokens, which wraps negative.
    """
    per = max(1, tokens // chunks)
    last = max(total - per, 0)
    n = min(chunks, last + 1)
    return np.linspace(0, last, n).astype(np.int64), per


def slice_bin(src: Path, dst: Path, tokens: int, chunks: int = 100) -> int:
    data = np.memmap(src, dtype=np.uint16, mode="r")
    if tokens >= len(data):
        raise SystemExit(f"{src} has only {len(data):,} tokens")
    starts, per = chunk_starts(len(data), tokens, chunks)
    with dst.open("wb") as fh:
        written = 0
        for s in starts:
            block = np.asarray(data[s:s + per], dtype=np.uint16)
            block.tofile(fh)
            written += len(block)
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--tokens", type=int, default=100_000_000)
    ap.add_argument("--chunks", type=int, default=100)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    n = slice_bin(src, dst, args.tokens, args.chunks)
    print(f"{src.name} {src.stat().st_size/1e9:.2f} GB "
          f"-> {dst.name} {dst.stat().st_size/1e6:.0f} MB "
          f"({n:,} tokens from {args.chunks} spread chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
