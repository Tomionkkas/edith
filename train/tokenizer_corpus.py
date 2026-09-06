#!/usr/bin/env python3
"""Build the tokenizer training sample: Marvel text mixed with general English.

The tokenizer serves BOTH training stages, so it must see both distributions.
Train it only on Marvel and general prose fragments badly; only on general
English and every Marvel entity name shatters into pieces.

Sampling is deterministic given a seed so the mix is reproducible.
"""
from __future__ import annotations
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CURATED = BASE_DIR / "curated"
GROUNDING = BASE_DIR / "data" / "raw" / "grounding"
SEPARATOR = "=" * 60


def marvel_records(path: Path):
    """Yield one curated record at a time without loading the whole file."""
    buf = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(SEPARATOR):
                rec = "".join(buf).strip()
                if rec:
                    yield rec
                buf = []
            else:
                buf.append(line)
    rec = "".join(buf).strip()
    if rec:
        yield rec


def take_chars(records, budget: int, rng: random.Random, keep_prob: float = 1.0):
    """Pull records until `budget` characters are collected.

    `keep_prob` < 1 samples across the whole file rather than taking a
    contiguous prefix, which would over-represent alphabetically early pages.
    """
    out, total = [], 0
    for rec in records:
        if total >= budget:
            break
        if keep_prob < 1.0 and rng.random() > keep_prob:
            continue
        out.append(rec)
        total += len(rec)
    return out, total


def general_docs(budget: int, rng: random.Random, files=None):
    """Documents from the FineWeb-Edu parquet shards, up to `budget` chars."""
    import pyarrow.parquet as pq
    files = files if files is not None else sorted(GROUNDING.glob("*.parquet"))
    out, total = [], 0
    for f in files:
        if total >= budget:
            break
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            for t in batch["text"]:
                s = t.as_py()
                if not s:
                    continue
                out.append(s)
                total += len(s)
                if total >= budget:
                    break
            if total >= budget:
                break
    return out, total


def interleave(marvel: list, general: list, rng: random.Random) -> list:
    """Shuffle the two pools together so neither dominates any file region."""
    merged = list(marvel) + list(general)
    rng.shuffle(merged)
    return merged


def write_sample(docs: list, dest: Path) -> int:
    """One line per non-empty line of each document. Returns lines written."""
    n = 0
    with dest.open("w", encoding="utf-8") as fh:
        for doc in docs:
            for line in doc.split("\n"):
                line = line.strip()
                if line:
                    fh.write(line + "\n")
                    n += 1
    return n
