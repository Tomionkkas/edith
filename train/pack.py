#!/usr/bin/env python3
"""Pack the corpora into tokenized uint16 shards for training.

Output is nanoGPT-style flat binaries: every document encoded, an EOS token
appended after each, concatenated, written as little-endian uint16. The
tokenizer's vocab (50,257) fits in uint16, halving both disk and the bytes the
data loader touches per step.

Two tiers, written separately because the two training stages consume them
separately:

    data/packed/marvel_{train,val}.bin      ~58M tokens   (stage 2)
    data/packed/general_{train,val}.bin   ~3.63B tokens   (stage 1)

The general tier is packed shard by shard and is resumable -- it is ~7 GB of
output and an hour of CPU, so it should survive being interrupted.

  py train/pack.py --marvel                # fast, ~2 min
  py train/pack.py --general               # slow, resumable, uses all cores
  py train/pack.py --general --workers 4   # leave cores free
  py train/pack.py --status
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CURATED = BASE_DIR / "curated"
GROUNDING = BASE_DIR / "data" / "raw" / "grounding"
OUT_DIR = BASE_DIR / "data" / "packed"
TOKENIZER = BASE_DIR / "tokenizer" / "marvel_bpe_50257.model"
SEPARATOR = "=" * 60
VAL_EVERY = 200                 # 1 doc in 200 held out
DTYPE = np.uint16


# ------------------------------------------------------------------ encoding

def encode_documents(docs, sp, eos_id: int) -> np.ndarray:
    """Documents -> one uint16 array, EOS after each document."""
    ids: list[int] = []
    for d in docs:
        if not d or not d.strip():
            continue
        ids.extend(sp.encode(d))
        ids.append(eos_id)
    return np.array(ids, dtype=DTYPE) if ids else np.empty(0, dtype=DTYPE)


def split_train_val(docs, val_every: int = VAL_EVERY):
    """Strided split. A contiguous tail would put one whole category in val."""
    train, val = [], []
    for i, d in enumerate(docs):
        (val if i % val_every == val_every - 1 else train).append(d)
    return train, val


# ------------------------------------------------------------------- writing

class ShardWriter:
    """Append uint16 arrays to one flat binary."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("ab")
        self.total = path.stat().st_size // 2 if path.exists() else 0

    def write(self, arr: np.ndarray) -> None:
        if arr.size:
            arr.astype(DTYPE, copy=False).tofile(self._fh)
            self.total += int(arr.size)

    def close(self) -> None:
        self._fh.close()


class Manifest:
    """Which source shards are already packed, so --general can resume."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def done(self, name: str) -> bool:
        return name in self.data

    def tokens(self, name: str) -> int:
        return int(self.data.get(name, 0))

    def mark(self, name: str, n_tokens: int) -> None:
        self.data[name] = int(n_tokens)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")


# ------------------------------------------------------------------- sources

def marvel_records(path: Path):
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


def load_tokenizer():
    import sentencepiece as spm
    if not TOKENIZER.exists():
        raise SystemExit(f"tokenizer missing: {TOKENIZER}\n"
                         f"run: py train/train_tokenizer.py")
    return spm.SentencePieceProcessor(model_file=str(TOKENIZER))


# --------------------------------------------------------------------- packs

def pack_marvel() -> int:
    sp = load_tokenizer()
    eos = sp.piece_to_id("<|endoftext|>")
    files = [f for f in sorted(CURATED.glob("*.txt")) if f.stat().st_size > 1000]
    tr_w = ShardWriter(OUT_DIR / "marvel_train.bin")
    va_w = ShardWriter(OUT_DIR / "marvel_val.bin")
    t0 = time.time()
    print(f"{'file':<24}{'records':>9}{'train tok':>13}{'val tok':>11}")
    print("-" * 57)
    try:
        for f in files:
            docs = list(marvel_records(f))
            tr, va = split_train_val(docs)
            a = encode_documents(tr, sp, eos)
            b = encode_documents(va, sp, eos)
            tr_w.write(a)
            va_w.write(b)
            print(f"{f.name:<24}{len(docs):>9,}{a.size:>13,}{b.size:>11,}", flush=True)
    finally:
        tr_w.close()
        va_w.close()
    print("-" * 57)
    print(f"{'TOTAL':<24}{'':>9}{tr_w.total:>13,}{va_w.total:>11,}")
    print(f"\nwrote {(tr_w.total + va_w.total) * 2 / 1e6:.0f} MB "
          f"in {(time.time() - t0) / 60:.1f} min -> {OUT_DIR}")
    return 0


_WORKER_SP = None


def _init_worker(model_path: str) -> None:
    """Load the tokenizer once per worker process, not once per chunk."""
    global _WORKER_SP
    import sentencepiece as spm
    _WORKER_SP = spm.SentencePieceProcessor(model_file=model_path)


def _encode_chunk(args):
    texts, eos = args
    return encode_documents(texts, _WORKER_SP, eos)


def pack_general(workers: int, limit_shards: int | None) -> int:
    import pyarrow.parquet as pq
    from concurrent.futures import ProcessPoolExecutor

    sp = load_tokenizer()
    eos = sp.piece_to_id("<|endoftext|>")
    man = Manifest(OUT_DIR / "general_manifest.json")
    shards = sorted(GROUNDING.glob("*.parquet"))
    if limit_shards:
        shards = shards[:limit_shards]
    if not shards:
        raise SystemExit(f"no parquet in {GROUNDING} — run download_grounding.py")

    tr_w = ShardWriter(OUT_DIR / "general_train.bin")
    va_w = ShardWriter(OUT_DIR / "general_val.bin")
    t0 = time.time()
    try:
        for shard in shards:
            if man.done(shard.name):
                print(f"  skip (done)  {shard.name}  "
                      f"{man.tokens(shard.name):,} tokens", flush=True)
                continue
            s0 = time.time()
            before = tr_w.total + va_w.total
            pf = pq.ParquetFile(shard)
            group_size = max(workers * 4, 4)

            def flush(group, ex):
                """Encode a group of batches in parallel, write in order."""
                trains, vals = [], []
                for texts in group:
                    tr, va = split_train_val(texts)
                    trains.append((tr, eos))
                    vals.append((va, eos))
                for arr in ex.map(_encode_chunk, trains):
                    tr_w.write(arr)
                for arr in ex.map(_encode_chunk, vals):
                    va_w.write(arr)

            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker,
                                     initargs=(str(TOKENIZER),)) as ex:
                group, n_groups = [], 0
                for b in pf.iter_batches(batch_size=2000, columns=["text"]):
                    group.append([t.as_py() for t in b["text"]])
                    if len(group) >= group_size:
                        flush(group, ex)
                        group = []
                        n_groups += 1
                        if n_groups % 5 == 0:
                            done = tr_w.total + va_w.total
                            el = max(time.time() - s0, 1)
                            print(f"    {shard.name}: {done/1e9:.2f}B tokens "
                                  f"({(done-before)/el/1e6:.2f}M tok/s)", flush=True)
                if group:
                    flush(group, ex)
            man.mark(shard.name, tr_w.total + va_w.total - before)
            print(f"  done  {shard.name}  "
                  f"+{tr_w.total + va_w.total - before:,} tokens "
                  f"in {(time.time()-s0)/60:.1f} min", flush=True)
    finally:
        tr_w.close()
        va_w.close()
    print(f"\ntrain {tr_w.total:,} tokens | val {va_w.total:,} tokens")
    print(f"{(tr_w.total + va_w.total) * 2 / 1e9:.2f} GB in "
          f"{(time.time()-t0)/60:.1f} min -> {OUT_DIR}")
    return 0


def status() -> int:
    if not OUT_DIR.exists():
        print("nothing packed yet")
        return 0
    print(f"{'file':<26}{'tokens':>16}{'size':>10}")
    print("-" * 52)
    for f in sorted(OUT_DIR.glob("*.bin")):
        n = f.stat().st_size // 2
        print(f"{f.name:<26}{n:>16,}{f.stat().st_size/1e9:>8.2f} GB")
    man = Manifest(OUT_DIR / "general_manifest.json")
    if man.data:
        print(f"\ngeneral shards packed: {len(man.data)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marvel", action="store_true")
    ap.add_argument("--general", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shards", type=int, default=None)
    args = ap.parse_args()
    if args.status:
        return status()
    if args.marvel:
        return pack_marvel()
    if args.general:
        return pack_general(args.workers, args.shards)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
