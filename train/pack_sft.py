#!/usr/bin/env python3
"""Pack the stage-3 dataset into tokens plus a loss mask.

Stage 3 must score **only the assistant turn**. Training on the whole sequence
teaches the model to generate `Context:` blocks of its own — exactly the
hallucination retrieval exists to prevent. So every token carries a mask bit:
0 across the context and the user turn, 1 across the answer and its EOS.

Output, alongside the stage-1/2 binaries:

    data/packed/sft_train.bin        uint16 tokens
    data/packed/sft_train_mask.bin   uint8  1 = score this token
    data/packed/sft_val.bin
    data/packed/sft_val_mask.bin

  py train/pack_sft.py --build
  py train/pack_sft.py --status
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SFT_FILE = BASE_DIR / "data" / "sft" / "stage3.jsonl"
OUT_DIR = BASE_DIR / "data" / "packed"
TOKENIZER = BASE_DIR / "tokenizer" / "marvel_bpe_50257.model"
VAL_EVERY = 200

_spec = importlib.util.spec_from_file_location(
    "sft_data", Path(__file__).resolve().parent / "sft_data.py")
_sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sd)


def encode_example(example: dict, sp, eos_id: int):
    """(tokens uint16, mask uint8). Mask is 1 only over the assistant turn."""
    text = _sd.format_example(example.get("context", ""),
                              example.get("user", ""),
                              example.get("assistant", ""))
    prompt, answer = _sd.split_prompt(text)
    p_ids = sp.encode(prompt)
    a_ids = sp.encode(answer) + [eos_id]
    toks = np.array(p_ids + a_ids, dtype=np.uint16)
    mask = np.concatenate([np.zeros(len(p_ids), dtype=np.uint8),
                           np.ones(len(a_ids), dtype=np.uint8)])
    return toks, mask


def load_tokenizer():
    import sentencepiece as spm
    if not TOKENIZER.exists():
        raise SystemExit(f"tokenizer missing: {TOKENIZER}")
    return spm.SentencePieceProcessor(model_file=str(TOKENIZER))


def build() -> int:
    if not SFT_FILE.exists():
        raise SystemExit(f"missing {SFT_FILE} — run: py train/sft_data.py --build")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = load_tokenizer()
    eos = sp.piece_to_id("<|endoftext|>")

    handles = {
        "train": (OUT_DIR / "sft_train.bin").open("wb"),
        "train_mask": (OUT_DIR / "sft_train_mask.bin").open("wb"),
        "val": (OUT_DIR / "sft_val.bin").open("wb"),
        "val_mask": (OUT_DIR / "sft_val_mask.bin").open("wb"),
    }
    totals = {"train": 0, "val": 0}
    scored = {"train": 0, "val": 0}
    t0 = time.time()
    n = 0
    try:
        with SFT_FILE.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                toks, mask = encode_example(ex, sp, eos)
                split = "val" if i % VAL_EVERY == VAL_EVERY - 1 else "train"
                toks.tofile(handles[split])
                mask.tofile(handles[f"{split}_mask"])
                totals[split] += len(toks)
                scored[split] += int(mask.sum())
                n += 1
                if n % 50000 == 0:
                    print(f"  {n:,} examples, {totals['train']/1e6:.1f}M train "
                          f"tokens ({(time.time()-t0):.0f}s)", flush=True)
    finally:
        for h in handles.values():
            h.close()

    print(f"\n{n:,} examples in {(time.time()-t0)/60:.1f} min")
    print(f"{'split':<8}{'tokens':>14}{'scored':>14}{'share':>8}")
    for split in ("train", "val"):
        t, s = totals[split], scored[split]
        print(f"{split:<8}{t:>14,}{s:>14,}{(s/max(t,1))*100:>7.1f}%")
    print(f"\n-> {OUT_DIR}")
    return 0


def status() -> int:
    for name in ("sft_train.bin", "sft_train_mask.bin",
                 "sft_val.bin", "sft_val_mask.bin"):
        p = OUT_DIR / name
        if p.exists():
            unit = 2 if name.endswith(".bin") and "mask" not in name else 1
            print(f"  {name:<24}{p.stat().st_size//unit:>14,} entries"
                  f"{p.stat().st_size/1e6:>10.1f} MB")
        else:
            print(f"  {name:<24}(missing)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        return status()
    if args.build:
        return build()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
