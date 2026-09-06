#!/usr/bin/env python3
"""Marvel SLM - Tier C: general-English grounding corpus.

Downloads FineWeb-Edu (sample-10BT) parquet shards from HuggingFace. Chosen
over Wikipedia because the Marvel corpus is already encyclopedic wiki prose --
more Wikipedia would reinforce that register rather than balance it -- and over
TinyStories, whose deliberately simple synthetic vocabulary suits 1-30M models,
not 250M.

Resumable: a shard already present at its full size is skipped, and a partial
file resumes via a Range request.

  py download_grounding.py            # default shard count
  py download_grounding.py --shards 7
  py download_grounding.py --list     # show what is on disk
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/10BT"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "grounding"
UA = {"User-Agent": "Mozilla/5.0 (marvel-slm corpus builder)"}
DEFAULT_SHARDS = 4
CHUNK = 1 << 20

# huggingface.co is blocked or throttled from some regions (notably mainland
# China, where a lot of cheap rented GPUs live). hf-mirror.com is the usual
# drop-in mirror. Override with:
#     HF_ENDPOINT=https://hf-mirror.com python crawl/download_grounding.py
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def api_tree() -> list:
    url = f"{ENDPOINT}/api/datasets/{REPO}/tree/main/{PREFIX}?recursive=1"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def shard_list() -> list:
    files = [f for f in api_tree() if f["path"].endswith(".parquet")]
    out = []
    for f in sorted(files, key=lambda x: x["path"]):
        size = f.get("size") or (f.get("lfs") or {}).get("size") or 0
        out.append((f["path"], size))
    return out


def download(path: str, size: int) -> bool:
    """Fetch one shard, resuming a partial file. True if it ended complete."""
    dest = OUT_DIR / Path(path).name
    have = dest.stat().st_size if dest.exists() else 0
    if have == size and size > 0:
        print(f"  skip (complete)  {dest.name}  {size/1e9:.2f} GB", flush=True)
        return True
    url = f"{ENDPOINT}/datasets/{REPO}/resolve/main/{path}"
    headers = dict(UA)
    mode = "wb"
    if 0 < have < size:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"  resuming at {have/1e9:.2f} GB  {dest.name}", flush=True)
    req = urllib.request.Request(url, headers=headers)
    t0 = time.time()
    got = have
    last = 0.0
    with urllib.request.urlopen(req, timeout=120) as r, dest.open(mode) as fh:
        while True:
            block = r.read(CHUNK)
            if not block:
                break
            fh.write(block)
            got += len(block)
            now = time.time()
            if now - last > 20:
                last = now
                rate = (got - have) / 1e6 / max(now - t0, 1)
                pct = got / size * 100 if size else 0
                print(f"    {dest.name}: {got/1e9:.2f}/{size/1e9:.2f} GB "
                      f"({pct:.0f}%) {rate:.1f} MB/s", flush=True)
    ok = dest.stat().st_size == size or size == 0
    print(f"  {'done' if ok else 'INCOMPLETE'}  {dest.name}  "
          f"{dest.stat().st_size/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min", flush=True)
    return ok


def show() -> int:
    if not OUT_DIR.exists():
        print("nothing downloaded yet")
        return 0
    tot = 0
    for f in sorted(OUT_DIR.glob("*.parquet")):
        tot += f.stat().st_size
        print(f"  {f.name:<28} {f.stat().st_size/1e9:>6.2f} GB")
    print(f"  {'TOTAL':<28} {tot/1e9:>6.2f} GB")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        return show()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shards = shard_list()[:args.shards]
    total = sum(s for _, s in shards)
    print(f"{REPO} / {PREFIX}  via {ENDPOINT}")
    print(f"{len(shards)} shards, {total/1e9:.1f} GB -> {OUT_DIR}\n", flush=True)

    ok = 0
    for path, size in shards:
        if download(path, size):
            ok += 1
    print(f"\n{ok}/{len(shards)} shards complete")
    return 0 if ok == len(shards) else 1


if __name__ == "__main__":
    sys.exit(main())

