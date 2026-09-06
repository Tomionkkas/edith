#!/usr/bin/env python3
"""Deduplicate the curated corpus.

Two passes, both exact-match (fuzzy dedup is deliberately out of scope):

  1. **Document level** - drop records whose whole text repeats. Catches pages
     duplicated within a category and the same subject curated identically.
  2. **Long-paragraph level** - drop prose blocks of >= MIN_PARA_CHARS seen
     before anywhere in the corpus. Catches shared boilerplate and the
     Fandom/Wikipedia overlap where one mirrors the other's prose.

**Line-level dedup would destroy this corpus** and is not done. Every curated
record repeats short schema lines by design (`Reality: Earth-616`,
`Gender: Male`), and 103,811 character records share them; deduping lines
would strip the schema from all but the first record.

  py train/dedupe.py                # dedupe curated/ -> curated_dedup/
  py train/dedupe.py --dry-run      # report only, write nothing
"""
from __future__ import annotations
import argparse
import hashlib
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CURATED = BASE_DIR / "curated"
OUT_DIR = BASE_DIR / "curated_dedup"
SEPARATOR = "=" * 60
MIN_PARA_CHARS = 200          # below this a block is schema/short prose, keep it

_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Whitespace-insensitive, case-SENSITIVE key for comparison only."""
    return _WS.sub(" ", text).strip()


def _key(text: str) -> str:
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()


def dedupe_documents(docs):
    """Drop whole documents already seen. Returns (kept, stats)."""
    seen, kept, dupes = set(), [], 0
    for d in docs:
        k = _key(d)
        if k in seen:
            dupes += 1
            continue
        seen.add(k)
        kept.append(d)
    return kept, {"exact_dupes": dupes, "kept": len(kept)}


def dedupe_paragraphs(doc: str, seen: set, min_chars: int = MIN_PARA_CHARS) -> str:
    """Drop long paragraphs already seen elsewhere; leave short ones alone.

    `seen` is carried across the whole corpus by the caller so cross-file
    boilerplate is caught.
    """
    out = []
    for para in doc.split("\n\n"):
        if len(normalise(para)) < min_chars:
            out.append(para)                 # schema / short prose: always keep
            continue
        k = _key(para)
        if k in seen:
            continue
        seen.add(k)
        out.append(para)
    return "\n\n".join(out)


def read_records(path: Path):
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


def run(dry_run: bool = False) -> int:
    files = [f for f in sorted(CURATED.glob("*.txt")) if f.stat().st_size > 1000]
    if not files:
        print(f"no curated files in {CURATED}")
        return 1
    if not dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    doc_seen, para_seen = set(), set()
    grand = {"records": 0, "doc_dupes": 0, "para_dropped": 0,
             "chars_in": 0, "chars_out": 0}

    print(f"{'file':<24}{'records':>9}{'doc dupes':>11}{'paras cut':>11}"
          f"{'MB in':>9}{'MB out':>9}")
    print("-" * 73)

    for f in files:
        recs_in = kept = doc_dupes = para_dropped = 0
        c_in = c_out = 0
        out_path = OUT_DIR / f.name
        fh = None if dry_run else out_path.open("w", encoding="utf-8")
        try:
            for rec in read_records(f):
                recs_in += 1
                c_in += len(rec)
                k = _key(rec)
                if k in doc_seen:
                    doc_dupes += 1
                    continue
                doc_seen.add(k)
                before = rec.count("\n\n") + 1
                trimmed = dedupe_paragraphs(rec, para_seen)
                para_dropped += before - (trimmed.count("\n\n") + 1)
                if not trimmed.strip():
                    continue
                kept += 1
                c_out += len(trimmed)
                if fh:
                    fh.write(trimmed + "\n" + SEPARATOR + "\n\n")
        finally:
            if fh:
                fh.close()

        grand["records"] += recs_in
        grand["doc_dupes"] += doc_dupes
        grand["para_dropped"] += para_dropped
        grand["chars_in"] += c_in
        grand["chars_out"] += c_out
        print(f"{f.name:<24}{recs_in:>9,}{doc_dupes:>11,}{para_dropped:>11,}"
              f"{c_in/1e6:>9.1f}{c_out/1e6:>9.1f}")

    print("-" * 73)
    print(f"{'TOTAL':<24}{grand['records']:>9,}{grand['doc_dupes']:>11,}"
          f"{grand['para_dropped']:>11,}{grand['chars_in']/1e6:>9.1f}"
          f"{grand['chars_out']/1e6:>9.1f}")
    removed = grand["chars_in"] - grand["chars_out"]
    pct = removed / max(grand["chars_in"], 1) * 100
    print(f"\nremoved {removed/1e6:.1f} MB ({pct:.2f}%)")
    if dry_run:
        print("(dry run - nothing written)")
    else:
        print(f"written to {OUT_DIR}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    return run(ap.parse_args().dry_run)


if __name__ == "__main__":
    sys.exit(main())
