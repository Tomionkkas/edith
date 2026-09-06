"""Fetch specific pages and curate them into the corpus.

For repairing gaps the main crawl left. It writes a separate raw/curated pair
rather than touching the originals, so a patch can be inspected, rebuilt or
discarded without re-running a crawl that takes an hour.

    py crawl/fetch_pages.py "Victor von Doom (Earth-616)"
    py crawl/fetch_pages.py --file crawl/missing_titles.txt

Then rebuild what reads the corpus:

    py retrieve/search.py --build
    py retrieve/build_names.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
CURATED = ROOT / "curated"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="*")
    ap.add_argument("--file", help="one title per line")
    ap.add_argument("--name", default="patch", help="raw/<name>.jsonl, curated/<name>.txt")
    ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()

    titles = list(args.titles)
    if args.file:
        titles += [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    titles = list(dict.fromkeys(titles))
    if not titles:
        print("no titles given")
        return 2

    crawl = _load("crawl", "crawl/crawl.py")
    curate = _load("curate", "crawl/curate.py")

    RAW.mkdir(exist_ok=True)
    CURATED.mkdir(exist_ok=True)
    raw_path = RAW / f"{args.name}.jsonl"
    got_total = 0
    with raw_path.open("w", encoding="utf-8") as fh:
        for i in range(0, len(titles), args.batch):
            chunk = titles[i:i + args.batch]
            got, _ = crawl.fetch_wikitext_batch(crawl.FANDOM, chunk)
            for title in chunk:
                wt = got.get(title)
                if not wt:
                    print(f"   MISSING on Fandom: {title}")
                    continue
                fh.write(json.dumps({"title": title, "wikitext": wt},
                                    ensure_ascii=False) + "\n")
                got_total += 1
            print(f"   fetched {got_total}/{len(titles)}")

    # Curate with the SAME function the main corpus used, so a patched record
    # is indistinguishable from an original one.
    out_path = CURATED / f"{args.name}.txt"
    n = empty = 0
    with raw_path.open(encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            text = curate.curate_character(rec)
            if not text:
                empty += 1
                continue
            fout.write(text + "\n" + "=" * 60 + "\n\n")
            n += 1
    print(f"\nfetched {got_total}/{len(titles)} -> {raw_path.name}")
    print(f"curated {n} ({empty} empty) -> {out_path.name}")
    print("\nnow rebuild:  py retrieve/search.py --build  &&  py retrieve/build_names.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
