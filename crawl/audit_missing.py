"""Find substantial character pages the crawl missed.

The crawl reached 187,580 records but dropped some canonical pages: there is
no Victor von Doom (Earth-616), only four small appearance stubs, so "who
created doctor doom" answers from a Gwenpool cameo. No retrieval change can
fix a page that was never fetched.

Comparing titles alone is not enough, because a stub with the right name looks
like a hit. This compares SIZE: Fandom reports page length, and a 132 KB page
represented locally by a 6 KB record means the real one is missing.

    py crawl/audit_missing.py --limit 3000 --min-bytes 20000
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def page_sizes(crawl, titles, batch=50):
    """title -> byte length, from the API rather than by fetching content."""
    sizes = {}
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        data = crawl.http_json(crawl.FANDOM, {
            "action": "query", "prop": "info", "format": "json",
            "titles": "|".join(chunk)})
        for page in data.get("query", {}).get("pages", {}).values():
            if "length" in page:
                sizes[page["title"]] = page["length"]
    return sizes


def local_best(index, resolve, names, title):
    """Size of the biggest local record for this title's character, or 0."""
    doc = resolve.resolve(names, PAREN.sub("", title))
    return len(index.text(doc)) if doc is not None else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Category:Earth-616/Characters")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--min-bytes", type=int, default=20000,
                    help="only audit pages at least this large on Fandom")
    ap.add_argument("--ratio", type=float, default=0.4,
                    help="flag when the local record is under this fraction")
    ap.add_argument("--out", default="crawl/missing_titles.txt")
    args = ap.parse_args()

    crawl = _load("crawl", "crawl/crawl.py")
    search = _load("search", "retrieve/search.py")
    resolve = _load("resolve", "retrieve/resolve.py")
    index = search.Index.load()
    names = resolve.load() or {}

    titles, cont = [], None
    while len(titles) < args.limit:
        batch, cont = crawl.enumerate_titles_page(crawl.FANDOM, args.category,
                                                  limit=500, cont=cont)
        titles.extend(batch)
        if not cont:
            break
    titles = titles[:args.limit]
    print(f"enumerated {len(titles):,} titles from {args.category}")

    sizes = page_sizes(crawl, titles)
    big = [t for t, n in sizes.items() if n >= args.min_bytes]
    print(f"{len(big):,} are at least {args.min_bytes:,} bytes on Fandom")

    missing = []
    for t in big:
        local = local_best(index, resolve, names, t)
        if local < sizes[t] * args.ratio:
            missing.append((sizes[t], local, t))
    missing.sort(reverse=True)

    print(f"{len(missing):,} substantial pages are missing or stubbed locally")
    out = ROOT / args.out
    out.write_text("\n".join(t for _, _, t in missing) + "\n", encoding="utf-8")
    print(f"-> {out}")
    for remote, local, t in missing[:15]:
        print(f"   fandom {remote:>7,}b  local {local:>7,}c  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
