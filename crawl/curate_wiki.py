#!/usr/bin/env python3
"""Marvel SLM - Wikipedia (Tier A) curation.

Wikipedia articles are prose under `== Section ==` headers, not Marvel Database
infoboxes, so `curate.py`'s field parser finds nothing in them. This is their
path: drop the sections that carry no in-universe knowledge, then reuse the
tested `strip_markup` from curate.py.

  py curate_wiki.py            # raw/wiki_marvel.jsonl -> curated/wiki_marvel.txt
  py curate_wiki.py --demo     # print a couple of curated articles
"""
from __future__ import annotations
import importlib.util
import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "raw" / "wiki_marvel.jsonl"
OUT_FILE = BASE_DIR / "curated" / "wiki_marvel.txt"
MIN_CHARS = 400

_spec = importlib.util.spec_from_file_location(
    "curate_core", Path(__file__).resolve().parent / "curate.py")
_cu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cu)
strip_markup = _cu.strip_markup

# Sections that are publishing metadata, real-world criticism or link dumps.
# "In other media" is deliberately NOT here: the corpus keeps adaptation facts
# (who played/voiced whom), consistent with keeping the MCU character pages.
DROP_SECTIONS = {
    "references", "external links", "see also", "further reading",
    "bibliography", "notes", "citations", "footnotes", "sources",
    "collected editions", "reception", "critical reception", "awards",
    "awards and nominations", "gallery", "production", "sales",
    "publication information", "issues",
}

HEADER_RE = re.compile(r"(?m)^(={2,6})\s*(.+?)\s*\1\s*$")


def trim_sections(wikitext: str) -> str:
    """Remove DROP_SECTIONS (and everything nested under them) from wikitext.

    A dropped `== X ==` runs until the next header at the same level or higher,
    so its `=== subsections ===` go with it.
    """
    if not wikitext:
        return ""
    headers = [(m.start(), m.end(), len(m.group(1)), m.group(2).strip().lower())
               for m in HEADER_RE.finditer(wikitext)]
    if not headers:
        return wikitext

    keep = []
    prev_end = 0
    i = 0
    while i < len(headers):
        start, end, level, name = headers[i]
        if name in DROP_SECTIONS:
            keep.append(wikitext[prev_end:start])
            # skip to the next header at this level or shallower
            j = i + 1
            while j < len(headers) and headers[j][2] > level:
                j += 1
            prev_end = headers[j][0] if j < len(headers) else len(wikitext)
            i = j
        else:
            i += 1
    keep.append(wikitext[prev_end:])
    return "".join(keep)


def curate_wiki(rec: dict, min_chars: int = MIN_CHARS) -> str:
    """Raw Wikipedia record -> clean training text for one article."""
    wt = rec.get("wikitext") or ""
    if not wt.strip():
        return ""
    # `== Publication history ==` should read as a prose heading, not markup
    body = HEADER_RE.sub(lambda m: m.group(2), trim_sections(wt))
    body = strip_markup(body)
    body = "\n".join(ln.strip() for ln in body.split("\n") if ln.strip()).strip()
    if len(body) < min_chars:
        return ""
    return f"{rec['title']}\nKind: article\n{body}\n"


def run() -> int:
    if not RAW_FILE.exists():
        print(f"no such raw file: {RAW_FILE}")
        return 1
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    n = empty = 0
    with RAW_FILE.open(encoding="utf-8") as fin, \
         OUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = curate_wiki(rec)
            if not text:
                empty += 1
                continue
            fout.write(text + "\n" + "=" * 60 + "\n\n")
            n += 1
    mb = OUT_FILE.stat().st_size / 1e6
    print(f"wiki_marvel: curated {n} articles -> {OUT_FILE} "
          f"({empty} empty skipped, {mb:,.1f} MB)")
    return 0


def demo() -> int:
    shown = 0
    for line in RAW_FILE.open(encoding="utf-8"):
        rec = json.loads(line)
        text = curate_wiki(rec)
        if not text or len(text) < 2000:
            continue
        print("=" * 70)
        print(text[:1200])
        shown += 1
        if shown >= 2:
            break
    return 0


if __name__ == "__main__":
    sys.exit(demo() if "--demo" in sys.argv else run())
