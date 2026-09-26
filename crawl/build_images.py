"""page title -> image filename, for every raw phase.

The corpus carries no images: `strip_file_links()` removes them during
curation, correctly, because an image filename is not text a model should
ever learn to emit. The web app needs them anyway, so they live in a sidecar
built from `raw/` and committed - 2.7 MB gzipped - rather than in `curated/`,
which stays exactly as the model was trained on.

Two things this file exists to get right, both measured 2026-09-26:

  The cover field on a comic is `Image1`, not `Image`. Searching for
  `| Image =` finds ONE comic in 72,295 and silently loses 36% of the corpus.

  13,336 pages carry the field and leave it EMPTY. They are not art. A
  pattern whose whitespace class runs past the newline captures the FOLLOWING
  field, which reported 198,556 pages with images where 185,220 have one.

Stdlib only, like every other crawl script.
"""
import gzip
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
OUT = ROOT / "retrieve" / "images.json.gz"

# Line-anchored, and the value must START with a non-space on the SAME line.
# `\s*` after the `=` would cross the newline; `[ \t]*` cannot.
IMAGE = re.compile(r"^\|\s*Image1?\s*=[ \t]*(\S[^\n]*)$", re.M)

# A filename longer than this is a parse that went wrong, not a file.
MAX_NAME = 200


def image_name(wikitext: str) -> str:
    """The `Image` (or `Image1`) value, or "" when absent or empty."""
    match = IMAGE.search(wikitext or "")
    if not match:
        return ""
    name = match.group(1).strip()
    return name if 0 < len(name) < MAX_NAME else ""


def build(raw_dir: pathlib.Path = RAW, quiet: bool = False) -> dict:
    """Every raw phase, in one pass. `title` is the wiki page title, which is
    what `retrieve/images.py:title_of()` reconstructs from a record."""
    out = {}
    for path in sorted(raw_dir.glob("*.jsonl")):
        found = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                name = image_name(record.get("wikitext", ""))
                if name:
                    out[record["title"]] = name
                    found += 1
        if not quiet:
            print(f"  {path.name:34} {found:>7,}")
    return out


def main(argv=None) -> int:
    if not RAW.is_dir():
        print(f"no {RAW} - the sidecar is built from raw wikitext, which is "
              f"gitignored. See CLAUDE.md for how to obtain it.")
        return 1
    images = build()
    OUT.write_bytes(gzip.compress(
        json.dumps(images, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        9))
    print(f"\n{len(images):,} entries -> {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
