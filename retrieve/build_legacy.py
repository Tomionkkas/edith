"""name -> every record that has GONE BY it, without being it.

The names index answers "which Hulk". This answers "who else has been the
Hulk", which is a different question and a better one: She-Hulk, Brawn, Red
Hulk and Maestro all carry `Hulk` on a Codename or alias line while being
their own characters.

`resolve.build()` cannot serve it. It keeps the best eight entries per name
(`del entries[8:]`), and the identity holders take most of those slots, so
`spider-man` surfaced two other characters out of dozens. Raising that cap
would change what `confidence()` contests and therefore when the picker
opens - a retrieval change needing `run_cases` and the `--picker` sweep.

So this is a SIDECAR: built here, read only by the web app, and invisible to
`resolve.py`. Nothing in retrieval knows it exists, which is the point.
Measured 2026-09-26: 26,130 names, 35,225 rows, 0.26 MB gzipped, 3 seconds.

Built from the INDEX rather than `raw/`, so it needs no crawl - `py
retrieve/search.py --build` is enough, the same input `build_names.py` takes.
"""
import collections
import gzip
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "legacy.json.gz"

# Per name, the biggest records win. 24 is a screenful of cards and holds
# every legacy anyone asks about: hulk, spider-man, iron man and captain
# america all fill it, and going deeper trades size for rows nobody scrolls
# to. Raising it costs about 10 KB per extra row across the file.
PER_NAME = 24


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(index, resolve) -> dict:
    """`{"name key": [[doc_id, provenance], ...]}`, biggest record first.

    A name is recorded for a record only when the record does NOT hold it as
    an identity: the identity holders are the variants list, and repeating
    them here is how a picker stops being a choice.
    """
    found = collections.defaultdict(list)
    for doc_id in range(len(index.docs)):
        record = index.text(doc_id)
        identity, codename, alias = resolve.names_by_provenance(
            index.headlines[doc_id], record)
        size = len(record)
        for names, provenance in ((codename, resolve.PROV_CODENAME),
                                  (alias, resolve.PROV_ALIAS)):
            for key in names:
                if key and key not in identity:
                    found[" ".join(key)].append((size, doc_id, provenance))

    out = {}
    for key, rows in found.items():
        rows.sort(reverse=True)
        out[key] = [[doc_id, provenance]
                    for _size, doc_id, provenance in rows[:PER_NAME]]
    return out


def main(argv=None) -> int:
    search = _load("search", "retrieve/search.py")
    resolve = _load("resolve", "retrieve/resolve.py")
    index = search.Index.load()
    legacy = build(index, resolve)
    OUT.write_bytes(gzip.compress(
        json.dumps(legacy, separators=(",", ":")).encode("utf-8"), 9))
    rows = sum(len(v) for v in legacy.values())
    print(f"{len(legacy):,} names, {rows:,} rows -> {OUT.name} "
          f"({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
