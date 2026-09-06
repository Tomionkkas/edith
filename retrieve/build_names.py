"""Build the name index the resolver uses.

    py retrieve/build_names.py

Reads retrieve/index.pkl and writes retrieve/names.pkl (~2 MB, ~2 s). Rerun
after rebuilding the index.
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


WIKI_TIER = HERE.parent / "curated" / "wiki_marvel.txt"


def notable(resolve, search):
    """The notability oracle, read from the Wikipedia tier's article titles.

    The file is the oracle, so it is read directly rather than inferred from
    the index: nothing in a built record says which tier it came from, and a
    doc-id range would silently mean something else the next time the corpus
    is rebuilt in a different order.
    """
    if not WIKI_TIER.exists():
        print(f"WARNING: {WIKI_TIER.name} missing - building without the "
              f"notability oracle")
        return frozenset()
    titles = (r.split(chr(10))[0] for r in search.read_records(WIKI_TIER))
    return resolve.notable_names(titles)


def main() -> int:
    search = _load("search")
    resolve = _load("resolve")
    index = search.Index.load()
    oracle = notable(resolve, search)
    names = resolve.build(index, notable=oracle)
    resolve.save(names)
    print(f"{len(names):,} names from {len(index):,} records "
          f"({len(oracle):,} notable) -> {resolve.NAMES_PATH.name} "
          f"({resolve.NAMES_PATH.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
