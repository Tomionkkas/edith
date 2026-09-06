"""The one reference test that needs the index.

REALITY_NICKNAMES is hand-written because Category:Realities was never
crawled. This is what stops it being an unfalsifiable table: every value has
to name a reality that records in the corpus actually carry.

THE TRAP: `reference.reality_of()` FALLS BACK to "616" for any headline with
no explicit reality suffix, and just over half the corpus (107,040 of
202,101 headlines) is unmarked. Build the comparison set with `reality_of()`
and "616" is guaranteed to be present no matter what reference.py claims -
the test would then only be checking reference.py's own default against
itself for `main`, `mainstream`, `prime` and `classic`, a third of the
table, tautologically. So this test collects ONLY explicit `(Earth-N)` /
`(Reality-N)` suffixes, via `reference.REALITY` directly, and never calls
`reality_of()`. Earth-616 does appear explicitly too (24,996 headlines), so
this is not a weaker check - it is the only version of this test with teeth
on all twelve entries.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = _load("reference", "infer/reference.py")
search = _load("search", "retrieve/search.py")

_paths_spec = importlib.util.spec_from_file_location(
    "edith_paths", ROOT / "paths.py")
paths = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(paths)

INDEX = paths.INDEX


@unittest.skipUnless(INDEX.exists(), "index.pkl not built")
class EveryNicknameNamesARealReality(unittest.TestCase):
    """Explicit suffixes only - see the module docstring for why. Using
    `reality_of()` here would let its own "616" fallback validate itself,
    since every unmarked headline (over half the corpus) counts as a hit."""

    @classmethod
    def setUpClass(cls):
        index = search.Index.load()
        cls.realities = {
            m.group(1).lower()
            for h in index.headlines
            for m in [R.REALITY.search(h)] if m
        }

    def test_every_nickname_maps_to_a_reality_in_the_corpus(self):
        for nick, code in R.REALITY_NICKNAMES.items():
            with self.subTest(nickname=nick):
                self.assertIn(code, self.realities,
                              f"{nick!r} -> Earth-{code}, which no record has")
