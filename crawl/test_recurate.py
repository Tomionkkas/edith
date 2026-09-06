"""Tests for which raw phases get re-curated.

recurate.py's PHASES list is hand-maintained; curate.py's PHASE_KINDS is the
source of truth for which raw phases curation knows a kind for. FINDING 1
2026-09-02: raw/patch.jsonl exists and PHASE_KINDS lists "patch": "character",
but PHASES never curated it -- 4 records (Doctor Doom, Iron Man, Magma,
Tombstone) were indexed carrying neither a `Kind:` nor a `Page:` line.

Importing recurate.py as written actually RUNS every phase (curate.run_phase
for each entry in PHASES, ~7 minutes, rewrites curated/) -- there is no
`if __name__ == "__main__":` guard. This test must never trigger that, so it
reads the PHASES list out of the source with `ast`, rather than importing the
module or invoking the script.

Run: py crawl/test_recurate.py
"""
import ast
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("curate", HERE / "curate.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)


def _phases_list():
    """The literal PHASES list in recurate.py, read statically.

    Never imports or executes recurate.py: doing so runs every curation
    phase for real, which this repo's working agreements forbid from a test.
    """
    tree = ast.parse((HERE / "recurate.py").read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PHASES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no PHASES assignment found in recurate.py")


class PhaseCoverage(unittest.TestCase):
    """Every raw phase curate.py knows a kind for must actually be
    re-curated, or a clean rebuild silently drops it and the indexed corpus
    stops matching what curate.py's own PHASE_KINDS promises."""

    def test_every_known_phase_is_recurated(self):
        phases = _phases_list()
        missing = [p for p in cu.PHASE_KINDS if p not in phases]
        self.assertEqual(missing, [],
                          f"PHASE_KINDS knows a kind for {missing} but "
                          f"recurate.py's PHASES never curates them")

    def test_patch_phase_is_recurated(self):
        """raw/patch.jsonl exists and PHASE_KINDS says 'patch': 'character',
        so a clean rebuild must curate it -- else Doctor Doom, Iron Man,
        Magma and Tombstone are indexed with neither Kind: nor Page:."""
        self.assertIn("patch", _phases_list())


if __name__ == "__main__":
    unittest.main()
