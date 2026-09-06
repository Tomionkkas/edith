"""Tests for the tokenizer sampling mix (offline, no corpus needed).

A bad mix is silent: the tokenizer trains fine and every downstream token count
is quietly wrong. These pin the properties that matter.

Run: py train/test_tokenizer_corpus.py
"""
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "tokenizer_corpus", Path(__file__).resolve().parent / "tokenizer_corpus.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

SEP = "=" * 60
SAMPLE_FILE = f"""Spider-Man
Reality: Earth-616
He climbs walls.
{SEP}

Iron Man (Earth-199999)
Reality: Earth-199999
He flies.
{SEP}

Doctor Strange
Reality: Earth-616
He casts spells.
{SEP}

"""


class MarvelRecords(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "c.txt"
        self.tmp.write_text(SAMPLE_FILE, encoding="utf-8")

    def test_splits_on_separator(self):
        recs = list(tc.marvel_records(self.tmp))
        self.assertEqual(len(recs), 3)

    def test_record_keeps_its_whole_body(self):
        first = list(tc.marvel_records(self.tmp))[0]
        self.assertIn("Spider-Man", first)
        self.assertIn("climbs walls", first)
        self.assertNotIn(SEP, first)

    def test_no_empty_records(self):
        self.assertTrue(all(r.strip() for r in tc.marvel_records(self.tmp)))


class TakeChars(unittest.TestCase):
    def test_stops_at_budget(self):
        recs = ["a" * 100 for _ in range(50)]
        out, total = tc.take_chars(iter(recs), 250, random.Random(0))
        self.assertGreaterEqual(total, 250)
        self.assertLess(len(out), 50)

    def test_is_deterministic_for_a_seed(self):
        recs = [f"record-{i}" for i in range(200)]
        a, _ = tc.take_chars(iter(recs), 300, random.Random(7), keep_prob=0.5)
        b, _ = tc.take_chars(iter(recs), 300, random.Random(7), keep_prob=0.5)
        self.assertEqual(a, b)

    def test_keep_prob_samples_beyond_a_contiguous_prefix(self):
        """Taking a prefix would over-represent alphabetically early pages."""
        recs = [f"record-{i:04d}" for i in range(2000)]
        out, _ = tc.take_chars(iter(recs), 300, random.Random(1), keep_prob=0.2)
        idx = [int(r.split("-")[1]) for r in out]
        self.assertGreater(max(idx), len(out))  # reached past a dense prefix

    def test_empty_input_is_safe(self):
        out, total = tc.take_chars(iter([]), 1000, random.Random(0))
        self.assertEqual((out, total), ([], 0))


class Interleave(unittest.TestCase):
    def test_keeps_every_document(self):
        m = [f"m{i}" for i in range(20)]
        g = [f"g{i}" for i in range(20)]
        merged = tc.interleave(m, g, random.Random(3))
        self.assertEqual(sorted(merged), sorted(m + g))

    def test_actually_mixes_the_two_pools(self):
        """Unmixed, every Marvel doc would precede every general doc."""
        m = [f"m{i}" for i in range(50)]
        g = [f"g{i}" for i in range(50)]
        merged = tc.interleave(m, g, random.Random(3))
        first_half = merged[:50]
        n_general = sum(1 for d in first_half if d.startswith("g"))
        self.assertGreater(n_general, 10)

    def test_is_deterministic_for_a_seed(self):
        m, g = ["m1", "m2", "m3"], ["g1", "g2", "g3"]
        self.assertEqual(tc.interleave(m, g, random.Random(9)),
                         tc.interleave(m, g, random.Random(9)))


class WriteSample(unittest.TestCase):
    def test_writes_one_line_per_source_line_and_drops_blanks(self):
        dest = Path(tempfile.mkdtemp()) / "s.txt"
        n = tc.write_sample(["alpha\n\nbeta", "gamma"], dest)
        self.assertEqual(n, 3)
        self.assertEqual(dest.read_text(encoding="utf-8").split("\n")[:3],
                         ["alpha", "beta", "gamma"])

    def test_no_blank_lines_in_output(self):
        dest = Path(tempfile.mkdtemp()) / "s.txt"
        tc.write_sample(["a\n\n\nb", "\n", "c"], dest)
        lines = dest.read_text(encoding="utf-8").split("\n")
        self.assertFalse(any(l == "" for l in lines[:-1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
