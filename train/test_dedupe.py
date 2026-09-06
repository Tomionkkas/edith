"""Tests for corpus deduplication.

The dangerous mistake here is line-level dedup. Every curated record repeats
short schema lines by design --

    Reality: Earth-616
    Gender: Male
    Identity status: Secret

-- and 103,811 character records share them. Deduping lines would delete the
schema from all but the first record and leave the corpus unlearnable. Dedup
therefore works on whole documents and on *long* prose paragraphs only.

Run: py train/test_dedupe.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "dedupe", Path(__file__).resolve().parent / "dedupe.py")
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)

LONG_A = ("Peter Parker was bitten by a radioactive spider during a science "
          "demonstration at Empire State University, granting him proportional "
          "strength, speed and the ability to cling to walls, along with a "
          "precognitive spider-sense that warns him of imminent danger. " * 2)
LONG_B = ("Tony Stark built the first Iron Man armour in a cave from a box of "
          "scraps while held captive, using it to escape and later refining the "
          "design into a suit powered by an arc reactor embedded in his chest. " * 2)


class DocumentDedup(unittest.TestCase):
    def test_exact_duplicate_document_removed(self):
        kept, st = dd.dedupe_documents(["alpha text", "beta text", "alpha text"])
        self.assertEqual(kept, ["alpha text", "beta text"])
        self.assertEqual(st["exact_dupes"], 1)

    def test_first_occurrence_is_the_one_kept(self):
        kept, _ = dd.dedupe_documents(["first", "second", "first"])
        self.assertEqual(kept[0], "first")

    def test_order_is_preserved(self):
        kept, _ = dd.dedupe_documents(["a", "b", "c", "b"])
        self.assertEqual(kept, ["a", "b", "c"])

    def test_whitespace_differences_still_count_as_duplicates(self):
        kept, st = dd.dedupe_documents(["Spider-Man  rules", "Spider-Man rules "])
        self.assertEqual(len(kept), 1)

    def test_near_duplicates_are_kept_this_is_exact_dedup_only(self):
        kept, _ = dd.dedupe_documents(["Spider-Man rules", "Spider-Man rules!"])
        self.assertEqual(len(kept), 2)

    def test_empty_input(self):
        self.assertEqual(dd.dedupe_documents([]), ([], {"exact_dupes": 0, "kept": 0}))


class ParagraphDedup(unittest.TestCase):
    """The critical safety property: schema lines must survive."""

    def test_short_schema_lines_are_never_deduped(self):
        doc_a = "Spider-Man\nReality: Earth-616\nGender: Male\nIdentity status: Secret"
        doc_b = "Iron Man\nReality: Earth-616\nGender: Male\nIdentity status: Secret"
        seen = set()
        out_a = dd.dedupe_paragraphs(doc_a, seen)
        out_b = dd.dedupe_paragraphs(doc_b, seen)
        self.assertIn("Reality: Earth-616", out_a)
        self.assertIn("Reality: Earth-616", out_b)   # <- would vanish under line dedup
        self.assertIn("Gender: Male", out_b)

    def test_long_repeated_paragraph_is_removed_the_second_time(self):
        seen = set()
        first = dd.dedupe_paragraphs(f"Header\n\n{LONG_A}", seen)
        second = dd.dedupe_paragraphs(f"Other\n\n{LONG_A}", seen)
        self.assertIn("radioactive spider", first)
        self.assertNotIn("radioactive spider", second)

    def test_distinct_long_paragraphs_both_survive(self):
        seen = set()
        a = dd.dedupe_paragraphs(LONG_A, seen)
        b = dd.dedupe_paragraphs(LONG_B, seen)
        self.assertTrue(a.strip())
        self.assertTrue(b.strip())

    def test_paragraph_just_under_threshold_is_left_alone(self):
        para = "x" * (dd.MIN_PARA_CHARS - 1)
        seen = set()
        dd.dedupe_paragraphs(para, seen)
        self.assertIn(para, dd.dedupe_paragraphs(para, seen))

    def test_document_with_no_long_paragraphs_is_unchanged(self):
        doc = "Spider-Man\nReality: Earth-616\nGender: Male"
        self.assertEqual(dd.dedupe_paragraphs(doc, set()).strip(), doc)


class Normalisation(unittest.TestCase):
    def test_collapses_whitespace_for_comparison_only(self):
        self.assertEqual(dd.normalise("a   b\n\nc"), dd.normalise("a b c"))

    def test_is_case_sensitive(self):
        self.assertNotEqual(dd.normalise("Spider-Man"), dd.normalise("spider-man"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
