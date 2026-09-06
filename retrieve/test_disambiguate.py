"""Tests for "which Spider-Man did you mean?".

Marvel is genuinely ambiguous: 18 records are headlined exactly "Spider-Man"
and 15 exactly "Venom", all different characters. Picking silently is what
made a reader doubt the corpus was scraped correctly.
"""
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


se = _load("search")
D = _load("disambiguate")


def rec(head, full="", reality="Earth-616", first="Some Comic Vol 1 1",
        pad=100, extra=""):
    lines = [head]
    if full:
        lines.append(f"Full name: {full}")
    if reality:
        lines.append(f"Reality: {reality}")
    if first:
        lines.append(f"First appearance: {first}")
    if extra:
        lines.append(extra)
    lines.append("History:")
    lines.append("body text about the character. " * pad)
    return "\n".join(lines)


class Identity(unittest.TestCase):
    def test_same_headline_different_people_are_different(self):
        a = rec("Captain America", "Steven Rogers")
        b = rec("Captain America", "Samuel Wilson")
        self.assertNotEqual(D.identity(a), D.identity(b))

    def test_same_record_is_the_same_entity(self):
        a = rec("Venom", "Eddie Brock")
        self.assertEqual(D.identity(a), D.identity(a))

    def test_reality_separates_variants(self):
        a = rec("Venom", "Anders Arnvidson", reality="Earth-113")
        b = rec("Venom", "Anders Arnvidson", reality="Earth-616")
        self.assertNotEqual(D.identity(a), D.identity(b))

    def test_history_body_does_not_leak_into_identity(self):
        a = rec("Storm", "Ororo Munroe", pad=10)
        b = rec("Storm", "Ororo Munroe", pad=900)
        self.assertEqual(D.identity(a), D.identity(b))


class Describe(unittest.TestCase):
    def test_prefers_the_real_name(self):
        self.assertIn("Ororo Munroe", D.describe(rec("Storm", "Ororo Munroe")))

    def test_includes_reality_and_first_appearance(self):
        out = D.describe(rec("Storm", "Ororo Munroe", first="Giant-Size X-Men Vol 1 1"))
        self.assertIn("Earth-616", out)
        self.assertIn("Giant-Size X-Men Vol 1 1", out)

    def test_species_is_never_used_as_a_name(self):
        """Deadpool's record has no Full name and rendered as
        "Human mutate after being a De..." - a different character entirely."""
        r = rec("Deadpool", full="", extra="Species / origin: Human mutate after being a Dep")
        self.assertNotIn("Human mutate", D.describe(r))

    def test_placeholder_names_are_ignored(self):
        self.assertNotIn("Inapplicable", D.describe(rec("Spider-Man", "Inapplicable")))

    def test_prose_record_falls_back_to_its_first_line(self):
        """Wikipedia records carry no fielded head at all."""
        r = ("Spider-Man\nSpider-Man is a superhero in American comic books "
             "published by Marvel Comics and created by Stan Lee.")
        self.assertIn("superhero", D.describe(r))

    def test_never_returns_empty(self):
        self.assertTrue(D.describe("Nameless\n"))


class Flagship(unittest.TestCase):
    """Among near-tied candidates, the most written-about holder wins."""

    def setUp(self):
        self.ix = se.Index.build([
            rec("Captain America", "Samuel Thomas Wilson", pad=300),
            rec("Captain America", "Steven Rogers", pad=340),      # bigger page
            rec("Captain America", "William Nasland", pad=60),
            rec("Ant-Man", "Scott Lang", pad=200),
        ])

    def cands(self, q):
        return D.candidates(self.ix, q, k=5)

    def test_bigger_page_wins_a_near_tie(self):
        top = self.cands("captain america")[0]
        self.assertIn("Steven Rogers", top[2])

    def test_all_variants_still_offered(self):
        names = " ".join(c[2] for c in self.cands("captain america"))
        self.assertIn("Samuel Thomas Wilson", names)
        self.assertIn("William Nasland", names)

    def test_a_clear_winner_is_not_displaced_by_a_longer_article(self):
        """Promotion applies only inside a scoring tie."""
        ix = se.Index.build([
            rec("Ant-Man", "Scott Lang", pad=40),
            rec("Wasp", "Janet van Dyne", pad=900),
        ])
        self.assertIn("Scott Lang", D.candidates(ix, "ant-man", k=3)[0][2])

    def test_only_same_headline_candidates_compete(self):
        top = D.candidates(self.ix, "ant-man", k=5)[0]
        self.assertIn("Scott Lang", top[2])

    def test_one_candidate_is_returned_unchanged(self):
        ix = se.Index.build([rec("Nova", "Richard Rider")])
        self.assertEqual(len(D.candidates(ix, "nova", k=5)), 1)


class Ambiguity(unittest.TestCase):
    def test_single_candidate_is_never_ambiguous(self):
        self.assertFalse(D.is_ambiguous([(0, "X", "d", 10.0)]))

    def test_a_dead_heat_is_ambiguous(self):
        self.assertTrue(D.is_ambiguous([(0, "X", "a", 10.0), (1, "X", "b", 9.99)]))

    def test_a_clear_winner_is_not(self):
        self.assertFalse(D.is_ambiguous([(0, "X", "a", 10.0), (1, "X", "b", 6.0)]))

    def test_margin_is_relative_not_absolute(self):
        """BM25 scales with query length; a fixed threshold would ask
        constantly on long questions and never on short ones."""
        small = [(0, "X", "a", 10.0), (1, "X", "b", 9.0)]
        large = [(0, "X", "a", 100.0), (1, "X", "b", 90.0)]
        self.assertEqual(D.is_ambiguous(small), D.is_ambiguous(large))


class AlternativesFooter(unittest.TestCase):
    def test_lists_the_others(self):
        c = [(0, "Venom", "Symbiote", 9.0), (1, "Venom", "Mary Jane Watson", 8.0)]
        self.assertEqual(D.alternatives_line(c), "Mary Jane Watson")

    def test_empty_when_there_is_only_one(self):
        self.assertEqual(D.alternatives_line([(0, "X", "d", 1.0)]), "")

    def test_counts_the_overflow(self):
        c = [(i, "Venom", f"Host {i}", 9.0 - i) for i in range(9)]
        self.assertIn("4 more", D.alternatives_line(c, limit=4))


if __name__ == "__main__":
    unittest.main()


class FieldRichness(unittest.TestCase):
    """A long prose record cannot answer "who created X".

    Page size alone promoted the Wikipedia article for "doctor doom" - by far
    the biggest page, and prose: no Created by, no Reality. The model then
    filled the gaps from memory, the exact failure retrieval prevents.
    """

    PROSE = ("Doctor Doom\n" +
             "Doctor Doom is a supervillain appearing in American comic books "
             "published by Marvel Comics. " * 90)
    FIELDED = rec("Doctor Doom", "Victor von Doom", first="Fantastic Four Vol 1 5",
                  pad=40)

    def test_prose_record_scores_zero_fields(self):
        self.assertEqual(D._field_count(self.PROSE), 0)

    def test_fielded_record_counts_its_fields(self):
        self.assertGreaterEqual(D._field_count(self.FIELDED), 3)

    def test_history_body_is_not_counted(self):
        r = rec("Storm", "Ororo Munroe", pad=500)
        self.assertLessEqual(D._field_count(r), 6)

    def test_fielded_record_beats_a_longer_prose_one(self):
        ix = se.Index.build([self.PROSE, self.FIELDED])
        top = D.candidates(ix, "doctor doom", k=3)[0]
        self.assertIn("Victor von Doom", top[2])

    def test_size_still_decides_between_equally_fielded_records(self):
        ix = se.Index.build([
            rec("Captain America", "Samuel Thomas Wilson", pad=300),
            rec("Captain America", "Steven Rogers", pad=340),
        ])
        self.assertIn("Steven Rogers", D.candidates(ix, "captain america", k=3)[0][2])
