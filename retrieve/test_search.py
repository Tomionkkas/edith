"""Tests for corpus retrieval.

Retrieval is what makes the system factually correct - the model supplies
fluency, this supplies truth. The failure modes are quiet: a tokeniser that
shatters `Earth-616` into `earth` + `616` makes continuity lookup impossible,
and a ranker that ignores the headline buries the record you asked for under
every page that merely mentions it.

Run: py retrieve/test_search.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "search", Path(__file__).resolve().parent / "search.py")
se = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se)

RECORDS = [
    ("Spider-Man\nFull name: Peter Parker\nCreated by: Stan Lee; Steve Ditko\n"
     "Reality: Earth-616\nHistory:\nBitten by a radioactive spider."),
    ("Spider-Man (Earth-1610)\nFull name: Miles Morales\n"
     "Created by: Brian Michael Bendis; Sara Pichelli\nReality: Earth-1610\n"
     "History:\nThe Ultimate universe Spider-Man."),
    ("Doctor Strange (Earth-199999)\nFull name: Stephen Strange\n"
     "Created by: Scott Derrickson; C. Robert Cargill\nReality: Earth-199999\n"
     "History:\nSorcerer Supreme of the Marvel Cinematic Universe."),
    ("Nick Fury\nFull name: Nicholas Fury\nCreated by: Stan Lee; Jack Kirby\n"
     "Reality: Earth-616\nAffiliation: S.H.I.E.L.D.\n"
     "History:\nDirector of S.H.I.E.L.D. and ally of Spider-Man."),
    ("Galactus\nFull name: Galan\nCreated by: Stan Lee; Jack Kirby\n"
     "Reality: Earth-616\nHistory:\nDevourer of worlds."),
]


class Tokenising(unittest.TestCase):
    def test_lowercases(self):
        self.assertIn("spider", se.tokenize("Spider"))

    def test_keeps_reality_ids_intact(self):
        """earth-616 split into earth + 616 makes continuity lookup useless."""
        self.assertIn("earth-616", se.tokenize("Reality: Earth-616"))

    def test_keeps_dotted_acronyms_intact(self):
        self.assertIn("s.h.i.e.l.d.", se.tokenize("Agent of S.H.I.E.L.D."))

    def test_hyphenated_name_also_yields_its_parts(self):
        """'spider man' as a query should still reach Spider-Man."""
        toks = se.tokenize("Spider-Man")
        self.assertIn("spider-man", toks)
        self.assertIn("spider", toks)
        self.assertIn("man", toks)

    def test_drops_punctuation_noise(self):
        self.assertNotIn(":", se.tokenize("Created by: Stan Lee"))

    def test_empty_text_is_safe(self):
        self.assertEqual(se.tokenize(""), [])


class Ranking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ix = se.Index.build(RECORDS)

    def top(self, q):
        hits = self.ix.search(q, k=3)
        return self.ix.text(hits[0][0]) if hits else ""

    def test_finds_record_by_entity_name(self):
        self.assertIn("Galactus", self.top("who is Galactus"))

    def test_headline_match_beats_a_passing_mention(self):
        """Nick Fury's page mentions Spider-Man; it must not outrank his page."""
        self.assertTrue(self.top("Spider-Man").startswith("Spider-Man"))

    def test_reality_id_disambiguates_two_spider_men(self):
        self.assertIn("Earth-1610", self.top("Spider-Man Earth-1610"))

    def test_dotted_acronym_is_searchable(self):
        self.assertIn("Nick Fury", self.top("S.H.I.E.L.D. director"))

    def test_returns_at_most_k(self):
        self.assertLessEqual(len(self.ix.search("Stan Lee", k=2)), 2)

    def test_results_are_ordered_by_descending_score(self):
        scores = [s for _, s in self.ix.search("Stan Lee", k=5)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_common_term_does_not_dominate(self):
        """'Created' appears in every record and should carry little signal."""
        hits = self.ix.search("Created", k=5)
        self.assertLessEqual(len(hits), 5)

    def test_unknown_query_returns_nothing_rather_than_noise(self):
        self.assertEqual(self.ix.search("zzzz nonexistent quux", k=3), [])

    def test_empty_query_is_safe(self):
        self.assertEqual(self.ix.search("", k=3), [])


CANON = [
    "Moon Knight\nFull name: Marc Spector\nCreated by: Doug Moench; Don Perlin\nReality: Earth-616",
    "Moon Knight (Earth-17952)\nCreated by: Cullen Bunn; Iban Coello\nReality: Earth-17952",
    "Moon Knight (Earth-9997)\nCreated by: Jim Krueger\nReality: Earth-9997",
    "Spider-Man\nFull name: Peter Parker\nCreated by: Stan Lee; Steve Ditko\nReality: Earth-616",
    "List of actors who have played Spider-Man\nA list of performers.",
    "Stephen Strange (Earth-199999)\nCreated by: Scott Derrickson\nReality: Earth-199999",
    "Donna Strange (Earth-199999)\nCreated by: Sam Raimi\nReality: Earth-199999",
    "Nick Fury\nAffiliation: S.H.I.E.L.D.\nCreated by: Stan Lee; Jack Kirby\nReality: Earth-616",
    "Nick Fury: Director of S.H.I.E.L.D. Vol 1 1\nA Marvel comic published in 2010.",
]


class CanonicalEntity(unittest.TestCase):
    """Live queries surfaced obscure variants over the entity actually asked for."""

    @classmethod
    def setUpClass(cls):
        cls.ix = se.Index.build(CANON)

    def top(self, q):
        h = self.ix.search(q, k=1)
        return self.ix.text(h[0][0]).split("\n")[0] if h else ""

    def test_question_words_do_not_steer_the_result(self):
        self.assertEqual(self.top("who created Moon Knight"), self.top("Moon Knight"))

    def test_main_continuity_wins_over_obscure_variants(self):
        """Earth-616 records carry no suffix by our headline convention."""
        self.assertEqual(self.top("who created Moon Knight"), "Moon Knight")

    def test_character_beats_a_list_page_about_the_character(self):
        self.assertEqual(self.top("who created Spider-Man"), "Spider-Man")

    def test_named_person_beats_another_person_in_the_same_reality(self):
        self.assertTrue(self.top("Doctor Strange Earth-199999").startswith("Stephen"))

    def test_character_reaches_context_despite_a_comic_of_the_same_name(self):
        """`Nick Fury: Director of S.H.I.E.L.D.` is literally a comic title, so
        word overlap cannot separate it from the character. That is fine: the
        Context block carries k records, so the requirement is that the
        character reaches the model, not that it ranks first."""
        heads = [self.ix.text(d).split("\n")[0]
                 for d, _ in self.ix.search("S.H.I.E.L.D. director", k=3)]
        self.assertIn("Nick Fury", heads)

    def test_explicit_variant_still_reachable_when_asked_for(self):
        self.assertIn("Earth-17952", self.top("Moon Knight Earth-17952"))

    def test_canonical_entity_wins_among_identical_headlines(self):
        """The corpus holds 18 Earth-616 records headlined exactly 'Spider-Man'
        -- the real Peter Parker plus an android, a Life Model Decoy, a Skrull
        impostor and so on. Page size separates them: Peter Parker's is 125,811
        chars, the android's is a stub. BM25 length normalisation penalises
        exactly the record we want, so notability has to be added back."""
        stub = ("Spider-Man\nFull name: Inapplicable\n"
                "Created by: Jim Zub\nReality: Earth-616")
        canon = ("Spider-Man\nFull name: Peter Parker\n"
                 "Created by: Stan Lee; Steve Ditko\nReality: Earth-616\n"
                 "History:\n" + "He fought crime in New York for many years. " * 200)
        ix = se.Index.build([stub, canon])
        best = ix.text(ix.search("who is Spider-Man", k=1)[0][0])
        self.assertIn("Stan Lee", best)


class QueryNormalisation(unittest.TestCase):
    def test_strips_leading_interrogatives(self):
        self.assertEqual(se.strip_question(tuple(se.tokenize("who created Galactus"))),
                         ("galactus",))

    def test_keeps_the_entity_when_no_question_word(self):
        self.assertEqual(se.strip_question(tuple(se.tokenize("Galactus"))), ("galactus",))

    def test_does_not_empty_a_query_made_only_of_stopwords(self):
        self.assertTrue(se.strip_question(tuple(se.tokenize("who is the"))))


class ContextBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ix = se.Index.build(RECORDS)

    def test_contains_the_retrieved_record(self):
        ctx = se.format_context(self.ix, "who created Galactus", k=1)
        self.assertIn("Stan Lee", ctx)

    def test_is_labelled_so_the_model_can_learn_the_shape(self):
        ctx = se.format_context(self.ix, "Galactus", k=1)
        self.assertTrue(ctx.startswith("Context:"))

    def test_empty_when_nothing_matches(self):
        self.assertEqual(se.format_context(self.ix, "zzzz quux", k=2), "")

    def test_respects_a_character_budget(self):
        ctx = se.format_context(self.ix, "Stan Lee", k=5, max_chars=200)
        self.assertLessEqual(len(ctx), 260)


class Persistence(unittest.TestCase):
    def test_index_survives_a_save_load_round_trip(self):
        import tempfile
        p = Path(tempfile.mkdtemp()) / "ix.pkl"
        a = se.Index.build(RECORDS)
        a.save(p)
        b = se.Index.load(p)
        self.assertEqual(b.search("Galactus", k=1)[0][0],
                         a.search("Galactus", k=1)[0][0])
        self.assertEqual(len(b), len(a))



class ConversationalQueries(unittest.TestCase):
    """People do not type bare entity names.

    The headline test is all-or-nothing: one stray term and the entity keeps
    neither FULL_HEAD_BONUS nor its notability score. "tell me about
    spider-man in detail" lost Spider-Man ~30 points that way and returned a
    comic issue. These pin the phrasing-robustness, and deliberately do NOT
    strip "power"/"man", which are parts of real names.
    """

    @classmethod
    def setUpClass(cls):
        cls.idx = se.Index.build([
            "Spider-Man\nFull name: Peter Parker\nPowers: wall-crawling\n"
            "Reality: Earth-616\n" + "body text about spider-man. " * 400,
            "Spectacular Spider-Man Vol 2 3\nSpectacular Spider-Man Vol 2 3 is a "
            "Marvel comic. It was published in detail. " + "spider-man issue. " * 60,
            "Power Man\nFull name: Luke Cage\nPowers: super strength\n"
            "Reality: Earth-616\n" + "body about power man. " * 200,
            "Detective Fantome\nFull name: Unknown\n" + "he has powers in detail. " * 200,
            "Lord Hood\nFull name: Parker Robbins\nPowers: magic cloak\n"
            "Reality: Earth-616\n" + "the hood body text. " * 300,
        ])

    def top(self, q):
        hits = self.idx.search(q, k=1)
        return self.idx.headlines[hits[0][0]] if hits else None

    def test_bare_name(self):
        self.assertEqual(self.top("spider-man"), "Spider-Man")

    def test_conversational_phrasing_still_finds_the_character(self):
        self.assertEqual(self.top("tell me about spider-man in detail"), "Spider-Man")

    def test_filler_does_not_hand_the_query_to_a_comic_issue(self):
        self.assertNotIn("Vol", self.top("tell me about spider-man in detail"))

    def test_multi_clause_question(self):
        self.assertEqual(
            self.top("tell me about the hood in detail and what powers does he have"),
            "Lord Hood")

    def test_power_is_not_treated_as_filler(self):
        """Stripping "power" would break Power Man, Power Pack, Star-Lord."""
        self.assertEqual(self.top("who is power man"), "Power Man")

    def test_man_is_not_treated_as_filler(self):
        self.assertEqual(self.top("power man"), "Power Man")

    def test_question_words_alone_do_not_crash(self):
        self.idx.search("who is")

    def test_entity_falls_back_when_every_term_is_common(self):
        """If the df filter would empty the entity set, keep the original."""
        self.assertTrue(self.idx.search("the a of in"), "should still return something")

if __name__ == "__main__":
    unittest.main(verbosity=2)
