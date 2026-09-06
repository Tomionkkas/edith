"""Tests for answering field questions in code.

A rendered answer is only worth having if it is right. These pin the routing
(which questions code should NOT try to answer) and the honesty (what happens
when the record lacks the field).
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("facts", ROOT / "infer" / "facts.py")
F = importlib.util.module_from_spec(spec)
sys.modules["facts"] = F
spec.loader.exec_module(F)

_R_SPEC = importlib.util.spec_from_file_location(
    "resolve", ROOT / "retrieve" / "resolve.py")
R = importlib.util.module_from_spec(_R_SPEC)
sys.modules["resolve"] = R
_R_SPEC.loader.exec_module(R)


class FakeIndex:
    """Minimal index shape `facts.variants()` needs: docs, headlines, text()."""

    def __init__(self, records):
        self.docs = list(range(len(records)))
        self.headlines = [r.split("\n")[0] for r in records]
        self._records = records

    def text(self, doc_id):
        return self._records[doc_id]

MOON = ("Moon Knight\n"
        "Full name: Marc Spector\n"
        "Created by: Doug Moench; Don Perlin\n"
        "First appearance: Werewolf by Night Vol 1 32\n"
        "Powers: Enhanced Strength; Lunar Empowerment; Regeneration\n"
        "Species / origin: Human\n"
        "Reality: Earth-616\n"
        "Affiliation: Avengers; Defenders\n"
        "History:\n" + "narrative " * 60)

BARE = "Nobody\nReality: Earth-616\nHistory:\nnarrative"


class Routing(unittest.TestCase):
    def test_creator_questions(self):
        for q in ("who created moon knight", "who made moon knight",
                  "who is the creator of moon knight", "who drew moon knight"):
            self.assertEqual(F.detect_intent(q)[0], "creator", q)

    def test_first_appearance_questions(self):
        for q in ("when did moon knight first appear",
                  "which comic did moon knight debut in",
                  "where should i start with moon knight"):
            self.assertEqual(F.detect_intent(q)[0], "first_appearance", q)

    def test_creator_beats_generic_who(self):
        """"who created X" must not fall into a broader "who is X" shape."""
        self.assertEqual(F.detect_intent("who created moon knight")[0], "creator")

    def test_open_ended_is_left_to_the_model(self):
        for q in ("tell me about moon knight",
                  "give me a rundown on moon knight",
                  "tell me everything about moon knight in detail",
                  "who is moon knight"):
            self.assertIsNone(F.detect_intent(q), q)

    def test_chitchat_is_left_alone(self):
        for q in ("hi", "who are you", "thanks", "how are you"):
            self.assertIsNone(F.detect_intent(q), q)

    def test_case_and_punctuation_do_not_matter(self):
        for q in ("Who created Moon Knight?", "WHO CREATED MOON KNIGHT",
                  "who created moon knight"):
            self.assertEqual(F.detect_intent(q)[0], "creator", q)


class Rendering(unittest.TestCase):
    def test_creators_read_as_prose(self):
        self.assertEqual(F.answer("who created moon knight", MOON),
                         "Moon Knight was created by Doug Moench and Don Perlin.")

    def test_semicolon_lists_become_and(self):
        out = F.answer("what powers does moon knight have", MOON)
        self.assertIn("Enhanced Strength, Lunar Empowerment and Regeneration", out)

    def test_single_value_has_no_and(self):
        self.assertEqual(F.answer("which earth is moon knight from", MOON),
                         "Moon Knight is from Earth-616.")

    def test_real_name(self):
        self.assertEqual(F.answer("what is moon knight's real name", MOON),
                         "Moon Knight's real name is Marc Spector.")

    def test_long_lists_are_capped(self):
        rec = MOON.replace("Affiliation: Avengers; Defenders",
                           "Affiliation: " + "; ".join(f"Team{i}" for i in range(30)))
        self.assertNotIn("Team20", F.answer("what teams is moon knight on", rec))

    def test_narrative_body_is_never_used_as_a_field(self):
        self.assertNotIn("narrative", F.answer("who created moon knight", MOON))


class Honesty(unittest.TestCase):
    def test_missing_field_refuses_rather_than_guessing(self):
        """Falling through to the model here is how invention happens."""
        out = F.answer("who created nobody", BARE)
        self.assertIn("don't have", out)
        self.assertIn("creators", out)

    def test_refusal_names_the_entity(self):
        self.assertIn("Nobody", F.answer("what powers does nobody have", BARE))

    def test_present_field_is_answered_even_in_a_sparse_record(self):
        self.assertEqual(F.answer("which earth is nobody from", BARE),
                         "Nobody is from Earth-616.")

    def test_unrecognised_question_returns_none_not_a_refusal(self):
        """None means "model, your turn"; a refusal would silence it."""
        self.assertIsNone(F.answer("tell me about moon knight", MOON))


class MissingFieldStillAnswers(unittest.TestCase):
    """Two questions in one sentence, and the record answers one of them."""

    RECORD = chr(10).join([

        "Lord Hood",
        "Full name: Parker Robbins",
        "Species / origin: Human",
        "Occupation: crime lord",
        "Gender: Male",
        "Created by: Brian K. Vaughan; Kyle Hotz",
        "First appearance: Hood Vol 1 1",
        "Reality: Earth-616",
        "History:",
        "Parker stole a cloak.",
    ])

    def test_the_missing_field_is_still_admitted(self):
        out = F.answer("who is the Hood, explain in detail his powers",
                       self.RECORD)
        self.assertIn("in my sources", out)

    def test_but_what_the_record_has_is_answered(self):
        out = F.answer("who is the Hood, explain in detail his powers",
                       self.RECORD)
        self.assertIn("Brian K. Vaughan", out)
        self.assertIn("Parker Robbins", out)

    def test_a_bare_field_question_gets_the_bare_refusal(self):
        """No profile shape in the question, no profile in the answer."""
        out = F.answer("what powers does the hood have", self.RECORD)
        self.assertNotIn("Brian K. Vaughan", out)
        self.assertIn("in my sources", out)


class SchemaLessRecords(unittest.TestCase):
    """38% of the corpus has no field schema.

    X-Men, Fantastic Four and Secret Wars are Wikipedia articles: prose, no
    "Created by:" line, nothing to compose a profile from. Falling through to
    the model there is exactly the case where it invents, and the article's
    own opening sentence says what the thing is.
    """

    PROSE = chr(10).join([
        "X-Men",
        "Marvel Comics superhero team of mutants",
        "the superhero team",
        "The X-Men are a superhero team in American comic books published by "
        "Marvel Comics. Created by Stan Lee and Jack Kirby, the team first "
        "appeared in The X-Men #1 in September 1963. A third sentence that "
        "should not be included when the cap is small.",
    ])

    def test_the_opening_claim_is_returned(self):
        out = F.profile(self.PROSE)
        self.assertIn("superhero team", out)
        self.assertIn("Stan Lee", out)

    def test_the_subtitle_lines_are_skipped(self):
        """The first two lines are a subtitle and a disambiguation hint."""
        out = F.profile(self.PROSE)
        self.assertFalse(out.startswith("Marvel Comics superhero team of"))
        self.assertFalse(out.startswith("the superhero team"))

    def test_it_stops_at_a_sentence_boundary(self):
        out = F.prose_summary(self.PROSE, max_chars=90)
        self.assertTrue(out.endswith("."), out)
        self.assertNotIn("third sentence", out)

    def test_a_record_with_no_prose_at_all_returns_none(self):
        self.assertIsNone(F.prose_summary("Name" + chr(10) + "short"))

    def test_a_record_with_fields_still_uses_them(self):
        """The fallback must not take over from the field renderer."""
        rec = chr(10).join([
            "Moon Knight", "Full name: Marc Spector",
            "Species / origin: Human", "Gender: Male",
            "Created by: Doug Moench; Don Perlin",
            "First appearance: Werewolf by Night Vol 1 32",
            "Reality: Earth-616", "History:", "He died in Egypt."])
        self.assertIn("Marc Spector", F.profile(rec))
        self.assertIn("Doug Moench", F.profile(rec))


class RenderedPunctuation(unittest.TestCase):
    """The template appends a full stop to a value that usually has one."""

    RECORD = ("Daredevil\n"
              "Full name: Matt Murdock\n"
              "Powers: Mutated Physiology: The full extent of this ability "
              "are unknown.\n"
              "Created by: Stan Lee; Bill Everett\n"
              "Reality: Earth-616")

    def test_no_doubled_full_stop(self):
        out = F.answer("what powers does daredevil have", self.RECORD)
        self.assertNotIn("..", out)
        self.assertTrue(out.endswith("."), out)

    def test_the_answer_is_otherwise_unchanged(self):
        out = F.answer("who created daredevil", self.RECORD)
        self.assertEqual(out, "Daredevil was created by Stan Lee and Bill Everett.")


class VariantsAlwaysReturnsAPair(unittest.TestCase):
    """`rows, total = variants(...)` is how all three callers use it, and a
    headline that normalises to nothing returned a bare list instead."""

    def test_a_headline_with_no_name_returns_an_empty_pair(self):
        ix = FakeIndex(["...\nKind: character\nReality: Earth-616\n"])
        rows, total = F.variants(ix, R, 0)
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)


if __name__ == "__main__":
    unittest.main()


class EntityExtraction(unittest.TestCase):
    """Wording must not choose the character.

    "what powers does storm have" retrieved Ororo Munroe, "what is storm
    skilled in" retrieved Of-Storm, and "what is storm good at" retrieved
    Prince of Good. Same entity, three records, purely from phrasing.
    """

    def test_intent_words_are_removed(self):
        self.assertNotIn("skilled", F.entity_text("what is storm skilled in").lower())
        self.assertNotIn("powers", F.entity_text("what powers does storm have").lower())

    def test_the_entity_survives(self):
        for q in ("what is storm skilled in", "what powers does storm have",
                  "who created storm", "what is storm good at"):
            self.assertIn("storm", F.entity_text(q).lower(), q)

    def test_no_fragments_are_left_behind(self):
        """Stripping "skill" from "skilled" would leave a stray "ed"."""
        for q in ("what is storm skilled in", "what abilities does storm have"):
            self.assertNotIn(" ed ", " " + F.entity_text(q).lower() + " ")

    def test_unmatched_questions_are_returned_whole(self):
        q = "tell me about storm"
        self.assertEqual(F.entity_text(q), q)

    def test_never_returns_empty(self):
        """An empty query retrieves nothing at all."""
        self.assertTrue(F.entity_text("who created"))
        self.assertTrue(F.entity_text("powers"))

    def test_multi_word_names_survive(self):
        self.assertIn("moon knight",
                      F.entity_text("who created moon knight").lower())
        self.assertIn("black panther",
                      F.entity_text("what powers does black panther have").lower())


class ProfileRows(unittest.TestCase):
    """The record as ordered label/value pairs - the answer, as structure."""

    RECORD = (
        "Namor McKenzie (Earth-616)\n"
        "Page: Namor McKenzie (Earth-616)\n"
        "Kind: character\n"
        "Reality: Earth-616\n"
        "Full name: Namor McKenzie\n"
        "Created by: Bill Everett\n"
        "First appearance: Motion Picture Funnies Weekly Vol 1 1\n"
        "Powers: Mutant/Atlantean Physiology\n"
        "Occupation: King of Atlantis\n"
        "History: Namor was born ...\n")

    def test_it_returns_pairs_in_a_fixed_order(self):
        rows = F.profile_rows(self.RECORD)
        self.assertEqual([label for label, _ in rows][:2],
                         ["Powers", "Created by"])

    def test_values_come_from_the_record(self):
        rows = dict(F.profile_rows(self.RECORD))
        self.assertEqual(rows["Created by"], "Bill Everett")
        self.assertEqual(rows["Occupation"], "King of Atlantis")

    def test_a_record_with_no_usable_field_returns_nothing(self):
        self.assertEqual(F.profile_rows("Some Headline\nKind: article\n"), [])

    def test_prose_is_never_a_row(self):
        labels = [label for label, _ in F.profile_rows(self.RECORD)]
        self.assertNotIn("History", labels)

    def test_a_stray_citation_is_filtered_out_of_powers(self):
        """4.12 REGRESSION, found by run_cases after Task 7 wired plan.rows
        into the answer. `answer()` (this file) and `_usable()`
        (train/sft_data.py) already strip a bare comic citation that
        unwrap_templates() can leave behind in Powers/Abilities - Squirrel
        Girl's opens 'Unbeatable Squirrel Girl Vol 2 40'. profile_rows() read
        the field raw and had no filter of its own, so the citation reached
        the screen verbatim - forbidden by answer_cases.py's corpus-hygiene
        case for this exact record. It was fixed as a third copy of the same
        one-line filter and REFACTORED in a later round: `_strip_citations()`
        is now shared by answer() and profile_rows(), so this test pins one
        implementation rather than one of two copies of it. See
        docs/4.12-rebaseline.md - this used to point at
        logs/4.12-rebaseline.txt, a gitignored path that no longer exists."""
        record = (
            "Squirrel Girl\n"
            "Page: Doreen Green (Earth-616)\n"
            "Kind: character\n"
            "Powers: Unbeatable Squirrel Girl Vol 2 40; Squirrel Powers: "
            "Squirrel Girl is a mutate with squirrel-like abilities.\n"
            "Created by: Will Murray; Steve Ditko\n")
        rows = dict(F.profile_rows(record))
        self.assertNotIn("Vol 2 40", rows["Powers"])
        self.assertIn("Squirrel Powers", rows["Powers"])


class ProfileRowsGetTheSameHygieneAsAFieldAnswer(unittest.TestCase):
    """FINDING 2026-09-05 (whole-branch review, Important 2).

    `answer()` and `profile_rows()` read the same fields out of the same
    record, and 4.12 promoted the LESS hygienic of the two: `answer()` reads
    every value through `sd.clean_value()` and bounds an over-long item with
    `_first_sentence`, while `profile_rows()` split the raw value and
    filtered citations out of Powers/Abilities only. Measured over
    `curated/characters.txt`, `clean_value()` changes 10.9% of Occupation
    values, 16.2% of Affiliation and 6.1% of Powers - almost all of them the
    `Formerly:;` sub-label, which is the exact string the phase's own spec
    quotes as the defect the field block exists to remove, and it was still
    rendering on Namor, the spec's own worked example.

    These are the two functions' shared pipeline, asserted on both at once
    so they cannot drift apart again.
    """

    RECORD = (
        "Namor McKenzie (Earth-616)\n"
        "Page: Namor McKenzie (Earth-616)\n"
        "Kind: character\n"
        "Created by: Bill Everett\n"
        "First appearance: Motion Picture Funnies Weekly Vol 1 1\n"
        "Occupation: King of Atlantis and adventurer; Formerly:; terrorist; "
        "warrior\n"
        "Affiliation: Avengers; Formerly:; Illuminati\n"
        "Powers: Mutant/Atlantean Physiology: Namor's powers come from his "
        "being a unique hybrid of Atlantean Homo mermanus and mutant Homo "
        "superior physiologies, which is a sentence long enough to run past "
        "the character bound a rendered answer is held to and therefore to "
        "need its opening claim kept and the rest dropped. Because of his "
        "unusual genetic heritage he is unique among both.; "
        "Superhuman Strength\n"
        "History: Namor was born ...\n")

    def rows(self):
        return dict(F.profile_rows(self.RECORD))

    def test_a_formerly_sub_label_never_reaches_a_row(self):
        """The artifact by name, in the two fields that carry it most."""
        self.assertNotIn("Formerly", self.rows()["Occupation"])

    def test_the_sub_label_leaves_the_real_values_alone(self):
        occupation = self.rows()["Occupation"]
        self.assertEqual(occupation,
                         "King of Atlantis and adventurer; terrorist; warrior")

    def test_a_row_is_cleaned_the_same_way_the_field_answer_is(self):
        """Not "cleaned somehow" - cleaned by the SAME function, so a third
        pipeline cannot quietly grow here. `answer()` humanises its output
        into prose and a row does not, so the comparison is per item."""
        cleaned = F._sd().clean_value(F.field_value(self.RECORD, "Occupation"))
        self.assertEqual(self.rows()["Occupation"], cleaned)

    def test_an_item_longer_than_the_answer_bound_keeps_its_first_sentence(self):
        powers = self.rows()["Powers"]
        self.assertIn("Mutant/Atlantean Physiology", powers)
        self.assertNotIn("Because of his", powers)
        self.assertIn("Superhuman Strength", powers)

    def test_the_fixture_really_exceeds_the_bound(self):
        """A first-sentence test whose item already fits certifies nothing."""
        first = F.field_value(self.RECORD, "Powers").split(";")[0]
        self.assertGreater(len(first), F.ANSWER_CHARS)


_paths_spec = importlib.util.spec_from_file_location(
    "edith_paths", ROOT / "paths.py")
paths = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(paths)

CURATED = paths.CORPUS / "characters.txt"


class ProfileFieldsExistInTheCorpus(unittest.TestCase):
    """Every PROFILE_FIELDS key must name a field real records carry.

    `Teams` shipped in the first draft and matched ZERO of 104,097 character
    records - the corpus calls it `Affiliation`. No hand-written fixture can
    catch that, because a fixture is written to match the code it tests.
    """

    SCAN_LIMIT = 200_000        # every real key appears within the first 20

    @unittest.skipUnless(CURATED.exists(), "curated corpus not present")
    def test_every_profile_field_key_appears_in_a_real_record(self):
        want = {key for key, _ in F.PROFILE_FIELDS}
        seen = set()
        with CURATED.open(encoding="utf-8") as handle:
            for count, line in enumerate(handle):
                head, sep, _ = line.partition(": ")
                if sep and head in want:
                    seen.add(head)
                    if seen == want:
                        break
                if count > self.SCAN_LIMIT:
                    break
        self.assertEqual(seen, want,
                         f"never seen in the corpus: {sorted(want - seen)}")
