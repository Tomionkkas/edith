"""Quote the prose, never paraphrase it. Phase 4.5b.

The safety property this whole phase rests on, from the 4.5b design:

    A verbatim quote can be the WRONG passage; it can never be a FALSE one.

Which is also the only thing that works. A spike on 2026-09-20 handed the
model 957 characters of real Civil War narrative through the `histories=`
argument `context.py` already takes - the same way stage 3 was trained - and
it still answered "Civil War is from Earth-616. Civil War first appeared in
Civil War Vol 1 1." A 250M model does not use prose it is given, so the
deterministic path is not merely the safe choice, it is the available one.

Selection is scoped to the RESOLVED RECORD, not a corpus-wide passage index.
Retrieval already answers "which record" at 97.5%; the open question is only
"which sentence of it", over 500-40,000 characters already in hand. That
needs no second index and no extra download.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _load("passages", "infer/passages.py")
R = _load("resolve", "retrieve/resolve.py")


RECORD = """Civil War (Event)
Kind: event
Page: Civil War (Event)
Reality: Earth-616
History:
Brief Summary; In a battle between Nitro and the New Warriors, Nitro exploded, killing sixty school children. This led the United States government to introduce a registration act for all super-powered individuals. Most heroes were divided on the issue. Iron Man led the pro-registration side and Captain America led the resistance. Captain America surrendered after seeing the damage the fighting caused."""

NO_PROSE = """Some Event
Kind: event
Page: Some Event
Reality: Earth-616
First appearance: A Comic Vol 1 1"""


def select(question, record=RECORD, **kw):
    return P.select(record, question, R.query_key, R.norm, **kw)


class WholeSentencesOnly(unittest.TestCase):
    """A passage that needs rewriting to make sense was the wrong passage."""

    def test_a_quote_ends_on_a_sentence_boundary(self):
        got = select("how did nitro die")
        self.assertTrue(got.rstrip().endswith("."), got)

    def test_a_quote_is_verbatim_from_the_record(self):
        """The property the whole phase rests on. If this fails, the feature
        can state something the source does not."""
        got = select("how did nitro die")
        for sentence in got.split(". "):
            self.assertIn(sentence.rstrip("."), RECORD, sentence)

    def test_it_never_returns_more_than_the_cap(self):
        got = select("how did nitro die", max_sentences=2)
        self.assertLessEqual(got.count(". ") + 1, 2, got)


class WhichSentence(unittest.TestCase):

    def test_it_picks_the_sentence_the_question_is_about(self):
        self.assertIn("Nitro exploded", select("how did nitro die"))

    def test_a_different_question_picks_a_different_sentence(self):
        got = select("why did captain america surrender")
        self.assertIn("surrendered", got)
        self.assertNotIn("school children", got)

    def test_the_entitys_own_name_does_not_count_as_overlap(self):
        """`what happened in the civil war` keys to ("civil", "war") - the
        record's own name, which appears throughout it and discriminates
        nothing. Scoring on it would rank by sentence length instead."""
        self.assertEqual(
            P.discriminating(("civil", "war"), RECORD, R.norm), frozenset())

    def test_a_question_naming_only_the_entity_falls_back_to_the_opening(self):
        """Asking what happened, with nothing more specific, wants the
        summary - and a narrative's opening IS its summary.

        CORRECTED after running against the real 39,821-character record:
        this first asserted the passage starts with "Brief Summary", which is
        a section heading `clean_headings` turned into a bare line, not prose.
        Quoting it is quoting debris.
        """
        got = select("what happened in the civil war")
        self.assertTrue(got.startswith("In a battle between Nitro"), got)


class TheJoinArtifactIsNotProse(unittest.TestCase):
    """Curation joins a list-valued narrative field with "; ", so one
    "sentence" can span several source lines and swallow their headings:

        ...a Civil War ensued.; Background and Casus Belli; Mutant
        Registration Act, Secret War, Hulk; The Superhuman Registration...

    Measured in ROADMAP 4.14 and deliberately not fixed there, because fixing
    it in the corpus means re-curating characters.txt and re-uploading 296 MB.
    It is fixed HERE instead, where it costs nothing: "; " is the join marker,
    so splitting on it recovers the lines curation collapsed, and the existing
    fragment filter then drops the headings on its own.
    """

    JOINED = """Some Record
History:
Brief Summary; In a battle between Nitro and the New Warriors, Nitro exploded, killing many people.; Background and Casus Belli; The Superhuman Registration Act had been a long time in the making."""

    def test_a_joined_heading_does_not_reach_the_quote(self):
        got = P.select(self.JOINED, "what happened", R.query_key, R.norm)
        self.assertNotIn("Background and Casus Belli", got)
        self.assertNotIn("Brief Summary", got)

    def test_the_prose_either_side_of_it_survives(self):
        got = P.select(self.JOINED, "what happened", R.query_key, R.norm)
        self.assertIn("Nitro exploded", got)

    def test_wikitext_heading_debris_is_dropped_too(self):
        """Character records still carry `===Early Life===` - 4.14 fixed the
        source, but those records have not been re-curated."""
        rec = """Spider-Man
History:
===Early Life===; Peter Benjamin Parker was born in Queens to CIA agents Richard and Mary Parker."""
        got = P.select(rec, "what happened", R.query_key, R.norm)
        self.assertNotIn("===", got)
        self.assertIn("Peter Benjamin Parker", got)


class EditorialBoilerplateIsNotNarrative(unittest.TestCase):
    """"This is an abridged version of X's history" is about the ARTICLE,
    not the subject, and quoting it answers a question with a cross-reference.

    Only 44 of 98,214 character records open this way (0.04%) - but they are
    Tony Stark, Ben Grimm and Peter Parker, the records most asked about. Low
    prevalence, high impact, and one fixed wiki template rather than a
    per-record special case.
    """

    REC = """Spider-Man
History:
This is an abridged version of Peter Parker's history. For a complete history see Peter Parker's Expanded History. Peter Benjamin Parker was born in Queens to CIA agents Richard and Mary Parker."""

    def test_the_cross_reference_is_not_quoted(self):
        got = P.select(self.REC, "who is spider-man", R.query_key, R.norm)
        self.assertNotIn("abridged version", got)
        self.assertNotIn("Expanded History", got)

    def test_the_real_narrative_after_it_survives(self):
        got = P.select(self.REC, "who is spider-man", R.query_key, R.norm)
        self.assertIn("born in Queens", got)


class RareTermsWeighMore(unittest.TestCase):
    """4.5b says "real overlap with the question's RARE terms". Scoring
    every term equally is not that, and real data showed the difference:

        how did nitro kill the new warriors
          -> "Chord attempted to kill himself. The Warriors also helped
              Scarlet Spider fight a mind controlled Spider-Man..."

    "kill" appears all over a 42,989-character record and "nitro" almost
    nowhere, so an unweighted count let the common word decide. Rarity is
    measured WITHIN THE RECORD - a term in most of its sentences cannot
    discriminate between them - which needs no index and no df table.
    """

    REC = """Some Team
History:
The team fought hard to kill the invaders in the first year of their run. Later they would kill again during a long and difficult campaign abroad. A third battle saw them kill several more enemies over many months. When Nitro detonated in Stamford he destroyed several city blocks."""

    def test_a_weak_match_does_not_fill_a_slot(self):
        """4.5b: "below threshold, no passage". Scoring alone is not enough -
        without a floor, sentences matching only the COMMON term fill the
        remaining slots and are then re-sorted into document order, putting
        the weak matches AHEAD of the strong one."""
        got = P.select(self.REC, "how did nitro kill the team",
                       R.query_key, R.norm)
        self.assertIn("Nitro detonated", got)
        self.assertNotIn("fought hard", got)
        self.assertNotIn("third battle", got)

    def test_the_rare_term_decides(self):
        got = P.select(self.REC, "how did nitro kill the team",
                       R.query_key, R.norm, max_sentences=1)
        self.assertIn("Nitro detonated", got)
        self.assertNotIn("fought hard", got)


class GoingOn(unittest.TestCase):
    """`explain in more detail` asked again must not re-read the opening.

    MEASURED 2026-09-26 in the web app: asking twice printed the SAME three
    sentences, because a question naming nobody takes the bare-narrative
    branch and that branch returns the record's opening. Identical input,
    identical output - correct, and useless as a conversation.
    """

    def test_skip_moves_past_what_was_already_quoted(self):
        first = select("what happened in the civil war")
        second = select("what happened in the civil war", skip=3)
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertNotIn(second.split(".")[0], first)

    def test_running_out_of_record_says_nothing(self):
        """A record has an end. Looping back to the opening would be the
        worst of the options - it reads as the system forgetting."""
        self.assertIsNone(select("what happened in the civil war", skip=99))

    def test_skip_zero_is_what_it_always_was(self):
        self.assertEqual(select("what happened in the civil war"),
                         select("what happened in the civil war", skip=0))

    def test_a_specific_ask_skips_the_sentences_it_already_showed(self):
        """The scored branch moves on too: the second-best match, not the
        best one again."""
        first = select("how did nitro explode", skip=0)
        second = select("how did nitro explode", skip=1)
        self.assertTrue(first)
        if second is not None:
            self.assertNotEqual(first, second)


class WhenToSayNothing(unittest.TestCase):
    """Silence is a valid answer. A passage is appended, so a wrong one is
    noise on top of a correct answer - but noise is still a cost."""

    def test_a_record_with_no_prose_yields_nothing(self):
        self.assertIsNone(select("how did nitro die", record=NO_PROSE))

    def test_a_question_this_record_does_not_address_yields_nothing(self):
        """Discriminating terms exist and none of them appear. The record is
        simply not about this, and the opening would be a non-answer."""
        self.assertIsNone(select("how did galactus devour arakko"))


if __name__ == "__main__":
    unittest.main()
