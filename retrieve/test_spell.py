"""Tests for typo recovery — Phase 4.13.

The phase's thesis in one line: **correction proposes, `rank()` disposes.**
`difflib` ranks by string similarity, which does not know who is famous — it
offers `mantego` before `magneto` and `spidermen` before `spiderman`. So
`spell` never picks a word. It hands candidate KEYS to the ranking loop that
already exists, and the ordering swept in 4.5a / 4.10 / 4.11 picks the winner.

The other half is the safety argument: 88.9% of single-edit typos resolve to
`None` today, so the fallback runs ONLY on the `None` branch and the 97.5%
round-trip cannot move. `test_a_query_that_already_resolves_never_corrects`
is what holds that property in place.

Spec: docs/superpowers/specs/2026-09-18-typo-recovery-design.md
"""
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path=None):
    spec = importlib.util.spec_from_file_location(name, path or HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = _load("resolve")
S = _load("spell")
# The fixtures are already written and already reviewed; rebuilding them here
# would be a second set to keep in step with build().
T = _load("test_resolve")
FakeIndex, rec = T.FakeIndex, T.rec


class DropNoise(unittest.TestCase):
    """Misspelled question words. 23% of typo damage, and the cheap half.

    `query_key` strips QUERY_NOISE by exact match, so `who created thor`
    keys as ("thor",). One typo in `created` and the whole thing collapses:
    ("crrreated", "thor") is not a subset of any name, and the unknown-word
    guard then refuses the tier-1 match outright.
    """

    def test_a_misspelled_question_word_is_dropped(self):
        self.assertEqual(S.drop_noise(("crrreated", "thor"), R.QUERY_NOISE), ("thor",))

    def test_a_correctly_spelled_word_is_untouched(self):
        """query_key already removed the exact ones; this must not re-scan."""
        self.assertEqual(S.drop_noise(("thor",), R.QUERY_NOISE), ("thor",))

    def test_a_known_name_is_never_dropped(self):
        """The guard that makes this safe.

        Without it a real but obscure character whose name happens to sit one
        edit from a question word would be deleted from their own query. A
        token the name index knows is a name, whatever it resembles.
        """
        self.assertEqual(
            S.drop_noise(("wanda",), R.QUERY_NOISE, vocab=frozenset({"wanda"})), ("wanda",))

    def test_a_word_resembling_nothing_survives(self):
        self.assertEqual(S.drop_noise(("zyxthalorax",), R.QUERY_NOISE), ("zyxthalorax",))

    def test_every_token_dropped_returns_the_key_unchanged(self):
        """Emptiness is not an improvement.

        resolve() treats an empty key as "names nobody" and refuses. Dropping
        the last token would turn a typo into a refusal, which is the outcome
        this phase exists to remove.
        """
        self.assertEqual(S.drop_noise(("crrreated",), R.QUERY_NOISE), ("crrreated",))


class Corrections(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ix = FakeIndex([rec("Magneto", pad=400), rec("Mantego", pad=2)])
        cls.names = R.build(cls.ix)
        cls.vocab = R.sorted_vocabulary(cls.names)

    def test_a_known_key_yields_nothing(self):
        """A query that spells everything right must not enter this code."""
        self.assertEqual(list(S.corrections(self.vocab, ("magneto",))), [])

    def test_a_typo_yields_the_corrected_key(self):
        got = list(S.corrections(self.vocab, ("magnteo",)))
        self.assertIn(("magneto",), got)

    def test_the_corrected_token_keeps_its_position(self):
        got = list(S.corrections(self.vocab, ("magnteo", "helmet")))
        self.assertTrue(all(c[1] == "helmet" for c in got), got)

    def test_at_most_two_tokens_are_corrected(self):
        """Three unknown tokens is a sentence, not a typo.

        Uncapped, the candidate set is the product of every token's matches
        and the cost stops being bounded.
        """
        got = list(S.corrections(self.vocab, ("aaaa", "bbbb", "cccc")))
        self.assertEqual(got, [])


class Respell(unittest.TestCase):
    """The question itself has to carry the correction downstream.

    FOUND BY PIPING A SESSION, 2026-09-18, which is why CLAUDE.md says a unit
    test under the UI is not a test of the product. Retrieval was already
    right - `deadpol` reached Wade Wilson (Earth-616) - but the PROMPT still
    said "deadpol", so the model answered "I couldn't find Deadpol" directly
    beneath "reading that as Deadpool". Incoherent, and worse than the
    silence 4.13 replaced.
    """

    def test_the_misspelled_word_is_replaced(self):
        self.assertEqual(
            S.respell("who created deadpol", ("deadpol",), ("deadpool",), R.norm),
            "who created deadpool")

    def test_nothing_changes_without_a_correction(self):
        self.assertEqual(
            S.respell("who created deadpool", ("deadpool",), None, R.norm),
            "who created deadpool")

    def test_other_words_are_left_alone(self):
        """Only the corrected token moves. Rewriting the whole question would
        put the model's words in the user's mouth."""
        self.assertEqual(
            S.respell("what powers does magnteo have", ("magnteo",),
                      ("magneto",), R.norm),
            "what powers does magneto have")

    def test_a_dropped_noise_word_does_not_shift_the_others(self):
        """drop_noise shortens the key, so positions do not line up; the
        match is by normalised WORD, not by index."""
        self.assertEqual(
            S.respell("who crrreated magnteo", ("crrreated", "magnteo"),
                      ("magneto",), R.norm, R.QUERY_NOISE),
            "who created magneto")

    def test_a_misspelled_question_word_is_respelled_not_left(self):
        """CORRECTED 2026-09-18 after piping a session.

        This test first asserted the misspelled word could stay, on the
        reasoning that only the entity matters. The session said otherwise:
        "who crrreated thor" resolved to Thor correctly and the model then
        answered "I don't have any information about Crrreated Thor", because
        the PROMPT still carried it. drop_noise deletes the token from the
        KEY; the question needs the word spelled right, not deleted - "who
        thor" is not an improvement on "who crrreated thor".
        """
        self.assertEqual(
            S.respell("who crrreated thor", ("crrreated", "thor"), ("thor",),
                      R.norm, R.QUERY_NOISE),
            "who created thor")


    def test_a_noise_word_after_the_entity_still_aligns(self):
        """Alignment must not be positional.

        Pairing the LAST spare token with the correction assumes the dropped
        noise word came first, which "who created thor" makes look safe.
        "thor varriants" puts it last, and the positional guess then maps
        `varriants` -> `thor` and leaves the entity misspelled.
        """
        question = "thoor varriants"
        key = R.query_key(question)      # ("thoor", "varriant") - normalised
        got = S.respell(question, key, ("thor",), R.norm, R.QUERY_NOISE)
        self.assertTrue(got.startswith("thor "), got)
        self.assertNotIn("thoor", got)


class RankingBeatsStringSimilarity(unittest.TestCase):
    """The thesis. If this test ever fails the phase has lost its argument."""

    def test_magnteo_reaches_magneto_not_mantego(self):
        """`difflib` offers mantego FIRST and magneto third.

        Taking the best string match answers Magneto questions with Mantego.
        Feeding both through rank() lets size decide, and it is not close.
        """
        ix = FakeIndex([rec("Mantego", pad=2), rec("Magneto", pad=400)])
        names = R.build(ix)
        self.assertEqual(ix.headlines[R.resolve(names, "magnteo")], "Magneto")


class TheNoneGate(unittest.TestCase):
    """Why the round-trip and the flagships cannot regress.

    Everything they measure resolves to a non-None doc today, so proving the
    fallback never runs on that path IS the regression proof. A number from a
    harness would only be evidence; this is the property itself.
    """

    def setUp(self):
        self.ix = FakeIndex([rec("Magneto", pad=400)])
        self.names = R.build(self.ix)

    def test_a_query_that_already_resolves_never_corrects(self):
        exploded = []

        def boom(*a, **k):
            exploded.append(True)
            raise AssertionError("the fallback ran on a query that resolved")

        original = R.spell.corrections
        R.spell.corrections = boom
        try:
            got = R.resolve(self.names, "magneto")
        finally:
            R.spell.corrections = original
        self.assertEqual(self.ix.headlines[got], "Magneto")
        self.assertFalse(exploded)

    def test_an_unknowable_query_still_returns_none(self):
        """Correction must not invent an answer out of nothing."""
        self.assertIsNone(R.resolve(self.names, "zyxthaloraxian the devourer"))


class ResolveSpelled(unittest.TestCase):
    """The sibling that reports WHAT was corrected, so the terminal can say."""

    def setUp(self):
        self.ix = FakeIndex([rec("Magneto", pad=400)])
        self.names = R.build(self.ix)

    def test_a_clean_query_reports_no_correction(self):
        doc, corrected = R.resolve_spelled(self.names, "magneto")
        self.assertEqual(self.ix.headlines[doc], "Magneto")
        self.assertIsNone(corrected)

    def test_a_typo_reports_the_key_it_used(self):
        doc, corrected = R.resolve_spelled(self.names, "magnteo")
        self.assertEqual(self.ix.headlines[doc], "Magneto")
        self.assertEqual(corrected, ("magneto",))

    def test_resolve_returns_what_resolve_spelled_returns(self):
        """resolve() is a thin wrapper; two code paths would drift."""
        for q in ("magneto", "magnteo", "nobody at all"):
            self.assertEqual(R.resolve(self.names, q),
                             R.resolve_spelled(self.names, q)[0], q)


if __name__ == "__main__":
    unittest.main()
