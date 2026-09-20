"""When a passage earns its place. Phase 4.5b.

A passage is APPENDED, never substituted, so a wrong one is noise on top of a
correct answer rather than an error. That makes the trigger a question of
taste rather than safety - but noise is still a cost, and a quote after every
answer would train the reader to skip it.

4.5b names three triggers. Measurement on 2026-09-20 found a fourth it needs
and a hole in one it has:

    'tell me about the civil war'  -> 91 chars, 1 row, field path
       "Civil War (Event) is from Earth-616. Civil War (Event) first
        appeared in Civil War Vol 1 1."

91 characters clears 4.5b's "thin (< ~80 chars)" bar while saying nothing.
Length is the wrong proxy for a profile question; ROW COUNT is the right one,
because a whole-entity question that yielded one field is thin whatever its
sentence happens to weigh.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    # Registered before exec: theme.py uses @dataclass, which resolves its
    # annotations through sys.modules and fails on an unregistered module.
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E = _load("engine", "infer/engine.py")


class NeverOnAFieldQuestion(unittest.TestCase):
    """One line is the right answer, and a paragraph after it is worse."""

    def test_who_created_is_left_alone(self):
        self.assertFalse(E.wants_passage("who created moon knight",
                                         "Moon Knight was created by Doug Moench.", None))

    def test_first_appearance_is_left_alone(self):
        self.assertFalse(E.wants_passage("when did blade first appear",
                                         "Blade first appeared in Tomb of Dracula Vol 1 10.", None))

    def test_a_narrative_word_does_not_override_it(self):
        """"who created X" wins even phrased as "how was X created" - the
        answer is still one credit line."""
        self.assertFalse(E.wants_passage("who created spider-man", "x" * 200, None))


class NarrativeQuestionsAlwaysWant(unittest.TestCase):
    """These are the questions no field can answer."""

    def test_what_happened(self):
        self.assertTrue(E.wants_passage("what happened in the civil war", None, None))

    def test_how(self):
        self.assertTrue(E.wants_passage("how did nitro die", None, None))

    def test_why(self):
        self.assertTrue(E.wants_passage("why did captain america surrender", None, None))

    def test_what_led_to(self):
        self.assertTrue(E.wants_passage("what led to the registration act", None, None))


class ThinAnswersWant(unittest.TestCase):

    def test_a_short_field_answer(self):
        self.assertTrue(E.wants_passage("tell me about wakanda", "Wakanda.", None))

    def test_a_profile_question_that_yielded_one_row(self):
        """The hole in 4.5b's length rule, and the reason this phase needed
        a fourth trigger: 91 characters of nothing."""
        thin = ("Civil War (Event) is from Earth-616. "
                "Civil War (Event) first appeared in Civil War Vol 1 1.")
        self.assertGreater(len(thin), E.THIN_TEXT)
        self.assertTrue(E.wants_passage("tell me about the civil war", thin,
                                        [("First appearance", "Civil War Vol 1 1")]))


class FullAnswersDoNot(unittest.TestCase):

    def test_a_rich_profile_is_left_alone(self):
        rows = [(f"Field {i}", "value") for i in range(8)]
        self.assertFalse(E.wants_passage("tell me about wolverine", "x" * 300, rows))

    def test_a_plain_question_on_the_model_path_is_left_alone(self):
        """text=None means the model is answering. That is not by itself a
        reason to quote - only a narrative question is."""
        self.assertFalse(E.wants_passage("what is adamantium made of", None, None))


class TheQuoteLeadsOnNarrativeAsks(unittest.TestCase):
    """Measured on a live run of 260 questions, 2026-09-20: 69% of quoted
    turns opened with a bibliographic credit before the passage.

        what happened in American Revolutionary War
          "American Revolutionary War was created by Ken Bald."
          > The American Revolutionary War ... was the armed struggle in
            which the thirteen North American colonies rejected ...

    4.5b's append-never-substitute rule exists so a passage cannot make a
    CORRECT answer worse. On a narrative ask the credit is not
    correct-but-incomplete - it answers a different question - so leading
    with it buries the answer. The safety property is untouched: the quote
    is still verbatim and still never generated.

    Only for the NARRATIVE trigger. A thin profile answer is still a real
    answer to what was asked, and keeps its place above the passage.
    """

    def test_a_narrative_ask_leads_with_the_quote(self):
        self.assertTrue(E.passage_leads("what happened in the civil war"))
        self.assertTrue(E.passage_leads("how did nitro die"))

    def test_a_thin_profile_keeps_its_answer_first(self):
        self.assertFalse(E.passage_leads("tell me about the civil war"))

    def test_a_field_question_never_leads_with_a_quote(self):
        self.assertFalse(E.passage_leads("who created spider-man"))
        self.assertFalse(E.passage_leads("when did blade first appear"))


class TheQuoteLooksQuoted(unittest.TestCase):
    """A quote must not read as EDITH's own sentence.

    It is the one part of the answer that came from the source verbatim, and
    the reader can only judge it if they can see which part that is.
    """

    def setUp(self):
        self.render = _load("render", "infer/render.py")
        self.theme = _load("theme", "infer/theme.py")
        self.t = self.theme.THEMES["plain"]

    def test_every_line_carries_the_quote_mark(self):
        text = ("In a battle between Nitro and the New Warriors, Nitro "
                "exploded, killing many people. This led the government to "
                "introduce a registration act.")
        out = self.render.quotation(text, self.t, 70)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        self.assertGreater(len(lines), 1)
        for line in lines:
            self.assertIn(self.render.QUOTE_MARK, line, line)

    def test_the_words_are_unchanged(self):
        out = self.render.quotation("Nitro exploded in Stamford.", self.t, 70)
        self.assertIn("Nitro exploded in Stamford.", out)

    def test_it_respects_the_width(self):
        out = self.render.quotation("word " * 60, self.t, 50)
        for line in out.splitlines():
            self.assertLessEqual(len(line), 50, line)


if __name__ == "__main__":
    unittest.main()
