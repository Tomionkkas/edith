"""The correction has to be VISIBLE. Phase 4.13.

Decided with the user 2026-09-18: say it, then answer. Silence was rejected
because a wrong correction then looks like the model hallucinating rather
than misreading, and asking (reusing the 4.7 picker) was rejected because it
puts a turn of friction on the thing this phase exists to make frictionless.

The line names the RECORD, not the corrected token: "reading that as
Spider-Man" tells the reader whether the guess was right; "reading that as
spiderman" tells them nothing.
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


engine = _load("engine", "infer/engine.py")
resolve = _load("resolve", "retrieve/resolve.py")
facts = _load("facts", "infer/facts.py")
T = _load("test_resolve", "retrieve/test_resolve.py")


class FakeIndex(T.FakeIndex):
    """The real Index exposes `postings`; resolved_doc passes it as the
    corpus vocabulary, which is the guard that refuses invented names."""

    def __init__(self, records):
        super().__init__(records)
        self.postings = {t for r in records for t in r.lower().split()}


def names_for(records):
    ix = FakeIndex(records)
    return ix, resolve.build(ix)


class PlanCarriesTheCorrection(unittest.TestCase):

    def setUp(self):
        self.ix, self.names = names_for([T.rec("Magneto", pad=400)])
        engine._NAMES = self.names

    def tearDown(self):
        engine._NAMES = None

    def test_a_clean_question_carries_no_correction(self):
        doc, corrected = engine.resolved_doc_spelled(
            "who is magneto", self.ix, facts, resolve)
        self.assertEqual(self.ix.headlines[doc], "Magneto")
        self.assertIsNone(corrected)

    def test_a_typo_carries_the_correction(self):
        doc, corrected = engine.resolved_doc_spelled(
            "who is magnteo", self.ix, facts, resolve)
        self.assertEqual(self.ix.headlines[doc], "Magneto")
        self.assertIsNotNone(corrected)

    def test_resolved_doc_still_returns_just_the_doc(self):
        """~40 callers read this. Its signature must not move."""
        self.assertEqual(
            engine.resolved_doc("who is magnteo", self.ix, facts, resolve),
            engine.resolved_doc_spelled(
                "who is magnteo", self.ix, facts, resolve)[0])


class PlanExposesIt(unittest.TestCase):
    """The terminal reads Plan, not resolved_doc. If it does not travel this
    far the user never sees it."""

    def setUp(self):
        self.ix, self.names = names_for([T.rec("Magneto", pad=400)])
        engine._NAMES = self.names
        self.sft = _load("sft_data", "train/sft_data.py")

    def tearDown(self):
        engine._NAMES = None

    def _plan(self, q):
        search = _load("search", "retrieve/search.py")
        disambiguate = _load("disambiguate", "retrieve/disambiguate.py")
        return engine.plan(q, self.ix, self.sft, search, disambiguate, facts,
                           resolve)

    def test_a_typo_reaches_the_plan(self):
        self.assertIsNotNone(self._plan("who is magnteo").corrected)

    def test_a_clean_question_leaves_it_none(self):
        self.assertIsNone(self._plan("who is magneto").corrected)


class TheLineItself(unittest.TestCase):
    """Rendering, kept away from the terminal's 860 lines of I/O."""

    def test_it_names_the_record_not_the_token(self):
        render = _load("render", "infer/render.py")
        self.assertEqual(render.correction_line("Spider-Man"),
                         "reading that as Spider-Man")

    def test_nothing_renders_without_a_correction(self):
        render = _load("render", "infer/render.py")
        self.assertIsNone(render.correction_line(None))


if __name__ == "__main__":
    unittest.main()
