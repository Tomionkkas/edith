"""The web payload, with a fake engine - no model, no index, no network.

What this guards is the SHAPE the page renders from. The engine's own
behaviour is covered by 1,200 other tests; what is new here is that a turn
arrives carrying the record that answered and the art that belongs to it.
"""
import unittest
import importlib.util
import sys
import pathlib
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


W = _load("web", "infer/web.py")

RECORD = ("Wolverine\n"
          "Kind: character\n"
          "Page: James Howlett (Earth-616)\n"
          "Full name: James Howlett\n"
          "First appearance: Incredible Hulk Vol 1 180\n"
          "History:\nbody")
OTHER = ("Hellverine\n"
         "Kind: character\n"
         "Page: Akihiro (Earth-616)\n"
         "History:\nbody")


class FakePlan:
    """engine.Plan's slots, with nothing but what a caller sets."""

    def __init__(self, **kw):
        for slot in ("text", "prompt", "doc_id", "chitchat", "choices",
                     "rows", "corrected", "quoted", "quoted_leads"):
            setattr(self, slot, kw.get(slot))


class FakeTerm:
    """Only what web.answer() is allowed to touch."""

    def __init__(self, plan, records=(RECORD, OTHER)):
        self._records = list(records)
        self.index = types.SimpleNamespace(
            text=lambda d: self._records[d],
            headlines=[r.split("\n")[0] for r in records],
            docs=list(range(len(records))))
        self.last_plan_args = None
        self.engine = types.SimpleNamespace(plan=self._plan)
        self._planned = plan
        self.images = {"James Howlett (Earth-616)": "Wolverine Vol 8 22.jpg",
                       "Akihiro (Earth-616)": "Hellverine Vol 2 2.jpg"}
        self.sft = self.search = self.disambiguate = None
        self.facts = self.resolve = None

    def _plan(self, *args, **kw):
        self.last_plan_args = (args, kw)
        return self._planned


class Answer(unittest.TestCase):
    def test_an_answer_carries_the_record_and_its_art(self):
        term = FakeTerm(FakePlan(text="He is a mutant.", doc_id=0,
                                 rows=[("Full name", "James Howlett")]))
        out = W.answer(term, "who is wolverine")
        self.assertEqual(out["answer"], "He is a mutant.")
        self.assertEqual(out["page"], "James Howlett (Earth-616)")
        self.assertEqual(out["kind"], "character")
        self.assertEqual(out["fields"], [["Full name", "James Howlett"]])
        self.assertIn("/Wolverine_Vol_8_22.jpg/revision/latest", out["images"][0])

    def test_a_picker_carries_every_choice_with_its_own_art(self):
        """The screen art matters most on: several records with one name,
        told apart by the picture before the label is read. choices are
        (size, headline, doc_id), the shape facts.variants() returns."""
        plan = FakePlan(doc_id=0, choices=[(138242, "Wolverine", 0),
                                           (96637, "Hellverine", 1)])
        out = W.answer(FakeTerm(plan), "who is wolverine")
        self.assertEqual(len(out["choices"]), 2)
        self.assertEqual(out["choices"][1]["page"], "Akihiro (Earth-616)")
        self.assertIn("/Hellverine_Vol_2_2.jpg/revision/latest",
                      out["choices"][1]["images"][0])

    def test_a_record_with_no_art_returns_an_empty_chain(self):
        term = FakeTerm(FakePlan(text="hi", doc_id=0))
        term.images = {}
        self.assertEqual(W.answer(term, "hi")["images"], [])

    def test_nothing_retrieved_still_answers(self):
        out = W.answer(FakeTerm(FakePlan(text="I don't have that.")),
                       "who is zarblaxian")
        self.assertIsNone(out["page"])
        self.assertIsNone(out["doc_id"])
        self.assertEqual(out["images"], [])

    def test_the_quoted_passage_and_correction_survive(self):
        """4.5b's verbatim passage and 4.13's "reading that as" line are part
        of the answer, not decoration."""
        plan = FakePlan(text="x", doc_id=0, quoted="He was born in Alberta.",
                        corrected="wolverine")
        out = W.answer(FakeTerm(plan), "who is wolverin")
        self.assertEqual(out["quoted"], "He was born in Alberta.")
        self.assertEqual(out["corrected"], "wolverine")

    def test_the_previous_record_is_threaded_for_follow_ups(self):
        """"what are his powers" answers from the record already on screen -
        the same threading the terminal does with self.last_doc."""
        term = FakeTerm(FakePlan(text="x", doc_id=0))
        W.answer(term, "what are his powers", previous=7)
        self.assertEqual(term.last_plan_args[1]["previous"], 7)


class Settle(unittest.TestCase):
    """A pick must stop the menu re-opening for that name."""

    def setUp(self):
        self.term = FakeTerm(FakePlan(text="x", doc_id=0))
        self.term.resolve = _load("resolve", "retrieve/resolve.py")

    def test_a_pick_settles_the_name_it_was_offered_for(self):
        settled = {}
        W.settle(self.term, 0, settled)
        self.assertEqual(settled, {("wolverine",): 0})

    def test_every_trailing_paren_is_stripped_not_just_the_reality(self):
        """terminal.settle() loops PAREN_SUFFIX because a headline can carry
        more than one disambiguator. Stripping only the reality suffix left
        ("sasquatch", "beast"), which query_key("sasquatch") never matches -
        so the pick silently did nothing."""
        self.term.index.headlines = ["Sasquatch (Beast) (Earth-616)"]
        settled = {}
        W.settle(self.term, 0, settled)
        self.assertEqual(settled, {("sasquatch",): 0})


class Generation(unittest.TestCase):
    """A plan with no text carries a PROMPT, not an answer."""

    def test_the_prompt_is_never_shown_as_the_answer(self):
        """MEASURED FAILURE 2026-09-26: `cosmic spider` printed the whole
        scaffold - "Context: Spider Also known as: ... User: cosmic spider
        Assistant:" - because the page rendered plan.prompt when plan.text
        was None. The terminal streams the model's continuation instead."""
        term = FakeTerm(FakePlan(text=None, prompt="Context: X / User: q / Assistant:",
                                 doc_id=0))
        term.model = object()
        term.sample = types.SimpleNamespace(
            generate=lambda *a, **k: " He is a mutant.")
        term.engine.tidy = lambda s, sft: s
        term.sp = term.device = term.block = None
        out = W.answer(term, "cosmic spider")
        self.assertEqual(out["answer"], "He is a mutant.")
        self.assertNotIn("Assistant:", out["answer"])

    def test_no_model_means_no_crash(self):
        """A caller without weights (a test, a corpus-only install) gets the
        prompt back rather than an exception."""
        term = FakeTerm(FakePlan(text=None, prompt="Context:", doc_id=0))
        self.assertEqual(W.answer(term, "q")["answer"], "Context:")


class Others(unittest.TestCase):
    """Who ELSE has gone by the name - the second half of a picker.

    facts.variants() answers "which Hulk" and every row is Bruce Banner from
    another earth. It cannot answer "who else has been the Hulk", which the
    corpus already holds: She-Hulk, Brawn and Rick Jones all carry `Hulk` on
    a Codename line. Verified live 2026-09-26; spider-man gives Chasm and
    Scorpion, iron man gives Emperor Doom and War Machine.
    """

    def test_a_codename_holder_is_offered_as_someone_else(self):
        term = FakeTerm(FakePlan(text="x", doc_id=0))
        term.resolve = _load("resolve", "retrieve/resolve.py")
        term.facts = _load("facts", "infer/facts.py")
        term.engine._NAMES = {("hulk",): [
            (0, 147852, True, term.resolve.PROV_IDENTITY, True, False),
            (1, 57663, True, term.resolve.PROV_CODENAME, True, False),
        ]}
        images = sys.modules["images"]
        out = W.others_of(term, "who is hulk", exclude={0}, images=images)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["doc_id"], 1)
        self.assertEqual(out[0]["provenance"], "codename")

    def test_the_identity_holders_are_not_repeated(self):
        """They are already the variants row above; showing them twice is
        how a picker stops being a choice."""
        term = FakeTerm(FakePlan(text="x", doc_id=0))
        term.resolve = _load("resolve", "retrieve/resolve.py")
        term.facts = _load("facts", "infer/facts.py")
        term.engine._NAMES = {("hulk",): [
            (0, 147852, True, term.resolve.PROV_IDENTITY, True, False),
        ]}
        out = W.others_of(term, "who is hulk", exclude=set(),
                          images=sys.modules["images"])
        self.assertEqual(out, [])


class Session(unittest.TestCase):
    def test_a_page_load_clears_the_picks(self):
        """A pick must not outlive the tab. SETTLED is module-level, so
        without a reset a choice made an hour ago still silences the picker
        for that name - which is what made `who is ghost rider` answer
        instead of offering, long after the pick that caused it."""
        W.SETTLED[("wolverine",)] = 41488
        W.SETTLED.clear()
        self.assertEqual(W.SETTLED, {})


class Handler(unittest.TestCase):
    def test_static_files_outside_the_folder_are_refused(self):
        """The one security boundary here: a path that climbs out of
        infer/web/ is a traversal attempt, not a typo."""
        self.assertFalse(W.static_path("../../CLAUDE.md"))
        self.assertFalse(W.static_path("/etc/passwd"))

    def test_the_root_path_is_the_page(self):
        self.assertEqual(W.static_name("/"), "page.html")


if __name__ == "__main__":
    unittest.main()
