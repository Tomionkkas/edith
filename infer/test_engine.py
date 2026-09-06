"""Routing tests: which questions get a context block, and which do not.

Sending a retrieved record with a greeting is a pairing the model never saw
in training, and it degenerates into a repetition loop rather than failing
visibly.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sft = _load("sft_data", "train/sft_data.py")
E = _load("engine", "infer/engine.py")
facts = _load("facts", "infer/facts.py")
resolve = _load("resolve", "retrieve/resolve.py")


class Routing(unittest.TestCase):
    def chit(self, q):
        return E.is_chitchat(q, sft)

    def test_bare_greetings(self):
        for q in ("hi", "hello", "hey", "yo", "thanks", "bye", "ok"):
            self.assertTrue(self.chit(q), q)

    def test_punctuation_does_not_matter(self):
        """Training saw "who are you?"; users type "who are you"."""
        self.assertTrue(self.chit("who are you"))
        self.assertTrue(self.chit("who are you?"))

    def test_case_does_not_matter(self):
        self.assertTrue(self.chit("HI"))
        self.assertTrue(self.chit("Who Are You?"))

    def test_trailing_whitespace(self):
        self.assertTrue(self.chit("  hi  "))

    def test_empty_input(self):
        self.assertTrue(self.chit(""))

    def test_marvel_questions_are_not_chitchat(self):
        for q in ("who created moon knight",
                  "tell me about the hood in detail",
                  "what powers does venom have",
                  "hulk"):
            self.assertFalse(self.chit(q), q)

    def test_a_character_named_like_a_greeting_still_looks_up(self):
        """"Hi-Lite" is a real Marvel character; "hi lite" must not route to
        small talk just because it starts with a greeting word."""
        self.assertFalse(self.chit("who is hi-lite"))

    def test_long_phrases_are_not_small_talk(self):
        self.assertFalse(self.chit("ok so tell me about spider-man please"))

    def test_every_trained_chitchat_prompt_routes_to_empty(self):
        """The router must cover everything the model was taught."""
        for user, _ in sft.CHITCHAT:
            self.assertTrue(self.chit(user), user)


class TidyingWhatTheDataTaught(unittest.TestCase):
    """Two defects reached the stage-3 dataset and the model learned both.

    32,345 doubled full stops (6.9% of examples) and 2,621 "They works".
    sft_data.emit() stops the NEXT dataset carrying them; until stage 3 is
    retrained the checkpoint we have keeps producing them, so the answer is
    repaired on the way out with the same functions that clean the data.
    """

    def test_a_doubled_stop_is_collapsed(self):
        self.assertEqual(E.tidy("The full extent is unknown..", sft),
                         "The full extent is unknown.")

    def test_a_plural_subject_agrees(self):
        self.assertEqual(E.tidy("They works as a Vigilante.", sft),
                         "They work as a Vigilante.")

    def test_a_clean_answer_is_untouched(self):
        good = "Moon Knight was created by Doug Moench and Don Perlin."
        self.assertEqual(E.tidy(good, sft), good)

    def test_a_real_title_keeps_its_ellipsis(self):
        s = "Wolverine first appeared in What If...? Vol 1 1."
        self.assertEqual(E.tidy(s, sft), s)

class ChitChatCarriesNoRecord(unittest.TestCase):
    """Found by talking to it, not by any test.

    "hi" resolved to Hi-Vo (Earth-616) and the terminal drew a full character
    card beside the greeting. Every unit test called try_facts directly, so
    none of them ever saw what the screen actually showed.
    """

    def test_plan_marks_small_talk(self):
        for greeting in ("hi", "hello", "thanks", "who are you"):
            plan = E.plan(greeting, _FakeIndex(), sft, None, None, _FakeFacts())
            self.assertTrue(plan.chitchat, greeting)
            self.assertIsNone(plan.doc_id, greeting)

    def test_a_real_question_is_not_small_talk(self):
        """Checked at the router rather than through plan(), which would need
        the whole pipeline standing up to answer it."""
        for question in ("who created moon knight", "what powers does storm have",
                         "tell me about the x-men"):
            self.assertFalse(E.is_chitchat(question, sft), question)

    def test_small_talk_still_gets_a_prompt_to_answer_with(self):
        """It loses the card, not the reply."""
        plan = E.plan("hi", _FakeIndex(), sft, None, None, _FakeFacts())
        self.assertIsNotNone(plan.text or plan.prompt)


class _FakeIndex:
    docs = [""]
    postings = {}

    def text(self, doc_id):
        return ""


class _FakeFacts:
    @staticmethod
    def entity_text(q):
        return q

    @staticmethod
    def detect_intent(q):
        return None

    @staticmethod
    def wants_variants(q):
        return False

    @staticmethod
    def wants_profile(q):
        return False

    def wants_whole_entity(self, q):
        """Composed from THIS fake's own three predicates, in the same
        expression facts.wants_whole_entity() composes the real ones from -
        so a subclass that overrides one of them (and every fake below
        overrides at least one) stays consistent with itself. An instance
        method, not a staticmethod, because the overrides are a mix of both
        and only `self.` dispatches to either."""
        return (self.detect_intent(q) is None
                and not self.wants_variants(q)
                and self.wants_profile(q))


class WantsChoice(unittest.TestCase):
    """wants_choice() is pure and is where the offer-or-answer decision
    actually lives, so it is tested directly rather than through plan()."""

    def test_a_broad_ask_returns_true(self):
        facts = _FakeFactsWantsVariants()
        self.assertTrue(E.wants_choice(
            "show me the variants of spider-man", _FakeIndex(), facts,
            object(), doc_id=1))

    def test_an_ordinary_question_returns_false(self):
        """Not a broad ask AND one record clearly dominates - moon knight
        measures 38.7, well past CONFIDENCE_TO_ASK."""
        saved = E._NAMES
        E._NAMES = {"seeded": True}
        try:
            self.assertFalse(E.wants_choice(
                "who created moon knight", _FakeIndex(), _PlainFacts(),
                _ConfidenceResolve(1, 38.7), doc_id=1))
        finally:
            E._NAMES = saved

    def test_no_doc_id_returns_false(self):
        facts = _FakeFactsWantsVariants()
        self.assertFalse(E.wants_choice(
            "show me the variants of spider-man", _FakeIndex(), facts,
            object(), doc_id=None))

    def test_no_resolver_returns_false(self):
        facts = _FakeFactsWantsVariants()
        self.assertFalse(E.wants_choice(
            "show me the variants of spider-man", _FakeIndex(), facts,
            None, doc_id=1))


class PlanChoicesSlot(unittest.TestCase):
    """The `choices` slot is new; most other tests build a Plan through
    plan(), so its default and its round-trip are covered directly here."""

    def test_defaults_to_no_choices(self):
        self.assertIsNone(E.Plan().choices)

    def test_choices_round_trip(self):
        rows = [(100, "Spider-Man", 0), (90, "Spider-Man (Earth-1610)", 1)]
        self.assertEqual(E.Plan(choices=rows).choices, rows)


class ChoicesInThePlan(unittest.TestCase):
    """A broad ask is a request to be OFFERED the choice, not answered.

    Today `facts.render_variants` turns it into a sentence listing ten
    headlines, which cannot be selected from - so "tell me about the ultimate
    one" matched the WORD ultimate and returned an Earth-6160 team.

    plan() cannot be driven with the module's shared `_FakeFacts`/`_FakeIndex`
    here: `_FakeFacts.wants_variants` is hardcoded False and `_FakeIndex` has
    no name index behind it. This extends them locally, and seeds/restores
    engine._NAMES directly - it is a module-level cache keyed on nothing but
    "has it been loaded yet", so leaving it seeded would feed the REAL
    pipeline tests in test_pipeline.py a fake name index.
    """

    def setUp(self):
        self._saved_names = E._NAMES
        E._NAMES = {"spider-man": "seeded so _names() skips resolve.load()"}

    def tearDown(self):
        E._NAMES = self._saved_names

    def test_a_broad_ask_returns_choices_not_text(self):
        rows = [(100, "Spider-Man", 0), (90, "Spider-Man (Earth-1610)", 1)]
        facts = _FakeFactsOffersVariants(rows, total=2)
        resolve = _FakeResolveForChoice(doc_id=0)
        p = E.plan("show me the variants of spider-man", _FakeIndex(), sft,
                   None, None, facts, resolve=resolve)
        self.assertEqual(p.choices, rows)
        self.assertIsNone(p.text)
        self.assertIsNone(p.prompt)
        self.assertEqual(p.doc_id, 0)

    def test_the_choices_path_does_not_run_try_facts(self):
        """facts.variants() walks all 201,815 headlines - 636 ms measured.

        With the choices check sitting AFTER `text = try_facts(...)`, a broad
        ask paid that walk twice: once inside try_facts to build a sentence
        that is then thrown away, once to build the rows. A second of latency
        on precisely the question the picker exists to answer.

        `render_variants` is the tell. `detect_intent` looks like the obvious
        one and is useless: try_facts short-circuits at its OWN variants
        branch long before reaching it, so a counter there reads zero either
        way. Only try_facts renders the sentence; the choices path calls
        `variants` for rows and never renders.
        """
        rows = [(100, "Spider-Man", 0), (90, "Spider-Man (Earth-1610)", 1)]

        class _Counting(_FakeFactsOffersVariants):
            seen = 0

            def render_variants(self, index, resolve, doc_id):
                type(self).seen += 1
                return super().render_variants(index, resolve, doc_id)

        facts = _Counting(rows, total=2)
        E.plan("show me the variants of spider-man", _FakeIndex(), sft,
               None, None, facts, resolve=_FakeResolveForChoice(doc_id=0))
        self.assertEqual(_Counting.seen, 0)


class TryFactsStillRendersVariants(unittest.TestCase):
    """Coverage gap closed after landing choices: plan() now routes a broad
    ask through wants_choice() and never reaches try_facts's own variants
    branch (see the comment on that branch in engine.py). But try_facts() is
    still called directly by engine.main()'s CLI, and before this test
    nothing exercised that branch THROUGH try_facts - test_facts.py covers
    facts.variants() the function, not this call site, and test_pipeline.py's
    only "variant" is an unrelated docstring.

    Placed here rather than in test_pipeline.py because it needs no real
    corpus: a 3-record fake index (two of them sharing a headline after the
    reality suffix) is enough to drive the REAL facts.py and resolve.py
    through the branch, the same seeded-engine._NAMES technique
    ChoicesInThePlan uses above.
    """

    def setUp(self):
        self._saved_names = E._NAMES
        # One name key, "spiderman" (how resolve.norm folds "Spider-Man"),
        # pointing at doc 0: (doc_id, size, main, primary, has_page, famous).
        E._NAMES = {("spiderman",): [(0, 999, True, True, True, False)]}

    def tearDown(self):
        E._NAMES = self._saved_names

    def test_a_broad_ask_renders_the_variants_sentence(self):
        answer = E.try_facts("show me the variants of spider-man",
                             _VariantsIndex(), sft, None, None, facts, resolve)
        self.assertIsNotNone(answer)
        self.assertIn("records in my sources are called Spider-Man", answer)


class _VariantsIndex:
    """Three records: doc 0 is the resolved "Spider-Man", docs 1 and 2 share
    a headline that collapses to the same name once REALITY_SUFFIX is
    stripped - real facts.variants() finds all three, dedupes 1 and 2's
    identical headline down to one row, and total(3) > len(rows)(2) is what
    makes render_variants() choose the "N records ... are called X, showing
    the M largest" phrasing this test checks for.
    """
    docs = [0, 1, 2]
    headlines = ["Spider-Man", "Spider-Man (Earth-1610)", "Spider-Man (Earth-1610)"]
    postings = {}
    _records = ["Spider-Man\n" + "a" * 90,
                "Spider-Man (Earth-1610)\n" + "b" * 40,
                "Spider-Man (Earth-1610)\n" + "c" * 40]

    def text(self, doc_id):
        return self._records[doc_id]


class _FakeFactsWantsVariants(_FakeFacts):
    @staticmethod
    def wants_variants(q):
        return True


class _FakeFactsOffersVariants(_FakeFacts):
    """wants_variants is always True, and variants() returns a fixed pair -
    what facts.variants(index, resolve, doc_id) returns for a contested name,
    without needing a real index or name index behind it.

    render_variants() is also stubbed, non-empty, and DIFFERENT text from
    variants() - so `try_facts`'s own variants branch (which runs first,
    before plan() ever reaches the new choices branch) has something real to
    render and discard. If plan()'s choices branch did not return before
    `if text is not None`, this rendered sentence would leak out as `p.text`.
    """

    def __init__(self, rows, total):
        self._rows = rows
        self._total = total

    def wants_variants(self, q):
        return True

    def render_variants(self, index, resolve, doc_id):
        return f"{self._total} records in my sources are called Spider-Man."

    def variants(self, index, resolve, doc_id):
        return self._rows, self._total


class _FakeResolveForChoice:
    """Just enough of retrieve/resolve.py for plan() to resolve a doc_id
    without a real name index or corpus behind it."""

    def __init__(self, doc_id):
        self._doc_id = doc_id

    @staticmethod
    def query_key(text):
        return ("spider-man",)

    def resolve(self, names, query, known_words=None):
        return self._doc_id


class _ConfidenceResolve(_FakeResolveForChoice):
    """A resolver whose confidence is whatever the test says it is.

    `wants_choice` asks a single question of the resolver - how clearly does
    one record win - so a stub that answers it is the whole fixture.

    `known_words=None` mirrors the real `resolve.confidence()` signature
    (FINDING 2026-09-02: confidence() now takes the same known_words escape
    hatch resolve() always had) - a subclass can capture what wants_choice()
    actually passes, as WantsChoiceUsesCorpusVocabulary does below.
    """

    def __init__(self, doc_id, confidence):
        super().__init__(doc_id)
        self._confidence = confidence

    def confidence(self, names, query, known_words=None):
        return self._confidence


class _PlainFacts(_FakeFacts):
    """Not a broad ask: the variants route must not be what fires here."""

    @staticmethod
    def wants_variants(q):
        return False

    @staticmethod
    def entity_text(q):
        return q


class UncertainSpecificAsksToo(unittest.TestCase):
    """"who is spider-man earth-1610" looks precise and is not: Peter Parker
    and Miles Morales are both Earth-1610. Precision in the question is not
    certainty in the answer.

    The threshold is measured, not chosen - see CONFIDENCE_TO_ASK's comment.
    """

    def setUp(self):
        # _names() caches on a module global, so seed it rather than let a
        # stub's load() decide what every later test sees.
        self._saved = E._NAMES
        E._NAMES = {"seeded": True}

    def tearDown(self):
        E._NAMES = self._saved

    def wants(self, confidence):
        return E.wants_choice("who is spider-man earth-1610", _FakeIndex(),
                              _PlainFacts(), _ConfidenceResolve(1, confidence), 1)

    def test_two_records_neck_and_neck_offer_a_choice(self):
        self.assertTrue(self.wants(1.9))

    def test_a_record_that_clearly_dominates_answers_directly(self):
        """The floor that stops "broader" becoming "always". Measured:
        moon knight 38.7, taskmaster 43.4, kitty pryde 67.9."""
        self.assertFalse(self.wants(38.7))

    def test_the_boundary_is_the_measured_threshold(self):
        self.assertTrue(self.wants(E.CONFIDENCE_TO_ASK - 0.01))
        self.assertFalse(self.wants(E.CONFIDENCE_TO_ASK))

    def test_a_follow_up_that_names_nobody_never_offers(self):
        """FOUND BY READING A SESSION, not by any test.

        After a pick the terminal asks "who is this", which names nobody -
        `resolved_doc` recognises that and carries the previously chosen
        record. But `confidence()` fell back to scoring the raw scaffolding,
        got a low number, and re-opened the picker. On screen: you chose row
        2, and the same menu came straight back.

        A question that names nobody is a follow-up about the record already
        in hand. There is nothing to disambiguate.
        """

        class _NamesNobody(_ConfidenceResolve):
            @staticmethod
            def query_key(text):
                return ()

        self.assertFalse(E.wants_choice(
            "who is this", _FakeIndex(), _PlainFacts(),
            _NamesNobody(1, 1.0), doc_id=1))

    def test_a_single_candidate_is_maximum_confidence(self):
        """confidence() returns infinity when only one record can answer."""
        self.assertFalse(self.wants(float("inf")))

    def test_confidence_is_measured_on_the_stripped_text(self):
        """`resolved_doc` resolves facts.entity_text(question), so measuring
        the raw question scores a different string than the one that will be
        answered - "what powers does moon knight have" reads 5.9 raw and 38.7
        stripped, which is the difference between a picker and a toll booth."""
        seen = []

        class _Recording(_ConfidenceResolve):
            def confidence(self, names, query, known_words=None):
                seen.append(query)
                return 99.0

        class _Stripping(_PlainFacts):
            @staticmethod
            def entity_text(q):
                return "moon knight"

        E.wants_choice("what powers does moon knight have", _FakeIndex(),
                       _Stripping(), _Recording(1, 99.0), 1)
        self.assertEqual(seen, ["moon knight"])


class WantsChoiceUsesCorpusVocabulary(unittest.TestCase):
    """FINDING 2026-09-02: `resolve.confidence()`'s tier-1 unknown-word guard
    must be judged against the CORPUS vocabulary, the same one resolve.py:145
    already passes resolve() as `index.postings` - not the NAME vocabulary
    confidence() fell back to reading on its own. `index` was already a
    parameter of `wants_choice`; it was simply never read. This pins the
    wiring, not the guard's behaviour (that lives in retrieve/test_resolve.py
    and infer/test_pipeline.py).
    """

    def setUp(self):
        self._saved = E._NAMES
        E._NAMES = {"seeded": True}

    def tearDown(self):
        E._NAMES = self._saved

    def test_confidence_is_called_with_index_postings(self):
        seen = []

        class _Recording(_ConfidenceResolve):
            def confidence(self, names, query, known_words=None):
                seen.append(known_words)
                return 99.0

        class _PostingsIndex(_FakeIndex):
            postings = {"a real corpus word": True}

        E.wants_choice("what powers does moon knight have", _PostingsIndex(),
                       _PlainFacts(), _Recording(1, 99.0), 1)
        self.assertEqual(seen, [{"a real corpus word": True}])


class SingleRowMenuGuard(unittest.TestCase):
    """FINDING 2026-09-02 (Minor): plan() gated the choices branch on
    `total`, facts.variants()'s RAW count before it dedupes same-headlined
    records into `rows` - but the menu infer/terminal.py's offer() draws is
    built from `rows`. A name whose every contending record shares one exact
    headline collapses to a single row while `total` stays > 1, so plan()
    still offered a choice: "1 records could be this. Which?" - a menu of
    one, and ungrammatical. Gate on len(rows) instead.
    """

    def setUp(self):
        self._saved_names = E._NAMES
        E._NAMES = {"spider-man": "seeded so _names() skips resolve.load()"}

    def tearDown(self):
        E._NAMES = self._saved_names

    def test_a_single_row_never_becomes_a_menu(self):
        rows = [(100, "Spider-Man", 0)]           # one row, deduped
        facts = _FakeFactsOffersVariants(rows, total=5)   # five raw records
        resolve = _FakeResolveForChoice(doc_id=0)
        p = E.plan("show me the variants of spider-man", _FakeIndex(), sft,
                   None, None, facts, resolve=resolve)
        self.assertIsNone(p.choices)

    def test_two_or_more_rows_still_offer(self):
        """The fix must not be "never offer" - a genuine multi-row contest
        still has to open the picker."""
        rows = [(100, "Spider-Man", 0), (90, "Spider-Man (Earth-1610)", 1)]
        facts = _FakeFactsOffersVariants(rows, total=2)
        resolve = _FakeResolveForChoice(doc_id=0)
        p = E.plan("show me the variants of spider-man", _FakeIndex(), sft,
                   None, None, facts, resolve=resolve)
        self.assertEqual(p.choices, rows)


class ASettledNameIsNotOfferedAgain(unittest.TestCase):
    """Picking a record is the user answering the question the picker asked.
    Asking the same name again, in the same session, is forgetting - not
    disambiguating.

    Scoped to the NAME, never global: a session that settled `spider-man`
    must still offer for `beast`, or one pick would silence every genuine
    ambiguity that follows it.

    `_FakeResolveForChoice.query_key()` already returns ("spider-man",), so
    that tuple is the settled key throughout."""

    KEY = ("spider-man",)

    def _asks(self, settled):
        """wants_choice on an UNCERTAIN specific ask - confidence 2.0 is
        inside CONFIDENCE_TO_ASK, so this offers unless something stops it."""
        saved = E._NAMES
        E._NAMES = {"seeded": True}
        try:
            return E.wants_choice("who is spider-man", _FakeIndex(),
                                  _PlainFacts(), _ConfidenceResolve(1, 2.0),
                                  doc_id=1, settled=settled)
        finally:
            E._NAMES = saved

    def test_a_settled_name_is_answered_not_offered(self):
        self.assertFalse(self._asks({self.KEY: 7}))

    def test_a_different_settled_name_does_not_suppress_this_one(self):
        self.assertTrue(self._asks({("beast",): 7}))

    def test_no_settled_dict_behaves_exactly_as_before(self):
        self.assertTrue(self._asks(None))

    def test_a_BROAD_ask_still_offers_even_when_settled(self):
        """`show me the variants of X` IS a request to be offered the choice.
        A pick must never suppress it - the user is explicitly asking to see
        the list again, which is the one thing settling must not override."""
        self.assertTrue(E.wants_choice(
            "show me the variants of spider-man", _FakeIndex(),
            _FakeFactsWantsVariants(), _ConfidenceResolve(1, 2.0),
            doc_id=1, settled={self.KEY: 7}))


class ResolvedDocRespectsASettledPick(unittest.TestCase):
    """`resolved_doc()` is the single function `plan()`, `try_facts()` and
    `build_prompt()` all call to decide which record a question is about -
    so this is the one place `settled` must be checked. Checking it only in
    `plan()` (round 1 of this fix) changed `Plan.doc_id` - and so the trace -
    but left `try_facts()`/`build_prompt()` re-resolving on their own,
    unaware of `settled`: the trace said the picked record while the printed
    text still described the one the user did NOT pick. Fixing it here makes
    the trace, the rendered field answer and the model prompt agree by
    construction, since all three now come from this one function.

    `_FakeResolveForChoice.query_key()` returns ("spider-man",) for any
    input, so that tuple is the settled key throughout.
    """

    KEY = ("spider-man",)

    def setUp(self):
        self._saved = E._NAMES
        E._NAMES = {"seeded": True}

    def tearDown(self):
        E._NAMES = self._saved

    def test_a_settled_key_returns_the_settled_doc(self):
        doc = E.resolved_doc("who is spider-man", _FakeIndex(), _PlainFacts(),
                             _FakeResolveForChoice(doc_id=0),
                             settled={self.KEY: 99})
        self.assertEqual(doc, 99)

    def test_an_unsettled_key_returns_the_freshly_resolved_doc(self):
        doc = E.resolved_doc("who is spider-man", _FakeIndex(), _PlainFacts(),
                             _FakeResolveForChoice(doc_id=0),
                             settled={("beast",): 99})
        self.assertEqual(doc, 0)

    def test_no_settled_dict_behaves_exactly_as_before(self):
        doc = E.resolved_doc("who is spider-man", _FakeIndex(), _PlainFacts(),
                             _FakeResolveForChoice(doc_id=0), settled=None)
        self.assertEqual(doc, 0)

    def test_a_follow_up_naming_nobody_is_not_redirected(self):
        """`facts.entity_text` strips a pronoun-only follow-up to nothing, so
        `resolve.query_key` returns an empty tuple - and an empty tuple must
        never collide with a settled name's key. `previous` (5), not the
        settled pick (99), must answer."""

        class _NamesNobody(_FakeResolveForChoice):
            @staticmethod
            def query_key(text):
                return ()

        doc = E.resolved_doc("who is this", _FakeIndex(), _PlainFacts(),
                             _NamesNobody(doc_id=0), previous=5,
                             settled={self.KEY: 99})
        self.assertEqual(doc, 5)


class ASettledNameAnswersFromThePick(unittest.TestCase):
    """End-to-end through plan(): wants_choice() alone only decides whether
    to OFFER a choice - it says nothing about which record the answer text
    comes from. `resolved_doc()` is what plan() calls to set `doc_id`, and it
    is now also what `try_facts()`/`build_prompt()` call to build the text -
    the same settled check in the one shared function, so plan()'s reported
    `doc_id` and the actual rendered answer cannot disagree.

    `_FakeResolveForChoice.query_key()` returns ("spider-man",) for any
    input, so that tuple is the settled key throughout. `_AnsweringFacts`
    routes try_facts through its own `detect_intent`/`answer` branch so these
    tests exercise the settled override without touching
    build_prompt/retrieve/context.py at all.
    """

    KEY = ("spider-man",)

    def setUp(self):
        self._saved = E._NAMES
        E._NAMES = {"seeded": True}

    def tearDown(self):
        E._NAMES = self._saved

    class _AnsweringFacts(_PlainFacts):
        @staticmethod
        def detect_intent(q):
            return "identity"

        @staticmethod
        def answer(question, record):
            return "answer text"

    def test_a_settled_name_answers_from_the_pick(self):
        p = E.plan("who is spider-man", _FakeIndex(), sft, None, None,
                   self._AnsweringFacts(),
                   resolve=_FakeResolveForChoice(doc_id=0),
                   settled={self.KEY: 99})
        self.assertEqual(p.doc_id, 99)

    def test_an_unsettled_name_keeps_the_freshly_resolved_doc(self):
        # _ConfidenceResolve, not _FakeResolveForChoice: an unsettled name
        # falls through wants_choice() as far as confidence(), which the
        # plain fixture does not implement. 38.7 is well past
        # CONFIDENCE_TO_ASK, so this does not open a menu either.
        p = E.plan("who is spider-man", _FakeIndex(), sft, None, None,
                   self._AnsweringFacts(),
                   resolve=_ConfidenceResolve(0, 38.7),
                   settled={("beast",): 99})
        self.assertEqual(p.doc_id, 0)

    def test_a_follow_up_naming_nobody_is_not_redirected(self):
        """`facts.entity_text` strips a pronoun-only follow-up to nothing, so
        `resolve.query_key` returns an empty tuple - and an empty tuple must
        never collide with a settled name's key. Confirmed by test rather
        than trusted: `previous` (5), not the settled pick (99), must answer."""

        class _NamesNobody(_FakeResolveForChoice):
            @staticmethod
            def query_key(text):
                return ()

        p = E.plan("who is this", _FakeIndex(), sft, None, None,
                   self._AnsweringFacts(), resolve=_NamesNobody(doc_id=0),
                   previous=5, settled={self.KEY: 99})
        self.assertEqual(p.doc_id, 5)


class PlanCarriesRows(unittest.TestCase):

    def test_a_plan_defaults_to_no_rows(self):
        self.assertIsNone(E.Plan(text="hi").rows)

    def test_rows_survive_the_constructor(self):
        rows = [("Powers", "Flight")]
        self.assertEqual(E.Plan(text="hi", rows=rows).rows, rows)


class PlanPopulatesRowsFromTheRecord(unittest.TestCase):
    """PlanCarriesRows only proves the `rows` slot round-trips through the
    constructor - it never calls plan() itself, so it cannot catch a bug in
    the branch that actually decides whether rows get attached. This drives
    plan() end-to-end, the same way ASettledNameAnswersFromThePick does,
    with `_ProfileFacts` routing try_facts through its profile-composition
    branch (detect_intent is None, wants_profile is True) so the rows really
    come back from facts.profile_rows().

    Which questions get rows is a different question, and a fake cannot
    answer it: see TheRowsGateMatchesTheProfileGate below, which drives the
    real predicates.
    """

    def setUp(self):
        self._saved = E._NAMES
        E._NAMES = {"seeded": True}

    def tearDown(self):
        E._NAMES = self._saved

    class _ProfileFacts(_PlainFacts):
        DETAIL_RE = facts.DETAIL_RE

        @staticmethod
        def detect_intent(q):
            return None

        @staticmethod
        def wants_profile(q):
            return True

        @staticmethod
        def profile(record, detail=False):
            return "a whole-entity answer"

        @staticmethod
        def profile_rows(record, limit=4):
            return [("Powers", "Flight")]

    def test_a_whole_entity_question_gets_populated_rows(self):
        p = E.plan("who is spider-man", _FakeIndex(), sft, None, None,
                   self._ProfileFacts(), resolve=_ConfidenceResolve(0, 38.7))
        self.assertEqual(p.rows, [("Powers", "Flight")])


class _RecordIndex(_FakeIndex):
    """An index of exactly one real-shaped record, for the real facts module.

    `_FakeIndex.text()` returns "", which every field predicate reads as an
    empty record - so a test that means to exercise facts.profile_rows() has
    to hand plan() something with fields in it.
    """

    RECORD = "\n".join((
        "Namor McKenzie (Earth-616)",
        "Kind: character",
        "Page: Namor McKenzie (Earth-616)",
        "Full name: Namor McKenzie",
        "First appearance: Motion Picture Funnies Weekly Vol 1 1",
        "Created by: Bill Everett",
        "Species / origin: Mutant Human/Atlantean hybrid",
        "Reality: Earth-616",
        "Occupation: King of Atlantis and adventurer; Formerly:; terrorist",
        "Powers: Mutant/Atlantean Physiology; Superhuman Strength",
        "History:",
        "Namor is the Sub-Mariner.",
    ))

    def text(self, doc_id):
        return self.RECORD


class TheRowsGateMatchesTheProfileGate(unittest.TestCase):
    """Which questions get a field block, driven by the REAL predicates.

    FINDING 2026-09-05 (whole-branch review, Critical 1). The test this
    replaces handed plan() a fake whose `wants_profile` returned False and
    then asserted no rows came back - true of a fake returning False against
    ANY plan() whatsoever, and it passed against the very bug it was written
    to guard. `PROFILE_RE` and `INTENTS` are what actually decide this, they
    overlap heavily, and only the real module can show the overlap:

        who is namor            detect_intent None    wants_profile True
        what are his powers     detect_intent powers  wants_profile True

    plan() used to gate rows on `wants_profile` alone, so the second
    question got a profile dump - and since terminal.py's answer path
    prefers rows over text, the targeted sentence try_facts had already
    computed was thrown away. `Plan.rows`' own docstring forbids exactly
    that. The gate is now the same three-part condition try_facts uses.

    The fakes here are the two things the real modules cannot supply without
    a corpus - an index and a resolver - and nothing else.
    """

    def setUp(self):
        self._saved = E._NAMES
        E._NAMES = {"seeded": True}

    def tearDown(self):
        E._NAMES = self._saved

    def plan(self, question):
        # confidence 38.7 is the measured "one record clearly dominates"
        # value used throughout this file, so wants_choice() cannot fire and
        # steal the turn before the rows gate is reached.
        return E.plan(question, _RecordIndex(), sft, None, None, facts,
                      resolve=_ConfidenceResolve(0, 38.7))

    def test_the_two_questions_really_do_overlap(self):
        """Pins the premise. If PROFILE_RE ever stopped matching "what are
        his powers", the pair below would still pass while no longer
        exercising the overlap the gate exists to resolve."""
        self.assertIsNone(facts.detect_intent("who is namor"))
        self.assertTrue(facts.wants_profile("who is namor"))
        self.assertIsNotNone(facts.detect_intent("what are his powers"))
        self.assertTrue(facts.wants_profile("what are his powers"))

    def test_a_whole_entity_question_gets_rows(self):
        p = self.plan("who is namor")
        self.assertEqual([label for label, _ in p.rows],
                         ["Powers", "Created by", "First appearance",
                          "Occupation"])

    def test_a_field_question_gets_no_rows(self):
        p = self.plan("what are his powers")
        self.assertIsNone(p.rows)

    def test_a_field_question_keeps_its_targeted_sentence(self):
        """The rows are not merely absent - the sentence they used to
        displace is what the terminal now prints."""
        p = self.plan("what are his powers")
        self.assertEqual(
            p.text,
            "Namor McKenzie's powers include Mutant/Atlantean Physiology "
            "and Superhuman Strength.")


class RefusalOffersANextStep(unittest.TestCase):

    def test_the_sentence_the_harness_asserts_on_is_unchanged(self):
        text = E.REFUSAL.format(name="zyxthaloraxian")
        self.assertTrue(text.startswith(
            "I don't have zyxthaloraxian in my sources."))

    def test_it_says_what_to_try_instead(self):
        text = E.REFUSAL.format(name="zyxthaloraxian")
        self.assertIn("codename", text)
        self.assertIn("reality", text)


class TryFactsActuallyReturnsTheModuleConstant(unittest.TestCase):
    """Fix round 1, Finding 2: RefusalOffersANextStep above only formats
    E.REFUSAL directly - it never calls try_facts(), so it would still pass
    even if try_facts()'s return were reverted to the old inline string
    while REFUSAL sat unused. `run_cases` cannot close this gap either: its
    `must_include: ["don't have", "sources"]` is satisfied equally by the
    old one-sentence refusal.

    This drives try_facts() itself with a resolver that resolves nothing -
    the same shape as the other resolve fakes in this file
    (_FakeResolveForChoice, _ConfidenceResolve) - and asserts the RETURNED
    TEXT equals E.REFUSAL.format(...), so a revert of the try_facts() return
    line fails this test even with REFUSAL still defined.
    """

    def setUp(self):
        # _names() caches on a module global; seed it truthy so try_facts's
        # resolver branch runs without needing a real resolve.load().
        self._saved = E._NAMES
        E._NAMES = {"seeded": True}

    def tearDown(self):
        E._NAMES = self._saved

    class _UnresolvableResolve:
        """query_key/resolve are stubbed to "this name resolves to
        nothing"; norm/QUERY_NOISE delegate to the real retrieve/resolve.py
        module (imported as `resolve` above) so display_name() computes the
        same name a real session would - not a second copy of that logic."""

        norm = staticmethod(resolve.norm)
        QUERY_NOISE = resolve.QUERY_NOISE

        @staticmethod
        def query_key(text):
            return ("zyxthaloraxian",)

        @staticmethod
        def resolve(names, query, known_words=None):
            return None

    def test_try_facts_returns_the_module_constant(self):
        question = "who is zyxthaloraxian the unmaker"
        fake_resolve = self._UnresolvableResolve()
        answer = E.try_facts(question, _FakeIndex(), sft, None, None,
                             _PlainFacts(), resolve=fake_resolve)
        self.assertEqual(
            answer,
            E.REFUSAL.format(name=E.display_name(question, fake_resolve)))


class _TwinHeadlineIndex(_FakeIndex):
    """Two records under ONE exact headline, for the real facts module.

    This is the shape that reaches try_facts()'s variants branch THROUGH
    plan(): facts.variants() finds both records, dedupes their identical
    headline down to a single row, and plan()'s choices branch gates on
    len(rows) > 1 - so no menu is offered and the turn falls through to
    try_facts(), which renders the variants sentence instead. Measured over
    curated/characters.txt, 680 name groups covering 1,702 records have
    total > 1 with a single distinct headline, so the shape is the corpus's
    own and not a contrivance built for this test.
    """

    docs = [0, 1]
    headlines = ["Beast", "Beast"]
    postings = {}
    RECORDS = (
        "\n".join((
            "Beast",
            "Kind: character",
            "Page: Henry McCoy (Earth-616)",
            "Full name: Henry Philip McCoy",
            "First appearance: X-Men Vol 1 1",
            "Created by: Stan Lee; Jack Kirby",
            "Species / origin: Mutant",
            "Reality: Earth-616",
            "Occupation: Adventurer; Formerly:; teacher",
            "Powers: Superhuman Strength; Superhuman Agility",
            "History:",
            "Henry McCoy is the Beast.",
        )),
        "\n".join((
            "Beast",
            "Kind: character",
            "Page: Beast (Earth-1610)",
            "Created by: Mark Millar",
            "Powers: Superhuman Strength",
            "History:",
            "The Ultimate Beast.",
        )),
    )

    def text(self, doc_id):
        return self.RECORDS[doc_id]


class _TwinResolve(_ConfidenceResolve):
    """_ConfidenceResolve plus the two attributes facts.variants() reads off
    a real resolver, delegated to retrieve/resolve.py rather than restated -
    the dedupe under test is then the corpus's own, not a second copy."""

    norm = staticmethod(resolve.norm)
    REALITY_SUFFIX = resolve.REALITY_SUFFIX


class PlanAndTryFactsAgreeOnWholeEntityQuestions(unittest.TestCase):
    """RESIDUAL 1 (2026-09-05): the two gates, pinned to each other.

    TheRowsGateMatchesTheProfileGate above fixed one disagreement between
    plan()'s rows gate and try_facts()'s profile branch by copying the
    condition across. A copy can drift again, and had already: try_facts
    reaches its profile branch through a sequence of early returns and one
    of them - facts.wants_variants() - sits ABOVE it, while plan()'s copy
    omitted it. Measured against the real facts module, "what are the other
    versions of beast" made try_facts return

        2 records in my sources are called Beast, showing the 1 largest: Beast.

    while plan() attached four profile rows beside it - and terminal.ask()
    prefers rows over text, so the variants answer was computed and thrown
    away. The identical failure shape the earlier fix closed.

    So this does not assert a condition; it asserts the AGREEMENT, over a
    battery, with the real predicates: rows are attached if and only if
    try_facts actually composed a profile. A future edit that re-splits the
    two gates fails here whichever side it edits.
    """

    QUESTIONS = (
        "who is namor",
        "what are the other versions of beast",
        "what are beast's variants",
        "tell me about beast",
        "who is beast in detail",
        "what are his powers",
        "who created beast",
        "when did beast first appear",
        "hi",
    )

    def setUp(self):
        self._saved = E._NAMES
        E._NAMES = {"seeded": True}
        self.index = _TwinHeadlineIndex()
        # 38.7 is the measured "one record clearly dominates" value used
        # throughout this file: wants_choice() must not fire on the ordinary
        # questions and steal the turn before either gate is reached.
        self.resolve = _TwinResolve(0, 38.7)

    def tearDown(self):
        E._NAMES = self._saved

    def _try_facts(self, question):
        return E.try_facts(question, self.index, sft, None, None, facts,
                           resolve=self.resolve)

    def _plan(self, question):
        return E.plan(question, self.index, sft, None, None, facts,
                      resolve=self.resolve)

    def _composed_a_profile(self, question, text) -> bool:
        """Whether try_facts's answer came from its profile branch.

        That branch's whole output is facts.profile() on the resolved
        record, so comparing against it identifies the branch without
        asking try_facts to report which one it took. No other branch can
        collide: a field sentence, the variants list and the refusal are
        all different strings.
        """
        if text is None:
            return False
        composed = facts.profile(
            self.index.text(0),
            detail=bool(facts.DETAIL_RE.search(question)))
        return composed is not None and text == composed

    def test_the_battery_covers_both_answers(self):
        """Pins the premise. If every question composed a profile - or none
        did - the agreement below would pass while proving nothing."""
        composed = [q for q in self.QUESTIONS
                    if self._composed_a_profile(q, self._try_facts(q))]
        self.assertIn("who is namor", composed)
        self.assertIn("tell me about beast", composed)
        self.assertNotIn("what are the other versions of beast", composed)
        self.assertNotIn("what are his powers", composed)

    def test_rows_are_attached_exactly_when_a_profile_was_composed(self):
        for question in self.QUESTIONS:
            with self.subTest(question=question):
                profiled = self._composed_a_profile(
                    question, self._try_facts(question))
                self.assertEqual(self._plan(question).rows is not None,
                                 profiled)

    def test_the_variants_answer_reaches_the_plan_intact(self):
        """The failure shape itself, stated once in full: the sentence
        try_facts computed is what the terminal will print, because rows -
        which ask() prefers over text - are not there to displace it."""
        p = self._plan("what are the other versions of beast")
        self.assertIsNone(p.rows)
        self.assertEqual(p.text, "2 records in my sources are called Beast, "
                                 "showing the 1 largest: Beast.")


if __name__ == "__main__":
    unittest.main()
