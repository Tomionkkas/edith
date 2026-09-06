"""Tests for the instrument, not for EDITH.

Nothing here spawns a subprocess or loads a checkpoint. The scorer is the
part that has to mean the same thing every time it runs: the throwaway this
runner replaces reported 23/27 and then 22/27 on a build that had not
changed, because its own rules had.

Run: py -m pytest infer/test_run_cases.py
"""
import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R = _load("run_cases", "infer/run_cases.py")
# The terminal itself, for the one coupling this harness silently depends on:
# what `--trace` pins the theme to. Loading it costs no model and no index.
T = _load("terminal", "infer/terminal.py")


class ParseTrace(unittest.TestCase):
    def test_a_well_formed_line(self):
        self.assertEqual(
            R.parse_trace("trace: doc=41416 page=Miguel O'Hara (Earth-928) offered=0"),
            {"doc": "41416", "page": "Miguel O'Hara (Earth-928)", "offered": 0})

    def test_nothing_resolved(self):
        self.assertEqual(R.parse_trace("trace: doc=none page=none offered=0"),
                         {"doc": "none", "page": "none", "offered": 0})

    def test_a_picker_reports_its_size(self):
        self.assertEqual(
            R.parse_trace("trace: doc=none page=none offered=10")["offered"], 10)

    def test_a_page_containing_an_equals_sign(self):
        # Fandom titles are arbitrary text. Splitting on every `=` would cut
        # the page in half and score a right answer as a wrong record.
        self.assertEqual(
            R.parse_trace("trace: doc=7 page=E=MC2 (Earth-982) offered=0")["page"],
            "E=MC2 (Earth-982)")

    def test_a_line_that_is_not_a_trace(self):
        self.assertIsNone(R.parse_trace("  ▌ edith"))
        self.assertIsNone(R.parse_trace(""))
        self.assertIsNone(R.parse_trace("tracing through 201,815 records"))

    def test_the_page_is_stripped_of_trailing_whitespace(self):
        # emit_trace's headline fallback is not itself stripped. An
        # unstripped trailing space would otherwise produce a false page
        # mismatch: 'Some Headline ' != 'Some Headline'.
        self.assertEqual(
            R.parse_trace("trace: doc=9 page=Some Headline  offered=0")["page"],
            "Some Headline")

    def test_a_windows_line_with_crlf(self):
        # line.strip() is the only reason a \r\n-terminated stderr line
        # parses at all, and Windows is the platform Task 4 runs on.
        self.assertEqual(
            R.parse_trace("trace: doc=12 page=X offered=0\r\n"),
            {"doc": "12", "page": "X", "offered": 0})


TRANSCRIPT = """
  ▌ you
  who is namor

  ▸ searched 201,815 records — 1 caught · 210 ms

  ▌ edith
  Namor McKenzie is the Atlantean king, and Category:Dictators
  is not something he would say.

  ◈ 1 source  ·  ⧗ 4.2s  ·  grounded

  ▌ you
  /quit
"""


class SplitAnswer(unittest.TestCase):
    def test_takes_the_text_between_the_speaker_and_the_footer(self):
        answer = R.split_answer(TRANSCRIPT)
        self.assertIn("Atlantean king", answer)
        self.assertNotIn("searched", answer)
        self.assertNotIn("◈", answer)
        self.assertNotIn("/quit", answer)

    def test_ignores_the_sources_box(self):
        # must_not_include is a claim EDITH made. The box is the record, and
        # the record is allowed to contain anything.
        with_box = TRANSCRIPT.replace(
            "  ▌ edith",
            "  ┌─ Namor McKenzie ─────\n"
            "  │ Occupation  Category:Dictators │\n"
            "  └──────────────────────\n"
            "  ▌ edith", 1)
        self.assertNotIn("│", R.split_answer(with_box))

    def test_the_last_answer_wins_on_a_multi_turn_transcript(self):
        two = TRANSCRIPT + """
  ▌ edith
  His powers are Atlantean physiology.

  ◈ 1 source  ·  ⧗ 3.0s  ·  grounded
"""
        self.assertIn("Atlantean physiology", R.split_answer(two))
        self.assertNotIn("Atlantean king", R.split_answer(two))

    def test_empty_when_nothing_was_answered(self):
        self.assertEqual(R.split_answer("  ▌ you\n  who is spider-man\n"), "")

    def test_a_wrapped_phrase_still_matches(self):
        # render.wrap() is a greedy word wrap, so a two-word fact lands
        # across a line break whenever it straddles the margin. A substring
        # match against the raw lines would score a right answer as a
        # missing string.
        wrapped = """
  ▌ edith
  Cable first appeared in The New
  Mutants #87.

  ◈ 1 source  ·  ⧗ 2.0s  ·  from the record
"""
        self.assertIn("new mutants", R.split_answer(wrapped).lower())

    def test_the_quit_turn_is_not_the_answer(self):
        # Every transcript run_case() captures ends with the /quit turn it
        # appends, and `▌` marks both speakers. Anchoring on the last speaker
        # bar returned "/quit bye." for every case in the suite.
        session = """
  ▌ you
  who is namor

  ▌ edith
  Namor McKenzie is the Atlantean king.

  ◈ 1 source  ·  ⧗ 4.2s  ·  grounded

  ▌ you
  /quit
  bye.
"""
        answer = R.split_answer(session)
        self.assertIn("atlantean king", answer.lower())
        self.assertNotIn("quit", answer.lower())
        self.assertNotIn("bye", answer.lower())

    def test_a_picker_with_no_footer_is_empty(self):
        # A picker turn returns from ask() before printing anything, so it
        # has no footer. score() judges a picker case on `offered`, never on
        # answer text, so split_answer must not synthesize an answer here.
        picker = """
  ▌ you
  who is beast

  ▸ searched 201,815 records — 10 caught · 88 ms

  1. Henry McCoy (Earth-616)
  2. Henry McCoy (Earth-928)
  ...

  ▌ you
  /quit
"""
        self.assertEqual(R.split_answer(picker), "")


class TheScoredWindowExcludesChrome(unittest.TestCase):
    """FINDING 2026-09-05 (whole-branch review, Important 5).

    split_answer() takes everything between the last `▌` and the last `◈`,
    and 4.12 moved a new line INTO that window: render.caption() prints
    between the answer and the footer. A caption is chrome - "it is never
    the answer, and never a word of the record" (render.caption's docstring)
    - but a case's must_not_include cannot tell, so a captioned theme would
    score its quip as something EDITH said.

    Nothing broke, because `--trace` pins the theme to `plain` and `plain`
    defines no caption. Nothing STATED that either, and this is the function
    the whole re-baseline turns on. These pin both halves of the coupling.
    """

    def test_the_trace_theme_is_pinned_and_has_no_caption(self):
        term = T.Terminal(theme_name="deadpool", trace=True)
        self.assertEqual(term.theme.name, "plain")
        self.assertEqual(T.theme.caption_of(term.theme), ())

    def test_a_caption_would_land_inside_the_scored_window(self):
        """Not a wish - a demonstration. If the pin above is ever removed,
        this is the string that starts appearing in must_not_include
        failures, so the test that guards it says what goes wrong."""
        captioned = """
  ▌ edith
  Namor McKenzie is the Atlantean king.

  we are in your terminal

  ◈ 1 source  ·  ⧗ 4.2s  ·  from the record
"""
        self.assertIn("we are in your terminal",
                      R.split_answer(captioned))

    def test_the_caption_ask_would_print_is_the_quip(self):
        """`plain` has no quip either, so the pin holds twice over. Named
        explicitly because ask() gates on BOTH caption_of(t) and t.quip -
        adding one without the other would not print a caption, and a
        future reader should not have to re-derive that."""
        plain = T.theme.get("plain")
        self.assertFalse(plain.quip)
        self.assertFalse(plain.caption)


OK_TRACE = {"doc": "41416", "page": "Namor McKenzie (Earth-616)", "offered": 0}


class Score(unittest.TestCase):
    """Every call states `traced` out loud; none may leave it out.

    R11 made the fourth parameter REQUIRED rather than defaulted. A default
    meaning "rule not applicable" is precisely how a rule gets silently
    skipped, and that is the bug class this phase has now found three times.
    The base fixture sends one turn, so `traced=1` is "every turn reached
    the engine" - the assumption each of these tests was always making, now
    written down.
    """

    def case(self, **over):
        base = {"ask": ["who is namor"], "page": "Namor McKenzie (Earth-616)",
                "must_include": ["atlantean"], "must_not_include": ["category:"],
                "why": "test"}
        base.update(over)
        return base

    def test_everything_right_is_no_reasons(self):
        self.assertEqual(
            R.score(self.case(), "He is an atlantean king.", OK_TRACE, 1), [])

    def test_a_missing_string_fails(self):
        reasons = R.score(self.case(), "He is a king.", OK_TRACE, 1)
        self.assertEqual(reasons, ["missing 'atlantean'"])

    def test_matching_is_case_insensitive(self):
        self.assertEqual(
            R.score(self.case(), "He is ATLANTEAN.", OK_TRACE, 1), [])

    def test_a_forbidden_string_fails(self):
        reasons = R.score(
            self.case(), "atlantean Category:Dictators", OK_TRACE, 1)
        self.assertEqual(reasons, ["forbidden 'category:'"])

    def test_the_wrong_page_fails(self):
        wrong = dict(OK_TRACE, page="Krahllak (Earth-616)")
        reasons = R.score(self.case(), "He is atlantean.", wrong, 1)
        self.assertEqual(
            reasons,
            ["page Krahllak (Earth-616), wanted Namor McKenzie (Earth-616)"])

    def test_no_record_at_all_fails_the_page_check(self):
        # The Domino case: a fluent answer composed over nothing.
        none = {"doc": "none", "page": "none", "offered": 0}
        reasons = R.score(self.case(), "He is atlantean.", none, 1)
        self.assertEqual(reasons,
                         ["page none, wanted Namor McKenzie (Earth-616)"])

    def test_a_picker_where_an_answer_was_wanted_fails_as_a_picker(self):
        # Reported as itself, not as a missing string. The picker firing too
        # often is a real defect and must be visible as one.
        offered = dict(OK_TRACE, doc="none", page="none", offered=10)
        self.assertEqual(R.score(self.case(), "", offered, 1),
                         ["offered a picker of 10; an answer was expected"])

    def test_a_picker_case_passes_when_a_choice_was_offered(self):
        picker = self.case(page=None, expect="picker", must_include=[],
                           must_not_include=[])
        offered = {"doc": "none", "page": "none", "offered": 10}
        self.assertEqual(R.score(picker, "", offered, 1), [])

    def test_a_picker_case_fails_when_it_answered_instead(self):
        # A record resolved (OK_TRACE's doc is 41416) and no choice offered:
        # it answered. The refusal case below is the one that must not be
        # described this way.
        picker = self.case(page=None, expect="picker", must_include=[],
                           must_not_include=[])
        self.assertEqual(R.score(picker, "Peter Parker is...", OK_TRACE, 1),
                         ["answered instead of offering a choice"])

    def test_a_picker_case_that_refused_says_refused(self):
        # F2, case 24 `who is spider-man earth-1610`: 0 caught, no source,
        # nothing resolved. It did not answer, and a report whose entire
        # purpose is not making false statements must not say that it did.
        # The verdict is unchanged - it still fails - only the words move.
        picker = self.case(page=None, expect="picker", must_include=[],
                           must_not_include=[])
        none = {"doc": "none", "page": "none", "offered": 0}
        self.assertEqual(R.score(picker, "", none, 1),
                         ["refused instead of offering a choice"])

    def test_a_picker_case_checks_nothing_else(self):
        # `page` is None and there is no answer text to match, so the strings
        # on a picker case describe the answer it must NOT have given.
        picker = self.case(page=None, expect="picker",
                           must_include=["never checked"],
                           must_not_include=["also never checked"])
        offered = {"doc": "none", "page": "none", "offered": 4}
        self.assertEqual(R.score(picker, "never checked", offered, 1), [])

    def test_a_case_with_a_null_page_still_checks_its_strings(self):
        # Case 7, `who is zarblaxian the unmaker`: page is None because
        # refusing is right, but the refusal still has to say something.
        refusal = self.case(page=None, must_include=["don't have"],
                            must_not_include=["created by"])
        none = {"doc": "none", "page": "none", "offered": 0}
        self.assertEqual(R.score(refusal, "I don't have that.", none, 1), [])
        self.assertEqual(R.score(refusal, "Created by Stan Lee.", none, 1),
                         ["missing 'don't have'", "forbidden 'created by'"])

    def test_a_missing_trace_fails_loudly(self):
        # A crashed or silent subprocess must not read as a pass.
        self.assertEqual(R.score(self.case(), "He is atlantean.", None, 1),
                         ["no trace: the terminal answered nothing"])

    def test_an_empty_answer_fails_even_with_nothing_to_match(self):
        # Cases 25 and 26 carry no must_include, so without this rule the
        # scorer makes no positive assertion and a subprocess that died
        # after emitting its trace scores as a pass. emit_trace runs at
        # terminal.py:267, the footer at :303 - everything in between can
        # produce a good trace and no answer.
        silent = self.case(must_include=[], must_not_include=[])
        self.assertEqual(R.score(silent, "", OK_TRACE, 1),
                         ["no answer: the turn produced no text"])

    def test_whitespace_only_is_also_no_answer(self):
        # "produced no text", not "printed no footer" (F4): this fires when
        # a footer DID print over an empty answer too, and the old wording
        # asserted something the scorer cannot know.
        silent = self.case(must_include=[], must_not_include=[])
        self.assertEqual(R.score(silent, "   \n  ", OK_TRACE, 1),
                         ["no answer: the turn produced no text"])

    def test_an_empty_answer_still_reports_a_wrong_page(self):
        wrong = dict(OK_TRACE, page="Krahllak (Earth-616)")
        silent = self.case(must_include=[], must_not_include=[])
        self.assertEqual(
            R.score(silent, "", wrong, 1),
            ["page Krahllak (Earth-616), wanted Namor McKenzie (Earth-616)",
             "no answer: the turn produced no text"])

    def test_a_picker_case_is_not_failed_for_an_empty_answer(self):
        # A picker prints no footer by design - ask() returns before the
        # answer block - so "" is the CORRECT answer text for these.
        picker = self.case(page=None, expect="picker", must_include=[],
                           must_not_include=[])
        self.assertEqual(R.score(picker, "", {"doc": "none", "page": "none",
                                              "offered": 10}, 1), [])

    def test_expectation_strings_are_also_lowercased(self):
        # Every must_include fixture elsewhere is already lowercase; this
        # exercises s.lower() on the EXPECTATION side, not just the answer.
        shouting = self.case(must_include=["ATLANTEAN"])
        self.assertEqual(R.score(shouting, "he is atlantean", OK_TRACE, 1), [])

    def test_every_reason_is_reported_not_just_the_first(self):
        wrong = dict(OK_TRACE, page="Krahllak (Earth-616)")
        self.assertEqual(
            R.score(self.case(), "Category:Dictators", wrong, 1),
            ["page Krahllak (Earth-616), wanted Namor McKenzie (Earth-616)",
             "missing 'atlantean'", "forbidden 'category:'"])


class ATurnThatNeverRan(unittest.TestCase):
    """R11, which reverses R7. A case whose turns did not all reach the
    engine FAILS.

    The spec's scoring section opens "for each case, run every turn in
    order, then judge the LAST turn". Through a pipe, once a picker opened,
    needs_a_pick (infer/terminal.py, since replaced by take_reference)
    intercepted every later line without calling ask(), so on cases 21 and
    23 the last turn never ran - and report() labels each case by ask[-1],
    so the log printed a PASS beside two strings the product never
    processed. R7 disclosed that in a NOTE and left the verdict alone; the
    NOTE told the truth and the score laundered it. The swallow is fixed
    now, but R11 still stands: it is the rule that a case whose turns did
    not all reach the engine FAILS, whatever the reason a turn is missing.
    """

    def case(self, **over):
        base = {"ask": ["who is namor"], "page": "Namor McKenzie (Earth-616)",
                "must_include": ["atlantean"], "must_not_include": ["category:"],
                "why": "test"}
        base.update(over)
        return base

    def test_case_21s_shape_fails_where_it_used_to_pass(self):
        # Two turns sent, one ran. A picker was offered, which is all the
        # case checks - and before R11 that was a pass.
        picker = self.case(ask=["who is spider-man",
                                "show me the variants of spider-man"],
                           page=None, expect="picker", must_include=[],
                           must_not_include=[])
        offered = {"doc": "none", "page": "none", "offered": 10}
        self.assertEqual(
            R.score(picker, "", offered, 1),
            ["1 turn(s) never reached the engine; scored off turn 1"])

    def test_case_23s_shape_counts_two_swallowed_turns(self):
        # Three sent, one ran. The count is the arithmetic, not a constant.
        picker = self.case(ask=["who is spider-man", "what are the variants",
                                "tell me about the ultimate one"],
                           page=None, expect="picker", must_include=[],
                           must_not_include=[])
        offered = {"doc": "none", "page": "none", "offered": 10}
        self.assertEqual(
            R.score(picker, "", offered, 1),
            ["2 turn(s) never reached the engine; scored off turn 1"])

    def test_a_multi_turn_case_whose_turns_all_ran_is_unaffected(self):
        # Case 18's shape. The rule must not fail the six multi-turn cases
        # that work.
        two = self.case(ask=["who is namor", "what are his powers"])
        self.assertEqual(R.score(two, "He is atlantean.", OK_TRACE, 2), [])

    def test_the_reason_comes_first(self):
        # It is what makes the others untrustworthy: they describe an
        # earlier turn than the one the case was written to measure.
        two = self.case(ask=["who is namor", "what are his powers"])
        wrong = dict(OK_TRACE, page="Krahllak (Earth-616)")
        self.assertEqual(
            R.score(two, "Category:Dictators", wrong, 1),
            ["1 turn(s) never reached the engine; scored off turn 1",
             "page Krahllak (Earth-616), wanted Namor McKenzie (Earth-616)",
             "missing 'atlantean'", "forbidden 'category:'"])

    def test_it_also_reports_ahead_of_a_picker_offered_where_one_was_not(self):
        two = self.case(ask=["who is namor", "what are his powers"])
        offered = dict(OK_TRACE, doc="none", page="none", offered=10)
        self.assertEqual(
            R.score(two, "", offered, 1),
            ["1 turn(s) never reached the engine; scored off turn 1",
             "offered a picker of 10; an answer was expected"])

    def test_it_also_reports_ahead_of_a_missing_trace(self):
        # A timeout returns traced=0 and no trace at all (R9). Both facts
        # are true, both are printed, and the rule takes no exception - an
        # exception is how a rule gets silently skipped.
        self.assertEqual(
            R.score(self.case(), "", None, 0),
            ["1 turn(s) never reached the engine; scored off turn 0",
             "no trace: the terminal answered nothing"])

    def test_the_parameter_is_required(self):
        # A defaulted fourth argument is how this rule would come back.
        with self.assertRaises(TypeError):
            R.score(self.case(), "He is atlantean.", OK_TRACE)


def _result(**over):
    """A synthetic run_case() result, shaped like main()'s results dicts."""
    base = {"case": {"ask": ["who is namor"]},
            "trace": {"page": "Namor McKenzie (Earth-616)", "offered": 0},
            "transcript": "  ▌ edith\n  He is atlantean.\n\n  ◈ 1 source\n",
            "traced": 1, "trouble": None, "reasons": []}
    base.update(over)
    return base


class Report(unittest.TestCase):
    """report() is pure given synthetic result dicts - no subprocess, no
    checkpoint. All 40 tests above this class are scorer-only; this is new
    ground for the NOTE and for R8/R9/F2/F3's fixes.
    """

    def run_report(self, results, show=0, always=False):
        buf = io.StringIO()
        with redirect_stdout(buf):
            passed = R.report(results, show, always)
        return buf.getvalue(), passed

    def test_a_swallowed_turn_notes_beside_its_failure(self):
        # Case 21 and 23's shape. Under R7 this fixture carried reasons=[]
        # and the assertion below was `passed == 1`: the NOTE told the truth
        # while the score contradicted it. R11 fails the case, and the NOTE
        # stays - the verdict says it happened, the NOTE says how many, and
        # a reader wants both. The NOTE is also outside the failure branch
        # still, so nothing can hide it.
        r = _result(case={"ask": ["who is spider-man",
                                  "show me the variants of spider-man"]},
                   traced=1,
                   reasons=["1 turn(s) never reached the engine; "
                            "scored off turn 1"])
        out, passed = self.run_report([r])
        self.assertIn("FAIL", out)
        self.assertIn("never reached the engine", out)
        self.assertIn("NOTE", out)
        self.assertIn("2 turns sent, 1 reached the engine", out)
        self.assertIn("infer/terminal.py:199", out)
        self.assertEqual(passed, 0)

    def test_no_gap_prints_no_note(self):
        r = _result(traced=1, reasons=[])  # 1 turn sent, 1 traced
        out, _ = self.run_report([r])
        self.assertNotIn("NOTE", out)

    def test_a_nonzero_return_code_notes(self):
        r = _result(trouble="subprocess exited 1", reasons=[])
        out, _ = self.run_report([r])
        self.assertIn("NOTE", out)
        self.assertIn("subprocess exited 1", out)

    def test_a_timeout_notes(self):
        r = _result(trace=None, trouble="timed out after 300s",
                   reasons=["no trace: the terminal answered nothing"])
        out, _ = self.run_report([r])
        self.assertIn("timed out after 300s", out)

    def test_the_page_line_prints_for_a_failure_on_the_right_record(self):
        # "right record, wrong words" and "wrong record" are different bugs,
        # and the score alone cannot tell them apart.
        r = _result(case={"ask": ["who is cable"],
                          "page": "Nathan Summers (Earth-616)"},
                   trace={"doc": "7", "page": "Nathan Summers (Earth-616)",
                          "offered": 0},
                   reasons=["missing 'new mutants'"])
        out, _ = self.run_report([r])
        self.assertIn("page  Nathan Summers (Earth-616)  ok", out)

    def test_no_page_line_for_a_case_that_has_no_page_to_check(self):
        # F2, case 24: `page  none  ok` said a record had been checked and
        # found right, on a case whose `page` is None and where nothing was
        # ever resolved. Two false statements in one line.
        r = _result(case={"ask": ["who is spider-man earth-1610"],
                          "page": None, "expect": "picker"},
                   trace={"doc": "none", "page": "none", "offered": 0},
                   reasons=["refused instead of offering a choice"])
        out, _ = self.run_report([r])
        self.assertIn("refused instead of offering a choice", out)
        self.assertNotIn("ok", out)

    def test_the_pass_count_is_right_when_a_note_fired(self):
        # The swallowed-turn fixture is the FAILING one now (R11), which is
        # the only shape a real run can produce: score() cannot return []
        # for a case whose turns did not all run. A NOTE still genuinely
        # fires, so the test's name stays true.
        passing = _result(case={"ask": ["c"]}, traced=1, reasons=[])
        failing = _result(case={"ask": ["a", "b"]}, traced=1,
                         reasons=["1 turn(s) never reached the engine; "
                                  "scored off turn 1"])
        out, passed = self.run_report([passing, failing])
        self.assertIn("NOTE", out)
        self.assertEqual(passed, 1)
        self.assertIn("1/2", out)

    def test_case_always_shows_the_transcript_on_a_pass(self):
        # R8: `--case N` prints the transcript whether it passes or fails -
        # `--case 8` is `who is beast`, which offers a picker and may well
        # pass, and a verification step reading the transcript would
        # otherwise have nothing to read.
        r = _result(reasons=[], transcript="THE TRANSCRIPT MARKER")
        out, _ = self.run_report([r], show=1, always=True)
        self.assertIn("THE TRANSCRIPT MARKER", out)

    def test_show_without_case_still_shows_only_failures(self):
        # `--show N` (always=False) keeps its old, narrower meaning.
        r = _result(reasons=[], transcript="SHOULD NOT APPEAR")
        out, _ = self.run_report([r], show=1, always=False)
        self.assertNotIn("SHOULD NOT APPEAR", out)


if __name__ == "__main__":
    unittest.main()
