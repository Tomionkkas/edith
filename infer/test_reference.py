"""Tests for turning a typed phrase into a row of the last menu.

Index-free on purpose: the module under test sees a list of tuples, so these
run in milliseconds and cover the edge cases exhaustively.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "reference", ROOT / "infer" / "reference.py")
R = importlib.util.module_from_spec(spec)
sys.modules["reference"] = R
spec.loader.exec_module(R)

# The live `who is spider-man` menu, measured 2026-09-04, plus one
# Earth-928 (Spider-Man 2099) row APPENDED at the end so
# test_2099_is_a_question_not_a_row_and_not_a_reality can prove the
# digits-only guard actually matters - see that test's docstring. Appended,
# not inserted, so no other test's expected index moves.
MENU = [
    (120000, "Spider-Man", 1),
    (75775, "Spider-Man (Earth-1610)", 2),
    (39730, "Spider-Man (Earth-1048)", 3),
    (30000, "Spider-Man (Earth-199999)", 4),
    (28000, "Spider-Man (Earth-12041)", 5),
    (27884, "Spider-Man (Earth-1610B)", 6),
    (20000, "Spider-Man (Earth-928)", 7),
]


class ANumber(unittest.TestCase):
    def test_a_bare_number_in_range_picks_that_row(self):
        self.assertEqual(R.pick_row("2", MENU), 1)

    def test_a_bare_number_out_of_range_is_not_a_pick(self):
        self.assertIsNone(R.pick_row("99", MENU))

    def test_2099_is_a_question_not_a_row_and_not_a_reality(self):
        """The rule select() carried: `2099` is a question about Spider-Man
        2099. A line that is nothing but digits never reaches the reality
        matcher, so the nickname 2099 -> Earth-928 cannot hijack it.

        MENU carries an Earth-928 row (index 6) precisely so this test has
        teeth: without the digits-only guard in matcher 1, "2099" would
        reach matcher 3, match the "2099" -> "928" nickname, find exactly
        that one row, and return 6 instead of None.
        """
        self.assertIsNone(R.pick_row("2099", MENU))

    def test_what_is_1_is_a_question(self):
        self.assertIsNone(R.pick_row("what is 1", MENU))


class AnOrdinal(unittest.TestCase):
    def test_the_second_one(self):
        self.assertEqual(R.pick_row("the second one", MENU), 1)

    def test_a_digit_ordinal(self):
        self.assertEqual(R.pick_row("2nd", MENU), 1)

    def test_an_ordinal_past_the_end_is_not_a_pick(self):
        self.assertIsNone(R.pick_row("the ninth one", MENU))


class AReality(unittest.TestCase):
    def test_the_ultimate_one(self):
        """The case that motivated the whole picker."""
        self.assertEqual(R.pick_row("tell me about the ultimate one", MENU), 1)

    def test_a_literal_reality(self):
        self.assertEqual(R.pick_row("the earth-1610b one", MENU), 5)

    def test_the_mcu_one(self):
        self.assertEqual(R.pick_row("the mcu one", MENU), 3)

    def test_the_main_one_is_the_unmarked_row(self):
        """The corpus writes Earth-616 without a suffix and marks every other
        reality - CLAUDE.md, 'the unmarked default'."""
        self.assertEqual(R.pick_row("the main one", MENU), 0)

    def test_a_nickname_matching_two_rows_declines(self):
        rows = [(10, "Spider-Man (Earth-1610)", 1),
                (9, "Spider-Man (Earth-1610)", 2)]
        self.assertIsNone(R.pick_row("the ultimate one", rows))

    def test_an_unknown_nickname_declines(self):
        self.assertIsNone(R.pick_row("the red one", MENU))


class NotAReference(unittest.TestCase):
    """These must fall through so the terminal treats them as questions -
    this is the swallowed-turn fix."""

    def test_a_broad_ask(self):
        self.assertIsNone(
            R.pick_row("show me the variants of spider-man", MENU))

    def test_another_broad_ask(self):
        self.assertIsNone(R.pick_row("what are the variants", MENU))

    def test_an_ordinary_question(self):
        self.assertIsNone(R.pick_row("who is doctor doom", MENU))

    def test_no_rows_declines(self):
        self.assertIsNone(R.pick_row("the ultimate one", []))

    def test_an_empty_line_declines(self):
        self.assertIsNone(R.pick_row("", MENU))


class AQuestionThatHappensToContainANickname(unittest.TestCase):
    """FINDING 2026-09-04: with a menu open, matchers 2 and 3 used to fire
    on ANY line carrying an ordinal or a reality nickname, no matter what
    else was in it. "who is the ultimate hulk" matched REALITY_NICKNAMES
    ["ultimate"] and silently answered from the Earth-1610 Spider-Man row -
    worse than the swallow this module exists to fix, because a wrong
    answer reads as a right one and a swallow does not. The fix: a leftover
    word that is not scaffolding makes the line a question, hard yes/no,
    for both matchers - not a heuristic that scores which word "wins".
    """

    def test_the_ultimate_one_still_picks_when_every_leftover_is_scaffolding(self):
        self.assertEqual(R.pick_row("tell me about the ultimate one", MENU), 1)

    def test_the_second_one_still_picks(self):
        self.assertEqual(R.pick_row("the second one", MENU), 1)

    def test_a_literal_reality_still_picks(self):
        self.assertEqual(R.pick_row("the earth-1610b one", MENU), 5)

    def test_a_nickname_beside_a_content_word_declines(self):
        """"hulk" carries identity of its own - the line is a question
        about Hulk, not a reference to the Ultimate row."""
        self.assertIsNone(R.pick_row("who is the ultimate hulk", MENU))

    def test_main_beside_a_content_word_declines(self):
        self.assertIsNone(R.pick_row("what is the main difference", MENU))

    def test_an_ordinal_beside_a_content_word_declines(self):
        self.assertIsNone(R.pick_row("give me the second book", MENU))

    def test_a_bare_number_is_unguarded(self):
        """Matcher 1 has no leftover to examine - the guard does not apply
        to it at all."""
        self.assertEqual(R.pick_row("2", MENU), 1)
