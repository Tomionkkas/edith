"""Piped sessions: the product, not a function under it.

`run_cases.py` proved the point - a unit test that calls the engine directly
cannot see that an open menu was eating turns before they reached it.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "model.safetensors"
INDEX = ROOT / "retrieve" / "index.pkl"


def session(*lines, timeout=300):
    """Pipe a conversation in, return (stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "infer" / "terminal.py"), "--trace"],
        input="".join(l + "\n" for l in lines) + "/quit\n",
        capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace", cwd=str(ROOT))
    return proc.stdout, proc.stderr


def traces(stderr):
    """One `trace:` line per turn that REACHED the engine."""
    return [l for l in stderr.split("\n") if l.startswith("trace:")]


@unittest.skipUnless(CKPT.exists() and INDEX.exists(), "model or index missing")
class EveryTurnReachesTheEngine(unittest.TestCase):
    """The swallowed-turn bug: `needs_a_pick` intercepted every line while a
    menu was open, so case 21 sent 2 turns and 1 arrived, and case 23 sent 3
    and 1 arrived. Both were scored off turn 1."""

    def test_case_21_shape_sends_two_turns_and_two_arrive(self):
        _out, err = session("who is spider-man",
                            "show me the variants of spider-man")
        self.assertEqual(len(traces(err)), 2, err)

    def test_case_23_shape_sends_three_turns_and_three_arrive(self):
        _out, err = session("who is spider-man", "what are the variants",
                            "tell me about the ultimate one")
        self.assertEqual(len(traces(err)), 3, err)


@unittest.skipUnless(CKPT.exists() and INDEX.exists(), "model or index missing")
class APhraseNamesARow(unittest.TestCase):
    def test_the_ultimate_one_lands_on_earth_1610(self):
        _out, err = session("who is spider-man", "what are the variants",
                            "tell me about the ultimate one")
        self.assertIn("Earth-1610", traces(err)[-1], err)


@unittest.skipUnless(CKPT.exists() and INDEX.exists(), "model or index missing")
class HistoryAndForget(unittest.TestCase):
    def test_history_lists_what_was_asked(self):
        out, _err = session("who is namor", "who is beast", "/history")
        self.assertIn("who is namor", out)
        self.assertIn("who is beast", out)

    def test_forget_says_what_it_cleared(self):
        out, _err = session("who is spider-man", "2", "/forget")
        self.assertIn("forgot", out.lower())

    def test_history_on_a_fresh_session_says_so(self):
        out, _err = session("/history")
        self.assertIn("nothing yet", out.lower())

    def test_history_after_a_pick_shows_what_was_typed_not_the_sentinel(self):
        """RULING 2: ask() is called internally with the sentinel "who is
        this" from both pick paths. /history must show the line the user
        actually typed - here, the phrase that named a row - never that
        sentinel."""
        out, _err = session("who is spider-man", "what are the variants",
                            "tell me about the ultimate one", "/history")
        self.assertIn("tell me about the ultimate one", out)
        self.assertNotIn("who is this", out)

    def test_forget_leaves_history_intact(self):
        """/forget clears settled picks only - the conversation-so-far stays
        readable."""
        out, _err = session("who is namor", "who is beast", "/forget",
                            "/history")
        self.assertIn("who is namor", out)
        self.assertIn("who is beast", out)
