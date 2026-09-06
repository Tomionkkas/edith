"""Tests for motion, and mostly for its off switch.

The off switch is the whole safety story of this module. run_cases.py pipes
questions into the terminal and reads stdout; test_terminal_session.py spawns
it as a subprocess. Both must get zero delay and identical bytes, without
knowing this module exists.

Run: py -m pytest infer/test_motion.py
"""
import importlib.util
import io
import os
import sys
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


M = _load("motion", "infer/motion.py")


class NotATty:
    """A stream that is not a terminal - what a pipe looks like."""

    def isatty(self):
        return False


class Tty(io.StringIO):
    def isatty(self):
        return True


@contextmanager
def _only(**overrides):
    """Run enable() with PYTEST_CURRENT_TEST out of the way.

    Pytest sets that variable for every test it runs, and enable() turns
    motion off whenever it is present. Without removing it, a test named
    for the TTY check passes even if the TTY check is deleted.
    """
    keys = ("PYTEST_CURRENT_TEST", "NO_COLOR") + tuple(overrides)
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.pop("PYTEST_CURRENT_TEST", None)
    os.environ.pop("NO_COLOR", None)
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TheOffSwitch(unittest.TestCase):

    def tearDown(self):
        M.ENABLED = False

    def test_a_terminal_with_nothing_in_the_way_gets_motion(self):
        # The control. Without it every other test in this class would pass
        # against an enable() that simply returned False forever.
        with _only():
            self.assertTrue(M.enable(Tty(), argv=[]))

    def test_a_pipe_gets_no_motion(self):
        with _only():
            self.assertFalse(M.enable(NotATty(), argv=[]))

    def test_no_color_turns_motion_off(self):
        with _only(NO_COLOR="1"):
            self.assertFalse(M.enable(Tty(), argv=[]))

    def test_the_no_anim_flag_turns_motion_off(self):
        with _only():
            self.assertFalse(M.enable(Tty(), argv=["--no-anim"]))

    def test_pytest_turns_motion_off_even_on_a_tty(self):
        # PYTEST_CURRENT_TEST is left in place deliberately: this is the one
        # test whose subject is the pytest check itself.
        os.environ["PYTEST_CURRENT_TEST"] = "test_motion.py::x"
        self.assertFalse(M.enable(Tty(), argv=[]))


class StillOutput(unittest.TestCase):
    """With motion off, every call is an ordinary print and costs nothing."""

    def setUp(self):
        M.ENABLED = False

    def test_wipe_prints_every_line_in_order(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            M.wipe(["one", "two", "three"])
        self.assertEqual(buf.getvalue(), "one\ntwo\nthree\n")

    def test_wipe_does_not_sleep_when_disabled(self):
        start = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            M.wipe([str(i) for i in range(40)], delay=0.5)
        self.assertLess(time.perf_counter() - start, 0.5)

    def test_spin_returns_the_work_result_and_prints_the_label(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            got = M.spin("model", lambda: 41 + 1)
        self.assertEqual(got, 42)
        self.assertIn("model", buf.getvalue())

    def test_spin_lets_an_exception_out(self):
        def boom():
            raise ValueError("load failed")
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                M.spin("model", boom)

    def test_count_prints_the_final_value_with_separators(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            M.count(202101, prefix="index  ", suffix=" records")
        self.assertIn("202,101", buf.getvalue())
        self.assertIn("index", buf.getvalue())

    def test_count_emits_no_carriage_return_when_disabled(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            M.count(202101, prefix="index  ", suffix=" records")
        self.assertNotIn("\r", buf.getvalue())


class TheSpinnerThread(unittest.TestCase):
    """The ENABLED=True path, which nothing else reaches.

    enable() cannot return True under pytest, so these set the flag directly.
    What matters is not what is drawn - it is that the thread always dies.
    """

    def setUp(self):
        M.ENABLED = True
        self.before = set(threading.enumerate())

    def tearDown(self):
        M.ENABLED = False

    def _leaked(self):
        return [t for t in threading.enumerate()
                if t not in self.before and t.is_alive()]

    def test_the_spinner_stops_when_work_returns(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(M.spin("model", lambda: 42), 42)
        self.assertEqual(self._leaked(), [])

    def test_the_spinner_stops_when_work_raises(self):
        def boom():
            raise ValueError("load failed")
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                M.spin("model", boom)
        self.assertEqual(self._leaked(), [])

    def test_a_failed_stage_is_not_marked_with_a_check(self):
        def boom():
            raise ValueError("load failed")
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(ValueError):
                M.spin("model", boom)
        drawn = buf.getvalue()
        self.assertNotIn("✓", drawn)
        self.assertIn("✗", drawn)
