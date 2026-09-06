"""Tests for the shell around the model.

Nothing here loads a checkpoint. What is worth pinning is the part that runs
before and between answers: which theme a name resolves to, that a command
cannot crash the loop, and that a missing checkpoint says what to do instead
of raising a traceback at someone who just cloned the repo.

Run: py -m pytest infer/test_terminal.py
"""
import importlib.util
import io
import os
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


theme = _load("theme", "infer/theme.py")
T = _load("terminal", "infer/terminal.py")
resolve = _load("resolve", "retrieve/resolve.py")
engine = _load("engine", "infer/engine.py")
motion = _load("motion", "infer/motion.py")
theme.COLOUR_ENABLED = False


class Themes(unittest.TestCase):
    def test_every_theme_has_a_blurb_for_the_picker(self):
        self.assertEqual(set(theme.names()), set(theme.BLURBS))

    def test_the_names_people_type(self):
        for typed, want in (("spidey", "spider-man"), ("SPIDER-MAN", "spider-man"),
                            ("Wolverine", "wolverine"), ("logan", "wolverine"),
                            ("dp", "deadpool"), ("banner", "hulk"),
                            ("none", "plain"), ("spider man", "spider-man")):
            self.assertEqual(theme.get(typed).name, want, typed)

    def test_an_unknown_name_is_none_not_a_crash(self):
        self.assertIsNone(theme.get("galactus"))
        self.assertIsNone(theme.get(""))

    def test_the_default_exists(self):
        self.assertIn(theme.DEFAULT, theme.THEMES)

    def test_plain_paints_nothing(self):
        """The plain theme must cost no special case at the call sites."""
        plain = theme.get("plain")
        self.assertEqual(theme.paint("text", plain.accent), "text")

    def test_a_palette_is_a_hex_colour_or_empty(self):
        for name in theme.names():
            t = theme.THEMES[name]
            for field in (t.accent, t.second, t.text, t.dim, t.faint):
                self.assertTrue(field == "" or
                                (field.startswith("#") and len(field) == 7), field)


class Config(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self._real, T.CONFIG = T.CONFIG, self.tmp

    def tearDown(self):
        T.CONFIG = self._real

    def test_round_trip(self):
        T.write_config({"theme": "hulk", "sources": False})
        self.assertEqual(T.read_config()["theme"], "hulk")

    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(T.read_config(), {})

    def test_a_corrupt_file_is_not_an_error(self):
        """A half-written config must not stop the terminal from opening."""
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self.tmp.write_text("{not json", encoding="utf-8")
        self.assertEqual(T.read_config(), {})


class Commands(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self._real, T.CONFIG = T.CONFIG, self.tmp
        self.term = T.Terminal("plain")

    def tearDown(self):
        T.CONFIG = self._real

    def run_cmd(self, line):
        out = io.StringIO()
        with redirect_stdout(out):
            keep = self.term.command(line)
        return keep, out.getvalue()

    def test_quit_leaves_the_loop(self):
        for line in ("/quit", "/exit", "/q"):
            self.assertFalse(self.run_cmd(line)[0], line)

    def test_help_lists_every_command(self):
        _, out = self.run_cmd("/help")
        for cmd, _hint in T.HELP:
            self.assertIn(cmd.split()[0], out)

    def test_theme_with_no_argument_lists_them(self):
        _, out = self.run_cmd("/theme")
        for name in theme.names():
            self.assertIn(name, out)

    def test_theme_switches_and_remembers(self):
        keep, _ = self.run_cmd("/theme hulk")
        self.assertTrue(keep)
        self.assertEqual(self.term.theme.name, "hulk")
        self.assertEqual(T.read_config()["theme"], "hulk")

    def test_an_unknown_theme_keeps_the_current_one(self):
        self.run_cmd("/theme galactus")
        self.assertEqual(self.term.theme.name, "plain")

    def test_sources_can_be_turned_off_and_back_on(self):
        self.run_cmd("/sources off")
        self.assertFalse(self.term.show_sources)
        self.run_cmd("/sources on")
        self.assertTrue(self.term.show_sources)

    def test_an_unknown_command_does_not_end_the_session(self):
        keep, out = self.run_cmd("/wakanda")
        self.assertTrue(keep)
        self.assertIn("/help", out)

    def test_a_command_with_odd_spacing_still_works(self):
        self.run_cmd("/theme    hulk   ")
        self.assertEqual(self.term.theme.name, "hulk")

    def test_a_saved_theme_is_picked_up_next_time(self):
        self.run_cmd("/theme venom")
        self.assertEqual(T.Terminal().theme.name, "venom")


class HeaderAfterATheme(unittest.TestCase):
    """Switching skins printed a shorter header than booting did.

    No record count, no rule under it - so it read as half-loaded, as though
    something was still on its way. Both paths print the same block now.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self._real, T.CONFIG = T.CONFIG, self.tmp
        self.term = T.Terminal("plain")

    def tearDown(self):
        T.CONFIG = self._real

    def shown(self, fn):
        out = io.StringIO()
        with redirect_stdout(out):
            fn()
        return out.getvalue()

    def test_a_theme_change_reprints_the_whole_header(self):
        after = self.shown(lambda: self.term.apply_theme(theme.get("hulk")))
        self.assertIn("even dead i'm the hero", after)
        self.assertIn("hulk", after)
        self.assertIn(theme.get("hulk").rule, after)

    def test_the_wordmark_comes_back_too(self):
        after = self.shown(lambda: self.term.apply_theme(theme.get("venom")))
        self.assertIn(T.render.BANNER[0], after)

    def test_the_two_headers_carry_the_same_parts(self):
        """Whatever boot shows, a theme change shows."""
        t = theme.get("wolverine")
        after = self.shown(lambda: self.term.apply_theme(t))
        boot = self.shown(lambda: self.term.header())
        chip = theme.strip(T.render.chip(t.name, t))
        for part in ("even dead i'm the hero", chip):
            self.assertIn(part, after, part)
            self.assertIn(part, boot, part)

    def test_the_record_count_is_left_out_before_the_index_loads(self):
        """The header is printed by /theme too, and that can precede a load."""
        self.assertNotIn("records indexed", self.shown(self.term.header))


class Boot(unittest.TestCase):
    def test_a_missing_checkpoint_explains_itself(self):
        """Someone who has just cloned this needs an instruction, not a
        traceback from torch.load. It used to say 'copy it from a machine
        that has it'; now it offers to fetch, and a non-TTY gets the retry
        instruction instead of a prompt it cannot answer."""
        term = T.Terminal("plain", ckpt=ROOT / "checkpoints" / "nope.pt")
        out = io.StringIO()
        with redirect_stdout(out):
            ok = term.boot()
        self.assertFalse(ok)
        self.assertIn("nope.pt", out.getvalue())
        self.assertIn("run it again", out.getvalue())

    def test_main_returns_nonzero_when_it_cannot_start(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = T.main(["--ckpt", str(ROOT / "checkpoints" / "nope.pt")])
        self.assertEqual(code, 1)


class FirstRun(unittest.TestCase):
    """The offer that replaced 'copy it from a machine that has it' - a
    sentence that stopped being true on publication day."""

    def setUp(self):
        self.term = T.Terminal("plain")
        self._missing = T.bootstrap.missing
        self._fetch = T.bootstrap.fetch_all
        self._download_mb = T.bootstrap.download_mb
        self._isatty = sys.stdin.isatty

    def tearDown(self):
        T.bootstrap.missing = self._missing
        T.bootstrap.fetch_all = self._fetch
        T.bootstrap.download_mb = self._download_mb
        sys.stdin.isatty = self._isatty

    def run_offer(self, answer="y", tty=True, gone=None, mb=810):
        """Returns (result, printed).

        `missing()` is stateful, not a fixed stub: offer_bootstrap() now
        re-checks it after fetch_all() runs (FIX ROUND 1), so a fetch that
        "succeeds" has to actually clear what was missing for the tests
        that expect a successful round trip to return True.
        """
        gone = [ROOT / "checkpoints" / "model.safetensors"] if gone is None else gone
        state = {"gone": gone}
        T.bootstrap.missing = lambda *_a, **_k: state["gone"]
        T.bootstrap.download_mb = lambda *_a, **_k: mb
        self.fetched = []

        def fake_fetch(log=None):
            self.fetched.append(1)
            state["gone"] = []

        T.bootstrap.fetch_all = fake_fetch
        sys.stdin.isatty = lambda: tty
        out = io.StringIO()
        with redirect_stdout(out):
            with unittest.mock.patch("builtins.input", lambda *_: answer):
                result = self.term.offer_bootstrap()
        return result, out.getvalue()

    def test_it_quotes_the_size_before_it_asks(self):
        ok, shown = self.run_offer(answer="y", mb=810)
        self.assertTrue(ok)
        self.assertEqual(self.fetched, [1])
        self.assertIn("810", shown)

    def test_when_only_the_index_is_missing_nothing_downloads(self):
        """FIX ROUND 1: quoting a fixed MB total regardless of what is
        actually missing asked someone who only lacks the index (which is
        never downloaded) to approve a download that would never happen."""
        ok, shown = self.run_offer(
            answer="n", gone=[ROOT / "retrieve" / "index.pkl"], mb=0)
        self.assertFalse(ok)
        self.assertIn("30 seconds", shown)
        self.assertNotIn("MB", shown)

    def test_ctrl_c_on_the_prompt_is_a_decline_not_a_crash(self):
        """FIX ROUND 1: terminal.py already catches KeyboardInterrupt at
        three other prompts (:447, :713, :756) and prints something
        friendly - this was the one place that missed the house pattern,
        and the worst place to miss it: the very first prompt a new user
        ever sees."""
        def interrupted(*_a, **_k):
            raise KeyboardInterrupt

        T.bootstrap.missing = lambda *_a, **_k: [ROOT / "checkpoints" / "model.safetensors"]
        T.bootstrap.download_mb = lambda *_a, **_k: 810
        self.fetched = []
        T.bootstrap.fetch_all = lambda log=None: self.fetched.append(1)
        sys.stdin.isatty = lambda: True
        out = io.StringIO()
        with redirect_stdout(out):
            with unittest.mock.patch("builtins.input", interrupted):
                ok = self.term.offer_bootstrap()
        self.assertFalse(ok)
        self.assertEqual(self.fetched, [])
        self.assertIn("run it again", out.getvalue())

    def test_a_failed_fetch_explains_itself_and_stops(self):
        """FIX ROUND 1: fetch_all() used to be called unwrapped, so a
        network drop, a 404, or a disk-full partway through an 810 MB fetch
        propagated as a raw exception through boot() and main()."""
        def boom():
            raise ConnectionError("connection reset by peer")

        T.bootstrap.missing = lambda *_a, **_k: [ROOT / "checkpoints" / "model.safetensors"]
        T.bootstrap.download_mb = lambda *_a, **_k: 810
        T.bootstrap.fetch_all = boom
        sys.stdin.isatty = lambda: True
        out = io.StringIO()
        with redirect_stdout(out):
            with unittest.mock.patch("builtins.input", lambda *_: "y"):
                ok = self.term.offer_bootstrap()
        self.assertFalse(ok)
        self.assertIn("connection reset by peer", out.getvalue())
        self.assertIn("running EDITH again resumes", out.getvalue())

    def test_a_fetch_that_leaves_something_missing_is_reported_not_trusted(self):
        """FIX ROUND 1 (reviewer finding): fetch_all() can return normally
        without actually finishing - offer_bootstrap() used to return True
        on trust rather than checking. Re-checking bootstrap.missing() after
        the fetch turns a confusing downstream model-load error into a
        clear statement of what did not arrive."""
        still_there = [ROOT / "checkpoints" / "model.safetensors"]
        # Every call to missing() - before AND after the fetch - reports the
        # same file still absent, simulating a fetch that returns without
        # raising but never actually delivers the weights.
        T.bootstrap.missing = lambda *_a, **_k: still_there
        T.bootstrap.download_mb = lambda *_a, **_k: 810
        T.bootstrap.fetch_all = lambda log=None: None  # "succeeds" but nothing changed
        sys.stdin.isatty = lambda: True
        out = io.StringIO()
        with redirect_stdout(out):
            with unittest.mock.patch("builtins.input", lambda *_: "y"):
                ok = self.term.offer_bootstrap()
        self.assertFalse(ok)
        self.assertIn("model.safetensors", out.getvalue())

    def test_declining_is_not_an_error(self):
        """Saying no prints what to run and stops. It does not raise."""
        ok, shown = self.run_offer(answer="n")
        self.assertFalse(ok)
        self.assertEqual(self.fetched, [])
        self.assertIn("run it again", shown)

    def test_a_non_tty_is_never_prompted(self):
        """run_cases.py drives this as a subprocess. input() on a closed stdin
        hangs the harness or raises EOFError inside a measurement."""
        def explode(*_):
            raise AssertionError("prompted on a non-TTY")

        T.bootstrap.missing = lambda *_a, **_k: [ROOT / "checkpoints" / "x"]
        sys.stdin.isatty = lambda: False
        out = io.StringIO()
        with redirect_stdout(out):
            with unittest.mock.patch("builtins.input", explode):
                ok = self.term.offer_bootstrap()
        self.assertFalse(ok)
        self.assertIn("run it again", out.getvalue())

    def test_nothing_missing_asks_nothing(self):
        ok, shown = self.run_offer(gone=[])
        self.assertTrue(ok)
        self.assertEqual(shown, "")

    def test_it_reports_the_checkpoint_the_caller_named(self):
        """--ckpt names a file; the message must be about THAT file."""
        term = T.Terminal("plain", ckpt=ROOT / "checkpoints" / "nope.pt")
        seen = []
        T.bootstrap.missing = lambda w=None: seen.append(w) or [Path(w)]
        sys.stdin.isatty = lambda: False
        with redirect_stdout(io.StringIO()):
            term.offer_bootstrap()
        self.assertEqual(seen, [term.ckpt])


class PublishedFormat(unittest.TestCase):
    def test_the_terminal_loads_what_we_publish(self):
        """One artefact: the file uploaded and the file loaded are the same."""
        self.assertEqual(T.CKPT.name, "model.safetensors")


class ALineThatNamesNoRowIsAQuestion(unittest.TestCase):
    """`needs_a_pick` used to intercept every line while a menu was open, on
    the theory that a console's picker blocks and so nothing could be typed
    past an open menu - true on a console, false through a pipe, where the
    list is only printed. That swallowed the exact turn that motivated the
    picker: `tell me about the ultimate one` never reached the engine.

    `take_reference` replaces it: a line either names a row from the last
    menu, or it does not, and if it does not the caller answers it as an
    ordinary question instead of refusing it.
    """

    def setUp(self):
        self.term = T.Terminal("plain")
        self.term.last_offer = [(100, "Spider-Man", 1),
                                (90, "Spider-Man (Earth-1610)", 2)]

    def test_a_line_naming_no_row_is_not_a_reference(self):
        self.assertIsNone(self.term.take_reference("who is wolverine"))

    def test_the_offer_survives_a_line_that_names_no_row(self):
        self.term.take_reference("who is wolverine")
        self.assertIsNotNone(self.term.last_offer)

    def test_a_bare_number_still_picks(self):
        self.assertEqual(self.term.take_reference("2"), 2)

    def test_a_command_is_untouched_by_the_reference_path(self):
        """take_reference() only looks at the offer; commands are handled
        earlier in run() and never reach it."""
        self.assertIsNone(self.term.take_reference("/quit"))
        self.assertIsNotNone(self.term.last_offer)

    def test_nothing_is_named_when_no_offer_is_open(self):
        self.term.last_offer = None
        self.assertIsNone(self.term.take_reference("who is wolverine"))


class BareNumberSelects(unittest.TestCase):
    """A printed list is only usable if a number picks from it.

    This is also the ONLY way the answer-case harness can drive the picker:
    it runs the terminal through a pipe, where render.interactive() is false
    and no keystroke can be read. `select()`/`needs_a_pick()` are gone;
    `take_reference()` carries this now, alongside phrases like "the
    ultimate one" that the old pair could never read at all.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self._real, T.CONFIG = T.CONFIG, self.tmp

    def tearDown(self):
        T.CONFIG = self._real

    def terminal(self):
        return T.Terminal("plain")

    def test_a_bare_number_selects_when_a_choice_is_pending(self):
        t = self.terminal()
        t.last_offer = [(100, "Spider-Man", 1), (90, "Spider-Man (Earth-1610)", 2)]
        self.assertEqual(t.take_reference("2"), 2)

    def test_a_number_out_of_range_does_not_select(self):
        t = self.terminal()
        t.last_offer = [(100, "Spider-Man", 1)]
        self.assertIsNone(t.take_reference("7"))
        self.assertIsNotNone(t.last_offer)

    def test_a_number_with_no_choice_pending_is_an_ordinary_question(self):
        t = self.terminal()
        t.last_offer = None
        self.assertIsNone(t.take_reference("2"))

    def test_a_year_is_not_a_selection_against_a_short_menu(self):
        """"2099" is a question about Spider-Man 2099, not row 2099 - and
        with only one row on offer it cannot be the "2099" reality nickname
        either, so it must still fall through to an ordinary question."""
        t = self.terminal()
        t.last_offer = [(100, "Spider-Man", 1)]
        self.assertIsNone(t.take_reference("2099"))

    def test_text_containing_a_number_is_not_a_selection(self):
        t = self.terminal()
        t.last_offer = [(100, "Spider-Man", 1)]
        self.assertIsNone(t.take_reference("what is 1"))

    def test_escaping_the_interactive_picker_leaves_the_rows_nameable(self):
        """render.pick() returns -1 on escape. offer() must leave last_doc
        alone and select nothing outright - but unlike the old behaviour
        (FINDING 2026-09-02, which cleared the choice on escape), the rows
        stay in last_offer: the user SAW them, and naming one next turn -
        "the ultimate one" - is a reasonable thing to type. This is the only
        way the reference feature is reachable on a console, where
        render.pick() blocks and nothing can be typed at an open menu.

        Drives the INTERACTIVE branch (render.interactive() patched true,
        matching this file's T.CONFIG swap-with-restore idiom) and feeds
        render.pick a real escape key through its own `keys` parameter -
        exactly the mechanism render.py's docstring says makes the widget
        testable without a console - rather than stubbing pick() away.
        """
        t = self.terminal()
        t.last_doc = "unchanged"
        real_interactive = T.render.interactive
        real_pick = T.render.pick
        T.render.interactive = lambda: True
        T.render.pick = (lambda rows, current, draw, keys=None:
                         real_pick(rows, current, draw, keys=[T.render.ESC]))
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                t.offer([(100, "Spider-Man", 1),
                         (90, "Spider-Man (Earth-1610)", 2)])
        finally:
            T.render.interactive = real_interactive
            T.render.pick = real_pick
        self.assertEqual(t.last_doc, "unchanged")
        self.assertEqual(t.last_offer,
                         [(100, "Spider-Man", 1), (90, "Spider-Man (Earth-1610)", 2)])
        self.assertEqual(t.take_reference("2"), 2)


class PickerRowsAreWipedAndChipped(unittest.TestCase):
    """Task 9: the picker's rows get the same wipe/chip treatment the rest
    of the terminal has - render.chip (Task 3), motion.wipe (Task 2, now
    actually enabled - Task 8)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self._real, T.CONFIG = T.CONFIG, self.tmp

    def tearDown(self):
        T.CONFIG = self._real

    def terminal(self):
        return T.Terminal("plain")

    def test_the_piped_listing_is_printed_through_motion_wipe(self):
        """render.interactive() is False under pytest (no tty), so offer()
        takes the numbered-list branch. Proven with a spy on motion.wipe,
        not by reading stdout - a bare print loop would also leave stdout
        looking right, and the point is that the ROWS are handed to wipe as
        a list, not printed one at a time."""
        t = self.terminal()
        calls = []
        real_wipe = T.motion.wipe

        def spy(lines, delay=0.055):
            calls.append(list(lines))
            for line in lines:
                print(line)
        T.motion.wipe = spy
        try:
            with redirect_stdout(io.StringIO()):
                t.offer([(100, "Spider-Man", 1),
                         (90, "Spider-Man (Earth-1610)", 2)])
        finally:
            T.motion.wipe = real_wipe
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 2)

    def test_the_first_row_is_chip_marked_in_the_piped_listing(self):
        """theme.COLOUR_ENABLED is False in this file (module top), so
        render.chip() falls back to bracket form - exactly what makes this
        assertable without a real terminal."""
        t = self.terminal()
        out = io.StringIO()
        with redirect_stdout(out):
            t.offer([(100, "Spider-Man", 1),
                     (90, "Spider-Man (Earth-1610)", 2)])
        text = out.getvalue()
        self.assertIn("[Spider-Man]", text)
        self.assertNotIn("[Spider-Man (Earth-1610)]", text)

    def test_the_cursor_row_is_chip_marked_not_arrow_marked(self):
        """Drives the interactive branch exactly as
        BareNumberSelects.test_escaping_the_interactive_picker_leaves_the_rows_nameable
        does: render.interactive() patched true, render.pick fed a real
        escape key through its own `keys` parameter so the widget stays
        testable without a console."""
        t = self.terminal()
        real_interactive = T.render.interactive
        real_pick = T.render.pick
        T.render.interactive = lambda: True
        T.render.pick = (lambda rows, current, draw, keys=None:
                         real_pick(rows, current, draw, keys=[T.render.ESC]))
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                t.offer([(100, "Spider-Man", 1),
                         (90, "Spider-Man (Earth-1610)", 2)])
        finally:
            T.render.interactive = real_interactive
            T.render.pick = real_pick
        text = out.getvalue()
        self.assertNotIn("▸", text)
        self.assertIn("[Spider-Man]", text)
        self.assertNotIn("[Spider-Man (Earth-1610)]", text)

    def test_the_interactive_redraw_does_not_stagger(self):
        """render.pick() calls draw() again on EVERY arrow key. A wipe there
        would pause ~0.055s x len(rows) after each one - on a ten-row offer,
        well over half a second per keystroke, in the one picker people
        actually navigate. The wipe belongs to the one-shot piped listing
        only; the chip already carries the visual change on redraw.

        Drives at least one arrow-key move before the escape, through the
        same `keys=` injection render.pick() exposes for exactly this - see
        test_escaping_the_interactive_picker_leaves_the_rows_nameable above.
        """
        t = self.terminal()
        calls = []
        real_wipe = T.motion.wipe
        T.motion.wipe = lambda lines, *a, **k: calls.append(lines)
        real_interactive = T.render.interactive
        real_pick = T.render.pick
        T.render.interactive = lambda: True
        T.render.pick = (lambda rows, current, draw, keys=None:
                         real_pick(rows, current, draw,
                                   keys=[T.render.UP, T.render.ESC]))
        try:
            with redirect_stdout(io.StringIO()):
                t.offer([(100, "Spider-Man", 1),
                         (90, "Spider-Man (Earth-1610)", 2)])
        finally:
            T.motion.wipe = real_wipe
            T.render.interactive = real_interactive
            T.render.pick = real_pick
        self.assertEqual(calls, [])

    def test_the_piped_listing_rows_stay_the_same_total_width(self):
        """render.chip() is exactly two columns wider than plain text (its
        own docstring). The chip'd row's label drops two columns to match -
        mirroring choice_row()'s 2-vs-4 prefix trade - so row 1 does not
        throw off the column a scanned list depends on."""
        t = self.terminal()
        out = io.StringIO()
        with redirect_stdout(out):
            t.offer([(100, "AAAAAAAAAA", 1), (90, "BBBBBBBBBB", 2)])
        lines = [l for l in out.getvalue().splitlines()
                 if "AAAAAAAAAA" in l or "BBBBBBBBBB" in l]
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0]), len(lines[1]))


class NothingOverflowsThePicker(unittest.TestCase):
    """The picker's half of test_render.py's NothingOverflows sweep.

    The spec's Testing section requires that no printable line exceeds the
    terminal width at 34/56/78/96 "for every answer shape, picker and
    refusal". NothingOverflows covers the banner, the answer rows and the
    caption; the picker was never given the same treatment, and until
    2026-09-05 `choice_row()` and `offer()` had no width bound at all.
    Measured over `curated/characters.txt`: 6,771 of 104,094 character
    headlines (6.50%) exceed 30 columns and so overflow at MIN_WIDTH = 34,
    81 exceed even width 56, and the longest is 406 characters.

    It lives here rather than beside NothingOverflows because both rows are
    built by Terminal methods, and test_render.py deliberately imports no
    REPL - "kept apart from the REPL so it can be tested without a model, a
    GPU or a terminal" (infer/render.py's own docstring). The widths are the
    same five NothingOverflows sweeps.
    """

    WIDTHS = (34, 40, 56, 78, 96)

    # Two headlines longer than the room at every width above, so the sweep
    # is a sweep and not a formality. Only ONE of them is a corpus headline:
    #  - the first is real - it occurs exactly once in
    #    curated/characters.txt, and is the longest headline there that a
    #    reader would recognise as a name (124 columns). It has spaces, so
    #    _fit cuts it at a word boundary;
    #  - the second is CONSTRUCTED (zero occurrences), and has to be. It
    #    carries no space, which is _fit's other branch - the only case
    #    where it may slice into a token - and no corpus headline can drive
    #    that branch across this sweep: the longest space-free headline in
    #    characters.txt is 32 columns ("User:ShawsAtNight/CharacterDraft"),
    #    far short of the premise test's bound of longer than every swept
    #    width. An earlier version of this comment called them "real corpus
    #    headlines, both of them"; measured, one of them is not.
    LONG = ("Valentina Ronzoni Anna Maria Albergetti Cecilia Andretti "
            "Rigaletto Merlo Allegro Dee Dee Sharp Joan Fontaine "
            "(Earth-TRN1656)")
    UNBROKEN = "Digitally-Enhanced-Autonomous-Telepathic-Host-Designed-" \
               "Only-for-Killing-and-Nothing-Else-(Earth-22925)"
    ROWS = [(900, LONG, 1), (800, UNBROKEN, 2), (700, "Spider-Man", 3)]

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "config.json"
        self._real, T.CONFIG = T.CONFIG, self.tmp

    def tearDown(self):
        T.CONFIG = self._real

    def terminal(self, t, width):
        term = T.Terminal("plain")
        term.theme, term.width = t, width
        return term

    def test_the_sweep_fixture_really_exceeds_every_swept_width(self):
        """A width sweep whose rows always fit certifies nothing. The
        picker's own overhead is six columns at most (a two-digit row
        number, its full stop and the 2-vs-4 prefix trade), so exceeding
        the widest swept width outright is the bound that matters."""
        for head in (self.LONG, self.UNBROKEN):
            with self.subTest(head=head[:20]):
                self.assertGreater(len(head), max(self.WIDTHS))
        self.assertNotIn(" ", self.UNBROKEN)

    def test_a_cursor_row_fits_every_width_in_every_theme(self):
        for width in self.WIDTHS:
            for name, t in T.theme.THEMES.items():
                term = self.terminal(t, width)
                for row in self.ROWS:
                    for cursor in (True, False):
                        with self.subTest(width=width, theme=name,
                                          cursor=cursor, row=row[1][:16]):
                            out = term.choice_row(row, cursor)
                            self.assertLessEqual(
                                len(T.theme.strip(out)), width)

    def test_the_piped_listing_fits_every_width_in_every_theme(self):
        """render.interactive() is False under pytest, so offer() takes the
        numbered-list branch. motion.wipe is replaced with a plain print
        loop: its per-line pause would cost ~0.055s x 3 rows x 5 widths x 7
        themes, and this test is about columns, not animation."""
        real_wipe = T.motion.wipe
        T.motion.wipe = lambda lines, *a, **k: [print(ln) for ln in lines]
        try:
            for width in self.WIDTHS:
                for name, t in T.theme.THEMES.items():
                    with self.subTest(width=width, theme=name):
                        term = self.terminal(t, width)
                        out = io.StringIO()
                        with redirect_stdout(out):
                            term.offer(list(self.ROWS))
                        for line in out.getvalue().split("\n"):
                            self.assertLessEqual(
                                len(T.theme.strip(line)), width, line)
        finally:
            T.motion.wipe = real_wipe

    def test_a_truncated_row_is_still_selectable(self):
        """Truncation is presentation only. `last_offer` keeps the rows
        exactly as facts.variants() produced them, so reference.pick_row()
        still sees the whole headline - a bound that clipped the stored row
        would turn a cosmetic fix into a lost selection."""
        real_wipe = T.motion.wipe
        T.motion.wipe = lambda lines, *a, **k: [print(ln) for ln in lines]
        try:
            term = self.terminal(T.theme.THEMES["plain"], 34)
            out = io.StringIO()
            with redirect_stdout(out):
                term.offer(list(self.ROWS))
        finally:
            T.motion.wipe = real_wipe
        # The row really was cut on screen ...
        self.assertIn("…", out.getvalue())
        self.assertNotIn("Joan Fontaine", out.getvalue())
        # ... and the row behind it is untouched.
        self.assertEqual(term.last_offer[0][1], self.LONG)
        self.assertEqual(term.take_reference("1"), 1)
        self.assertEqual(term.take_reference("the second one"), 2)


class Trace(unittest.TestCase):
    """--trace is what makes the answer harness possible: nothing EDITH
    prints says WHICH record answered, so `who is beast` reads correctly
    while answering from an obscure alien called Krahllak."""

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def emit(self, doc_id, offered, trace=True):
        term = T.Terminal("plain", trace=trace)
        term.index = _FakeIndex()
        term.resolve = resolve
        err = io.StringIO()
        with redirect_stderr(err):
            term.emit_trace(doc_id, offered)
        return err.getvalue()

    def test_names_the_record_that_answered(self):
        self.assertEqual(
            self.emit(41416, 0),
            "trace: doc=41416 page=Miguel O'Hara (Earth-928) offered=0\n")

    def test_says_none_when_nothing_was_resolved(self):
        # Not "the last record still in last_doc". A turn that resolved
        # nothing and was answered by the model anyway is the exact defect
        # the harness exists to catch.
        self.assertEqual(self.emit(None, 0),
                         "trace: doc=none page=none offered=0\n")

    def test_falls_back_to_the_headline_when_there_is_no_page_line(self):
        # Wikipedia prose and comic issues carry no `Page:`.
        self.assertEqual(self.emit(2, 0),
                         "trace: doc=2 page=Spider-Man offered=0\n")

    def test_reports_how_many_records_were_offered(self):
        self.assertEqual(self.emit(None, 10),
                         "trace: doc=none page=none offered=10\n")

    def test_silent_without_the_flag(self):
        self.assertEqual(self.emit(41416, 0, trace=False), "")

    def test_goes_to_stderr_not_stdout(self):
        term = T.Terminal("plain", trace=True)
        term.index = _FakeIndex()
        term.resolve = resolve
        out = io.StringIO()
        with redirect_stdout(out):
            term.emit_trace(41416, 0)
        self.assertEqual(out.getvalue(), "")

    def test_machine_mode_pins_sources_on(self):
        # This assertion used to read False. The box printed raw record text,
        # so `category:` and `she is` - both forbidden by cases - leaked in
        # from a machine whose saved config had sources on.
        #
        # The field block is the ANSWER now, not chrome around one, so
        # machine mode pins it ON: suppressing it would make run_cases
        # measure a view no user ever sees. What has NOT changed is that the
        # value is PINNED - a harness that scores differently depending on
        # saved preferences is not a measurement. See the 4.12 spec, R1.
        T.CONFIG.parent.mkdir(parents=True, exist_ok=True)
        T.CONFIG.write_text('{"sources": false}', encoding="utf-8")
        self.assertTrue(T.Terminal("", trace=True).show_sources)
        self.assertFalse(T.Terminal("", trace=False).show_sources)

    def test_config_sources_is_read_when_not_traced(self):
        # Named for what this actually checks: a non-trace terminal still
        # reads a saved config value. The real prose-fallback coverage is
        # RowsThroughAsk.test_sources_off_falls_back_to_the_prose, below.
        T.CONFIG.parent.mkdir(parents=True, exist_ok=True)
        T.CONFIG.write_text('{"sources": true}', encoding="utf-8")
        self.assertTrue(T.Terminal("", trace=False).show_sources)

    def test_machine_mode_pins_the_theme(self):
        T.CONFIG.parent.mkdir(parents=True, exist_ok=True)
        T.CONFIG.write_text('{"theme": "hulk"}', encoding="utf-8")
        self.assertEqual(T.Terminal("", trace=True).theme.name, "plain")

    def test_machine_mode_pins_the_width(self):
        # render.wrap() breaks the answer at self.width. A harness
        # `must_include` check is a substring match, so an expectation like
        # "new mutants" would fail whenever the measured console width
        # happened to wrap between the two words - a false failure that
        # moves with the terminal size, not with the answer.
        self.assertEqual(T.Terminal("", trace=True).width, 80)

    def test_without_the_flag_the_width_is_still_measured(self):
        self.assertEqual(T.Terminal("plain", trace=False).width,
                         T.render.terminal_width())

    def test_the_flag_reaches_the_terminal(self):
        parsed = T.build_parser().parse_args(["--trace"])
        self.assertTrue(parsed.trace)
        self.assertFalse(T.build_parser().parse_args([]).trace)

    def test_theme_is_refused_under_trace(self):
        """/theme would un-pin machine mode mid-session and write a config
        file - both forbidden by --trace."""
        term = T.Terminal("", trace=True)
        before = term.theme.name
        with redirect_stdout(io.StringIO()):
            term.command("/theme hulk")
        self.assertEqual(term.theme.name, before)
        self.assertFalse(T.CONFIG.exists())

    def test_sources_is_refused_under_trace(self):
        term = T.Terminal("", trace=True)
        before = term.show_sources
        with redirect_stdout(io.StringIO()):
            term.command("/sources on")
        self.assertEqual(term.show_sources, before)
        self.assertFalse(T.CONFIG.exists())

    def test_refusing_a_command_under_trace_says_so(self):
        term = T.Terminal("", trace=True)
        out = io.StringIO()
        with redirect_stdout(out):
            term.command("/theme hulk")
        self.assertIn("pinned", out.getvalue())

    def test_theme_and_sources_still_work_without_trace(self):
        """The refusal is trace-specific - an ordinary session is untouched."""
        term = T.Terminal("plain", trace=False)
        with redirect_stdout(io.StringIO()):
            term.command("/theme hulk")
            term.command("/sources off")
        self.assertEqual(term.theme.name, "hulk")
        self.assertFalse(term.show_sources)


class _FakeIndex:
    """Only what emit_trace touches: text(doc_id) and headlines."""

    TEXT = {
        41416: "Spider-Man 2099\nPage: Miguel O'Hara (Earth-928)\nCreated by: PD\n",
        2: "Spider-Man\nWikipedia prose, no Page line.\n",
    }
    headlines = {41416: "Spider-Man 2099", 2: "Spider-Man"}

    def text(self, doc_id):
        return self.TEXT[doc_id]

    def __len__(self):
        # ask()'s retrieval line prints len(self.index) whenever the turn is
        # not chitchat. Needed so TraceThroughAsk can drive a real (non-
        # chitchat) turn through ask() rather than stubbing that line away.
        return len(self.TEXT)


class _StubEngine:
    """Only what ask() calls. plan() ignores its arguments and returns
    whatever this stub was built with - this is what lets TraceThroughAsk
    drive ask() itself, with no real index, model, or retrieval path.

    Takes one plan per call, in order, then keeps returning the last one -
    so the common single-plan case (`_StubEngine(plan)`) just answers the
    same way every turn, and a multi-plan case (`_StubEngine(p1, p2)`) can
    drive a turn that resolves followed by one that doesn't."""

    def __init__(self, *plans):
        self._plans = list(plans)

    def plan(self, *a, **kw):
        if len(self._plans) > 1:
            return self._plans.pop(0)
        return self._plans[0]


class TraceThroughAsk(unittest.TestCase):
    """The 9 `Trace` tests above call emit_trace() directly, which proves the
    method is correct in isolation but cannot catch a regression at its ONE
    call site: rewriting `self.emit_trace(plan.doc_id, ...)` as
    `self.emit_trace(self.last_doc, ...)`, or moving that call below the
    `plan.choices` early return, would leave every one of those tests
    passing unchanged. These two drive `ask()` itself, through a stubbed
    engine, to pin both invariants where they can actually break.
    """

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def terminal(self, plan):
        term = T.Terminal("", trace=True)
        term.index = _FakeIndex()
        term.resolve = resolve
        term.sft = term.search = term.disambiguate = term.facts = None
        term.engine = _StubEngine(plan)
        return term

    def traced(self, term, question="who is this"):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            term.ask(question)
        return err.getvalue()

    def test_traces_doc_id_not_the_sticky_last_doc(self):
        """last_doc is sticky ON PURPOSE - ask() only overwrites it when a
        record is found - so a turn that resolves nothing still leaves the
        PREVIOUS turn's doc sitting in last_doc. If emit_trace ever read
        that instead of plan.doc_id, this is the turn that would report a
        hallucination as though it were grounded.
        """
        plan = engine.Plan(text="No source for that.", doc_id=None,
                           chitchat=False)
        term = self.terminal(plan)
        term.last_doc = 41416          # a real, resolvable doc id - if the
                                        # bug were present this would leak in
        self.assertEqual(self.traced(term),
                         "trace: doc=none page=none offered=0\n")

    def test_a_picker_turn_still_traces(self):
        """offer() returns without printing an answer. A trace emitted AFTER
        the `plan.choices` branch's `return` would never run on this path -
        exactly the four picker cases the brief calls out.
        """
        plan = engine.Plan(doc_id=41416, chitchat=False,
                           choices=[(3, "Beast", 1), (2, "Other Beast", 2)])
        term = self.terminal(plan)
        self.assertEqual(
            self.traced(term),
            "trace: doc=41416 page=Miguel O'Hara (Earth-928) offered=2\n")


class _RowsIndex(_FakeIndex):
    """_FakeIndex plus a record with fields, for the field-block tests."""

    TEXT = dict(_FakeIndex.TEXT)
    TEXT[7] = ("Chairman (Earth-616)\n"
               "Page: Henry McCoy (Earth-616)\n"
               "Kind: character\nReality: Earth-616\n"
               "Powers: Superhuman strength\n")
    headlines = dict(_FakeIndex.headlines)
    headlines[7] = "Chairman"


class RowsThroughAsk(unittest.TestCase):
    """The field block, driven through ask() itself.

    run_cases cannot reach this: it always runs --trace, which pins
    show_sources True, so the harness never exercises the `/sources off`
    side at all. Without these two tests, dropping the `and
    self.show_sources` clause would break a documented requirement and
    every gate in the project would stay green.
    """

    RECORD = ("Chairman (Earth-616)\n"
              "Page: Henry McCoy (Earth-616)\n"
              "Kind: character\nReality: Earth-616\n"
              "Powers: Superhuman strength\n")

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def _answer(self, show_sources):
        plan = engine.Plan(text="Henry McCoy is a mutant.", doc_id=7,
                           chitchat=False, rows=[("Powers", "Superhuman strength")])
        term = T.Terminal("", trace=True)
        term.index = _RowsIndex()
        term.resolve = resolve
        term.sft = term.search = term.disambiguate = term.facts = None
        term.engine = _StubEngine(plan)
        term.show_sources = show_sources
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            term.ask("who is beast")
        return theme.strip(out.getvalue())

    def test_sources_on_renders_the_field_block_not_the_prose(self):
        answered = self._answer(True)
        self.assertIn("Powers", answered)
        self.assertIn("Henry McCoy", answered)
        self.assertNotIn("Henry McCoy is a mutant.", answered)

    def test_sources_off_falls_back_to_the_prose(self):
        answered = self._answer(False)
        self.assertIn("Henry McCoy is a mutant.", answered)
        self.assertNotIn("Superhuman strength", answered)


class LastOfferHasAOneTurnLifetime(unittest.TestCase):
    """FINDING 2026-09-04: the whole "survives exactly ONE following turn"
    contract lives in the two `self.last_offer = None` lines inside
    handle_line() (run()'s per-turn body, pulled out so it is reachable
    here without driving `input()`) - and nothing pinned it. Deleting
    either line left the suite green. These two tests drive handle_line()
    itself, through the same stubbed engine as TraceThroughAsk, and fail if
    either clearing disappears: a menu from three questions ago must never
    stay readable for a fourth.
    """

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def terminal(self, plan):
        term = T.Terminal("", trace=True)
        term.index = _FakeIndex()
        term.resolve = resolve
        term.sft = term.search = term.disambiguate = term.facts = None
        term.engine = _StubEngine(plan)
        return term

    def handle(self, term, line):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            term.handle_line(line)

    def test_clears_after_a_reference_resolves(self):
        """A bare number against an open offer is the reference path - the
        `self.last_offer = None` beside `self.ask("who is this")`.

        plan.doc_id is set to the SAME id the reference picked (2, a real
        key in _FakeIndex): ask() overwrites last_doc from the plan it gets
        back, so this is what a working round trip looks like, not a
        coincidence worth removing.
        """
        plan = engine.Plan(doc_id=2, text="Spider-Man.", chitchat=False)
        term = self.terminal(plan)
        term.last_offer = [(100, "Spider-Man", 1),
                           (90, "Spider-Man (Earth-1610)", 2)]
        self.handle(term, "2")
        self.assertIsNone(term.last_offer)
        self.assertEqual(term.last_doc, 2)
        # settle() runs live here, through the real resolve module - this is
        # the only test in the suite that derives a key from a headline
        # rather than injecting `settled` directly. _FakeIndex.headlines[2]
        # is "Spider-Man", so the key is what query_key("Spider-Man")
        # actually returns.
        self.assertEqual(term.settled, {("spiderman",): 2})

    def test_clears_after_an_ordinary_question_is_processed(self):
        """A line naming no row falls through to ask(line) - the
        `self.last_offer = None` above the try/ask block."""
        plan = engine.Plan(doc_id=None, text="No source for that.",
                           chitchat=False)
        term = self.terminal(plan)
        term.last_offer = [(100, "Spider-Man", 1)]
        self.handle(term, "who is doctor doom")
        self.assertIsNone(term.last_offer)

    def test_untouched_when_no_offer_is_open(self):
        """Not a regression risk this finding is about, but worth pinning
        alongside it: an ordinary session with nothing offered stays that
        way."""
        plan = engine.Plan(doc_id=None, text="Hello.", chitchat=True)
        term = self.terminal(plan)
        term.last_offer = None
        self.handle(term, "hello")
        self.assertIsNone(term.last_offer)


class HistoryIsBounded(unittest.TestCase):
    """A long session must not grow state without limit. Drives handle_line()
    through the same stubbed engine as LastOfferHasAOneTurnLifetime - fast,
    no real model or index - well past HISTORY_MAX and checks the cap
    actually holds, not just that trimming code exists."""

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def terminal(self, plan):
        term = T.Terminal("", trace=True)
        term.index = _FakeIndex()
        term.resolve = resolve
        term.sft = term.search = term.disambiguate = term.facts = None
        term.engine = _StubEngine(plan)
        return term

    def handle(self, term, line):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            term.handle_line(line)

    def test_history_stops_growing_past_history_max(self):
        plan = engine.Plan(doc_id=None, text="Hello.", chitchat=True)
        term = self.terminal(plan)
        total = T.Terminal.HISTORY_MAX + 5
        for i in range(total):
            self.handle(term, f"question {i}")
        self.assertEqual(len(term.history), T.Terminal.HISTORY_MAX)
        # the newest turns survive; the oldest are the ones dropped.
        self.assertEqual(term.history[-1][0], f"question {total - 1}")
        self.assertEqual(term.history[0][0], f"question {total - T.Terminal.HISTORY_MAX}")


class HistoryRecordsTheTurnsOwnOutcome(unittest.TestCase):
    """REVIEW FINDING 2026-09-04 (Task 5, fix round 1): `self.last_doc`
    deliberately PERSISTS across a non-resolving turn (chitchat, an
    unresolved question, a menu) so a follow-up still has an entity to
    attach to. Reading `last_doc` in `_record_turn` therefore made a
    chitchat turn right after a resolved one show the PRIOR turn's record
    for a question that never touched it - reproduced with ordinary input:
    `who is spider-man` then `hello` then `/history` printed
    `hello   Spider-Man`. /history's whole purpose is reporting what
    actually happened, so a non-resolving turn must show no record, not a
    stale one. Fixed by a separate `self.last_turn_doc`, set fresh (INCLUDING
    to None) by every ask() call and read by `_record_turn` instead."""

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def terminal(self, *plans):
        term = T.Terminal("", trace=True)
        term.index = _FakeIndex()
        term.resolve = resolve
        term.sft = term.search = term.disambiguate = term.facts = None
        term.engine = _StubEngine(*plans)
        return term

    def handle(self, term, line):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            term.handle_line(line)

    def test_a_chitchat_turn_after_a_resolved_one_shows_no_record(self):
        resolved = engine.Plan(doc_id=2, text="Spider-Man.", chitchat=False)
        chitchat = engine.Plan(doc_id=None, text="Hello yourself.",
                               chitchat=True)
        term = self.terminal(resolved, chitchat)
        self.handle(term, "who is spider-man")
        self.handle(term, "hello")
        line, doc, head = term.history[-1]
        self.assertEqual(line, "hello")
        self.assertIsNone(doc)
        self.assertIsNone(head)
        # last_doc itself is untouched by the chitchat turn - the earlier
        # resolved record is still there for a follow-up to attach to.
        self.assertEqual(term.last_doc, 2)

    def test_the_resolved_turn_before_it_still_shows_its_record(self):
        """Not a regression risk this finding is about, but worth pinning
        alongside it: the fix must not make a resolving turn show nothing."""
        resolved = engine.Plan(doc_id=2, text="Spider-Man.", chitchat=False)
        chitchat = engine.Plan(doc_id=None, text="Hello yourself.",
                               chitchat=True)
        term = self.terminal(resolved, chitchat)
        self.handle(term, "who is spider-man")
        self.handle(term, "hello")
        line, doc, head = term.history[0]
        self.assertEqual(line, "who is spider-man")
        self.assertEqual(doc, 2)
        self.assertEqual(head, "Spider-Man")


class ForgetAndHistoryHaveRealTeeth(unittest.TestCase):
    """REVIEW FINDING 2026-09-04 (Task 5, fix round 1): a piped session
    echoes every typed line to stdout (terminal.py run()), so asserting
    `assertIn("who is namor", out)` against a whole transcript passes from
    that echo alone - it would pass even if /history printed nothing, and
    even if /forget wiped history too. These tests assert on Terminal state
    directly, and on stdout captured around ONE command in isolation (no
    earlier turns' echo anywhere near it), so the guarantee is the command's
    own behaviour, not an artefact of piping input."""

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def test_forget_leaves_history_populated(self):
        term = T.Terminal("", trace=True)
        term.history = [("who is namor", 5, "Namor"),
                        ("who is beast", 7, "Beast")]
        term.settled = {("namor",): 5}
        with redirect_stdout(io.StringIO()):
            term.command("/forget")
        self.assertEqual(term.settled, {})
        self.assertEqual(len(term.history), 2)
        self.assertEqual(term.history[0][0], "who is namor")
        self.assertEqual(term.history[1][0], "who is beast")

    def test_history_command_prints_the_rows(self):
        term = T.Terminal("", trace=True)
        term.history = [("who is namor", 5, "Namor"),
                        ("who is beast", 7, "Beast (Krahllak)")]
        out = io.StringIO()
        with redirect_stdout(out):
            term.command("/history")
        text = out.getvalue()
        self.assertIn("who is namor", text)
        self.assertIn("Namor", text)
        self.assertIn("who is beast", text)
        self.assertIn("Beast (Krahllak)", text)

    def test_history_command_on_an_empty_history_says_so(self):
        term = T.Terminal("", trace=True)
        out = io.StringIO()
        with redirect_stdout(out):
            term.command("/history")
        self.assertIn("nothing yet", out.getvalue().lower())


class _HeadlineIndex:
    """Only what settle() touches: headlines[doc_id]."""

    def __init__(self, headlines):
        self.headlines = headlines


class SettleDerivesAKeyFromTheHeadline(unittest.TestCase):
    """settle()'s only real logic is the headline -> key derivation, and its
    only failure mode is silent: get it wrong and settling becomes a no-op
    that no engine test can catch, since every engine test injects `settled`
    directly rather than deriving it from a headline. Only these tests (and
    the hand run) pin the derivation itself.
    """

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def terminal(self, headlines):
        term = T.Terminal("plain")
        term.index = _HeadlineIndex(headlines)
        term.resolve = resolve
        return term

    def test_a_headline_with_no_reality_suffix_settles(self):
        term = self.terminal({5: "Wolverine"})
        term.settle(5)
        self.assertEqual(term.settled, {("wolverine",): 5})

    def test_a_headline_with_a_reality_suffix_settles_under_the_bare_name(self):
        term = self.terminal({7: "Spider-Man (Earth-1610)"})
        term.settle(7)
        self.assertEqual(term.settled, {("spiderman",): 7})

    def test_a_double_disambiguator_strips_both(self):
        """FINDING: a single REALITY_SUFFIX strip left "(Beast)" attached to
        "Sasquatch (Beast) (Earth-616)", settling it under
        ("sasquatch", "beast") - a key query_key("beast") can never match.
        The pick then had no effect at all, with no error anywhere. Fixed by
        looping PAREN_SUFFIX, the same pattern resolve.py's names_of() uses
        on the Page: line."""
        term = self.terminal({9: "Sasquatch (Beast) (Earth-616)"})
        term.settle(9)
        self.assertEqual(term.settled, {("sasquatch",): 9})

    def test_settling_a_second_name_does_not_erase_the_first(self):
        term = self.terminal({5: "Wolverine", 7: "Spider-Man (Earth-1610)"})
        term.settle(5)
        term.settle(7)
        self.assertEqual(term.settled, {("wolverine",): 5, ("spiderman",): 7})


class ConsoleCursorPickSettlesToo(unittest.TestCase):
    """offer()'s interactive branch - reached only via render.pick() on a
    real console - is a second place a pick is made, and round 1 added its
    own settle() call there (terminal.py, beside that branch's
    last_doc/last_offer lines) with nothing exercising it. Same technique as
    BareNumberSelects.test_escaping_the_interactive_picker_leaves_the_rows_nameable:
    patch render.interactive() true and force render.pick()'s `keys` to a
    real selection (DOWN then ENTER, picking row 2) rather than stubbing
    pick() away entirely - a pipe-driven console never reaches this branch.
    """

    def setUp(self):
        self._real = T.CONFIG
        T.CONFIG = Path(tempfile.mkdtemp()) / "config.json"

    def tearDown(self):
        T.CONFIG = self._real

    def test_a_cursor_pick_settles(self):
        term = T.Terminal("plain")
        term.index = _FakeIndex()
        term.resolve = resolve
        term.sft = term.search = term.disambiguate = term.facts = None
        term.show_sources = False
        term.engine = _StubEngine(
            engine.Plan(doc_id=2, text="Spider-Man.", chitchat=False))
        real_interactive = T.render.interactive
        real_pick = T.render.pick
        T.render.interactive = lambda: True
        T.render.pick = (lambda rows, current, draw, keys=None:
                         real_pick(rows, current, draw,
                                   keys=[T.render.DOWN, T.render.ENTER]))
        try:
            with redirect_stdout(io.StringIO()):
                term.offer([(100, "Spider-Man 2099", 41416),
                           (90, "Spider-Man", 2)])
        finally:
            T.render.interactive = real_interactive
            T.render.pick = real_pick
        self.assertEqual(term.last_doc, 2)
        self.assertIsNone(term.last_offer)
        # _FakeIndex.headlines[2] is "Spider-Man", no reality suffix.
        self.assertEqual(term.settled, {("spiderman",): 2})


class NoAnimFlag(unittest.TestCase):
    """The flag has to be BOTH accepted by the parser and obeyed.

    The parser entry is not redundant: without it, `terminal.py --no-anim`
    dies with "unrecognized arguments". But the entry alone changes nothing -
    motion.enable() makes the decision by scanning raw argv - so asserting
    only that argparse parsed it tests the wrong half.
    """

    def tearDown(self):
        motion.ENABLED = False

    def test_the_parser_accepts_no_anim(self):
        self.assertTrue(T.build_parser().parse_args(["--no-anim"]).no_anim)

    def test_no_anim_defaults_off(self):
        self.assertFalse(T.build_parser().parse_args([]).no_anim)

    def test_passing_no_anim_actually_disables_motion(self):
        class Tty(io.StringIO):
            def isatty(self):
                return True
        saved = os.environ.pop("PYTEST_CURRENT_TEST", None)
        try:
            self.assertTrue(motion.enable(Tty(), argv=[]))
            self.assertFalse(motion.enable(Tty(), argv=["--no-anim"]))
        finally:
            if saved is not None:
                os.environ["PYTEST_CURRENT_TEST"] = saved


class Timings(unittest.TestCase):
    def test_a_rendered_answer_reports_milliseconds(self):
        """"0.0s" would hide the fastest path in the system."""
        self.assertEqual(T._took(0.0004), "0 ms")
        self.assertEqual(T._took(0.12), "120 ms")

    def test_a_generated_answer_reports_seconds(self):
        self.assertEqual(T._took(1.24), "1.2s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
