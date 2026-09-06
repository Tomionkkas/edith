"""Colour depth: what escape a palette entry becomes, and when.

The bug these guard is not a wrong shade. A terminal that does not understand
`38;2;r;g;b` drops the `38;2` and reads the rest as separate legacy SGR codes,
where 40-47 mean BACKGROUND - so `#e62429` (ending 41) filled the wordmark
solid red and `#2c3c62` (containing 44) turned the rules into blue bars. Every
theme in this repo has at least one such number.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("theme", ROOT / "infer" / "theme.py")
theme = importlib.util.module_from_spec(spec)
sys.modules["theme"] = theme
spec.loader.exec_module(theme)


class Detection(unittest.TestCase):
    def setUp(self):
        self.saved = theme.TRUECOLOR

    def tearDown(self):
        theme.TRUECOLOR = self.saved

    def _detect(self, **env):
        for key in ("COLORTERM", "EDITH_COLOR", "TERM"):
            theme.os.environ.pop(key, None)
        theme.os.environ.update(env)
        try:
            return theme.detect_truecolor()
        finally:
            for key in env:
                theme.os.environ.pop(key, None)

    def test_colorterm_truecolor_is_believed(self):
        self.assertTrue(self._detect(COLORTERM="truecolor"))
        self.assertTrue(self._detect(COLORTERM="24bit"))
        self.assertTrue(self._detect(COLORTERM="TrueColor"))

    def test_a_terminal_that_says_nothing_gets_256(self):
        """Terminal.app is the case: it sets no COLORTERM and cannot do
        24-bit. Optimism here is what produced the filled blocks."""
        if sys.platform == "win32":
            self.skipTest("win32 is 24-bit regardless")
        self.assertFalse(self._detect())

    def test_edith_color_overrides_either_way(self):
        self.assertFalse(self._detect(COLORTERM="truecolor", EDITH_COLOR="256"))
        self.assertTrue(self._detect(EDITH_COLOR="truecolor"))

    @unittest.skipUnless(sys.platform == "win32", "windows only")
    def test_windows_keeps_24_bit_without_colorterm(self):
        """Windows Terminal sets no COLORTERM, and this all rendered
        correctly there before any of it existed."""
        self.assertTrue(self._detect())


class Downsampling(unittest.TestCase):
    def test_the_cube_index_is_in_range(self):
        for colour in ("#000000", "#ffffff", "#e62429", "#2f63e0", "#7f7f7f"):
            with self.subTest(colour=colour):
                n = theme._to_256(*theme._channels(colour))
                self.assertGreaterEqual(n, 16)
                self.assertLessEqual(n, 255)

    def test_a_grey_picks_the_grey_ramp_not_a_tinted_cube_cell(self):
        """`faint` and `dim` are near-neutral; the cube's nearest neighbour
        to a grey can be visibly tinted."""
        self.assertGreaterEqual(theme._to_256(0x80, 0x80, 0x80), 232)

    def test_pure_black_and_white_land_on_the_cube_corners(self):
        self.assertEqual(theme._to_256(0, 0, 0), 16)
        self.assertEqual(theme._to_256(255, 255, 255), 231)

    def test_a_saturated_red_stays_red(self):
        n = theme._to_256(0xE6, 0x24, 0x29)
        self.assertLess(n, 232)                       # not a grey
        i = n - 16
        r, g, b = i // 36, (i // 6) % 6, i % 6
        self.assertGreater(r, g)
        self.assertGreater(r, b)


class Escapes(unittest.TestCase):
    def setUp(self):
        self.saved = theme.TRUECOLOR

    def tearDown(self):
        theme.TRUECOLOR = self.saved

    def test_24_bit_terminals_still_get_24_bit(self):
        theme.TRUECOLOR = True
        self.assertEqual(theme.rgb("#e62429"), "\033[38;2;230;36;41m")
        self.assertEqual(theme.bg("#16295c"), "\033[48;2;22;41;92m")

    def test_256_colour_terminals_get_the_indexed_form(self):
        theme.TRUECOLOR = False
        self.assertTrue(theme.rgb("#e62429").startswith("\033[38;5;"))
        self.assertTrue(theme.bg("#16295c").startswith("\033[48;5;"))

    def test_no_shipped_colour_emits_a_legacy_background_code_in_256_mode(self):
        """The whole point. In 256 mode the only numbers on the wire are the
        layer, the 5, and one index - none of which a terminal can mistake
        for SGR 40-47."""
        theme.TRUECOLOR = False
        for name, t in theme.THEMES.items():
            for field in ("accent", "second", "text", "dim", "faint"):
                colour = getattr(t, field)
                if not colour:
                    continue
                with self.subTest(theme=name, field=field):
                    esc = theme.rgb(colour)
                    parts = esc.lstrip("\033[").rstrip("m").split(";")
                    self.assertEqual(parts[0], "38")
                    self.assertEqual(parts[1], "5")
                    self.assertEqual(len(parts), 3)

    def test_an_empty_colour_is_still_no_escape_at_either_depth(self):
        for depth in (True, False):
            theme.TRUECOLOR = depth
            with self.subTest(truecolor=depth):
                self.assertEqual(theme.rgb(""), "")
                self.assertEqual(theme.bg(""), "")

    def test_strip_removes_the_indexed_form_too(self):
        """Width calculations measure stripped text; a form strip() missed
        would throw every alignment in the program off."""
        theme.TRUECOLOR = False
        painted = theme.paint("Earth-616", "#e62429")
        self.assertEqual(theme.strip(painted), "Earth-616")


if __name__ == "__main__":
    unittest.main()
