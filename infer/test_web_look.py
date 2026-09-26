"""The approved comps are the design. This fails when the page drifts.

`docs/superpowers/specs/2026-09-26-web-app/` holds the three boards that were
approved on 2026-09-26 with "i dont want it to look any different than this".
They are the target, not inspiration, so a colour or typeface appearing in
the stylesheet that was never approved is a regression - the way two earlier
passes rotted, one ornament at a time.
"""
import unittest
import pathlib
import re
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPS = ROOT / "docs" / "superpowers" / "specs" / "2026-09-26-web-app"
CSS = ROOT / "infer" / "web" / "app.css"
HTML = ROOT / "infer" / "web" / "page.html"
JS = ROOT / "infer" / "web" / "app.js"

TOKENS = ["#0B0B0C", "#121214", "#1F1F22", "#2A2A2E",
          "#F2F0EC", "#D6D2C9", "#9A968E", "#6E6A63", "#E8362F"]
FAMILIES = ["Archivo Black", "Archivo", "JetBrains Mono"]


class Look(unittest.TestCase):
    def setUp(self):
        self.css = CSS.read_text(encoding="utf-8")

    def test_every_approved_colour_is_still_in_the_stylesheet(self):
        upper = self.css.upper()
        for token in TOKENS:
            self.assertIn(token.upper(), upper, f"{token} left the design")

    def test_no_colour_outside_the_approved_set(self):
        """One accent. A second one appearing is how a design rots."""
        found = {c.upper() for c in re.findall(r"#[0-9a-fA-F]{6}", self.css)}
        allowed = {t.upper() for t in TOKENS} | {"#FFFFFF", "#C9C5BC", "#7C786F"}
        self.assertEqual(found - allowed, set())

    def test_the_three_typefaces_are_the_approved_ones(self):
        for family in FAMILIES:
            self.assertIn(family, self.css)

    def test_no_typeface_outside_the_approved_set(self):
        """Inter, Roboto and Arial are what a page reaches for when nobody
        chose. Two passes were rejected for looking generated."""
        for banned in ("Inter", "Roboto", "Arial", "Helvetica"):
            self.assertNotIn(banned, self.css)

    def test_the_art_column_keeps_its_height(self):
        """470px, and the plate matches it, so a record with no art does not
        reflow the column - about one record in twenty-one takes that path."""
        self.assertIn("height: 470px", self.css)

    def test_the_two_columns_keep_the_comps_proportion(self):
        """The comp is 1280 wide at 760 transcript / 520 record - about 60/40.
        Held as a ratio with a cap, because on a 2560px screen a record column
        that takes everything left over turns a cover into a mural. Found by
        opening it on a wide monitor."""
        self.assertIn("max-width: 1760px", self.css)
        self.assertIn("clamp(420px, 40%, 640px)", self.css)


class Wiring(unittest.TestCase):
    def test_the_page_loads_only_its_own_two_files(self):
        html = HTML.read_text(encoding="utf-8")
        self.assertIn('href="app.css"', html)
        self.assertIn('src="app.js"', html)

    def test_the_only_remote_thing_is_the_typeface(self):
        """No framework, no build step, no CDN. Fonts are the one exception
        the design allows; images come from the wiki at render time."""
        html = HTML.read_text(encoding="utf-8")
        remote = re.findall(r'(?:href|src)="(https?://[^"]+)"', html)
        self.assertTrue(all("fonts.g" in url for url in remote), remote)

    def test_hidden_actually_hides(self):
        """MEASURED FAILURE 2026-09-26, in a browser and nowhere else: the
        markup said <main hidden> and the stylesheet said display:flex, and
        the RULE wins over the attribute - so the cold open, the answer and
        the picker all rendered at once, stacked, and the page scrolled
        through three full-height sections."""
        css = CSS.read_text(encoding="utf-8")
        self.assertIn("[hidden]", css)
        self.assertIn("display: none !important", css)

    def test_the_page_sends_no_referer_with_image_requests(self):
        """Fandom's CDN refuses a hotlink carrying a Referer: from
        http://127.0.0.1:8420/ it returns a 1,976-byte 404 placeholder, with
        no Referer it serves the file. Every image on the page was that
        placeholder until this was set."""
        html = HTML.read_text(encoding="utf-8")
        self.assertIn('name="referrer"', html)
        self.assertIn("no-referrer", html)
        self.assertIn('referrerPolicy = "no-referrer"', JS.read_text(encoding="utf-8"))

    def test_a_long_field_folds_instead_of_being_truncated(self):
        """Wolverine's Powers field is 3,000 characters. The terminal
        truncates; this page exists not to, so a long value folds into a
        native <details> - no library, no read-more state to keep."""
        js = JS.read_text(encoding="utf-8")
        self.assertIn("details", js)
        self.assertIn("LONG", js)

    def test_the_script_parses(self):
        """MEASURED FAILURE 2026-09-26: a stray newline inside a string made
        app.js a syntax error, so NO listener attached, so every form did a
        native GET and the page answered 404 - with nothing in the console
        of the tab that mattered. A page whose script does not parse is not
        a page, and it looked like a server bug for two rounds."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        run = subprocess.run([node, "--check", str(JS)],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr[:400])

    def test_a_question_in_flight_is_visible(self):
        """A question takes seconds on CPU. Without a pending state that is
        indistinguishable from a page that ignored you - which it looked
        like, twice, to the person this was built for."""
        js = JS.read_text(encoding="utf-8")
        self.assertIn("thinking", js)
        self.assertIn("setBusy", js)

    def test_a_failed_request_says_so(self):
        """fetch() rejecting used to leave the page silent for ever."""
        self.assertIn("failed", JS.read_text(encoding="utf-8"))

    def test_the_session_survives_a_reload(self):
        """Every read and write of localStorage is guarded: it throws in a
        private window, and a session that cannot be saved is still a
        session."""
        js = JS.read_text(encoding="utf-8")
        self.assertIn("localStorage", js)
        self.assertEqual(js.count("try {"), js.count("catch"))

    def test_the_image_chain_is_wired_in_the_page(self):
        """onerror walking to the next rung is what covers a file the wiki
        serves under a different name."""
        js = JS.read_text(encoding="utf-8")
        self.assertIn("onerror", js)
        self.assertIn("plate", js)


class Comps(unittest.TestCase):
    """The approved comps live in docs/, which the public export does not
    carry - they are the private design record, not something a user needs.
    So these skip rather than fail there: a public checkout that cannot see
    them has nothing to compare, which is different from having drifted."""

    def setUp(self):
        if not COMPS.is_dir():
            self.skipTest("comps are private; not in an exported tree")

    def test_the_comps_still_exist_to_compare_against(self):
        for name in ("cold-open.html", "answer.html", "picker.html"):
            self.assertTrue((COMPS / name).is_file(), name)

    def test_the_comps_load_art_from_the_wiki_not_from_a_mockup(self):
        """They were rewritten from the artifact's /_blob/ urls so they open
        in a browser and show the real thing."""
        comp = (COMPS / "answer.html").read_text(encoding="utf-8")
        self.assertIn("static.wikia.nocookie.net", comp)
        self.assertNotIn("/_blob/", comp)
        # The wiki host refuses browser hotlinks; a comp pointing there
        # opens with no art, which is exactly the bug it should demonstrate
        # the absence of.
        self.assertNotIn("Special:FilePath", comp)


if __name__ == "__main__":
    unittest.main()
