"""Tests for what the terminal draws.

The rendering is where a correct answer can still come out wrong: a box that
does not line up, a wrap that eats a word, or - the one that matters - a
streamed answer that shows "unknown.." and then cannot take it back, because
the clean-up only runs on the finished text.

Run: py -m pytest infer/test_render.py
"""
import importlib.util
import random
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


R = _load("render", "infer/render.py")
theme = _load("theme", "infer/theme.py")
sft = _load("sft_data", "train/sft_data.py")

theme.COLOUR_ENABLED = False
PLAIN = theme.get("plain")

RECORD = ("Moon Knight\n"
          "Full name: Marc Spector\n"
          "Created by: Doug Moench; Don Perlin\n"
          "First appearance: Werewolf by Night Vol 1 32\n"
          "Powers: Strength that waxes with the moon\n"
          "Reality: Earth-616\n"
          "History:\nMarc Spector died and came back.")


class Wrapping(unittest.TestCase):
    def test_no_line_exceeds_the_width(self):
        text = "Moon Knight was created by Doug Moench and Don Perlin. " * 6
        for line in R.wrap(text, width=40, indent="  ").split("\n"):
            self.assertLessEqual(len(line), 40, line)

    def test_every_word_survives(self):
        text = "Spider-Man is from Earth-616 and was created by Stan Lee."
        self.assertEqual(R.wrap(text, 20, "  ").split(), text.split())

    def test_each_line_carries_the_indent(self):
        out = R.wrap("one two three four five six seven eight", 16, "  ")
        for line in out.split("\n"):
            self.assertTrue(line.startswith("  "), repr(line))

    def test_a_word_longer_than_the_width_is_not_lost(self):
        long = "Supercalifragilistic" * 3
        self.assertIn(long, R.wrap(f"a {long} b", 12, "  "))

    def test_empty_text(self):
        self.assertEqual(R.wrap("", 40, "  "), "")


class SourcesBox(unittest.TestCase):
    """The box is the project's claim to being grounded, so it has to be
    exactly right: every row the same width, and nothing in it invented."""

    def box(self, width=60, record=RECORD):
        return R.sources_box(record, PLAIN, width).split("\n")

    def test_every_row_is_the_same_width(self):
        rows = self.box()
        widths = {len(theme.strip(r)) for r in rows}
        self.assertEqual(len(widths), 1, rows)

    def test_it_fits_the_width_given(self):
        for width in (40, 52, 60, 78):
            for row in self.box(width):
                self.assertLessEqual(len(theme.strip(row)), width)

    def test_the_headline_and_reality_are_in_the_top_rule(self):
        top = theme.strip(self.box()[0])
        self.assertIn("Moon Knight", top)
        self.assertIn("Earth-616", top)

    def test_fields_are_shown_with_readable_labels(self):
        body = "\n".join(theme.strip(r) for r in self.box())
        self.assertIn("Created by", body)
        self.assertIn("Doug Moench", body)
        self.assertIn("Real name", body)      # not "Full name"

    def test_history_is_never_in_the_box(self):
        body = "\n".join(theme.strip(r) for r in self.box())
        self.assertNotIn("came back", body)

    def test_a_long_value_is_truncated_not_wrapped(self):
        rec = "X\nReality: Earth-616\nPowers: " + "very long power; " * 20
        rows = self.box(50, rec)
        self.assertEqual(len({len(theme.strip(r)) for r in rows}), 1)

    def test_a_record_with_no_usable_field_yields_nothing(self):
        self.assertEqual(R.sources_box("Someone\nHistory:\nstuff", PLAIN, 60), "")

    def test_a_formal_name_repeating_the_headline_is_dropped(self):
        """Locations and items carry `Formal name`, not `Full name` - the
        rename must not let the same headline-repeating row back into the
        box under its new label."""
        rec = ("Wakanda" + chr(10) + "Kind: location" + chr(10)
               + "Formal name: Wakanda" + chr(10) + "Created by: Stan Lee" + chr(10))
        out = R.sources_box(rec, PLAIN, 60)
        self.assertNotIn("Formal name", out)

    def test_a_headline_longer_than_the_box_still_closes_it(self):
        rec = ("A" * 90) + "\nCreated by: Stan Lee\nReality: Earth-616"
        rows = self.box(50, rec)
        self.assertEqual(len({len(theme.strip(r)) for r in rows}), 1)
        self.assertTrue(theme.strip(rows[-1]).endswith("┘"))


class BadgeByKind(unittest.TestCase):
    """The Earth-616 badge is a character's property. It appeared on the
    Cosmic Cube and the Savage Land, and on Adamantium it showed the FIRST of
    five realities, because sources_box split the field on ";" and took [0]."""

    def box(self, kind, reality):
        nl = chr(10)
        rec = nl.join(["Thing", f"Kind: {kind}", "Page: Thing",
                       f"Reality: {reality}",
                       "Created by: Stan Lee", "First appearance: A Vol 1 1"])
        return R.sources_box(rec, PLAIN)

    def test_a_character_keeps_its_badge(self):
        self.assertIn("Earth-1610", self.box("character", "Earth-1610"))

    def test_an_item_has_no_badge(self):
        self.assertNotIn("Earth-1610", self.box("item", "Earth-1610"))

    def test_a_location_has_no_badge(self):
        self.assertNotIn("Earth-616", self.box("location", "Earth-616"))

    def test_an_unknown_kind_keeps_todays_behaviour(self):
        self.assertIn("Earth-1610", self.box("unknown", "Earth-1610"))


class BadgeSuppressedWhenMainContinuityIsListed(unittest.TestCase):
    """FINDING 5 2026-09-02: the spec's second reality-badge bullet was never
    implemented. `sources_box` took `Reality.split(";")[0]`, so a record
    listing several realities including Earth-616 badged whichever came
    first -- `Godbomb (Story Arc)` with `Earth-14412; Earth-616` badged the
    ALTERNATE reality, though the record IS main continuity. Matches
    `crawl/curate.py`'s `is_main_continuity_reality()`: Earth-616 ANYWHERE in
    the semicolon-separated list means no badge -- but by exact stripped
    equality, never substring, since Earth-6160 and Earth-61610 are both
    real realities in this corpus."""

    def box(self, reality, kind="character"):
        nl = chr(10)
        rec = nl.join(["Thing", f"Kind: {kind}", "Page: Thing",
                       f"Reality: {reality}",
                       "Created by: Stan Lee", "First appearance: A Vol 1 1"])
        return R.sources_box(rec, PLAIN)

    def test_earth_616_anywhere_in_the_list_suppresses_the_badge(self):
        out = self.box("Earth-14412; Earth-616")
        self.assertNotIn("Earth-14412", out)
        self.assertNotIn("Earth-616", out)

    def test_a_list_without_earth_616_still_badges_the_first(self):
        out = self.box("Earth-14412; Earth-1610")
        self.assertIn("Earth-14412", out)

    def test_a_lookalike_reality_does_not_falsely_match_earth_616(self):
        """Earth-6160 and Earth-61610 are real realities in this corpus and
        must not be treated as Earth-616 by a substring check."""
        out = self.box("Earth-6160; Earth-61610")
        self.assertIn("Earth-6160", out)

    def test_a_lone_earth_616_reality_still_shows_its_badge(self):
        """The suppression is for a record that LISTS SEVERAL realities and
        includes Earth-616 among them, not for the ordinary single-reality
        Earth-616 case - that badge already read correctly and stays."""
        out = self.box("Earth-616")
        self.assertIn("Earth-616", out)

    def test_suppressing_the_badge_still_yields_a_valid_box(self):
        """The box ends with a hard width assert on every row; the tail must
        become genuinely empty, not merely blank text still occupying the
        row, or that assert fails."""
        rows = self.box("Earth-14412; Earth-616").split("\n")
        widths = {len(theme.strip(r)) for r in rows}
        self.assertEqual(len(widths), 1, rows)


class Streaming(unittest.TestCase):
    """Tokens arrive one at a time; the clean-up only makes sense on whole
    words. Printing raw and repairing later is not an option in a terminal -
    the characters are already on the screen."""

    def run_stream(self, text, chunks):
        out = []
        stream = R.TidyStream(out.append, lambda s: sft.one_terminator(sft.agree(s)),
                              width=40, indent="  ")
        for chunk in chunks:
            stream.write(chunk)
        stream.close()
        return "".join(out)

    def by_char(self, text):
        return self.run_stream(text, list(text))

    def expected(self, text):
        return R.wrap(sft.one_terminator(sft.agree(text)), 40, "  ")

    def test_streamed_output_equals_the_finished_answer(self):
        text = ("Moon Knight was created by Doug Moench and Don Perlin. "
                "His real name is Marc Spector.")
        self.assertEqual(self.by_char(text), self.expected(text))

    def test_a_doubled_stop_is_never_shown(self):
        text = "The full extent of this ability are unknown.. He is a Human."
        self.assertNotIn("..", self.by_char(text))
        self.assertEqual(self.by_char(text), self.expected(text))

    def test_a_disagreeing_verb_is_never_shown(self):
        text = "They works as a Vigilante and Agent of the Cosmos."
        self.assertNotIn("They works", self.by_char(text))
        self.assertEqual(self.by_char(text), self.expected(text))

    def test_the_result_does_not_depend_on_how_the_chunks_fall(self):
        text = ("Venom is a Symbiote.. They works as a Vigilante. "
                "Venom first appeared in Amazing Spider-Man Vol 1 252.")
        want = self.expected(text)
        rnd = random.Random(4)
        for _ in range(20):
            chunks, i = [], 0
            while i < len(text):
                n = rnd.randint(1, 7)
                chunks.append(text[i:i + n])
                i += n
            self.assertEqual(self.run_stream(text, chunks), want)

    def test_nothing_is_printed_that_the_clean_up_would_have_changed(self):
        """The invariant: only text no later token can alter goes to screen."""
        out = []
        stream = R.TidyStream(out.append, lambda s: sft.one_terminator(s),
                              width=40, indent="  ")
        for ch in "unknown..":
            stream.write(ch)
        self.assertNotIn(".", "".join(out))      # held back until it is settled
        stream.close()
        self.assertTrue("".join(out).rstrip().endswith("unknown."))

    def test_an_empty_answer_prints_nothing(self):
        self.assertEqual(self.run_stream("", []), "")


class BannerSpacing(unittest.TestCase):
    """The wordmark's letters must be evenly spaced.

    Before this test the banner had ZERO blank columns in it: E and D
    touched, D and I had one gap, I and T touched, T and H touched. Four
    joins, three different spacings, which is what read as crooked.
    """

    def test_all_rows_are_the_same_width(self):
        self.assertEqual({len(r) for r in R.BANNER}, {R.BANNER_WIDTH})

    def test_every_join_is_exactly_one_blank_column(self):
        n = len(R.BANNER[0])
        blank = [all(row[c] == " " for row in R.BANNER) for c in range(n)]
        runs, c = [], 0
        while c < n:
            if blank[c]:
                start = c
                while c < n and blank[c]:
                    c += 1
                runs.append((start, c - start))
            else:
                c += 1
        self.assertEqual(len(runs), 4, f"EDITH has 4 joins, found {runs}")
        self.assertTrue(all(w == 1 for _, w in runs),
                        f"every join must be one column, found {runs}")


class Banner(unittest.TestCase):
    def test_every_row_is_the_same_width(self):
        lines = [theme.strip(l) for l in R.banner(PLAIN).split("\n")]
        self.assertEqual(len({len(l) for l in lines}), 1, lines)

    def test_it_is_tall_enough_to_read_at_a_console_line_height(self):
        """The two-row half-block face was reported unreadable: it is drawn
        for a browser's 19px, and a console line height crushes it."""
        self.assertGreaterEqual(len(R.banner(PLAIN).split("\n")), 5)

    def test_the_letters_never_change_with_the_theme(self):
        """A theme colours the wordmark and locks its own emblem up beside
        it. The letters are the PRODUCT and stay put, or the terminal reads
        as seven different tools.

        The lockup is emblem-left, wordmark-right (Task 4), so the letters
        now sit at the END of each row rather than the start - this used to
        read `row.startswith(letters)` when the sigil was appended after.
        """
        for name in theme.names():
            drawn = theme.strip(R.banner(theme.get(name))).split(chr(10))
            self.assertEqual(len(drawn), len(R.BANNER), name)
            for row, letters in zip(drawn, R.BANNER):
                self.assertTrue(row.endswith(letters), (name, row))

    def test_a_sigil_sits_beside_the_letters_not_over_them(self):
        wide = theme.strip(R.banner(theme.get("hulk"))).split(chr(10))
        plain = theme.strip(R.banner(PLAIN)).split(chr(10))
        self.assertGreater(max(len(r) for r in wide),
                           max(len(r) for r in plain))


class ChipsAndCaptions(unittest.TestCase):
    """Painted backgrounds - the first use of theme.bg() in the program.

    The width rule is the one that matters: a chip is always exactly two
    columns wider than its text, coloured or not, or every line that
    contains one drifts.
    """

    def setUp(self):
        self.colour = theme.COLOUR_ENABLED

    def tearDown(self):
        theme.COLOUR_ENABLED = self.colour

    def test_every_painted_pair_carries_a_background_and_a_foreground(self):
        """FINDING 2026-09-05 (whole-branch review, Important 4).

        This used to assert `isinstance(..., str)` on three values that a
        frozen dataclass with `str` fields makes strings by construction, so
        it could not fail: `plain` defines none of the eight and passed. The
        spec asked for something the type cannot give - "every chip and
        caption pair carries both a background and an explicit foreground" -
        so the assertion is now non-emptiness.

        Scoped to the themes that actually define a pair, because a theme
        with no caption is meaningful (theme.caption_of's own docstring: it
        keeps its quip on the header line instead), and `plain` deliberately
        paints nothing at all - render.chip and render.caption both fall
        back to an uncoloured form when `back` is empty, and
        test_a_chip_is_two_columns_wider_than_its_text_without_colour pins
        that. What must not exist is a HALF-filled pair: a foreground escape
        emitted with no background behind it, which is what render.caption's
        docstring warns about.
        """
        painted = 0
        for name, t in theme.THEMES.items():
            with self.subTest(theme=name):
                for label, pair in (("chip", t.chip), ("caption", t.caption)):
                    if not pair:
                        continue
                    painted += 1
                    self.assertEqual(len(pair), 2, (name, label))
                    back, fore = pair
                    self.assertTrue(back, (name, label, "background"))
                    self.assertTrue(fore, (name, label, "foreground"))
        # The sweep must have found pairs to check, or it is the old
        # vacuous test wearing a new assertion.
        self.assertGreater(painted, 0)

    def test_a_derived_chip_pair_is_still_a_usable_pair(self):
        """chip_of() falls back to (faint, text) for a theme that defines no
        chip - "derived rather than required, so a theme nobody has designed
        yet still renders". For every theme that HAS colours at all, that
        fallback must itself be a filled pair, or render.chip() paints a
        foreground onto nothing."""
        for name, t in theme.THEMES.items():
            if name == "plain":                 # paints nothing, by design
                continue
            with self.subTest(theme=name):
                back, fore = theme.chip_of(t)
                self.assertTrue(back, name)
                self.assertTrue(fore, name)
                self.assertTrue(theme.bright_of(t), name)

    def test_a_chip_is_two_columns_wider_than_its_text_in_colour(self):
        theme.COLOUR_ENABLED = True
        for name, t in theme.THEMES.items():
            with self.subTest(theme=name):
                out = R.chip("Earth-616", t)
                self.assertEqual(len(theme.strip(out)), len("Earth-616") + 2)

    def test_a_chip_is_two_columns_wider_than_its_text_without_colour(self):
        theme.COLOUR_ENABLED = False
        t = theme.THEMES["spider-man"]
        self.assertEqual(len(theme.strip(R.chip("Earth-616", t))),
                         len("Earth-616") + 2)

    def test_a_caption_fills_exactly_the_width_it_is_given(self):
        theme.COLOUR_ENABLED = True
        t = theme.THEMES["deadpool"]
        for width in (34, 56, 78):
            with self.subTest(width=width):
                out = R.caption("i was there.", t, width)
                self.assertEqual(len(theme.strip(out)), width)

    def test_a_caption_too_long_for_the_width_is_cut_not_overflowed(self):
        theme.COLOUR_ENABLED = True
        t = theme.THEMES["deadpool"]
        out = R.caption("x" * 200, t, 40)
        self.assertEqual(len(theme.strip(out)), 40)

    def test_a_caption_paints_when_the_theme_actually_defines_one(self):
        # Every shipped theme still defaults caption=(), so a test that used
        # one would run the uncoloured branch and prove nothing about colour.
        theme.COLOUR_ENABLED = True
        painted = theme.Theme(
            name="t", label="t", accent="#ffffff", second="#000000",
            text="#cccccc", dim="#888888", faint="#444444",
            glyph="x", verb="x", caption=("#f2c94c", "#1a1206"))
        out = R.caption("i was there.", painted, 40)
        self.assertEqual(len(theme.strip(out)), 40)
        self.assertNotEqual(out, theme.strip(out))     # colour really applied
        # Pinned to 24-bit: the escape's exact form now depends on the
        # terminal, and this test is about caption painting a background
        # at all, not about which depth the machine running it reports.
        saved, theme.TRUECOLOR = theme.TRUECOLOR, True
        try:
            out = R.caption("i was there.", painted, 40)
            self.assertIn("48;2;242;201;76", out)      # the BACKGROUND escape
        finally:
            theme.TRUECOLOR = saved

    def test_a_half_defined_caption_pair_paints_nothing(self):
        # A foreground escape with no background behind it is the failure
        # mode: bg("") contributes nothing while rgb() would still emit.
        theme.COLOUR_ENABLED = True
        half = theme.Theme(
            name="t", label="t", accent="#ffffff", second="#000000",
            text="#cccccc", dim="#888888", faint="#444444",
            glyph="x", verb="x", caption=("", "#ffffff"))
        out = R.caption("hello", half, 20)
        self.assertEqual(out, theme.strip(out))        # no escapes at all
        self.assertEqual(len(out), 20)

    def test_bright_falls_back_to_text_only_when_unset(self):
        unset = theme.Theme(
            name="t", label="t", accent="#ffffff", second="#000000",
            text="#cccccc", dim="#888888", faint="#444444",
            glyph="x", verb="x")
        self.assertEqual(theme.bright_of(unset), "#cccccc")
        lit = theme.Theme(
            name="t", label="t", accent="#ffffff", second="#000000",
            text="#cccccc", dim="#888888", faint="#444444",
            glyph="x", verb="x", bright="#ffeecc")
        self.assertEqual(theme.bright_of(lit), "#ffeecc")


class TheLockup(unittest.TestCase):
    """Emblem beside the wordmark, at the wordmark's own height.

    The old sigil was three rows by five characters - fifteen cells to draw
    a spider in. Nothing readable fits in fifteen cells.
    """

    def test_every_sigil_is_six_rows_of_thirteen(self):
        for name, t in theme.THEMES.items():
            with self.subTest(theme=name):
                if not t.sigil:
                    continue
                self.assertEqual(len(t.sigil), 6)
                self.assertEqual({len(r) for r in t.sigil}, {13})

    def test_a_wide_terminal_gets_the_lockup(self):
        t = theme.THEMES["spider-man"]
        rows = theme.strip(R.banner(t, width=78)).split("\n")
        self.assertEqual(len(rows), 6)
        self.assertEqual({len(r) for r in rows}, {R.LOCKUP_WIDTH})

    def test_a_medium_terminal_gets_the_wordmark_alone(self):
        t = theme.THEMES["spider-man"]
        rows = theme.strip(R.banner(t, width=48)).split("\n")
        self.assertEqual(len(rows), 6)
        self.assertEqual({len(r) for r in rows}, {R.BANNER_WIDTH})

    def test_a_narrow_terminal_gets_one_line(self):
        t = theme.THEMES["spider-man"]
        out = theme.strip(R.banner(t, width=34))
        self.assertNotIn("\n", out)
        self.assertLessEqual(len(out), 34)

    def test_every_theme_locks_up_to_equal_width_rows(self):
        for name, t in theme.THEMES.items():
            with self.subTest(theme=name):
                rows = theme.strip(R.banner(t, width=78)).split("\n")
                self.assertEqual(len({len(r) for r in rows}), 1,
                                 f"{name} lockup has ragged rows")

    def test_a_mis_sized_emblem_is_dropped_not_drawn_over_the_wordmark(self):
        # zip() truncates to the shorter sequence, so a short sigil used to
        # cost wordmark rows. Better to draw no emblem than half a wordmark.
        stunted = theme.Theme(
            name="t", label="t", accent="#ffffff", second="#000000",
            text="#cccccc", dim="#888888", faint="#444444",
            glyph="x", verb="x", sigil=("###", "###"))
        rows = theme.strip(R.banner(stunted, width=78)).split(chr(10))
        self.assertEqual(len(rows), len(R.BANNER))
        self.assertEqual({len(r) for r in rows}, {R.BANNER_WIDTH})


class AnswerRows(unittest.TestCase):
    """The record as the answer. The box's two defects must not survive it.

    Defect one: `_clip` cut mid-value and shipped
    `King of Atlantis and adventurer; Formerly:; terrorist;…` to the user.
    Defect two: the title was the story-current alias, so Henry McCoy's
    card read `Chairman`.
    """

    RECORD = (
        "Chairman (Earth-616)\n"
        "Page: Henry McCoy (Earth-616)\n"
        "Kind: character\n"
        "Reality: Earth-616\n"
        "Full name: Henry Philip McCoy\n"
        "Created by: Stan Lee; Jack Kirby\n"
        "Powers: Superhuman strength and agility; a genius intellect in "
        "genetics and biophysics; prehensile feet\n"
        "History: ...\n")

    def _lines(self, width=78):
        t = theme.THEMES["spider-man"]
        rows = [("Powers", "Superhuman strength and agility; a genius "
                           "intellect in genetics and biophysics"),
                ("Created by", "Stan Lee; Jack Kirby")]
        return R.answer_rows(self.RECORD, rows, t, width)

    def test_no_line_exceeds_the_width(self):
        for width in (34, 56, 78, 96):
            with self.subTest(width=width):
                for line in self._lines(width):
                    self.assertLessEqual(len(theme.strip(line)), width)

    def test_a_long_value_wraps_instead_of_being_cut(self):
        body = " ".join(theme.strip(l) for l in self._lines(56))
        self.assertNotIn("…", body)

    def test_a_wrapped_value_is_indented_under_its_label(self):
        lines = [theme.strip(l) for l in self._lines(56)]
        powers = [i for i, l in enumerate(lines) if "Powers" in l][0]
        label_col = lines[powers].index("Powers")
        value_col = lines[powers].index("Superhuman")
        self.assertGreater(value_col, label_col)
        continuation = lines[powers + 1]
        self.assertEqual(len(continuation) - len(continuation.lstrip()),
                         value_col)

    def test_the_title_is_the_person_not_the_story_current_alias(self):
        head = theme.strip(self._lines(78)[0])
        self.assertIn("Henry McCoy", head)
        self.assertLess(head.index("Henry McCoy"), head.index("Chairman"))

    def test_the_reality_is_a_chip_on_the_title_line(self):
        head = theme.strip(self._lines(78)[0])
        self.assertIn("Earth-616", head)

    def test_a_value_longer_than_the_cap_is_cut_at_a_separator(self):
        t = theme.THEMES["spider-man"]
        long = "; ".join(f"clause number {i}" for i in range(40))
        lines = R.answer_rows(self.RECORD, [("Powers", long)], t, 56)
        body = " ".join(theme.strip(l) for l in lines)
        self.assertNotIn("clause number 39", body)
        self.assertNotRegex(body, r"clause numbe\b")

    def test_a_double_space_does_not_produce_a_false_ellipsis(self):
        # The first draft inferred truncation by comparing a whitespace-
        # normalised rejoin against the original, so any value carrying a
        # double space reported as cut when nothing had been dropped.
        self.assertNotIn("…", " ".join(R._wrap_value("Hello    world", 100)))

    def test_a_token_longer_than_the_room_is_broken_not_overflowed(self):
        # At width 34 with a long label the room is 12 columns, and
        # `Mutant/Atlantean` is 16. Overflowing breaks the width rule.
        for line in R._wrap_value("Mutant/Atlantean Physiology", 12):
            self.assertLessEqual(len(line), 12)

    def test_wrapping_never_emits_a_leading_blank_line(self):
        self.assertNotEqual(
            R._wrap_value("Mutant/Atlantean Physiology", 12)[0], "")

    def test_a_truncated_value_and_its_ellipsis_both_fit_the_room(self):
        long = "; ".join(f"clause number {i}" for i in range(40))
        for line in R._wrap_value(long, 20):
            self.assertLessEqual(len(line), 20)

    def test_a_long_token_cannot_overflow_a_rendered_answer(self):
        # The end-to-end version of the above, through answer_rows at the
        # narrowest supported width.
        t = theme.THEMES["spider-man"]
        rows = [("First appearance", "Mutant/Atlantean Physiology Vol 1")]
        for line in R.answer_rows(self.RECORD, rows, t, 34):
            self.assertLessEqual(len(theme.strip(line)), 34)

    def test_a_very_long_person_name_cannot_overflow(self):
        # 707 of 126,652 corpus records have a person longer than width 34
        # allows, and 8 exceed even 96 - the widest width the app supports.
        t = theme.THEMES["spider-man"]
        record = ("A\nPage: " + "Verylongname " * 12 + "(Earth-616)\n"
                  "Kind: character\nReality: Earth-616\n")
        for width in (34, 56, 78, 96):
            with self.subTest(width=width):
                for line in R.answer_rows(record, [("Powers", "Flight")],
                                          t, width):
                    self.assertLessEqual(len(theme.strip(line)), width)

    def test_a_long_label_cannot_overflow(self):
        # room = max(width - indent, 12) used to hand _wrap_value 12 columns
        # when the real space was 8, producing 38-column lines at width 34.
        t = theme.THEMES["spider-man"]
        rows = [("A" * 20, "some words that will need to wrap around")]
        for width in (34, 56):
            with self.subTest(width=width):
                for line in R.answer_rows(self.RECORD, rows, t, width):
                    self.assertLessEqual(len(theme.strip(line)), width)

    def test_truncation_never_slices_a_whole_word(self):
        value = "Xyzxyzxyzxy2 Abcabcabcab2 Superhumanly cc dd ee ff gg hh"
        out = " ".join(R._wrap_value(value, 12))
        self.assertNotIn("Superhumanl…", out)


class MoonKnight(unittest.TestCase):

    def test_moon_knight_exists_and_resolves_from_what_people_type(self):
        self.assertIn("moon-knight", theme.THEMES)
        for typed in ("moon knight", "moonknight", "moon-knight",
                      "khonshu", "spector"):
            with self.subTest(typed=typed):
                self.assertIsNotNone(theme.get(typed))
                self.assertEqual(theme.get(typed).name, "moon-knight")

    def test_moon_knight_has_a_blurb_like_every_other_theme(self):
        for name in theme.names():
            with self.subTest(theme=name):
                self.assertIn(name, theme.BLURBS)


class NothingOverflows(unittest.TestCase):
    """The box never drifted a column because it asserted its own width.

    Nothing that replaced it may be looser than that.
    """

    RECORD = ("Namor McKenzie (Earth-616)\n"
              "Page: Namor McKenzie (Earth-616)\n"
              "Kind: character\nReality: Earth-616\n"
              "Created by: Bill Everett\n"
              "Powers: " + "very long clause; " * 30
              + "Mutant/Atlantean Physiology\n")

    WIDTHS = (34, 40, 56, 78, 96)

    # `pad = max(len(label) for label, _ in ROWS)` is what drives `room` in
    # answer_rows - "First appearance" (16 chars) is the label that makes
    # room@34 = 12, smaller than "Mutant/Atlantean" (16 chars), so the sweep
    # below genuinely exercises _wrap_value's break_long_words path instead
    # of merely having a long token that happens to fit. Shared with
    # test_the_sweep_fixture_really_exceeds_its_room so the two cannot drift
    # apart into a fixture that looks potent but isn't.
    ROWS = [("Powers", "very long clause; " * 30),
            ("Created by", "Bill Everett"),
            ("First appearance", "Mutant/Atlantean Physiology")]

    def test_the_banner_fits_every_width(self):
        for width in self.WIDTHS:
            for name, t in theme.THEMES.items():
                with self.subTest(width=width, theme=name):
                    for line in theme.strip(R.banner(t, width)).split("\n"):
                        self.assertLessEqual(len(line), width)

    def test_the_answer_fits_every_width_in_every_theme(self):
        for width in self.WIDTHS:
            for name, t in theme.THEMES.items():
                with self.subTest(width=width, theme=name):
                    for line in R.answer_rows(self.RECORD, self.ROWS, t,
                                              width):
                        self.assertLessEqual(len(theme.strip(line)), width)

    def test_the_sweep_fixture_really_exceeds_its_room(self):
        """A width sweep whose tokens always fit certifies nothing.

        This pins the fixture itself: at the narrowest swept width, the
        room must be smaller than the longest token in the rows, so
        _wrap_value's break_long_words path is genuinely exercised by the
        theme x width matrix rather than merely being available.
        """
        pad = max(len(label) for label, _ in self.ROWS)
        room = min(self.WIDTHS) - (4 + pad + 2)
        longest = max(len(word) for _, value in self.ROWS
                      for word in value.split())
        self.assertLess(room, longest, (room, longest))

    def test_a_caption_fits_every_width_in_every_theme(self):
        for width in self.WIDTHS:
            for name, t in theme.THEMES.items():
                with self.subTest(width=width, theme=name):
                    out = R.caption("a quip that runs on and on and on",
                                    t, width)
                    self.assertEqual(len(theme.strip(out)), width)


if __name__ == "__main__":
    unittest.main(verbosity=2)
