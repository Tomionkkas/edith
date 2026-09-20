"""Events and story arcs carry their narrative in `Synopsis`, not `History`.

`curator_for()` sends both phases to `curate_character`, correctly - they are
character-shaped templates. But that curator's narrative block reads the
`History` field and the `== History ==` section, and events have NEITHER.
Measured over the full population of both raw files:

    phase         pages   History fld   History sec   Synopsis
    events          379             0             0        317
    story_arcs    1,046             0             0        287

So 100% of the narrative on 604 pages was dropped, including all of Civil
War - whose raw page is 59,790 characters opening with the whole story.
`curated/events.txt` was 373 records and 0.0 MB of prose.

This is the 1,304-character `Civil War (Event)` record that Phase 4.5b's
first bullet complains about. The prose was on the page the whole time.

The reverse direction is what makes the fix safe: characters, items and
locations have **zero** Synopsis fields between them (0 of 118,423), so a
Synopsis fallback cannot touch the 98,213 records that already carry History.

Run: py crawl/test_curate_events.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "curate", Path(__file__).resolve().parent / "curate.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)


EVENT_WIKITEXT = """{{Marvel Database:Event Template
| Image                   = Civil War Vol 1 1.jpg
| Name                    = [[Civil War]]
| Aliases                 = Superhuman Civil War
| Reality                 = Earth-616
| First                   = Civil War Vol 1 1
| Last                    = Civil War Vol 1 7

| Synopsis                =
===Brief Summary===
In a battle between [[Robert Hunter (Earth-616)|Nitro]] and the New Warriors,
Nitro exploded, seemingly killing the entire team and a huge number of
civilians. This led the United States government to introduce a registry for
all super-powered individuals, and a '''Civil War''' ensued.
}}
"""

# A character page: History present, Synopsis absent. The fallback must not
# fire, and this record must come out byte-identical to before the change.
CHARACTER_WIKITEXT = """{{Marvel Database:Character Template
| CurrentAlias            = Spider-Man
| Reality                 = Earth-616
| History                 = Peter Parker was bitten by a radioactive spider.
}}
"""


class EventNarrative(unittest.TestCase):

    def setUp(self):
        self.out = cu.curate_character(
            {"title": "Civil War (Event)", "wikitext": EVENT_WIKITEXT},
            kind="event")

    def test_the_synopsis_becomes_the_records_narrative(self):
        self.assertIn("Nitro exploded", self.out)

    def test_it_is_labelled_History_like_every_other_curate_character_record(self):
        """One curator, one narrative label.

        Events could equally be labelled `Synopsis:` - the corpus already has
        that label on comics, and every reader of it stops at both. `History:`
        is chosen because it keeps ALL of curate_character's output uniform,
        and because it is the smaller change: the fallback joins the existing
        `hist` chain rather than adding a second branch and a second label.
        """
        self.assertIn("History:", self.out)

    def test_the_wikilink_markup_is_stripped(self):
        """`[[Robert Hunter (Earth-616)|Nitro]]` is not training text."""
        self.assertNotIn("[[", self.out)
        self.assertNotIn("Earth-616)|", self.out)

    def test_the_fields_still_survive(self):
        self.assertIn("Civil War (Event)", self.out)
        self.assertIn("Kind: event", self.out)


class HeadingMarkupIsNotTrainingText(unittest.TestCase):
    """`===Brief Summary===` is wikitext, not prose.

    FOUND ON REAL DATA, not by a unit test: the first re-curate produced
    "History:\n===Brief Summary===; In a battle between Nitro and ...".

    Pre-existing and corpus-wide - 5,768 character records (5.5%) and 483
    team records (7.1%) already carry it - because the narrative FIELD path
    goes through joinval(), which strips wikilinks but not headings, while
    strip_markup() only ever saw the body-section path. Fixing it where the
    three narrative sources converge covers all of them at once.
    """

    def test_a_heading_keeps_its_words_and_loses_its_equals(self):
        self.assertEqual(cu.clean_headings("===Brief Summary==="),
                         "Brief Summary")

    def test_it_works_at_any_depth(self):
        self.assertEqual(cu.clean_headings("==Aftermath=="), "Aftermath")

    def test_surrounding_prose_is_untouched(self):
        self.assertEqual(
            cu.clean_headings("==War==\nNitro exploded."),
            "War\nNitro exploded.")

    def test_an_equals_sign_in_ordinary_prose_survives(self):
        self.assertEqual(cu.clean_headings("E = mc^2 is not a heading"),
                         "E = mc^2 is not a heading")

    def test_the_curated_event_carries_no_heading_markup(self):
        out = cu.curate_character(
            {"title": "Civil War (Event)", "wikitext": EVENT_WIKITEXT},
            kind="event")
        self.assertNotIn("===", out)
        self.assertIn("Brief Summary", out)


class ComicsGetTheSameTreatment(unittest.TestCase):
    """The heading defect lives in BOTH curators, so it is fixed in both.

    curate_issue has its own narrative block and its own joinval() call, and
    183 comic records (0.3%) carry the markup. Fixing only curate_character
    would leave the same bug in the other half of the corpus.

    Takes effect on the next re-curate of comics.txt, which B1 does not run -
    the shipped file keeps its 183 until then.
    """

    def test_a_comic_synopsis_loses_its_heading_markup(self):
        wt = """{{Marvel Database:Comic Template
| StoryTitle1 = The Final Chapter
| Synopsis1 = ==Part One==
Spider-Man lifts the machinery.
}}
"""
        out = cu.curate_issue({"title": "Amazing Spider-Man Vol 1 33",
                               "wikitext": wt})
        self.assertNotIn("==", out)
        self.assertIn("Part One", out)
        self.assertIn("lifts the machinery", out)


class GalleriesAreNotProse(unittest.TestCase):
    """An image gallery is a list of filenames, not narrative.

    FOUND BY A LIVE RUN of 260 questions on 2026-09-20, not by a test:

        what happened in Marvel Legacy
          -> "...in the wake of Secret Empire. FOOM Vol 2 1 Textless.jpg|
              FOOM Vol 2 (One-Shot) Marvel Legacy Vol 1 1.jpg|..."

    strip_markup ran TAG_RE over it, which removes the <gallery> tags and
    leaves every filename between them. 11% of event records carry this in
    their prose (35 of 317) against 0.0% of character records - because 4.14
    is what brought event prose into the corpus at all, so this arrived with
    it.
    """

    WT = """{{Marvel Database:Event Template
| Name = Marvel Legacy
| Synopsis =
Marvel Legacy relaunched numerous titles in the wake of Secret Empire.
<gallery position="center">
FOOM Vol 2 1 Textless.jpg|FOOM Vol 2 (One-Shot)
Marvel Legacy Vol 1 1.jpg|Marvel Legacy Vol 1 (One-Shot)
</gallery>
The initiative ran through the following year.
}}"""

    def setUp(self):
        self.out = cu.curate_character(
            {"title": "Marvel Legacy", "wikitext": self.WT}, kind="event")

    def test_no_filenames_reach_the_record(self):
        self.assertNotIn(".jpg", self.out)
        self.assertNotIn("FOOM", self.out)

    def test_the_prose_either_side_survives(self):
        self.assertIn("relaunched numerous titles", self.out)
        self.assertIn("ran through the following year", self.out)


class TheFallbackDoesNotTouchCharacters(unittest.TestCase):
    """0 of 118,423 character/item/location pages carry a Synopsis field, so
    this is the whole regression surface and it is empty."""

    def test_a_record_with_History_is_unchanged(self):
        out = cu.curate_character(
            {"title": "Peter Parker (Earth-616)", "wikitext": CHARACTER_WIKITEXT})
        self.assertIn("History:", out)
        self.assertIn("bitten by a radioactive spider", out)

    def test_History_wins_when_a_page_somehow_has_both(self):
        """Order is History first. No page in the corpus has both today, so
        this pins the intent rather than a measured case - if one ever
        appears, the field the other 98,213 records use must not lose."""
        both = CHARACTER_WIKITEXT.replace(
            "}}", "| Synopsis = A synopsis that must not win.\n}}")
        out = cu.curate_character({"title": "X", "wikitext": both})
        self.assertIn("bitten by a radioactive spider", out)
        self.assertNotIn("must not win", out)


if __name__ == "__main__":
    unittest.main()
