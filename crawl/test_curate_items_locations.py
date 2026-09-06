"""Items and Locations: the two record kinds the first crawl never had.

The gap was invisible in the corpus and obvious in the answers. Nothing in
187,584 records owned the name "Infinity Stones", so "what are the infinity
stones" fell through to a character called Stone; Wakanda existed only as a
Wikipedia article, so "what is wakanda" was answered by `Makers (Wakanda)
(Earth-616)`, a 2,769-character Earth-616 stub.

Both pages exist on Fandom with full schemas - 78 KB and 76 KB of wikitext -
under their own templates, whose fields CHAR_FIELD_MAP does not mention. Run
through the character path they would keep a name and a History and silently
drop everything that answers the question: what the stones are, who owns
them, what country Wakanda is, what its capital is.

Run: py crawl/test_curate_items_locations.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "curate", Path(__file__).resolve().parent / "curate.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)


ITEM_WIKITEXT = """{{Marvel Database:Item Template
| Name                    = Infinity Stones
| Aliases                 = * Ina Abanayyartu
* Infinity Gems
* Soul Gems
| CurrentOwner            = [[Infinity Watch (Earth-616)|Infinity Watch]]
| PreviousOwners          = [[Thanos (Earth-616)|Thanos]], [[Illuminati (Earth-616)|Illuminati]]
| Type                    = Infinity Stone
| Material                = Cosmic singularity
| Reality                 = Earth-616
| Creators                = Jim Starlin
| First                   = Marvel Team-Up Vol 1 55
| History                 = Six singularities forged before the universe began.
}}"""

LOCATION_WIKITEXT = """{{Marvel Database:Location Template
| Title                   = Wakanda
| Name                    = Kingdom of Wakanda
| Aliases                 = [[Wakanda Prime]]
| Reality                 = Earth-616
| Galaxy                  = Milky Way
| StarSystem              = Solar System
| Planet                  = Earth
| Continent               = Africa
| Country                 = Wakanda
| Region                  = Horn of Africa
| Capital                 = [[Birnin Zana]]
| Demonym                 = Wakandans
| Language                = [[Wakandan Dictionary|Wakandan]]
| Population              = 6000000
| Creators                = Stan Lee; Jack Kirby
| First                   = Fantastic Four Vol 1 52
| History                 = An African nation never conquered by a colonial power.
}}"""


class ItemRecords(unittest.TestCase):
    def setUp(self):
        self.text = cu.curate_item(
            {"title": "Infinity Stones", "wikitext": ITEM_WIKITEXT})

    def test_headline_is_the_page_title(self):
        """Items carry no CurrentAlias, so the title is the name people use."""
        self.assertEqual(self.text.split(chr(10))[0], "Infinity Stones")

    def test_the_older_name_survives_as_an_alias(self):
        """People still type "infinity gems"; Fandom files it under Stones."""
        self.assertIn("Infinity Gems", self.text)

    def test_type_is_kept(self):
        self.assertIn("Type: Infinity Stone", self.text)

    def test_material_is_kept(self):
        self.assertIn("Material: Cosmic singularity", self.text)

    def test_current_owner_is_kept_without_its_wikilink(self):
        self.assertIn("Current owner: Infinity Watch", self.text)
        self.assertNotIn("Earth-616|", self.text)

    def test_history_is_kept(self):
        self.assertIn("Six singularities", self.text)

    def test_page_title_is_recorded(self):
        self.assertIn("Page: Infinity Stones", self.text)

    def test_formal_name_is_labelled_as_one(self):
        """A country has a formal name; so does an artifact's official title.
        `Full name` is what makes the composer say "real name"."""
        self.assertIn("Formal name: Infinity Stones", self.text)
        self.assertNotIn("Full name:", self.text)


class LocationRecords(unittest.TestCase):
    def setUp(self):
        self.text = cu.curate_location(
            {"title": "Wakanda", "wikitext": LOCATION_WIKITEXT})

    def test_headline_is_the_page_title(self):
        self.assertEqual(self.text.split(chr(10))[0], "Wakanda")

    def test_country_is_kept(self):
        self.assertIn("Country: Wakanda", self.text)

    def test_capital_is_kept_without_its_wikilink(self):
        self.assertIn("Capital: Birnin Zana", self.text)

    def test_continent_is_kept(self):
        self.assertIn("Continent: Africa", self.text)

    def test_population_is_kept(self):
        self.assertIn("Population: 6000000", self.text)

    def test_formal_name_is_labelled_as_one(self):
        """"Latveria's real name is Königruch Latverien" - a country has a
        formal name. `Full name` is what makes the composer say "real name"."""
        self.assertIn("Formal name: Kingdom of Wakanda", self.text)
        self.assertNotIn("Full name:", self.text)

    def test_history_is_kept(self):
        self.assertIn("never conquered", self.text)


class SchemaIsRichEnoughToResolve(unittest.TestCase):
    """MIN_FIELDS = 3 in the resolver: below it a record is treated as prose
    and never wins its own name, which is the whole point of adding these."""

    def test_item_carries_at_least_three_fields(self):
        text = cu.curate_item(
            {"title": "Infinity Stones", "wikitext": ITEM_WIKITEXT})
        fields = [ln for ln in text.split(chr(10))[1:]
                  if ": " in ln and not ln.startswith("History")]
        self.assertGreaterEqual(len(fields), 3)

    def test_location_carries_at_least_three_fields(self):
        text = cu.curate_location(
            {"title": "Wakanda", "wikitext": LOCATION_WIKITEXT})
        fields = [ln for ln in text.split(chr(10))[1:]
                  if ": " in ln and not ln.startswith("History")]
        self.assertGreaterEqual(len(fields), 3)


class RedirectsAreDropped(unittest.TestCase):
    """`Infinity Gems` is a redirect to `Infinity Stones`. A redirect has no
    content, and emitted as a record it is a headline owning a famous name
    with nothing behind it - exactly the shape that loses answers."""

    def test_a_redirect_page_produces_nothing(self):
        rec = {"title": "Infinity Gems",
               "wikitext": "#REDIRECT [[Infinity Stones]]"}
        self.assertEqual(cu.curate_item(rec), "")

    def test_a_lowercase_redirect_also_produces_nothing(self):
        rec = {"title": "Infinity Gems",
               "wikitext": "#redirect [[Infinity Stones]]"}
        self.assertEqual(cu.curate_item(rec), "")


class PhaseRouting(unittest.TestCase):
    """A curator that exists but is never called is worth nothing, and the
    old dispatch was a single `is_issue` boolean that could not grow."""

    def test_items_route_to_the_item_curator(self):
        self.assertIs(cu.curator_for("items"), cu.curate_item)

    def test_locations_route_to_the_location_curator(self):
        self.assertIs(cu.curator_for("locations"), cu.curate_location)

    def test_comics_still_route_to_the_issue_curator(self):
        self.assertIs(cu.curator_for("comics"), cu.curate_issue)

    def test_anything_else_routes_to_the_character_curator(self):
        self.assertIs(cu.curator_for("characters"), cu.curate_character)
        self.assertIs(cu.curator_for("teams"), cu.curate_character)


class KindIsRecorded(unittest.TestCase):
    """Kind is data, written where it is known.

    run_phase() already knows which raw file it is reading, and that file IS
    the kind. By the time a record reaches the index the information is gone,
    which is why a country was rendered with a "real name" and a pronoun.
    """

    def test_a_location_says_it_is_a_location(self):
        text = cu.curate_location(
            {"title": "Wakanda", "wikitext": LOCATION_WIKITEXT}, kind="location")
        self.assertIn("Kind: location", text)

    def test_an_item_says_it_is_an_item(self):
        text = cu.curate_item(
            {"title": "Infinity Stones", "wikitext": ITEM_WIKITEXT}, kind="item")
        self.assertIn("Kind: item", text)

    def test_kind_precedes_the_page_line(self):
        text = cu.curate_location(
            {"title": "Wakanda", "wikitext": LOCATION_WIKITEXT}, kind="location")
        lines = text.split(chr(10))
        self.assertLess(lines.index("Kind: location"),
                        lines.index("Page: Wakanda"))

    def test_phase_names_map_to_kinds(self):
        self.assertEqual(cu.kind_for("locations"), "location")
        self.assertEqual(cu.kind_for("items"), "item")
        self.assertEqual(cu.kind_for("characters"), "character")
        self.assertEqual(cu.kind_for("comics"), "issue")
        self.assertEqual(cu.kind_for("teams"), "team")
        self.assertEqual(cu.kind_for("story_arcs"), "story arc")

    def test_an_unknown_phase_is_a_character(self):
        """Teams, events and arcs all use the character template, and a new
        phase is far more likely to be character-shaped than not."""
        self.assertEqual(cu.kind_for("some_new_phase"), "character")

    def test_an_empty_record_is_still_dropped(self):
        """The Kind line must not resurrect a record with no fields, the way
        the Page line nearly did."""
        self.assertEqual(
            cu.curate_item({"title": "X", "wikitext": "#REDIRECT [[Y]]"},
                           kind="item"), "")


class IssueKindDoesNotDisplaceTheHeadline(unittest.TestCase):
    """curate_issue has no separate `display` variable the way
    curate_character does - its headline sentence is out[0]. search.py builds
    every index headline as rec.split(chr(10), 1)[0], and resolve.untitled()
    extracts a schema-less record's title from that first line, so Kind must
    land at line 1, never displace the headline off line 0."""

    def test_headline_is_still_line_zero(self):
        text = cu.curate_issue(
            {"title": "Amazing Spider-Man Vol 1 300", "wikitext": ""},
            kind="issue")
        lines = text.split(chr(10))
        self.assertIn("Amazing Spider-Man Vol 1 300", lines[0])
        self.assertFalse(lines[0].startswith("Kind:"))

    def test_kind_is_line_one(self):
        text = cu.curate_issue(
            {"title": "Amazing Spider-Man Vol 1 300", "wikitext": ""},
            kind="issue")
        lines = text.split(chr(10))
        self.assertEqual(lines[1], "Kind: issue")


class MultiRealityHeadlines(unittest.TestCase):
    """MEASURED FAILURE 2026-09-02: the card read `Adamantium (Earth-616;
    Earth-1610; Earth-41578; Earth-TRN1400; Earth-199999)` and the prose
    repeated all five. An item that exists in many realities is not an
    alternate continuity; it is a thing that gets around."""

    def item(self, title, reality):
        wt = ("{{Marvel Database:Item Template" + chr(10)
              + f"| Name = {title}" + chr(10)
              + f"| Reality = {reality}" + chr(10)
              + "| Type = Metal" + chr(10)
              + "| Creators = Roy Thomas" + chr(10)
              + "| History = A very hard metal." + chr(10) + "}}")
        return cu.curate_item({"title": title, "wikitext": wt}, kind="item")

    def test_a_list_containing_main_continuity_gets_no_suffix(self):
        text = self.item("Adamantium", "Earth-616; Earth-1610; Earth-199999")
        self.assertEqual(text.split(chr(10))[0], "Adamantium")

    def test_a_genuinely_alternate_reality_still_gets_its_suffix(self):
        text = self.item("Some Gadget", "Earth-1610")
        self.assertEqual(text.split(chr(10))[0], "Some Gadget (Earth-1610)")

    def test_a_list_without_main_continuity_still_gets_its_suffix(self):
        text = self.item("Other Gadget", "Earth-1610; Earth-199999")
        self.assertIn("Earth-1610", text.split(chr(10))[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
