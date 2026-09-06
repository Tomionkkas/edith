"""Unit tests for wikitext -> clean text conversion (no network).

Covers two corpus-quality defects that matter for a language-model corpus:
  1. apostrophes must survive (wiki bold/italic is 2+ quotes, not 1)
  2. link-style templates must keep their display label instead of vanishing

Run: py crawl/test_curate_text.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "curate", Path(__file__).resolve().parent / "curate.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)


class Apostrophes(unittest.TestCase):
    def test_possessive_apostrophe_preserved(self):
        self.assertIn("DC's", cu.strip_markup("a take on DC's hero"))

    def test_contraction_apostrophe_preserved(self):
        self.assertIn("It's", cu.strip_markup("It's possible that he knows"))

    def test_apostrophe_inside_proper_noun_preserved(self):
        self.assertIn("O'Brien", cu.strip_markup("voiced by Liam O'Brien"))

    def test_wiki_bold_markup_removed(self):
        self.assertEqual(cu.strip_markup("'''Spider-Man''' swings"), "Spider-Man swings")

    def test_wiki_italic_markup_removed(self):
        self.assertEqual(cu.strip_markup("''Amazing Fantasy'' #15"), "Amazing Fantasy #15")


class LinkTemplates(unittest.TestCase):
    def test_two_arg_link_template_keeps_display_label(self):
        out = cu.strip_markup("a take on {{DC|Kal-El (New Earth)|Superman}}.")
        self.assertIn("Superman", out)
        self.assertNotIn("Kal-El", out)
        self.assertNotIn("|", out)

    def test_one_arg_link_template_keeps_target(self):
        self.assertIn("Iron Man Vol 1 55", cu.strip_markup("seen in {{cl|Iron Man Vol 1 55}}"))

    def test_power_template_preserved(self):
        self.assertIn("Flight", cu.strip_markup("{{Power|Flight}}"))

    def test_cross_wiki_link_keeps_last_label(self):
        out = cu.strip_markup("{{wc|intothespiderverse|Aaron Davis (Earth-1610)|Aaron Davis}}")
        self.assertIn("Aaron Davis", out)
        self.assertNotIn("intothespiderverse", out)

    def test_reference_template_dropped(self):
        self.assertNotIn("Vol 1 1", cu.strip_markup("Text.{{r|Marvel Mini-Book Vol 1 1}}"))

    def test_navigation_template_dropped(self):
        out = cu.strip_markup("{{Navigation\n| title = Former Powers\n| body = stuff}}")
        self.assertNotIn("Former Powers", out)



class WikiLinks(unittest.TestCase):
    """Guards the [[Target|Label]] unwrap that the template rewrite sits next to."""

    def test_piped_wikilink_keeps_label(self):
        out = cu.strip_markup("he and [[Clark Kent (Earth-616)|Clark Kent]] are one")
        self.assertIn("Clark Kent", out)
        self.assertNotIn("Earth-616", out)

    def test_plain_wikilink_keeps_target(self):
        self.assertIn("Mjolnir", cu.strip_markup("wielding [[Mjolnir]]"))

    def test_file_link_dropped(self):
        self.assertNotIn("Cover.jpg", cu.strip_markup("[[File:Cover.jpg|thumb]]text"))

    def test_file_link_with_linked_caption_fully_removed(self):
        """A caption containing [[...]] used to end the match early, leaving debris."""
        wt = ("[[File:Abomination.jpg|thumb|200px|The Abomination in "
              "''[[The Incredible Hulk (film)|The Incredible Hulk]]'' (2008).]]Body text.")
        out = cu.strip_markup(wt)
        self.assertIn("Body text.", out)
        self.assertNotIn("]]", out)
        self.assertNotIn("2008", out)
        self.assertNotIn("thumb", out)


INFOBOX = """{{Marvel Database:Character Template
| Name                    = Test Character
| Notes                   = * Was a take on DC's {{DC|Kal-El (New Earth)|Superman}}. It's possible.
| Powers                  = {{Power|Flight}}
| Affiliation             = {{Navigation
| title = Former
| body = junk}}
}}
"""


class InfoboxFields(unittest.TestCase):
    """The infobox path deleted every {{...}} before strip_markup saw it."""

    def setUp(self):
        self.fields = cu.parse_template_fields(INFOBOX)

    def test_notes_field_keeps_linked_entity(self):
        self.assertIn("Superman", cu.joinval(self.fields["Notes"]))

    def test_notes_field_preserves_apostrophes(self):
        v = cu.joinval(self.fields["Notes"])
        self.assertIn("DC's", v)
        self.assertIn("It's", v)

    def test_powers_field_keeps_power_name(self):
        self.assertIn("Flight", cu.joinval(self.fields["Powers"]))

    def test_navbox_field_still_dropped(self):
        self.assertNotIn("junk", cu.joinval(self.fields.get("Affiliation", "")))


def _infobox(alias=None, reality=None, name="Stephen Strange"):
    lines = ["{{Marvel Database:Character Template", f"| Name = {name}"]
    if alias:
        lines.append(f"| CurrentAlias = {alias}")
    if reality:
        lines.append(f"| Reality = {reality}")
    lines.append("| History = He studied the mystic arts for many years afterwards.")
    lines.append("}}")
    return "\n".join(lines)


class RealityInHeadline(unittest.TestCase):
    """Three Doctor Stranges must not all open with the same line.

    The reality label sits on line 5 of a record, but the first line is where a
    small model anchors. Alternate continuities carry their reality up front;
    Earth-616 is the unmarked default so it stays clean.
    """

    def headline(self, wikitext, title):
        return cu.curate_character({"title": title, "wikitext": wikitext}).split("\n")[0]

    def test_alternate_continuity_headline_carries_reality(self):
        wt = _infobox(alias="Doctor Strange", reality="Earth-199999")
        self.assertEqual(self.headline(wt, "Stephen Strange (Earth-199999)"),
                         "Doctor Strange (Earth-199999)")

    def test_main_continuity_headline_stays_unmarked(self):
        wt = _infobox(alias="Doctor Strange", reality="Earth-616")
        self.assertEqual(self.headline(wt, "Stephen Strange (Earth-616)"),
                         "Doctor Strange")

    def test_bare_numeric_reality_is_normalised(self):
        """Source data sometimes says `Reality = 6160` with no Earth- prefix."""
        wt = _infobox(alias="Doctor Strange", reality="6160")
        text = cu.curate_character({"title": "Stephen Strange (Earth-6160)", "wikitext": wt})
        self.assertEqual(text.split("\n")[0], "Doctor Strange (Earth-6160)")
        self.assertIn("Reality: Earth-6160", text)

    def test_reality_taken_from_title_when_field_missing(self):
        wt = _infobox(alias="Spider-Man", reality=None)
        self.assertEqual(self.headline(wt, "Peter Parker (Earth-1610)"),
                         "Spider-Man (Earth-1610)")

    def test_reality_not_duplicated_when_alias_already_has_it(self):
        wt = _infobox(alias="Doctor Strange (Earth-6160)", reality="Earth-6160")
        self.assertEqual(self.headline(wt, "Stephen Strange (Earth-6160)"),
                         "Doctor Strange (Earth-6160)")

    def test_character_with_no_reality_anywhere_is_unchanged(self):
        wt = _infobox(alias="Some Hero", reality=None)
        self.assertEqual(self.headline(wt, "Some Hero"), "Some Hero")


class CategoryLinksAreDropped(unittest.TestCase):
    """`[[Category:X]]` is filing, not prose, and it was leaking as text.

    A wikilink is unwrapped to its label, which is right for `[[Storm]]` and
    wrong for a namespace link: `[[Category:Human Hybrids]]` became the words
    "Category:Human Hybrids", glued to whatever preceded it. 3,269 lines of
    the character corpus carried one, and they land in exactly the fields an
    answer card shows - `Species / origin: Human hybrid Category:Human
    Hybrids`.
    """

    def test_a_category_link_leaves_nothing_behind(self):
        self.assertEqual(cu.strip_markup("[[Category:Human Hybrids]]"), "")

    def test_a_category_link_does_not_glue_itself_to_the_text(self):
        self.assertEqual(
            cu.strip_markup("Human hybrid[[Category:Human Hybrids]]"),
            "Human hybrid")

    def test_a_sort_keyed_category_link_is_dropped_whole(self):
        self.assertEqual(
            cu.strip_markup("Gary[[Category:Gary (Earth-6160)/Items|Gary]]"),
            "Gary")

    def test_the_namespace_is_matched_case_insensitively(self):
        self.assertEqual(cu.strip_markup("x[[category:Battlesuits]]"), "x")

    def test_a_colon_prefixed_category_keeps_its_label(self):
        """`[[:Category:X]]` is a REFERENCE, not filing.

        The leading colon is what separates "put this page in that category"
        from "link to that category". 62 lines survived the first fix in this
        form, and they carry real content: `Relatives: :Category:Anu Family`
        should read `Anu Family`, not vanish and leave the field empty.
        """
        self.assertEqual(cu.strip_markup("Relatives: [[:Category:Anu Family]]"),
                         "Relatives: Anu Family")

    def test_a_colon_prefixed_category_with_a_label_keeps_the_label(self):
        self.assertEqual(cu.strip_markup("[[:Category:Asia|Asia]]: China"),
                         "Asia: China")

    def test_filing_and_reference_forms_are_told_apart(self):
        """One drops, the other stays, in the same string."""
        self.assertEqual(
            cu.strip_markup("Kin: [[:Category:Tiamat Family]][[Category:Gods]]"),
            "Kin: Tiamat Family")

    def test_a_seealso_template_carries_no_prose(self):
        """The last 42 leaks were `{{Seealso|:Category: Argentinians}}`.

        Not a wikilink at all, so neither category rule could reach it -
        unwrap_templates kept the last positional argument and turned a
        cross-reference into the words "Category: Argentinians" at the head
        of Argentina's Residents list.
        """
        out = cu.strip_markup(
            "Residents: {{Seealso|:Category: Argentinians}}\n* Black Tarantula")
        self.assertNotIn("Category", out)
        self.assertIn("Black Tarantula", out)

    def test_an_ordinary_link_that_merely_starts_with_the_word_survives(self):
        self.assertEqual(cu.strip_markup("[[Categorical Imperative]]"),
                         "Categorical Imperative")


class PageTitleIsKept(unittest.TestCase):
    """The headline is the CurrentAlias, and the page title is the loss.

    Fandom titles Earth-616 pages by real name and puts the famous name in
    CurrentAlias, so the headline rule is right: 90.9% of character pages
    disagree with their title, and it is the alias people type. But the title
    is a name too, and curation dropped it.

    Jean Grey is the case that proves it. Her page is `Jean Grey (Earth-616)`,
    224 KB, headlined `Phoenix`, and her Full name is the married form
    `Jean Elaine Grey-Summers` - four tokens where the query is two, which is
    past what the resolver will bridge. Nothing in the record said "Jean Grey",
    so the biggest record about her was unreachable by her name.
    """

    def record(self, title, alias, name):
        wt = _infobox(alias=alias, reality="Earth-616", name=name)
        return cu.curate_character({"title": title, "wikitext": wt})

    def test_page_title_is_emitted(self):
        text = self.record("Jean Grey (Earth-616)", "Phoenix",
                           "Jean Elaine Grey-Summers")
        self.assertIn("Page: Jean Grey (Earth-616)", text)

    def test_headline_is_still_the_current_alias(self):
        text = self.record("Jean Grey (Earth-616)", "Phoenix",
                           "Jean Elaine Grey-Summers")
        self.assertEqual(text.split(chr(10))[0], "Phoenix")

    def test_page_line_precedes_the_history(self):
        """names_of() stops reading at History:, so the line must come first."""
        text = self.record("Jean Grey (Earth-616)", "Phoenix",
                           "Jean Elaine Grey-Summers")
        lines = text.split(chr(10))
        self.assertLess(lines.index("Page: Jean Grey (Earth-616)"),
                        lines.index("History:"))

    def test_page_title_kept_when_it_matches_the_alias(self):
        """No special case: a page titled like its alias still records it."""
        text = self.record("Wolverine (Earth-616)", "Wolverine", "James Howlett")
        self.assertIn("Page: Wolverine (Earth-616)", text)


class EmptyAliasFallsBackToTitle(unittest.TestCase):
    """FINDING 3 2026-09-02: a CurrentAlias that strips to nothing (its value
    is an HTML comment, e.g. `<!-- Psykos -->`) made `display` empty, and the
    record came out as `"\\nKind: character\\nPage: ...\\n..."`. Whatever
    later `.strip()`s the record (curated/ is re-read for training and for
    the index) then promotes `Kind: character` to be line 0. 4 records
    (docs 29759, 40982, 75697, 85706) were indexed with the literal headline
    `Kind: character`, and `frank mcgee` / `psykos` resolved to them."""

    def record(self, title, alias, name):
        wt = _infobox(alias=alias, reality="Earth-616", name=name)
        return cu.curate_character({"title": title, "wikitext": wt})

    def test_a_current_alias_that_strips_to_nothing_falls_back_to_the_title(self):
        text = self.record("Psykos (Earth-616)", "<!-- Psykos -->", "Sarah Vale")
        self.assertEqual(text.split(chr(10))[0], "Psykos (Earth-616)")
        self.assertNotEqual(text.split(chr(10))[0], "Kind: character")

    def test_a_present_and_usable_alias_is_unaffected(self):
        """Not a regression on the ordinary case: a real alias still wins."""
        text = self.record("Stephen Strange (Earth-616)", "Doctor Strange",
                           "Stephen Strange")
        self.assertEqual(text.split(chr(10))[0], "Doctor Strange")


class Codenames(unittest.TestCase):
    """The name people actually type lives in `Codenames`, which curation
    never read - `who is beast` could not reach Henry McCoy's 105,792-character
    record because "Beast" appeared in no field of it."""

    def test_a_plain_codename(self):
        self.assertEqual(
            cu.codenames({"Codenames": "[[Beast]]{{r|X-Men Vol 1 1}}"}),
            "Beast")

    def test_a_bulleted_codename(self):
        self.assertEqual(
            cu.codenames({"Codenames": "* [[Power Man]]{{r|Power Man Vol 1 17}}"}),
            "Power Man")

    def test_several_codenames_are_joined(self):
        self.assertEqual(
            cu.codenames({"Codenames": "* [[Prowler]]\n* [[Spider-Men]]"}),
            "Prowler; Spider-Men")

    def test_editorial_names_are_included(self):
        # Miguel O'Hara carries "Spider-Man 2099" here and nowhere else.
        self.assertEqual(
            cu.codenames({"EditorialNames": "* [[Spider-Man 2099]]"}),
            "Spider-Man 2099")

    def test_a_publication_title_is_dropped(self):
        # EditorialNames also carries comic titles. Indexing one as a
        # character name would answer a question about that comic with a
        # character card.
        self.assertEqual(
            cu.codenames({"EditorialNames": "* [[2020 Machine Man Vol 1]]"}),
            "")

    def test_a_publication_title_is_dropped_from_a_mixed_list(self):
        self.assertEqual(
            cu.codenames({"EditorialNames":
                          "* [[Spider-Man 2099]]\n* [[Adam: Legend of the Blue Marvel Vol 1]]"}),
            "Spider-Man 2099")

    def test_both_fields_merge(self):
        self.assertEqual(
            cu.codenames({"Codenames": "[[Beast]]",
                          "EditorialNames": "* [[Beast (comics)]]"}),
            "Beast; Beast (comics)")

    def test_a_duplicate_appears_once(self):
        self.assertEqual(
            cu.codenames({"Codenames": "[[Beast]]",
                          "EditorialNames": "* [[Beast]]"}),
            "Beast")

    def test_neither_field_gives_nothing(self):
        self.assertEqual(cu.codenames({"Name": "Henry McCoy"}), "")

    def test_an_empty_field_gives_nothing(self):
        # A field can be present and strip to nothing - its value is an HTML
        # comment. curate_character already guards CurrentAlias this way.
        self.assertEqual(cu.codenames({"Codenames": "<!-- none -->"}), "")


class CodenameLine(unittest.TestCase):
    def record(self, title, wikitext):
        return cu.curate_character({"title": title, "wikitext": wikitext})

    def test_a_codename_reaches_the_record(self):
        wt = ("{{Marvel Database:Character Template\n"
              "| CurrentAlias = Chairman\n"
              "| Codenames = [[Beast]]{{r|X-Men Vol 1 1}}\n"
              "| Aliases = * Beastmaster\n"
              "| History = He was born in Illinois.\n"
              "}}")
        text = self.record("Henry McCoy (Earth-616)", wt)
        self.assertIn("Codename: Beast", text)

    def test_it_sits_with_the_other_names(self):
        # Directly after `Also known as`, so a reader sees the names together
        # and names_of() finds them in one pass.
        wt = ("{{Marvel Database:Character Template\n"
              "| CurrentAlias = Chairman\n"
              "| Codenames = [[Beast]]\n"
              "| History = x\n"
              "}}")
        lines = [l.split(":")[0] for l in self.record("H (Earth-616)", wt).split("\n")]
        self.assertLess(lines.index("Also known as"), lines.index("Codename"))

    def test_editorial_names_alone_still_emit_a_codename(self):
        # Miguel O'Hara has no Codenames field at all, so the line must not
        # be gated on that key being present.
        wt = ("{{Marvel Database:Character Template\n"
              "| CurrentAlias = Spider-Man\n"
              "| EditorialNames = * [[Spider-Man 2099]]\n"
              "| History = x\n"
              "}}")
        self.assertIn("Codename: Spider-Man 2099",
                      self.record("Miguel O'Hara (Earth-928)", wt))

    def test_no_codename_no_line(self):
        wt = ("{{Marvel Database:Character Template\n"
              "| CurrentAlias = Spider-Man\n"
              "| History = x\n"
              "}}")
        self.assertNotIn("Codename:", self.record("P (Earth-616)", wt))

    def test_a_publication_does_not_reach_the_record(self):
        wt = ("{{Marvel Database:Character Template\n"
              "| CurrentAlias = Machine Man\n"
              "| EditorialNames = * [[2020 Machine Man Vol 1]]\n"
              "| History = x\n"
              "}}")
        self.assertNotIn("Codename:", self.record("Aaron Stack (Earth-616)", wt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
