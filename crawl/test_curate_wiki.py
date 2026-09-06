"""Tests for Wikipedia article curation (offline).

Wikipedia articles are prose with `== Section ==` headers, not Marvel Database
infoboxes, so they need their own path: trim the sections that carry no
in-universe knowledge, then strip markup.

Run: py crawl/test_curate_wiki.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "curate_wiki", Path(__file__).resolve().parent / "curate_wiki.py")
cw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw)


class SectionTrimming(unittest.TestCase):
    def test_drops_references_section(self):
        wt = ("Intro prose.\n"
              "== References ==\n{{reflist}}\ncitation junk\n"
              "== Legacy ==\nHe inspired clones.")
        out = cw.trim_sections(wt)
        self.assertNotIn("citation junk", out)
        self.assertIn("inspired clones", out)

    def test_drops_external_links_section(self):
        wt = "Intro.\n== External links ==\n* http://example.com\n"
        self.assertNotIn("example.com", cw.trim_sections(wt))

    def test_keeps_publication_history(self):
        """The creator facts live here - dropping this loses Stan Lee."""
        wt = ("Intro.\n== Publication history ==\n"
              "Created by Stan Lee and Steve Ditko.\n"
              "== References ==\njunk")
        out = cw.trim_sections(wt)
        self.assertIn("Stan Lee", out)
        self.assertNotIn("junk", out)

    def test_keeps_creation_section(self):
        wt = ("Intro.\n== Creation ==\nCreated by Stan Lee.\n"
              "== Story arcs ==\nHe fights villains.")
        out = cw.trim_sections(wt)
        self.assertIn("Stan Lee", out)
        self.assertIn("fights villains", out)

    def test_dropped_section_takes_its_subsections_with_it(self):
        wt = ("Intro.\n== Reception ==\nreviews\n=== Awards ===\nprizes\n"
              "== Powers and abilities ==\nHe can fly.")
        out = cw.trim_sections(wt)
        self.assertNotIn("reviews", out)
        self.assertNotIn("prizes", out)
        self.assertIn("He can fly", out)

    def test_keeps_lead_before_any_header(self):
        wt = "Spider-Man is a superhero.\n== References ==\njunk"
        out = cw.trim_sections(wt)
        self.assertIn("Spider-Man is a superhero", out)

    def test_subsection_of_a_kept_section_survives(self):
        wt = ("Intro.\n== Fictional character biography ==\nbio\n"
              "=== Early years ===\nchildhood\n== References ==\njunk")
        out = cw.trim_sections(wt)
        self.assertIn("childhood", out)
        self.assertNotIn("junk", out)


SAMPLE = """{{Infobox comics character
| character_name = Test Hero
| debut = ''Amazing Fantasy'' #15
}}
'''Test Hero''' is a [[superhero]] appearing in {{DC|Kal-El|American}} comic books.

== Publication history ==
Created by [[Stan Lee]] and Steve Ditko, he first appeared in 1962. It's notable.

== Powers and abilities ==
He can climb walls.

== Reception ==
IGN ranked him 20th.

== References ==
{{reflist}}

== External links ==
* [http://example.com Site]
"""


class CurateWikiArticle(unittest.TestCase):
    def setUp(self):
        # min_chars=0: this sample is deliberately small, the length filter is
        # exercised separately below.
        self.text = cw.curate_wiki({"title": "Test Hero", "wikitext": SAMPLE},
                                   min_chars=0)

    def test_starts_with_the_article_title(self):
        self.assertEqual(self.text.split("\n")[0], "Test Hero")

    def test_keeps_knowledge_sections(self):
        self.assertIn("Stan Lee", self.text)
        self.assertIn("climb walls", self.text)

    def test_drops_metadata_sections(self):
        self.assertNotIn("IGN ranked", self.text)
        self.assertNotIn("example.com", self.text)

    def test_strips_wiki_markup(self):
        for marker in ("'''", "[[", "]]", "{{", "}}", "reflist"):
            self.assertNotIn(marker, self.text)

    def test_section_headers_become_plain_lines(self):
        """`== Publication history ==` should read as prose, not markup."""
        self.assertNotIn("==", self.text)
        self.assertIn("Publication history", self.text)

    def test_preserves_apostrophes(self):
        self.assertIn("It's notable", self.text)

    def test_empty_article_yields_empty_string(self):
        self.assertEqual(cw.curate_wiki({"title": "X", "wikitext": ""}), "")

    def test_stub_below_min_chars_is_dropped(self):
        stub = {"title": "Stub", "wikitext": "A tiny article.\n== References ==\njunk"}
        self.assertEqual(cw.curate_wiki(stub, min_chars=400), "")
        self.assertNotEqual(cw.curate_wiki(stub, min_chars=0), "")


class ArticlesDeclareTheirKind(unittest.TestCase):
    """Prose already wins names and already answers questions, so "this is an
    encyclopedia article, not a database record" is worth saying out loud."""

    def test_an_article_says_it_is_an_article(self):
        text = cw.curate_wiki(
            {"title": "Jean Grey",
             "wikitext": "'''Jean Grey''' is a superhero appearing in "
                         "American comic books published by Marvel Comics. "
                         "She is a founding member of the X-Men and one of "
                         "the most powerful mutants in the Marvel Universe, "
                         "known for her telepathic and telekinetic abilities "
                         "and her eventual transformation into the cosmic "
                         "entity known as the Dark Phoenix, a storyline "
                         "considered one of the most influential in the "
                         "history of superhero comics."},
            min_chars=0)
        self.assertIn("Kind: article", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
