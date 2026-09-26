"""The image join, and why it is keyed the way it is.

The join is the product: the picture is keyed on the record that ANSWERED,
so `who is spider-man 2099` shows Miguel O'Hara and not Peter Parker.
"""
import unittest
import importlib.util
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


I = _load("images", "retrieve/images.py")

CHARACTER = ("Wolverine\n"
             "Kind: character\n"
             "Page: James Howlett (Earth-616)\n"
             "First appearance: Incredible Hulk Vol 1 180\n"
             "History:\nbody")
ISSUE = ("Incredible Hulk Vol 1 180 is a Marvel comic. It was published on "
         "January 1, 1974.\n"
         "Kind: issue\n"
         "History:\nbody")


class TitleOf(unittest.TestCase):
    def test_a_record_with_a_page_line_uses_it(self):
        self.assertEqual(I.title_of(CHARACTER, "Wolverine"),
                         "James Howlett (Earth-616)")

    def test_an_issue_has_no_page_line_and_uses_untitled(self):
        """MEASURED FAILURE 2026-09-26. Keying every record on its `Page:`
        line scored 55.98% coverage and looked like a data problem. It was a
        join problem: issue records carry no `Page:` line at all, so all
        72,295 of them fell through to a PROSE headline ("Incredible Hulk Vol
        1 180 is a Marvel comic. It was published on...") and matched nothing.
        `resolve.untitled()` already solves this and must stay the rule.
        """
        headline = ISSUE.split("\n")[0]
        self.assertEqual(I.title_of(ISSUE, headline),
                         "Incredible Hulk Vol 1 180")


class UrlsFor(unittest.TestCase):
    def setUp(self):
        self.images = {
            "James Howlett (Earth-616)": "Wolverine Vol 8 22.jpg",
            "Incredible Hulk Vol 1 180": "Incredible Hulk Vol 1 180.jpg",
        }

    def test_the_records_own_art_comes_first(self):
        urls = I.urls_for(CHARACTER, "Wolverine", self.images)
        self.assertTrue(urls[0].startswith(I.CDN))
        # md5("Wolverine_Vol_8_22.jpg") -> the shard it is stored under.
        self.assertIn("/Wolverine_Vol_8_22.jpg/revision/latest", urls[0])

    def test_first_appearance_is_the_fallback(self):
        """A record with no art of its own still arrives with the issue it
        debuted in - the comics tier is 99.997% covered, which is what makes
        this rung worth 4.37% of all records."""
        bare = CHARACTER.replace("Page: James Howlett (Earth-616)",
                                 "Page: Nobody (Earth-616)")
        urls = I.urls_for(bare, "Nobody", self.images)
        self.assertEqual(len(urls), 1)
        self.assertIn("/Incredible_Hulk_Vol_1_180.jpg/revision/latest", urls[0])

    def test_both_rungs_are_offered_when_both_exist(self):
        """The browser needs the second rung even when the first resolves:
        some files are served under a different name and only fail on load."""
        self.assertEqual(len(I.urls_for(CHARACTER, "Wolverine", self.images)), 2)

    def test_nothing_is_an_empty_chain_not_a_crash(self):
        """4.79% of records have no art anywhere - mostly Wikipedia-sourced
        articles and pages whose Image field is empty on the wiki itself.
        The page draws a plate; this must not raise."""
        self.assertEqual(
            I.urls_for("X\nKind: article\nHistory:\nbody", "X", {}), [])

    def test_a_first_appearance_list_takes_the_first_entry(self):
        """Fields are joined with "; " corpus-wide."""
        many = CHARACTER.replace(
            "First appearance: Incredible Hulk Vol 1 180",
            "First appearance: Incredible Hulk Vol 1 180; X-Men Vol 1 94")
        bare = many.replace("Page: James Howlett (Earth-616)",
                            "Page: Nobody (Earth-616)")
        urls = I.urls_for(bare, "Nobody", self.images)
        self.assertIn("/Incredible_Hulk_Vol_1_180.jpg/revision/latest", urls[0])

    def test_a_name_with_an_apostrophe_survives_quoting(self):
        images = {"Miguel O'Hara (Earth-928)": "Spider-Man 2099 #1.jpg"}
        record = "Spider-Man 2099\nKind: character\nPage: Miguel O'Hara (Earth-928)\n"
        urls = I.urls_for(record, "Spider-Man 2099", images)
        self.assertEqual(len(urls), 1)
        self.assertNotIn(" ", urls[0])


class CdnPath(unittest.TestCase):
    def test_the_shard_is_the_md5_of_the_underscored_name(self):
        """MEASURED 2026-09-26 against what Special:FilePath redirects to:
        `Wolverine Vol 8 22 Virgin Variant.jpg` is stored at `b/bb/`, and
        `Alpha Flight Vol 2 16.jpg` at `c/c5/`. Getting this wrong 404s every
        image, and a browser shows it as no art at all."""
        url = I.url_for_name("Wolverine Vol 8 22 Virgin Variant.jpg")
        self.assertIn("/images/b/bb/Wolverine_Vol_8_22_Virgin_Variant.jpg/", url)
        other = I.url_for_name("Alpha Flight Vol 2 16.jpg")
        self.assertIn("/images/c/c5/Alpha_Flight_Vol_2_16.jpg/", other)

    def test_it_does_not_go_through_the_wiki_host(self):
        """marvel.fandom.com refuses browser-initiated hotlinks; the CDN
        serves them. curl gets both, which is why this was invisible until
        the page ran in a browser."""
        self.assertNotIn("fandom.com", I.url_for_name("X.jpg"))


class Load(unittest.TestCase):
    def test_a_missing_sidecar_is_empty_not_fatal(self):
        """A user who installed the package without the sidecar gets an app
        with no art, not a traceback."""
        self.assertEqual(I.load(ROOT / "retrieve" / "no-such-file.json.gz"), {})

    def test_the_shipped_sidecar_loads_and_is_big(self):
        shipped = I.load()
        if not I.IMAGES_PATH.exists():
            self.skipTest("sidecar not built")
        self.assertGreater(len(shipped), 180_000)
        self.assertIn("James Howlett (Earth-616)", shipped)


if __name__ == "__main__":
    unittest.main()
