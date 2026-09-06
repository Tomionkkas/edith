"""Tests for multi-title batch fetching (hits the live Fandom API).

Batching is the difference between a 3-day crawl and a 2-hour one, so these
verify against the real API rather than a mock: the failure mode we care about
(MediaWiki silently normalising a title so it never matches our state key)
only exists on the real service.

Run: py crawl/test_batch_fetch.py
"""
import importlib.util
import time
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "crawl", Path(__file__).resolve().parent / "crawl.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

API = "https://marvel.fandom.com/api.php"
REAL = ["Spider-Man", "Iron Man", "Thor", "Hulk", "Black Widow"]


class BatchFetch(unittest.TestCase):
    def tearDown(self):
        time.sleep(1.0)  # stay polite while the suite runs

    def test_returns_wikitext_for_every_requested_title(self):
        pages, errors = cr.fetch_wikitext_batch(API, REAL)
        self.assertEqual(set(pages) | set(errors), set(REAL))
        self.assertEqual(errors, {})
        for t in REAL:
            self.assertTrue(pages[t].strip(), f"{t} came back empty")

    def test_batch_content_matches_single_fetch(self):
        title = "Iron Man"
        single, err = cr.fetch_wikitext(API, title)
        self.assertIsNone(err)
        pages, _ = cr.fetch_wikitext_batch(API, [title])
        self.assertEqual(pages[title], single)

    def test_results_keyed_by_requested_title_not_normalised(self):
        """MediaWiki turns 'Iron_Man' into 'Iron Man'; state keys must survive."""
        requested = ["Iron_Man"]
        pages, errors = cr.fetch_wikitext_batch(API, requested)
        self.assertIn("Iron_Man", set(pages) | set(errors))
        self.assertTrue(pages.get("Iron_Man", "").strip())

    def test_missing_page_reported_as_error_not_silently_dropped(self):
        bogus = "Zzzz No Such Marvel Page 90210"
        pages, errors = cr.fetch_wikitext_batch(API, [bogus, "Thor"])
        self.assertIn(bogus, errors)
        self.assertNotIn(bogus, pages)
        self.assertIn("Thor", pages)

    def test_batches_against_wikipedia_too(self):
        """Tier A uses the same batcher against a different MediaWiki host."""
        wiki = "https://en.wikipedia.org/w/api.php"
        titles = ["Spider-Man", "Iron Man", "Thanos", "Galactus", "Doctor Doom"]
        pages, errors = cr.fetch_wikitext_batch(wiki, titles)
        self.assertEqual(errors, {})
        self.assertEqual(set(pages), set(titles))
        self.assertGreater(len(pages["Spider-Man"]), 10_000)

    def test_large_batch_of_fifty_returns_all(self):
        titles, _ = cr.enumerate_titles_page(API, "Category:Characters", limit=50)
        self.assertEqual(len(titles), 50)
        pages, errors = cr.fetch_wikitext_batch(API, titles)
        self.assertEqual(len(pages) + len(errors), 50)
        self.assertGreaterEqual(len(pages), 45)


if __name__ == "__main__":
    unittest.main(verbosity=2)
