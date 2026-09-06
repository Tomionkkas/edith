"""Tests for the recursive Wikipedia category walk (offline, fake API).

The walk is the only genuinely tricky part of the Wikipedia crawler: the real
category graph contains cycles (A -> B -> A), so a naive BFS never terminates.
These use an injected fetch function so the logic is tested without network.

Run: py crawl/test_wiki_walk.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "crawl_wiki", Path(__file__).resolve().parent / "crawl_wiki.py")
cw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw)


# fake category graph: root -> sub -> subsub, plus a cycle back to root
GRAPH = {
    "Category:Root": [("Alpha", "page"), ("Category:Sub", "subcat")],
    "Category:Sub": [("Beta", "page"), ("Category:SubSub", "subcat"),
                     ("Category:Root", "subcat")],          # <-- cycle
    "Category:SubSub": [("Gamma", "page"), ("Alpha", "page")],  # <-- duplicate page
    "Category:Deep1": [("D1", "page"), ("Category:Deep2", "subcat")],
    "Category:Deep2": [("D2", "page"), ("Category:Deep3", "subcat")],
    "Category:Deep3": [("D3", "page")],
}


def fake_fetch(cat):
    """Stand-in for one paged categorymembers call."""
    return GRAPH.get(cat, [])


class CategoryWalk(unittest.TestCase):
    def test_collects_pages_from_nested_subcategories(self):
        pages, cats = cw.walk_categories(fake_fetch, ["Category:Root"], max_depth=5, allow=None)
        self.assertIn("Alpha", pages)
        self.assertIn("Beta", pages)
        self.assertIn("Gamma", pages)

    def test_terminates_on_cyclic_category_graph(self):
        """Root -> Sub -> Root would loop forever without a seen-set."""
        pages, cats = cw.walk_categories(fake_fetch, ["Category:Root"], max_depth=5, allow=None)
        self.assertEqual(sorted(cats), ["Category:Root", "Category:Sub", "Category:SubSub"])

    def test_deduplicates_pages_seen_in_two_categories(self):
        pages, _ = cw.walk_categories(fake_fetch, ["Category:Root"], max_depth=5, allow=None)
        self.assertEqual(len(pages), len(set(pages)))
        self.assertEqual(pages.count("Alpha"), 1)

    def test_respects_max_depth(self):
        """depth 1 reaches Deep2's pages but must not descend into Deep3."""
        pages, cats = cw.walk_categories(fake_fetch, ["Category:Deep1"], max_depth=1, allow=None)
        self.assertIn("D1", pages)
        self.assertIn("D2", pages)
        self.assertNotIn("D3", pages)

    def test_returns_sorted_stable_page_list(self):
        a, _ = cw.walk_categories(fake_fetch, ["Category:Root"], max_depth=5, allow=None)
        b, _ = cw.walk_categories(fake_fetch, ["Category:Root"], max_depth=5, allow=None)
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))

    def test_unknown_root_yields_nothing_without_crashing(self):
        pages, cats = cw.walk_categories(fake_fetch, ["Category:DoesNotExist"], max_depth=3, allow=None)
        self.assertEqual(pages, [])



class SubcategoryGate(unittest.TestCase):
    """The topical gate stops the walk leaking into all of Wikipedia."""

    def test_descends_into_marvel_named_subcategory(self):
        graph = {"Category:Root": [("Category:Marvel villains", "subcat")],
                 "Category:Marvel villains": [("Doom", "page")]}
        pages, _ = cw.walk_categories(lambda c: graph.get(c, []), ["Category:Root"],
                                      max_depth=3)
        self.assertIn("Doom", pages)

    def test_skips_offtopic_subcategory(self):
        graph = {"Category:Root": [("Category:American culture", "subcat")],
                 "Category:American culture": [("Jazz", "page")]}
        pages, cats = cw.walk_categories(lambda c: graph.get(c, []), ["Category:Root"],
                                         max_depth=3)
        self.assertNotIn("Jazz", pages)
        self.assertNotIn("Category:American culture", cats)


if __name__ == "__main__":
    unittest.main(verbosity=2)

