"""`Category:Events` hides most of its events in SUBCATEGORIES. Phase B3.

FOUND BY TESTING THE FAMOUS EVENTS, 2026-09-20, after a random 260-question
live run under-represented them: `what happened in age of ultron` offered a
picker of ten Ultron characters, because `Age of Ultron (Event)` is not in
the corpus at all.

The crawl was not wrong about what it asked for. `enumerate_titles` sends
`cmtype=page`, `Category:Events` has 379 member pages, and all 379 were
crawled. But that category ALSO has **425 subcategories**, each named after
an event - `Age of Ultron (Event)`, `Avengers vs. X-Men (Event)`,
`Age of Apocalypse (Event)` - and 416 of those names are real article pages.
94 of them were never fetched.

`crawl_wiki.py` already walks `cmtype=page|subcat`; the Fandom crawler is
deliberately flat, and for every other target that is right - Characters,
Comics, Items and Locations put their pages in the category directly. Events
is the one that does not, so the behaviour is opt-in per target rather than
turned on for all of them.

Run: py crawl/test_enumerate_subcats.py
"""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "crawl", Path(__file__).resolve().parent / "crawl.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


class FakeAPI:
    """One paged categorymembers response per call, keyed by cmtype."""

    def __init__(self, pages, subcats):
        self.pages, self.subcats = pages, subcats
        self.calls = []

    def __init_containers__(self, containers):
        self.containers = containers

    def __call__(self, api, params):
        self.calls.append((params.get("cmtitle"), params.get("cmtype")))
        want = params.get("cmtype")
        title = params.get("cmtitle")
        members = []
        if title == "Category:Events" or title == "Category:Characters":
            if "page" in want:
                members += [{"title": t} for t in self.pages]
            if "subcat" in want:
                members += [{"title": "Category:" + t} for t in self.subcats]
        else:
            # a container category being walked one level deep
            members += [{"title": t} for t in
                        getattr(self, "containers", {}).get(title, [])]
        return {"query": {"categorymembers": members}}


class SubcategoryNamesAreTitles(unittest.TestCase):

    def setUp(self):
        self.api = FakeAPI(pages=["Civil War (Event)", "Annihilation (Event)"],
                           subcats=["Age of Ultron (Event)", "Avengers Events"])
        self._real = cr.http_get_json
        cr.http_get_json = self.api

    def tearDown(self):
        cr.http_get_json = self._real

    def test_flat_by_default(self):
        """Every other target puts its pages in the category directly, and
        walking subcategories there would pull in thousands of unrelated
        pages."""
        titles, _ = cr.enumerate_titles("api", "Category:Characters")
        self.assertEqual(titles, ["Civil War (Event)", "Annihilation (Event)"])

    def test_subcategory_names_are_added_when_asked(self):
        titles, _ = cr.enumerate_titles("api", "Category:Events", subcats=True)
        self.assertIn("Age of Ultron (Event)", titles)

    def test_the_category_prefix_is_stripped(self):
        """The subcategory is `Category:Age of Ultron (Event)`; the ARTICLE
        is `Age of Ultron (Event)`. Fetching the former gets a category
        listing, not an event."""
        titles, _ = cr.enumerate_titles("api", "Category:Events", subcats=True)
        self.assertFalse(any(t.startswith("Category:") for t in titles), titles)

    def test_the_member_pages_still_come_too(self):
        titles, _ = cr.enumerate_titles("api", "Category:Events", subcats=True)
        self.assertIn("Civil War (Event)", titles)

    def test_a_title_is_not_listed_twice(self):
        """A page can be both a member and a subcategory name. Fetching it
        twice is a wasted request against a rate-limited API."""
        self.api.subcats.append("Civil War (Event)")
        titles, _ = cr.enumerate_titles("api", "Category:Events", subcats=True)
        self.assertEqual(titles.count("Civil War (Event)"), 1)


class ContainerCategoriesAreWalked(unittest.TestCase):
    """A subcategory whose name ends in "Events" is a CONTAINER of events,
    not an event. `Age of Ultron (Event)` lives inside `Editorial Events`,
    so taking that name as a title fetches nothing - the container has no
    article of its own. Its MEMBERS are the events.

    Measured 2026-09-20: 15 such containers under Category:Events, adding 15
    event pages, including both `Age of Ultron (Event)` and
    `Avengers vs. X-Men (Event)`.

    Only containers are walked. `Category:Civil War II` is a subcategory too,
    and it holds every tie-in issue and character appearance of that event -
    walking it would drag thousands of non-events into the phase.
    """

    def setUp(self):
        self.api = FakeAPI(pages=["Civil War (Event)"],
                           subcats=["Editorial Events", "Civil War II"])
        self.api.containers = {
            "Category:Editorial Events": ["Age of Ultron (Event)"],
            "Category:Civil War II": ["Civil War II Vol 1 1"],
        }
        self._real = cr.http_get_json
        cr.http_get_json = self.api

    def tearDown(self):
        cr.http_get_json = self._real

    def test_a_container_yields_its_members(self):
        titles, _ = cr.enumerate_titles("api", "Category:Events", subcats=True)
        self.assertIn("Age of Ultron (Event)", titles)

    def test_a_non_container_subcategory_is_not_walked(self):
        titles, _ = cr.enumerate_titles("api", "Category:Events", subcats=True)
        self.assertNotIn("Civil War II Vol 1 1", titles)

    def test_the_non_container_name_is_still_a_title(self):
        """`Civil War II` IS an event article - only its members are
        skipped, not the name itself."""
        titles, _ = cr.enumerate_titles("api", "Category:Events", subcats=True)
        self.assertIn("Civil War II", titles)


if __name__ == "__main__":
    unittest.main()
