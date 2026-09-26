"""One wiki page, one record.

MEASURED 2026-09-26: 52 pages were curated twice, 104 records, 0.05% of the
corpus. `who is iron man` read confidence 1.001 against Anthony Stark's own
twin and offered a menu, and `who is doctor doom` did the same.

The cause was NOT a resumed crawl appending a range twice, which is what the
alphabetical clustering of the titles suggested. Every raw file holds unique
titles. The wiki files a page under more than one CATEGORY - All-Black is a
character and an item, Arakko a character and a location, American Nightmare
an event and a story arc - so two phases crawl it and two phases curate it.

    events + story_arcs   25        characters + patch       4
    characters + items    10        items + locations        2
    characters + locations 9        characters + teams       1
                                    items + teams            1

`run_phase()` is per-phase and cannot know what another phase claimed, so
this is a pass over the finished files rather than a rule inside curation.
"""
import unittest
import importlib.util
import sys
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("curate", ROOT / "crawl/curate.py")
C = importlib.util.module_from_spec(spec)
sys.modules["curate"] = C
spec.loader.exec_module(C)

SEP = "=" * 60


def block(page, kind, pad=0):
    return (f"{page}\nKind: {kind}\nPage: {page}\n"
            f"History:\n{'body ' * (pad + 1)}").rstrip()


def write(folder, name, blocks):
    (folder / name).write_text(
        "".join(b + "\n" + SEP + "\n\n" for b in blocks), encoding="utf-8")


def read_pages(folder, name):
    text = (folder / name).read_text(encoding="utf-8")
    return [b.strip("\n") for b in text.split(SEP) if b.strip()]


class DedupePages(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_richer_record_is_the_one_kept(self):
        """Size is the proxy for "the template that matched": the losing
        copies of Doom and Stark were pre-4.11 curations with no Codename
        line at all, and the winner carried every name they did."""
        write(self.dir, "characters.txt", [block("Arakko (Earth-616)", "character", pad=40)])
        write(self.dir, "locations.txt", [block("Arakko (Earth-616)", "location", pad=5)])
        dropped = C.dedupe_pages(self.dir, log=lambda *a: None)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(read_pages(self.dir, "characters.txt")), 1)
        self.assertEqual(read_pages(self.dir, "locations.txt"), [])

    def test_a_page_in_one_file_only_is_untouched(self):
        write(self.dir, "characters.txt",
              [block("Wolverine", "character", pad=3),
               block("Storm", "character", pad=3)])
        before = (self.dir / "characters.txt").read_bytes()
        self.assertEqual(C.dedupe_pages(self.dir, log=lambda *a: None), 0)
        self.assertEqual((self.dir / "characters.txt").read_bytes(), before,
                         "a file with nothing to drop must not be rewritten")

    def test_the_separator_format_round_trips(self):
        """The corpus is parsed by splitting on the separator. A rewrite that
        changed the trailing blank line would move every record boundary in
        the file."""
        write(self.dir, "a.txt", [block("X", "character", pad=9)])
        write(self.dir, "b.txt", [block("X", "character", pad=1),
                                  block("Y", "character", pad=1)])
        C.dedupe_pages(self.dir, log=lambda *a: None)
        text = (self.dir / "b.txt").read_text(encoding="utf-8")
        self.assertTrue(text.endswith(SEP + "\n\n"))
        self.assertEqual(read_pages(self.dir, "b.txt"), [block("Y", "character", pad=1)])

    def test_running_it_twice_changes_nothing(self):
        """It runs at the end of every re-curation, so it has to be a
        no-op on a corpus that is already clean."""
        write(self.dir, "characters.txt", [block("Arakko", "character", pad=40)])
        write(self.dir, "items.txt", [block("Arakko", "item", pad=5)])
        self.assertEqual(C.dedupe_pages(self.dir, log=lambda *a: None), 1)
        after = {f.name: f.read_bytes() for f in self.dir.glob("*.txt")}
        self.assertEqual(C.dedupe_pages(self.dir, log=lambda *a: None), 0)
        self.assertEqual({f.name: f.read_bytes() for f in self.dir.glob("*.txt")},
                         after)

    def test_a_record_with_no_page_line_is_left_alone(self):
        """Issue records carry no `Page:` line - 72,295 of them - and two
        different comics may share a headline. Nothing here may touch them."""
        issue = "Some Comic Vol 1 1 is a Marvel comic.\nKind: issue\nHistory:\nbody"
        write(self.dir, "comics.txt", [issue, issue])
        self.assertEqual(C.dedupe_pages(self.dir, log=lambda *a: None), 0)
        self.assertEqual(len(read_pages(self.dir, "comics.txt")), 2)

    def test_ties_are_broken_the_same_way_every_time(self):
        """Amara Aquilla's two copies are byte-identical in length. A rule
        that picked by dict order would produce a different corpus on a
        different machine."""
        write(self.dir, "b.txt", [block("Amara", "character", pad=7)])
        write(self.dir, "a.txt", [block("Amara", "character", pad=7)])
        C.dedupe_pages(self.dir, log=lambda *a: None)
        self.assertEqual(len(read_pages(self.dir, "a.txt")), 1)
        self.assertEqual(read_pages(self.dir, "b.txt"), [])


if __name__ == "__main__":
    unittest.main()
