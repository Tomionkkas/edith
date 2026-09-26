"""What counts as an image, and what does not.

The two cases that cost a measurement each on 2026-09-26 are pinned here:
the cover field is `Image1`, and an EMPTY `| Image =` is not art.
"""
import unittest
import importlib.util
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_images", ROOT / "crawl/build_images.py")
B = importlib.util.module_from_spec(spec)
sys.modules["build_images"] = B
spec.loader.exec_module(B)


class ImageName(unittest.TestCase):
    def test_reads_a_character_image(self):
        w = ("{{Marvel Database:Character Template\n"
             "| Image = Logan.jpg\n| Name = Logan\n}}")
        self.assertEqual(B.image_name(w), "Logan.jpg")

    def test_reads_a_comic_cover_under_Image1(self):
        """The cover field is Image1, NOT Image. A search for `| Image =`
        finds ONE comic in 72,295, which nearly lost the entire tier - and
        that tier is 36% of the corpus."""
        w = ("{{Comic Template\n"
             "| Image1 = Hulk Vol 1 180.jpg\n| Month = January\n}}")
        self.assertEqual(B.image_name(w), "Hulk Vol 1 180.jpg")

    def test_an_empty_field_is_not_an_image(self):
        """13,336 pages carry `| Image =` with nothing after it. A pattern
        whose whitespace class runs past the newline captures the NEXT field
        and invents art: that bug reported 198,556 pages with images when
        185,220 have one."""
        w = ("{{Character Template\n"
             "| Image                   = \n| Name = Aala\n}}")
        self.assertEqual(B.image_name(w), "")

    def test_a_field_holding_only_spaces_is_not_an_image(self):
        w = "{{Character Template\n| Image =    \n| Name = Aala\n}}"
        self.assertEqual(B.image_name(w), "")

    def test_no_field_at_all(self):
        self.assertEqual(B.image_name("Prose about a Wikipedia topic."), "")

    def test_an_absurd_value_is_refused(self):
        """A run-on value is a parse that went wrong, not a filename."""
        self.assertEqual(B.image_name("| Image = " + "x" * 300), "")


class Build(unittest.TestCase):
    def test_builds_a_title_to_filename_map(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            raw = pathlib.Path(tmp)
            (raw / "characters.jsonl").write_text(
                json.dumps({"title": "James Howlett (Earth-616)",
                            "wikitext": "{{T\n| Image = Wolverine.jpg\n}}"})
                + "\n"
                + json.dumps({"title": "Nobody (Earth-616)",
                              "wikitext": "{{T\n| Image = \n}}"}) + "\n",
                encoding="utf-8")
            got = B.build(raw)
        self.assertEqual(got, {"James Howlett (Earth-616)": "Wolverine.jpg"})


if __name__ == "__main__":
    unittest.main()
