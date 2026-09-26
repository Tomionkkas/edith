"""The legacy sidecar: who has GONE BY a name without being it."""
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


R = _load("resolve", "retrieve/resolve.py")
B = _load("build_legacy", "retrieve/build_legacy.py")


class FakeIndex:
    def __init__(self, records):
        self.docs = list(range(len(records)))
        self.headlines = [r.split("\n")[0] for r in records]
        self._records = records

    def text(self, doc_id):
        return self._records[doc_id]

    def __len__(self):
        return len(self.docs)


def rec(headline, full="", codename="", aliases="", pad=10):
    lines = [headline, f"Page: {headline}"]
    if full:
        lines.append(f"Full name: {full}")
    if codename:
        lines.append(f"Codename: {codename}")
    if aliases:
        lines.append(f"Other aliases: {aliases}")
    lines += ["Reality: Earth-616", "Created by: Someone",
              "First appearance: A Comic Vol 1 1", "History:", "body " * pad]
    return "\n".join(lines)


class Build(unittest.TestCase):
    def test_a_codename_holder_is_recorded_under_that_name(self):
        """She-Hulk has gone by Hulk. That is the whole feature."""
        index = FakeIndex([rec("She-Hulk", full="Jennifer Walters",
                               codename="Hulk", pad=400)])
        got = B.build(index, R)
        self.assertEqual(got["hulk"], [[0, R.PROV_CODENAME]])

    def test_the_name_holder_itself_is_not_recorded(self):
        """Bruce Banner IS the Hulk, so he belongs to the variants list, not
        to "others who have gone by this name". Listing him in both is how a
        picker stops being a choice."""
        index = FakeIndex([rec("Hulk", full="Bruce Banner", pad=800)])
        self.assertEqual(B.build(index, R), {})

    def test_the_biggest_record_comes_first(self):
        index = FakeIndex([
            rec("Small One", codename="Hulk", pad=10),
            rec("Big One", codename="Hulk", pad=900),
        ])
        got = B.build(index, R)
        self.assertEqual([doc for doc, _prov in got["hulk"]], [1, 0])

    def test_an_alias_is_kept_and_marked_as_one(self):
        index = FakeIndex([rec("Someone", aliases="Hulk", pad=100)])
        self.assertEqual(B.build(index, R)["hulk"], [[0, R.PROV_ALIAS]])

    def test_the_list_is_capped(self):
        """24 is a screenful. Without a cap a common word carries thousands
        of rows and the sidecar stops being small."""
        index = FakeIndex([rec(f"Holder {i}", codename="Hulk", pad=i + 1)
                           for i in range(30)])
        self.assertEqual(len(B.build(index, R)["hulk"]), B.PER_NAME)


class Shipped(unittest.TestCase):
    def test_the_shipped_sidecar_knows_the_famous_legacies(self):
        import gzip
        import json
        if not B.OUT.exists():
            self.skipTest("sidecar not built")
        legacy = json.loads(gzip.decompress(B.OUT.read_bytes()).decode("utf-8"))
        for name in ("hulk", "spider man", "iron man", "captain america"):
            self.assertIn(name, legacy, name)
            self.assertGreater(len(legacy[name]), 1, name)


if __name__ == "__main__":
    unittest.main()
