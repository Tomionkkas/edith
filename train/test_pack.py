"""Tests for tokenized shard packing.

Packing is where a silent bug is most expensive: it runs once, produces a
binary blob, and every training step afterwards reads it. A wrong dtype, a
missing document separator, or a train/val split that leaks would all look
fine and quietly ruin the run.

Run: py train/test_pack.py
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "pack", Path(__file__).resolve().parent / "pack.py")
pk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pk)


class FakeTokenizer:
    """One token per word, ids offset so they never collide with EOS."""
    def encode(self, text):
        return [10 + (hash(w) % 1000) for w in text.split()]


EOS = 1


class Encoding(unittest.TestCase):
    def setUp(self):
        self.sp = FakeTokenizer()

    def test_produces_uint16(self):
        arr = pk.encode_documents(["alpha beta"], self.sp, EOS)
        self.assertEqual(arr.dtype, np.uint16)

    def test_appends_eos_after_each_document(self):
        arr = pk.encode_documents(["a b", "c d"], self.sp, EOS)
        self.assertEqual(int(arr[2]), EOS)
        self.assertEqual(int(arr[-1]), EOS)
        self.assertEqual(int((arr == EOS).sum()), 2)

    def test_length_is_tokens_plus_one_eos_per_document(self):
        arr = pk.encode_documents(["a b c", "d"], self.sp, EOS)
        self.assertEqual(len(arr), 3 + 1 + 1 + 1)

    def test_empty_document_is_skipped(self):
        arr = pk.encode_documents(["a b", "", "   ", "c"], self.sp, EOS)
        self.assertEqual(int((arr == EOS).sum()), 2)

    def test_no_documents_yields_empty_array(self):
        arr = pk.encode_documents([], self.sp, EOS)
        self.assertEqual(len(arr), 0)
        self.assertEqual(arr.dtype, np.uint16)


class TrainValSplit(unittest.TestCase):
    def test_split_is_disjoint(self):
        docs = [f"doc {i}" for i in range(1000)]
        tr, va = pk.split_train_val(docs, val_every=50)
        self.assertEqual(len(tr) + len(va), 1000)
        self.assertEqual(set(tr) & set(va), set())

    def test_val_fraction_matches_val_every(self):
        docs = [f"doc {i}" for i in range(1000)]
        _, va = pk.split_train_val(docs, val_every=50)
        self.assertEqual(len(va), 20)

    def test_split_is_deterministic(self):
        docs = [f"doc {i}" for i in range(500)]
        self.assertEqual(pk.split_train_val(docs, val_every=25),
                         pk.split_train_val(docs, val_every=25))

    def test_val_is_strided_not_a_contiguous_tail(self):
        """A tail split would put one whole category in val for our corpus."""
        docs = [f"doc {i}" for i in range(1000)]
        _, va = pk.split_train_val(docs, val_every=50)
        idx = [int(d.split()[1]) for d in va]
        self.assertLess(min(idx), 100)
        self.assertGreater(max(idx), 900)


class ShardWriter(unittest.TestCase):
    def test_appends_across_calls_and_reads_back(self):
        dest = Path(tempfile.mkdtemp()) / "out.bin"
        w = pk.ShardWriter(dest)
        w.write(np.array([1, 2, 3], dtype=np.uint16))
        w.write(np.array([4, 5], dtype=np.uint16))
        w.close()
        got = np.fromfile(dest, dtype=np.uint16)
        self.assertEqual(got.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(w.total, 5)

    def test_written_file_size_is_two_bytes_per_token(self):
        dest = Path(tempfile.mkdtemp()) / "out.bin"
        w = pk.ShardWriter(dest)
        w.write(np.arange(100, dtype=np.uint16))
        w.close()
        self.assertEqual(dest.stat().st_size, 200)


class Manifest(unittest.TestCase):
    def test_records_and_reloads_completed_shards(self):
        d = Path(tempfile.mkdtemp())
        m = pk.Manifest(d / "m.json")
        self.assertFalse(m.done("shard0"))
        m.mark("shard0", 123)
        m2 = pk.Manifest(d / "m.json")
        self.assertTrue(m2.done("shard0"))
        self.assertEqual(m2.tokens("shard0"), 123)

    def test_unknown_shard_is_not_done(self):
        m = pk.Manifest(Path(tempfile.mkdtemp()) / "m.json")
        self.assertFalse(m.done("never-seen"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
