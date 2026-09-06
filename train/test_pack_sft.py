"""Tests for stage-3 packing and masked loss.

Stage 3 must score ONLY the assistant turn. Score the whole sequence and the
model learns to generate `Context:` blocks — which is precisely the
hallucination retrieval exists to prevent. The mask is the thing that stops
it, and a mask that is silently wrong looks exactly like one that is right.

Run: py train/test_pack_sft.py
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pack_sft", _here / "pack_sft.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

spec2 = importlib.util.spec_from_file_location("trainer", _here / "trainer.py")
tr = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(tr)


class FakeTokenizer:
    """One id per word; ids never collide with EOS."""
    def encode(self, text):
        return [10 + (abs(hash(w)) % 900) for w in text.split()]


EOS = 1


class Encoding(unittest.TestCase):
    def setUp(self):
        self.sp = FakeTokenizer()
        self.ex = {"context": "Context:\nA B", "user": "C D", "assistant": "E F"}

    def test_tokens_and_mask_are_the_same_length(self):
        toks, mask = ps.encode_example(self.ex, self.sp, EOS)
        self.assertEqual(len(toks), len(mask))

    def test_prompt_is_masked_out(self):
        toks, mask = ps.encode_example(self.ex, self.sp, EOS)
        self.assertEqual(int(mask[0]), 0)
        self.assertEqual(int(mask[1]), 0)

    def test_answer_is_masked_in(self):
        toks, mask = ps.encode_example(self.ex, self.sp, EOS)
        self.assertEqual(int(mask[-1]), 1)   # EOS: the model must learn to stop
        self.assertEqual(int(mask[-2]), 1)

    def test_some_of_each(self):
        _, mask = ps.encode_example(self.ex, self.sp, EOS)
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), len(mask))

    def test_ends_with_eos(self):
        toks, _ = ps.encode_example(self.ex, self.sp, EOS)
        self.assertEqual(int(toks[-1]), EOS)

    def test_dtypes_are_compact(self):
        toks, mask = ps.encode_example(self.ex, self.sp, EOS)
        self.assertEqual(toks.dtype, np.uint16)
        self.assertEqual(mask.dtype, np.uint8)

    def test_chitchat_without_context_still_masks_the_prompt(self):
        ex = {"context": "", "user": "hi", "assistant": "Hello there friend"}
        toks, mask = ps.encode_example(ex, self.sp, EOS)
        self.assertEqual(int(mask[0]), 0)
        self.assertEqual(int(mask[-1]), 1)

    def test_masked_fraction_is_mostly_prompt_for_a_long_context(self):
        ex = {"context": "Context:\n" + "w " * 200, "user": "q", "assistant": "a"}
        _, mask = ps.encode_example(ex, self.sp, EOS)
        self.assertLess(mask.mean(), 0.1)


class MaskedLoss(unittest.TestCase):
    def test_equals_plain_loss_when_everything_is_unmasked(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 5, 11)
        y = torch.randint(0, 11, (2, 5))
        mask = torch.ones(2, 5)
        plain = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 11), y.reshape(-1))
        self.assertAlmostEqual(float(tr.masked_loss(logits, y, mask)),
                               float(plain), places=5)

    def test_ignores_masked_positions(self):
        """Corrupting a masked-out target must not change the loss."""
        torch.manual_seed(0)
        logits = torch.randn(1, 6, 11)
        y = torch.randint(0, 11, (1, 6))
        mask = torch.tensor([[0., 0., 0., 1., 1., 1.]])
        before = float(tr.masked_loss(logits, y, mask))
        y2 = y.clone()
        y2[0, 0] = (int(y2[0, 0]) + 5) % 11
        self.assertAlmostEqual(before, float(tr.masked_loss(logits, y2, mask)),
                               places=6)

    def test_responds_to_unmasked_positions(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 6, 11)
        y = torch.randint(0, 11, (1, 6))
        mask = torch.tensor([[0., 0., 0., 1., 1., 1.]])
        before = float(tr.masked_loss(logits, y, mask))
        y2 = y.clone()
        y2[0, 5] = (int(y2[0, 5]) + 5) % 11
        self.assertNotAlmostEqual(before, float(tr.masked_loss(logits, y2, mask)),
                                  places=4)

    def test_all_masked_batch_does_not_nan(self):
        logits = torch.randn(1, 4, 11)
        y = torch.randint(0, 11, (1, 4))
        loss = tr.masked_loss(logits, y, torch.zeros(1, 4))
        self.assertFalse(torch.isnan(loss).any())


class MaskedLoading(unittest.TestCase):
    def setUp(self):
        d = Path(tempfile.mkdtemp())
        self.tok = d / "sft_train.bin"
        self.msk = d / "sft_train_mask.bin"
        np.arange(5000, dtype=np.uint16).tofile(self.tok)
        (np.arange(5000) % 2).astype(np.uint8).tofile(self.msk)

    def test_returns_x_y_and_mask(self):
        dl = tr.MaskedDataLoader(self.tok, self.msk, 32, 2, "cpu")
        x, y, m = dl.get_batch()
        self.assertEqual(tuple(x.shape), (2, 32))
        self.assertEqual(tuple(m.shape), (2, 32))

    def test_mask_aligns_with_targets_not_inputs(self):
        """y[i] is tokens[i+1], so the mask must be shifted the same way or
        the loss is scored one position off."""
        dl = tr.MaskedDataLoader(self.tok, self.msk, 8, 1, "cpu")
        for _ in range(20):
            x, y, m = dl.get_batch()
            start = int(x[0, 0])
            expected = torch.tensor([(start + 1 + i) % 2 for i in range(8)],
                                    dtype=m.dtype)
            self.assertTrue(torch.equal(m[0], expected))

    def test_rejects_mismatched_lengths(self):
        bad = self.msk.with_name("short_mask.bin")
        np.zeros(10, dtype=np.uint8).tofile(bad)
        with self.assertRaises(ValueError):
            tr.MaskedDataLoader(self.tok, bad, 32, 2, "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
