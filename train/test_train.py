"""Tests for the training loop's non-GPU logic.

These cover the parts that fail silently: a data loader that misaligns targets
trains a model to predict the wrong token, an LR schedule that is wrong at the
boundaries wastes days, and a checkpoint that does not round-trip makes resume
a lie. All run on CPU in under a second.

Run: py train/test_train.py
"""
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "trainer", Path(__file__).resolve().parent / "trainer.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def _bin(n=10000):
    p = Path(tempfile.mkdtemp()) / "d.bin"
    np.arange(n, dtype=np.uint16).tofile(p)
    return p


class DataLoading(unittest.TestCase):
    def setUp(self):
        self.path = _bin()

    def test_batch_shapes(self):
        dl = tr.DataLoader(self.path, block_size=64, batch_size=4, device="cpu")
        x, y = dl.get_batch()
        self.assertEqual(tuple(x.shape), (4, 64))
        self.assertEqual(tuple(y.shape), (4, 64))

    def test_targets_are_inputs_shifted_by_one(self):
        """The whole objective depends on this. Off by one = training garbage."""
        dl = tr.DataLoader(self.path, block_size=32, batch_size=2, device="cpu")
        x, y = dl.get_batch()
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_dtype_is_long_for_embedding_lookup(self):
        dl = tr.DataLoader(self.path, block_size=16, batch_size=2, device="cpu")
        x, y = dl.get_batch()
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_never_reads_past_the_end(self):
        dl = tr.DataLoader(self.path, block_size=64, batch_size=8, device="cpu")
        for _ in range(200):
            x, y = dl.get_batch()
            self.assertLess(int(y.max()), 10000)

    def test_reports_token_count(self):
        dl = tr.DataLoader(self.path, block_size=16, batch_size=2, device="cpu")
        self.assertEqual(len(dl), 10000)

    def test_handles_corpora_larger_than_int32(self):
        """numpy randint defaults to int32 on Windows; the general tier is
        2.94e9 tokens and overflows it. Marvel (5.5e7) does not, so a smoke
        test on Marvel passes while stage 1 dies at step 1."""
        idx = tr.DataLoader.sample_starts(2_941_946_794, 8)
        self.assertEqual(idx.dtype, np.int64)
        self.assertEqual(len(idx), 8)
        self.assertTrue((idx >= 0).all() and (idx < 2_941_946_794).all())

    def test_sample_starts_spans_the_whole_range(self):
        idx = tr.DataLoader.sample_starts(2_941_946_794, 500)
        self.assertGreater(int(idx.max()), 2_000_000_000)

    def test_rejects_a_file_shorter_than_one_block(self):
        p = Path(tempfile.mkdtemp()) / "tiny.bin"
        np.arange(10, dtype=np.uint16).tofile(p)
        with self.assertRaises(ValueError):
            tr.DataLoader(p, block_size=64, batch_size=2, device="cpu")


class LRSchedule(unittest.TestCase):
    ARGS = dict(warmup=100, max_steps=1000, lr=6e-4, min_lr=6e-5)

    def test_starts_near_zero(self):
        self.assertLess(tr.lr_at(0, **self.ARGS), 1e-5)

    def test_peaks_at_end_of_warmup(self):
        self.assertAlmostEqual(tr.lr_at(100, **self.ARGS), 6e-4, places=7)

    def test_warmup_is_linear(self):
        self.assertAlmostEqual(tr.lr_at(50, **self.ARGS), 3e-4, places=6)

    def test_decays_to_min_at_the_end(self):
        self.assertAlmostEqual(tr.lr_at(1000, **self.ARGS), 6e-5, places=7)

    def test_stays_at_min_past_the_end(self):
        self.assertAlmostEqual(tr.lr_at(5000, **self.ARGS), 6e-5, places=7)

    def test_is_monotonic_after_warmup(self):
        vals = [tr.lr_at(s, **self.ARGS) for s in range(100, 1001, 50)]
        self.assertEqual(vals, sorted(vals, reverse=True))


class Initialisation(unittest.TestCase):
    """A model whose loss does not start near ln(vocab) is already broken.

    PyTorch gives nn.Embedding std=1.0 by default, and lm_head is tied to wte,
    so without explicit init the logits explode and training starts at ~91
    instead of 10.8. A smoke run caught this; these tests keep it caught.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        s = importlib.util.spec_from_file_location(
            "benchmark", Path(__file__).resolve().parent / "benchmark.py")
        cls.b = importlib.util.module_from_spec(s)
        s.loader.exec_module(cls.b)

    def test_embedding_std_is_small(self):
        m = self.b.GPT(2, 2, 128)
        self.assertLess(float(m.wte.weight.std()), 0.05)

    def test_loss_at_init_is_near_uniform_over_the_vocab(self):
        torch.manual_seed(0)
        m = self.b.GPT(2, 2, 128)
        x = torch.randint(0, self.b.VOCAB, (2, 64))
        y = torch.randint(0, self.b.VOCAB, (2, 64))   # targets != inputs
        loss = float(m(x, y).detach())
        expected = math.log(self.b.VOCAB)
        self.assertGreater(loss, expected - 1.5)
        self.assertLess(loss, expected + 1.5)

    def test_residual_projections_are_depth_scaled(self):
        shallow = self.b.GPT(2, 2, 128)
        deep = self.b.GPT(8, 2, 128)
        s = float(shallow.blocks[0].attn.c_proj.weight.std())
        d = float(deep.blocks[0].attn.c_proj.weight.std())
        self.assertLess(d, s)


class Checkpointing(unittest.TestCase):
    def test_round_trips_weights_optimizer_and_step(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(8, 8)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()

        path = Path(tempfile.mkdtemp()) / "ckpt.pt"
        tr.save_checkpoint(path, model, opt, step=42, config={"n_layer": 2})

        fresh = torch.nn.Linear(8, 8)
        fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
        step, cfg = tr.load_checkpoint(path, fresh, fresh_opt, device="cpu")

        self.assertEqual(step, 42)
        self.assertEqual(cfg["n_layer"], 2)
        for a, b in zip(model.parameters(), fresh.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_resume_continues_the_step_count(self):
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters())
        path = Path(tempfile.mkdtemp()) / "c.pt"
        tr.save_checkpoint(path, model, opt, step=1234, config={})
        step, _ = tr.load_checkpoint(path, model, opt, device="cpu")
        self.assertEqual(step, 1234)

    def test_best_tracker_accepts_the_first_score(self):
        b = tr.BestTracker()
        self.assertTrue(b.improved(3.5))

    def test_best_tracker_accepts_a_lower_loss(self):
        b = tr.BestTracker()
        b.improved(3.5)
        self.assertTrue(b.improved(3.1))

    def test_best_tracker_rejects_a_higher_loss(self):
        """Over 4 epochs on 55M tokens the model can start memorising; the
        final step must not overwrite a better middle."""
        b = tr.BestTracker()
        b.improved(2.58)
        self.assertFalse(b.improved(2.66))

    def test_best_tracker_remembers_the_best_value(self):
        b = tr.BestTracker()
        for v in (3.5, 3.1, 3.3, 2.9, 3.0):
            b.improved(v)
        self.assertAlmostEqual(b.best, 2.9)

    def test_missing_checkpoint_starts_from_zero(self):
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters())
        missing = Path(tempfile.mkdtemp()) / "nope.pt"
        step, _ = tr.load_checkpoint(missing, model, opt, device="cpu")
        self.assertEqual(step, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

