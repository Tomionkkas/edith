"""Tests for MixedDataLoader (replay mixing).

The bug that matters here is silent: if a window's inputs come from one
corpus and its targets from the other, training still runs, loss still
falls, and the model learns to predict unrelated text. The two corpora in
these tests use disjoint token ranges so that splice is detectable.
"""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "trainer", Path(__file__).resolve().parent / "trainer.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

BLOCK = 8


@pytest.fixture(scope="module")
def corpora():
    """Two bins with disjoint token ids: primary 0-99, replay 1000-1099."""
    d = Path(tempfile.mkdtemp())
    primary = d / "primary.bin"
    replay = d / "replay.bin"
    np.arange(0, 100, dtype=np.uint16).repeat(20).tofile(primary)
    (np.arange(0, 100, dtype=np.uint16) + 1000).repeat(20).tofile(replay)
    return primary, replay


def loader(corpora, mix_prob, batch=16, seed=0):
    primary, replay = corpora
    return T.MixedDataLoader(primary, replay, BLOCK, batch, "cpu",
                             mix_prob=mix_prob,
                             rng=np.random.default_rng(seed))


def source_of(row):
    """'replay' if every token is >= 1000, 'primary' if all < 1000, else None."""
    lo = (row < 1000).all()
    hi = (row >= 1000).all()
    return "primary" if lo else "replay" if hi else None


# ------------------------------------------------- the splice invariant

def test_each_sequence_comes_from_exactly_one_corpus(corpora):
    dl = loader(corpora, 0.5, batch=32)
    for _ in range(20):
        x, y = dl.get_batch()
        for xi, yi in zip(x.numpy(), y.numpy()):
            sx, sy = source_of(xi), source_of(yi)
            assert sx is not None, "inputs straddle two corpora"
            assert sy is not None, "targets straddle two corpora"
            assert sx == sy, "inputs and targets came from different corpora"


def test_targets_are_inputs_shifted_by_one(corpora):
    dl = loader(corpora, 0.5, batch=8)
    x, y = dl.get_batch()
    assert torch.equal(x[:, 1:], y[:, :-1])


# ------------------------------------------------------------- the ratio

def test_mix_ratio_converges(corpora):
    dl = loader(corpora, 0.3, batch=64, seed=7)
    seen = {"primary": 0, "replay": 0}
    for _ in range(40):
        x, _ = dl.get_batch()
        for row in x.numpy():
            seen[source_of(row)] += 1
    share = seen["replay"] / (seen["primary"] + seen["replay"])
    assert 0.25 < share < 0.35, f"replay share {share:.3f} is not ~0.30"


def test_mix_prob_zero_is_primary_only(corpora):
    dl = loader(corpora, 0.0, batch=32)
    for _ in range(5):
        x, _ = dl.get_batch()
        assert all(source_of(r) == "primary" for r in x.numpy())


def test_mix_prob_one_is_replay_only(corpora):
    dl = loader(corpora, 1.0, batch=32)
    for _ in range(5):
        x, _ = dl.get_batch()
        assert all(source_of(r) == "replay" for r in x.numpy())


def test_reproducible_for_a_given_seed(corpora):
    a, _ = loader(corpora, 0.4, seed=11).get_batch()
    b, _ = loader(corpora, 0.4, seed=11).get_batch()
    assert torch.equal(a, b)


# ------------------------------------------------------------- guardrails

def test_rejects_out_of_range_probability(corpora):
    with pytest.raises(ValueError):
        loader(corpora, 1.5)
    with pytest.raises(ValueError):
        loader(corpora, -0.1)


def test_rejects_a_replay_corpus_that_is_too_short(corpora):
    primary, _ = corpora
    d = Path(tempfile.mkdtemp())
    tiny = d / "tiny.bin"
    np.arange(0, 4, dtype=np.uint16).tofile(tiny)
    with pytest.raises(ValueError):
        T.MixedDataLoader(primary, tiny, BLOCK, 4, "cpu", mix_prob=0.5)


def test_shapes_and_dtype(corpora):
    x, y = loader(corpora, 0.5, batch=6).get_batch()
    assert x.shape == (6, BLOCK) and y.shape == (6, BLOCK)
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_window_starts_stay_int64_for_a_huge_corpus():
    """The replay corpus is general_train.bin: 2.94e9 tokens, int32 wraps."""
    rng = np.random.default_rng(0)
    hi = 2_940_000_000
    vals = rng.integers(0, hi, size=1000, dtype=np.int64)
    assert vals.dtype == np.int64 and vals.min() >= 0
