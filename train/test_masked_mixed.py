"""Tests for stage-3 replay (MaskedMixedDataLoader).

Two silent failures live here. A window whose inputs come from one corpus and
targets from the other trains fine and teaches nonsense. And a replay window
that inherits the SFT mask would be scored on ~13% of its tokens, quietly
delivering a fraction of the replay the ratio promises.
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
    """Positionally unique tokens, so a window can be located exactly.

    data[i] == i means the window starting at i is identifiable from its
    first target token, which is what makes the alignment test unambiguous.
    """
    d = Path(tempfile.mkdtemp())
    sft, mask, replay = d / "s.bin", d / "m.bin", d / "r.bin"
    np.arange(0, 5000, dtype=np.uint16).tofile(sft)
    m = np.zeros(5000, dtype=np.uint8)
    m[::2] = 1                                  # alternating, clearly not all-ones
    m.tofile(mask)
    (np.arange(0, 5000, dtype=np.uint16) + 10000).tofile(replay)
    return sft, mask, replay


def loader(corpora, mix_prob, batch=16, seed=0):
    s, m, r = corpora
    return T.MaskedMixedDataLoader(s, m, r, BLOCK, batch, "cpu",
                                   mix_prob=mix_prob,
                                   rng=np.random.default_rng(seed))


def source_of(row):
    lo = (row < 10000).all()
    hi = (row >= 10000).all()
    return "sft" if lo else "replay" if hi else None


# ------------------------------------------------- the splice invariant

def test_each_sequence_comes_from_one_corpus(corpora):
    dl = loader(corpora, 0.5, batch=32)
    for _ in range(20):
        x, y, _ = dl.get_batch()
        for xi, yi in zip(x.numpy(), y.numpy()):
            sx, sy = source_of(xi), source_of(yi)
            assert sx is not None and sy is not None
            assert sx == sy, "inputs and targets from different corpora"


def test_targets_are_inputs_shifted_by_one(corpora):
    x, y, _ = loader(corpora, 0.5, batch=8).get_batch()
    assert torch.equal(x[:, 1:], y[:, :-1])


# ------------------------------------------------------------ the mask

def test_replay_rows_are_scored_in_full(corpora):
    """Ordinary text has no prompt to exclude; a fraction-scored replay row
    would deliver far less replay than the ratio promises."""
    dl = loader(corpora, 1.0, batch=16)
    _, _, m = dl.get_batch()
    assert m.sum() == m.numel()


def test_sft_rows_keep_their_own_mask(corpora):
    dl = loader(corpora, 0.0, batch=16)
    _, _, m = dl.get_batch()
    assert 0 < m.sum() < m.numel(), "SFT rows must not be fully scored"


def test_mask_aligns_with_targets_not_inputs(corpora):
    """Scoring one position off is a subtly wrong run that still converges."""
    _, mp, _ = corpora
    raw_mask = np.memmap(mp, dtype=np.uint8, mode="r")
    _, y, m = loader(corpora, 0.0, batch=8, seed=5).get_batch()
    for yi, mi in zip(y.numpy(), m.numpy()):
        # data[i] == i, so the first target token IS the target start index
        start = int(yi[0])
        expected = raw_mask[start:start + BLOCK]
        assert np.array_equal(expected, mi.astype(np.uint8)), (
            "mask must shift with the TARGETS; scoring one position off is a "
            "subtly wrong run that still converges")


def test_mask_dtype_is_float(corpora):
    _, _, m = loader(corpora, 0.5).get_batch()
    assert m.dtype == torch.float32


# ------------------------------------------------------------ the ratio

def test_ratio_converges(corpora):
    dl = loader(corpora, 0.25, batch=64, seed=3)
    seen = {"sft": 0, "replay": 0}
    for _ in range(40):
        x, _, _ = dl.get_batch()
        for row in x.numpy():
            seen[source_of(row)] += 1
    share = seen["replay"] / sum(seen.values())
    assert 0.20 < share < 0.30, f"replay share {share:.3f} is not ~0.25"


def test_zero_prob_is_pure_sft(corpora):
    x, _, _ = loader(corpora, 0.0, batch=32).get_batch()
    assert all(source_of(r) == "sft" for r in x.numpy())


def test_rejects_bad_probability(corpora):
    with pytest.raises(ValueError):
        loader(corpora, 1.5)


# ---------------------------------------------- the scored-token maths

def test_replay_prob_accounts_for_masking():
    """13.2% scored means 30% replay by sequence is ~76% by gradient."""
    p = T.replay_prob_for(0.132, 0.7)
    assert 0.05 < p < 0.06, f"expected ~0.054, got {p:.4f}"


def test_replay_prob_is_identity_when_everything_is_scored():
    """With no mask, sequence share and scored share are the same thing."""
    assert T.replay_prob_for(1.0, 0.7) == pytest.approx(0.3)


def test_replay_prob_hits_its_target():
    for scored in (0.05, 0.132, 0.5, 1.0):
        for target in (0.5, 0.7, 0.9):
            p = T.replay_prob_for(scored, target)
            got = ((1 - p) * scored) / ((1 - p) * scored + p)
            assert got == pytest.approx(target, abs=1e-9)


def test_replay_prob_rejects_nonsense():
    with pytest.raises(ValueError):
        T.replay_prob_for(0.0)
    with pytest.raises(ValueError):
        T.replay_prob_for(0.5, 1.0)
