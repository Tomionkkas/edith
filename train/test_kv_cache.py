"""The cache must be a pure speedup, never a behaviour change.

A wrong cache does not crash: it produces fluent text from subtly wrong
attention, which no loss curve or perplexity number would reveal.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parent / "benchmark.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = B.GPT(n_layer=2, n_head=2, n_embd=32)
    m.eval()
    return m


def ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, B.VOCAB, (1, n), generator=g)


# ------------------------------------------------ equivalence with no cache

@torch.no_grad()
def test_prefill_matches_uncached(model):
    x = ids(12)
    plain = model.logits(x)
    cached = model.logits(x, cache=model.new_cache())
    assert torch.allclose(plain, cached, atol=1e-5)


@torch.no_grad()
def test_incremental_decode_matches_full_forward(model):
    """The whole point: token-by-token must equal one full pass."""
    x = ids(16, seed=3)
    reference = model.logits(x)[:, -1, :]

    cache = model.new_cache()
    model.logits(x[:, :-1], cache=cache)          # prefill
    stepped = model.logits(x[:, -1:], cache=cache)[:, -1, :]
    assert torch.allclose(reference, stepped, atol=1e-5)


@torch.no_grad()
def test_many_steps_stay_equivalent(model):
    """Drift would compound silently over a long generation."""
    x = ids(20, seed=7)
    cache = model.new_cache()
    model.logits(x[:, :8], cache=cache)
    for t in range(8, 20):
        got = model.logits(x[:, t:t + 1], cache=cache)[:, -1, :]
        want = model.logits(x[:, :t + 1])[:, -1, :]
        assert torch.allclose(got, want, atol=1e-5), f"diverged at step {t}"


# ------------------------------------------------------------- mechanics

@torch.no_grad()
def test_cache_grows_by_one_per_step(model):
    cache = model.new_cache()
    model.logits(ids(5), cache=cache)
    assert cache[0]["k"].shape[2] == 5
    model.logits(ids(1, seed=9), cache=cache)
    assert cache[0]["k"].shape[2] == 6


@torch.no_grad()
def test_every_layer_caches(model):
    cache = model.new_cache()
    model.logits(ids(4), cache=cache)
    assert all(c.get("k") is not None for c in cache)


def test_new_cache_is_empty(model):
    assert all(c == {} for c in model.new_cache())


def test_new_cache_is_not_shared_between_calls(model):
    a, b = model.new_cache(), model.new_cache()
    a[0]["k"] = torch.zeros(1)
    assert b[0] == {}


@torch.no_grad()
def test_position_embeddings_continue_past_the_cache(model):
    """Restarting positions at 0 would give every generated token the first
    token's position embedding - fluent output, wrong attention."""
    x = ids(6, seed=11)
    cache = model.new_cache()
    model.logits(x[:, :5], cache=cache)
    with_cache = model.logits(x[:, 5:6], cache=cache)[:, -1, :]
    from_scratch = model.logits(x[:, 5:6])[:, -1, :]   # position 0, wrong
    assert not torch.allclose(with_cache, from_scratch, atol=1e-4)


@torch.no_grad()
def test_exceeding_block_size_raises(model):
    cache = model.new_cache()
    with pytest.raises(ValueError, match="block size"):
        model.logits(ids(B.BLOCK + 1), cache=cache)


@torch.no_grad()
def test_training_path_is_untouched(model):
    """cache=None must behave exactly as before."""
    x = ids(9, seed=13)
    a = model.logits(x)
    b = model.logits(x)
    assert torch.equal(a, b)
