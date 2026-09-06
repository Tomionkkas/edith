"""Tests for the sampling filters.

Sampling bugs are silent: a broken top-p still returns fluent-looking text,
it just samples from the wrong distribution. These pin the filter semantics
so a wrong-but-plausible refactor fails loudly.
"""
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location(
    "sample", Path(__file__).resolve().parent / "sample.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def kept(row):
    """Indices the filter left alive."""
    return sorted(i for i, v in enumerate(row.tolist()) if v != float("-inf"))


# ------------------------------------------------------------------ top-k

def test_top_k_keeps_exactly_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    out = S.filter_logits(logits, top_k=2, top_p=1.0)
    assert kept(out[0]) == [1, 4]          # the 5.0 and the 4.0


def test_top_k_one_leaves_only_argmax():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    out = S.filter_logits(logits, top_k=1, top_p=1.0)
    assert kept(out[0]) == [1]


def test_top_k_zero_disables():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    out = S.filter_logits(logits, top_k=0, top_p=1.0)
    assert kept(out[0]) == [0, 1, 2]


def test_top_k_larger_than_vocab_is_harmless():
    logits = torch.tensor([[1.0, 5.0, 3.0]])
    out = S.filter_logits(logits, top_k=99, top_p=1.0)
    assert kept(out[0]) == [0, 1, 2]


# ------------------------------------------------------------------ top-p

def test_top_p_keeps_smallest_set_reaching_threshold():
    # probs after softmax of log() values: .5 .25 .15 .10
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.10]]))
    out = S.filter_logits(logits, top_k=0, top_p=0.85)
    # .5 -> .75 -> .90 first crosses .85 at the third token, so keep three
    assert kept(out[0]) == [0, 1, 2]


def test_top_p_one_keeps_everything():
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.15, 0.10]]))
    out = S.filter_logits(logits, top_k=0, top_p=1.0)
    assert kept(out[0]) == [0, 1, 2, 3]


def test_top_p_always_keeps_at_least_one():
    """A peak above p must not filter the whole row away."""
    logits = torch.log(torch.tensor([[0.99, 0.005, 0.005]]))
    out = S.filter_logits(logits, top_k=0, top_p=0.5)
    assert kept(out[0]) == [0]
    assert torch.isfinite(out).any()


def test_top_p_drops_the_tail():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.05, 0.05]]))
    out = S.filter_logits(logits, top_k=0, top_p=0.85)
    assert kept(out[0]) == [0, 1]


# ------------------------------------------------------- shared invariants

def test_argmax_is_never_filtered_out():
    torch.manual_seed(0)
    for _ in range(50):
        logits = torch.randn(1, 200)
        best = int(logits.argmax())
        for k, p in ((0, 0.3), (5, 1.0), (10, 0.7), (1, 0.1)):
            out = S.filter_logits(logits.clone(), top_k=k, top_p=p)
            assert out[0, best] != float("-inf")


def test_rows_are_filtered_independently():
    logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    out = S.filter_logits(logits, top_k=1, top_p=1.0)
    assert kept(out[0]) == [0]
    assert kept(out[1]) == [2]


def test_top_k_and_top_p_compose():
    logits = torch.log(torch.tensor([[0.4, 0.3, 0.2, 0.1]]))
    # top_k=3 drops the 0.1; top_p=0.7 then stops after 0.4+0.3
    out = S.filter_logits(logits, top_k=3, top_p=0.7)
    assert kept(out[0]) == [0, 1]


def test_filter_does_not_mutate_relative_order():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
    out = S.filter_logits(logits, top_k=3, top_p=1.0)
    alive = [(i, out[0, i].item()) for i in kept(out[0])]
    assert alive == [(1, 5.0), (2, 3.0), (3, 4.0)]


# ------------------------------------------------- repetition penalty

def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([[2.0, 2.0, 2.0]])
    out = S.apply_repetition_penalty(logits, [1], 2.0)
    assert out[0, 1] < out[0, 0]
    assert out[0, 0] == 2.0 and out[0, 2] == 2.0


def test_repetition_penalty_handles_negative_logits():
    """Dividing a negative logit would RAISE it; must multiply instead."""
    logits = torch.tensor([[-2.0, -2.0]])
    out = S.apply_repetition_penalty(logits, [0], 2.0)
    assert out[0, 0] < out[0, 1]


def test_repetition_penalty_of_one_is_a_noop():
    logits = torch.tensor([[2.0, -2.0]])
    out = S.apply_repetition_penalty(logits.clone(), [0, 1], 1.0)
    assert torch.equal(out, logits)


# ------------------------------------------------------- stop conditions

def test_a_cycling_tail_is_detected():
    """The degenerate loop: the same short phrase, over and over."""
    tokens = [7, 8, 9, 1, 2, 3, 4] * 5
    assert S.looping(tokens, window=8, repeats=3)


def test_a_phrase_repeated_twice_is_not_a_loop():
    """Real answers repeat: "she is an Omega-level mutant" legitimately shows
    up in both the species sentence and the powers sentence."""
    phrase = [5, 6, 7, 8, 9, 10, 11, 12]
    # long enough that the length guard is not what makes this pass
    tokens = phrase + list(range(20, 70)) + phrase
    assert len(tokens) > 8 * 3
    assert not S.looping(tokens, window=8, repeats=3)


def test_ordinary_text_is_not_a_loop():
    assert not S.looping(list(range(200)), window=8, repeats=3)


def test_too_short_to_judge():
    assert not S.looping([1, 2, 3], window=8, repeats=3)
    assert not S.looping([], window=8, repeats=3)


def test_a_long_repeat_still_needs_the_full_window():
    """A 7-token cycle read through an 8-token window still repeats."""
    assert S.looping([1, 2, 3, 4, 5, 6, 7] * 6, window=8, repeats=3)


# --------------------------------------------------- finishing an answer

def test_a_finished_sentence_is_left_alone():
    for s in ("Storm is a Mutant.", "Who knows?", "Stop!", 'He said "yes."'):
        assert S.clip_to_sentence(s) == s


def test_a_trailing_fragment_is_dropped():
    """Cut at the token cap, an answer ends mid-clause: "...in ways that"."""
    text = ("Thanos is a Titanian Eternal. He is skilled in cosmic energy. "
            "He has developed an impressive mastery of matter in ways that")
    out = S.clip_to_sentence(text)
    assert out.endswith("cosmic energy.")
    assert "ways that" not in out


def test_an_answer_with_no_sentence_end_is_kept_whole():
    """Better a long fragment than nothing at all."""
    text = "a" * 200
    assert S.clip_to_sentence(text) == text


def test_an_early_full_stop_does_not_gut_the_answer():
    """One period near the start is not a reason to throw away the rest."""
    text = "Yes. " + "and then something long happened without any full stop " * 4
    assert S.clip_to_sentence(text) == text


def test_whitespace_only_is_safe():
    assert S.clip_to_sentence("   ") == "   "
    assert S.clip_to_sentence("") == ""


# ---------------------------------------- checkpoint format reading

def test_read_checkpoint_reads_a_pt(tmp_path):
    sd = {"wte.weight": torch.randn(4, 2)}
    p = tmp_path / "m.pt"
    torch.save({"model": sd, "step": 7, "config": {"params": 250}}, p)
    got, cfg, step = S._read_checkpoint(p)
    assert torch.equal(got["wte.weight"], sd["wte.weight"])
    assert cfg["params"] == 250 and step == 7


def test_read_checkpoint_reties_the_head_from_safetensors(tmp_path):
    wte = torch.randn(4, 2)
    save_file({"wte.weight": wte}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"params": 250, "step": 9}), encoding="utf-8")
    sd, cfg, step = S._read_checkpoint(tmp_path / "model.safetensors")
    assert "lm_head.weight" in sd
    assert torch.equal(sd["lm_head.weight"], wte)
    assert cfg["params"] == 250 and step == 9


def test_read_checkpoint_without_a_config_says_which_file_is_missing(tmp_path):
    """A wrong architecture loads without complaining and produces noise, so
    a missing config is an error with a path in it, never a silent default."""
    save_file({"wte.weight": torch.randn(4, 2)},
              str(tmp_path / "model.safetensors"))
    try:
        S._read_checkpoint(tmp_path / "model.safetensors")
    except FileNotFoundError as e:
        assert "config.json" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_read_checkpoint_accepts_a_string_path(tmp_path):
    """engine.py passes Path, the terminal passes Path, but --ckpt is a str
    and suffix on a str raises AttributeError."""
    sd = {"wte.weight": torch.randn(4, 2)}
    p = tmp_path / "m.pt"
    torch.save({"model": sd, "step": 0, "config": {}}, p)
    got, _, _ = S._read_checkpoint(str(p))
    assert torch.equal(got["wte.weight"], sd["wte.weight"])
