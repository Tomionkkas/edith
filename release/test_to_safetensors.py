"""Tests for the safetensors converter.

The tie is the whole hazard here: two keys, one storage. save_file refuses
it, and a converter that "fixes" that by duplicating the tensor ships a
model whose head can drift from its embedding. These pin the drop-and-retie
contract, on tensors small enough to run in milliseconds.
"""
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

spec = importlib.util.spec_from_file_location(
    "to_safetensors", Path(__file__).resolve().parent / "to_safetensors.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def tied_checkpoint(tmp_path):
    """A miniature of the real thing: tied head, a config, a step."""
    wte = torch.randn(8, 4)
    sd = {"wte.weight": wte, "wpe.weight": torch.randn(6, 4)}
    sd["lm_head.weight"] = sd["wte.weight"]      # the tie: same object
    src = tmp_path / "mini.pt"
    torch.save({"model": sd, "step": 42, "config": {"params": 250, "stage": 3}}, src)
    return src, sd


def test_tied_head_is_dropped_from_the_saved_file(tmp_path):
    src, _ = tied_checkpoint(tmp_path)
    C.convert(src, tmp_path / "out", fp16=False)
    saved = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert "lm_head.weight" not in saved
    assert "wte.weight" in saved


def test_retying_reproduces_the_original_state_dict(tmp_path):
    src, original = tied_checkpoint(tmp_path)
    C.convert(src, tmp_path / "out", fp16=False)
    saved = load_file(str(tmp_path / "out" / "model.safetensors"))
    saved["lm_head.weight"] = saved["wte.weight"]
    assert set(saved) == set(original)
    for k in original:
        assert torch.equal(saved[k], original[k]), k


def test_an_untied_head_survives(tmp_path):
    """The guard reads data_ptr, so it must not drop a head that is genuinely
    its own tensor."""
    sd = {"wte.weight": torch.randn(8, 4), "lm_head.weight": torch.randn(8, 4)}
    src = tmp_path / "untied.pt"
    torch.save({"model": sd, "step": 1, "config": {"params": 250}}, src)
    C.convert(src, tmp_path / "out", fp16=False)
    saved = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert "lm_head.weight" in saved
    assert torch.equal(saved["lm_head.weight"], sd["lm_head.weight"])


def test_config_json_carries_the_architecture_and_the_checkpoint(tmp_path):
    src, _ = tied_checkpoint(tmp_path)
    C.convert(src, tmp_path / "out", fp16=False)
    cfg = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    assert cfg["n_layer"] == 16 and cfg["n_head"] == 16 and cfg["n_embd"] == 1024
    assert cfg["vocab_size"] == 50257 and cfg["block_size"] == 1024
    assert cfg["tied_embeddings"] is True
    assert cfg["params"] == 250 and cfg["stage"] == 3
    assert cfg["step"] == 42
    assert cfg["dtype"] == "float32"


def test_fp16_halves_the_tensors_and_says_so(tmp_path):
    src, _ = tied_checkpoint(tmp_path)
    C.convert(src, tmp_path / "out", fp16=True)
    saved = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert saved["wte.weight"].dtype == torch.float16
    cfg = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    assert cfg["dtype"] == "float16"
