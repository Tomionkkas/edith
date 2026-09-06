#!/usr/bin/env python3
"""Convert a stripped EDITH checkpoint to model.safetensors + config.json.

    py release/to_safetensors.py checkpoints/stage3_edith.pt out/
    py release/to_safetensors.py checkpoints/stage3_edith.pt out16/ --fp16

Why safetensors at all: torch.load unpickles, i.e. executes arbitrary code
from a downloaded file. Nothing we publish may require that of a stranger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# The only architecture this project ships. Mirrors benchmark.py CONFIGS[250]
# with VOCAB and BLOCK. The loader reads this back to know what to build, so
# it is load-bearing, not documentation.
ARCH = {"n_layer": 16, "n_head": 16, "n_embd": 1024,
        "vocab_size": 50257, "block_size": 1024, "tied_embeddings": True}


def convert(src: Path, out: Path, fp16: bool = False) -> None:
    src, out = Path(src), Path(out)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["model"]

    # benchmark.py:97 ties lm_head to wte: two keys, ONE storage, and
    # save_file refuses shared storage rather than quietly duplicating it.
    # Drop the tied key; the loader re-ties. Read data_ptr rather than
    # assuming, so an untied checkpoint keeps its own head.
    if ("lm_head.weight" in sd and "wte.weight" in sd
            and sd["lm_head.weight"].data_ptr() == sd["wte.weight"].data_ptr()):
        sd = {k: v for k, v in sd.items() if k != "lm_head.weight"}

    dtype = torch.float16 if fp16 else torch.float32
    sd = {k: v.to(dtype).contiguous() for k, v in sd.items()}

    out.mkdir(parents=True, exist_ok=True)
    # safetensors metadata is str->str only; the real config goes in its own
    # file, which is also what HF tooling expects to find.
    save_file(sd, str(out / "model.safetensors"),
              metadata={"format": "pt", "tied_embeddings": "true"})

    cfg = dict(ARCH)
    ck_cfg = ck.get("config", {})
    cfg.update({"params": int(ck_cfg.get("params", 250)),
                "stage": ck_cfg.get("stage"),
                "step": int(ck.get("step", 0)),
                "dtype": "float16" if fp16 else "float32"})
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"wrote {out / 'model.safetensors'} ({mb:.0f} MB) + config.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="a stripped checkpoint, e.g. checkpoints/stage3_edith.pt")
    ap.add_argument("out", help="directory to write into")
    ap.add_argument("--fp16", action="store_true", help="half the size, some fidelity")
    a = ap.parse_args()
    convert(Path(a.src), Path(a.out), a.fp16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
