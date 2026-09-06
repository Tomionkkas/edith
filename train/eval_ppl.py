"""Perplexity of a checkpoint on a packed .bin, on FIXED windows.

Why fixed windows: this exists to compare checkpoints against each other, so
the evaluation set must be byte-identical between runs. Random sampling would
make a 0.05 difference between two models indistinguishable from noise.

The point of pointing it at general_val.bin is that stage 2's own validation
set is Marvel-only, so it cannot see general-English forgetting: a model that
specialises harder scores BETTER on Marvel val while getting worse at English.

    py train/eval_ppl.py --ckpt <path> --bin data/packed/general_val.bin
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch


def window_starts(n_tokens: int, block: int, n_windows: int) -> np.ndarray:
    """Evenly spaced, deterministic, and int64.

    int64 is not cosmetic: numpy defaults to int32 on Windows and
    general_train.bin holds 2.94e9 tokens, which silently wraps negative.
    """
    last = max(n_tokens - block - 1, 0)
    n = max(1, min(n_windows, last + 1))
    return np.linspace(0, last, n).astype(np.int64)


@torch.no_grad()
def evaluate(model, path: Path, block: int, device: str, n_windows: int = 200,
             batch: int = 4) -> tuple[float, float]:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    starts = window_starts(len(data), block, n_windows)
    amp = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) \
        else torch.float16 if device == "cuda" else torch.float32

    model.eval()
    total, seen = 0.0, 0
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        x = torch.from_numpy(np.stack(
            [data[s:s + block].astype(np.int64) for s in chunk])).to(device)
        y = torch.from_numpy(np.stack(
            [data[s + 1:s + 1 + block].astype(np.int64) for s in chunk])).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp,
                            enabled=(device == "cuda")):
            loss = model(x, y)
        total += float(loss) * len(chunk)
        seen += len(chunk)
    mean = total / max(seen, 1)
    return mean, math.exp(mean)


def main() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sample import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, block, step, _ = load_model(Path(args.ckpt), device)
    loss, ppl = evaluate(model, Path(args.bin), block, device,
                         args.windows, args.batch)
    print(f"{Path(args.ckpt).name} on {Path(args.bin).name}: "
          f"loss {loss:.4f}  perplexity {ppl:.2f}  "
          f"({args.windows} fixed windows, step {step:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
