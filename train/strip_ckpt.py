"""Drop optimizer state from a checkpoint.

A training checkpoint is ~3x the size of the weights: 250M params in fp32 is
1 GB, and AdamW's two moments are another 2 GB. Inference needs none of it,
so stripping makes the download 1 GB instead of 3. load_checkpoint reads
`optimizer` with .get(), so a stripped file still loads.

Only strip a run you will not resume - the optimizer state is unrecoverable.

    py train/strip_ckpt.py checkpoints/stage2_mix/best.pt
"""
import sys
from pathlib import Path

import torch


def strip(src: Path, dst: Path | None = None) -> Path:
    dst = dst or src.with_name(src.stem + "_weights.pt")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    torch.save({"model": ck["model"], "step": ck.get("step", 0),
                "config": ck.get("config", {})}, dst)
    return dst


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for arg in sys.argv[1:]:
        src = Path(arg)
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        dst = strip(src)
        print(f"{src.name} {src.stat().st_size/1e9:.2f} GB -> "
              f"{dst.name} {dst.stat().st_size/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
