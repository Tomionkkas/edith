#!/usr/bin/env python3
"""One-command EDITH setup.

    git clone https://github.com/Tomionkkas/edith && cd edith
    py install.py            # Windows
    python3 install.py       # macOS / Linux

Installs dependencies, fetches the weights and the corpus from HuggingFace,
and rebuilds the retrieval index. Idempotent - anything already in place is
skipped, and the hub cache makes a repeated download free.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "bootstrap.py")
bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bootstrap)


def pip_install(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", *args]
    print("  pip install -q " + " ".join(args), flush=True)
    subprocess.run(cmd, check=True)


def _dependencies_installed() -> bool:
    """Check if all required dependencies are importable.

    A second run is free if dependencies are already in place. We use
    find_spec rather than import to avoid the torch startup cost.

    Note: if torch is installed as CPU but later an NVIDIA card is added,
    the CPU wheel will be kept until manually reinstalled. This is a fair
    trade for a re-run working offline.
    """
    for pkg in ("torch", "safetensors", "huggingface_hub", "sentencepiece", "numpy"):
        if importlib.util.find_spec(pkg) is None:
            return False
    return True


def main() -> int:
    print("\nEDITH setup\n")

    print(" dependencies")
    if _dependencies_installed():
        print("  already installed")
    else:
        # torch first and on its own: the wheel choice depends on the machine and
        # the CUDA index url must not apply to the other four packages.
        try:
            pip_install(*bootstrap.torch_install_args())
        except subprocess.CalledProcessError:
            print("\nFailed to install torch. Check:")
            print("  - Network connection or proxy settings")
            print("  - A compatible wheel exists for Python", sys.version.split()[0], "on", sys.platform)
            return 1

        try:
            pip_install("safetensors", "huggingface_hub", "sentencepiece", "numpy")
        except subprocess.CalledProcessError:
            print("\nFailed to install runtime dependencies. Check:")
            print("  - Network connection or proxy settings")
            print("  - A compatible wheel exists for Python", sys.version.split()[0], "on", sys.platform)
            return 1

    print("\n artefacts")
    bootstrap.migrate_legacy(log=print)
    bootstrap.fetch_all()

    print("\nEDITH is ready.\n")
    print("  Windows      .\\edith      (or: py infer\\terminal.py)")
    print("  macOS/Linux  ./edith      (or: python3 infer/terminal.py)")
    print("\n  To type `edith` from anywhere on Windows:")
    print("      powershell -ExecutionPolicy Bypass -File .\\install.ps1\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
