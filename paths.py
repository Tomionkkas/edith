"""Where EDITH keeps the 1.3 GB it downloads.

Two layouts, one file, because EDITH ships as two things.

An INSTALLED EDITH is a command: its code lives inside a virtualenv's
site-packages, so "beside me" means half a gigabyte of weights landing
somewhere the user has never heard of and cannot find to delete. Those go to
`~/.edith`, where config.json already lived - one directory, one thing to
delete.

A git CHECKOUT is a development tree, and its artefacts belong beside the
code: `train/` writes `checkpoints/`, `retrieve/search.py --build` writes
`retrieve/index.pkl`, `crawl/` writes `curated/`, and every doc and harness
in the repo names those paths. Moving them into `~/.edith` would split the
workflow in half - trained here, read from there.

EDITH_HOME overrides either, for a machine whose home is small or on a
network share.

Loaded by file path, like every other cross-directory import in this repo -
see infer/terminal.py's load of bootstrap.py. Re-executing it costs nothing:
it is constants and no side effects.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def data_home(root: Path, env: str | None, home: Path) -> Path:
    """The directory the downloaded artefacts live in.

    A `.git` is the signal, because it is exactly what separates a tree
    somebody is working IN from a package an installer put somewhere. It is
    a file rather than a directory inside a worktree, so this tests only that
    it exists.
    """
    if env:
        return Path(env)
    return root if (root / ".git").exists() else home / ".edith"


CHECKOUT = (ROOT / ".git").exists()
HOME = data_home(ROOT, os.environ.get("EDITH_HOME"), Path.home())

# Fetched from HuggingFace by bootstrap.fetch_all().
CHECKPOINTS = HOME / "checkpoints"
WEIGHTS = CHECKPOINTS / "model.safetensors"
CONFIG_JSON = CHECKPOINTS / "config.json"
CORPUS = HOME / "curated"

# Built locally, never shipped: both are pickles, and unpickling executes
# whatever is in the file. In a checkout they stay in retrieve/, beside the
# script that writes them.
_BUILT = HOME / "retrieve" if CHECKOUT else HOME
INDEX = _BUILT / "index.pkl"
NAMES = _BUILT / "names.pkl"

# The theme preference. It lived in ~/.edith before anything else did, and it
# stays there even for a checkout: a developer's theme is not repo state, and
# writing it into the tree would leave an untracked file behind.
CONFIG = (Path.home() / ".edith" / "config.json" if CHECKOUT
          else HOME / "config.json")
