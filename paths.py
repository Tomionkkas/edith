"""Where EDITH keeps the 1.3 GB it downloads.

Deliberately not next to the code. These paths used to be written as "beside
me" - `ROOT / "checkpoints"`, `Path(__file__).parent / "index.pkl"` - which is
correct for exactly one layout: a git clone the user chose the location of.
Once `edith` is a command rather than a folder, the code lives inside a
virtualenv's site-packages, and "beside me" means half a gigabyte of weights
landing somewhere the user has never heard of and cannot find to delete.

~/.edith is where config.json already lived, so the weights, the corpus and
the index move in beside it. One directory, one thing to delete.

EDITH_HOME overrides it, for a machine whose home is small or on a network
share.

Loaded by file path, like every other cross-directory import in this repo -
see install.py's load of bootstrap.py. Re-executing it costs nothing: it is
constants and no side effects.
"""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("EDITH_HOME") or Path.home() / ".edith")

# Fetched from HuggingFace by bootstrap.fetch_all().
CHECKPOINTS = HOME / "checkpoints"
WEIGHTS = CHECKPOINTS / "model.safetensors"
CONFIG_JSON = CHECKPOINTS / "config.json"
CORPUS = HOME / "curated"

# Built locally, never shipped: both are pickles, and unpickling executes
# whatever is in the file.
INDEX = HOME / "index.pkl"
NAMES = HOME / "names.pkl"

# The theme preference. Already lived here before anything else did.
CONFIG = HOME / "config.json"
