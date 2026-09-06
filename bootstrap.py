#!/usr/bin/env python3
"""Everything EDITH needs that is not in git: weights, corpus, index.

Imported by infer/terminal.py's first-run prompt: Terminal.boot() calls
fetch_all() when something is missing. One module on purpose - a second
fetcher would eventually name a different repo.

Nothing here writes a .pt. The published format is safetensors precisely
because torch.load executes whatever is inside the file it opens.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_REPO = "Tomionkkas/edith-250m"
CORPUS_REPO = "Tomionkkas/edith-marvel-corpus"

# The data directory, loaded by file path like bootstrap.py itself is - see
# infer/terminal.py's load of this module. Constants only, so re-executing
# it per module costs nothing.
_paths_spec = importlib.util.spec_from_file_location("edith_paths", ROOT / "paths.py")
paths = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(paths)

WEIGHTS = paths.WEIGHTS
CONFIG_JSON = paths.CONFIG_JSON
INDEX = paths.INDEX
CORPUS = paths.CORPUS

# Per-artefact sizes, quoted to the user before they agree to a download.
# Update alongside the published files. The index carries no entry here on
# purpose - see download_mb() below.
WEIGHTS_MB = 508     # fp16 weights on HuggingFace
CORPUS_MB = 296      # curated/ corpus

# The corpus files that must exist for a complete corpus. This mirrors the
# list that retrieve/search.py:454 globs to build the index, so a future
# editor knows why this exact list is here when they update one.
CORPUS_FILES = (
    "characters.txt",
    "comics.txt",
    "events.txt",
    "items.txt",
    "locations.txt",
    "mw_character_backstop.txt",
    "mw_marvel_backstop.txt",
    "patch.txt",
    "story_arcs.txt",
    "teams.txt",
    "wiki_marvel.txt",
)

# Imported lazily elsewhere; bound at module level so tests can replace it.
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:                # before huggingface_hub is installed
    hf_hub_download = snapshot_download = None


def _ensure_hub() -> None:
    """Bind the hub functions on first use.

    They are imported at module level inside a try/except so bootstrap.py can
    be imported before huggingface_hub is installed - a bare clone that has
    not set up its dependencies yet. If something installs the package
    afterward in the SAME process (as the test suite simulates), the names
    are still None: installing a package does not rebind a name that was
    already resolved, so resolve it again here, on first use.

    Only rebinds when a name is still None, so a caller (or test) that has
    already replaced hf_hub_download/snapshot_download with its own callable
    is left alone.
    """
    global hf_hub_download, snapshot_download
    if hf_hub_download is None or snapshot_download is None:
        try:
            from huggingface_hub import hf_hub_download as _dl, snapshot_download as _snap
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to fetch EDITH's weights/corpus "
                "but is not installed. Run `uv tool install --force "
                "git+https://github.com/Tomionkkas/edith`, or "
                "`py -m pip install huggingface_hub` yourself."
            ) from e
        hf_hub_download, snapshot_download = _dl, _snap


def has_nvidia_gpu() -> bool:
    """nvidia-smi on PATH and exiting 0. Not torch.cuda.is_available(): this
    runs BEFORE torch is installed, and choosing the wheel is the point."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True,
                              timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def torch_install_args() -> list[str]:
    """pip argv for this machine. The cu126 wheel is ~2.5 GB and pointless
    without an NVIDIA card; the default wheel is ~200 MB and gives MPS on
    Apple silicon. Most EDITH answers are composed from the record and never
    reach the model, so a CPU install is a real one."""
    if has_nvidia_gpu():
        return ["torch", "--index-url", "https://download.pytorch.org/whl/cu126"]
    return ["torch"]


def corpus_complete() -> bool:
    """True only when all 11 corpus files exist. snapshot_download creates
    CORPUS/ before files land, so a partial corpus (e.g., from an interrupted
    download) appears as "existing" to a bare .exists() check and would never
    be re-fetched. This checks for the actual output of the glob that
    retrieve/search.py:454 runs."""
    if not CORPUS.exists():
        return False
    return all((CORPUS / name).exists() for name in CORPUS_FILES)


def missing(weights: Path | None = None) -> list[Path]:
    """The required artefacts that are absent, in fetch order.

    `weights` is passed rather than read from the constant because Terminal
    takes --ckpt: a caller who named a file deserves to be told about THAT
    file, not about the one we would have downloaded.
    """
    weights = Path(weights) if weights else WEIGHTS
    return [p for p in (weights, INDEX) if not p.exists()]


def download_mb(weights: Path | None = None) -> int:
    """MB that fetch_all() will actually pull over the network, for the
    given checkpoint path.

    The index is deliberately priced at zero: it is never downloaded - it
    rebuilds locally from the corpus in about 30 s - so counting it here
    would ask someone to approve a download that will not happen. A person
    who already has the weights and a complete corpus but deleted their
    index sees 0, not a fixed total that was never true for their case.
    """
    weights = Path(weights) if weights else WEIGHTS
    total = 0
    if not weights.exists():
        total += WEIGHTS_MB
    if not corpus_complete():
        total += CORPUS_MB
    return total


def fetch_weights() -> None:
    """The two stage-3 files by name. The repo also holds stage 2, so a
    snapshot pull would cost a spare gigabyte."""
    _ensure_hub()
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json"):
        hf_hub_download(repo_id=MODEL_REPO, filename=name,
                        local_dir=str(WEIGHTS.parent))


def fetch_corpus() -> None:
    """allow_patterns keeps the `curated/` prefix, so local_dir is the data
    directory itself and the corpus lands in CORPUS. Passing ROOT here put it
    in the clone while CORPUS read from the data dir - the index build then
    failed on a corpus that had downloaded perfectly well."""
    _ensure_hub()
    snapshot_download(repo_id=CORPUS_REPO, repo_type="dataset",
                      local_dir=str(paths.HOME), allow_patterns="curated/*")


def build_index() -> None:
    """~30 s. Never shipped: index.pkl is a pickle, it is 521 MB, and it is
    CC BY-SA prose stored close to verbatim."""
    if not corpus_complete():
        missing_files = [name for name in CORPUS_FILES if not (CORPUS / name).exists()]
        raise FileNotFoundError(
            f"Cannot build index: corpus incomplete. Missing: {missing_files}"
        )
    for script in ("retrieve/search.py", "retrieve/build_names.py"):
        cmd = [sys.executable, str(ROOT / script)]
        if script.endswith("search.py"):
            cmd.append("--build")
        subprocess.run(cmd, check=True, cwd=str(ROOT))


# The layout before the data directory existed: everything beside the code.
# A clone that predates it holds ~1.3 GB that is still perfectly good, and a
# `git pull` must not turn that into a re-download plus a gigabyte of orphans.
def _legacy_pairs():
    """(old location, current constant), read at call time so a test that
    monkeypatches a constant is not migrating against the real one."""
    return (
        (ROOT / "checkpoints" / "model.safetensors", WEIGHTS),
        (ROOT / "checkpoints" / "config.json", CONFIG_JSON),
        (ROOT / "curated", CORPUS),
        (ROOT / "retrieve" / "index.pkl", INDEX),
        (ROOT / "retrieve" / "names.pkl", paths.NAMES),
    )


_migrated = False


def migrate_legacy(log=lambda *a: None) -> list:
    """Move a pre-data-dir clone's artefacts into the data directory.

    Called explicitly by Terminal.boot()'s first-run prompt and by
    fetch_all(), never from missing() or download_mb(): those are pure
    queries that tests monkeypatch the constants of, and a query with a
    filesystem move inside it would run against a real clone during pytest.

    Only ever moves what the data directory does not already have, so it
    cannot overwrite a newer download with an older one, and it is a handful
    of exists() calls on every run after the first. Returns what it moved.

    A failure here is not fatal: the artefact simply stays where it was and
    gets re-fetched. Losing a gigabyte to a half-finished move would be worse
    than downloading it again.
    """
    global _migrated
    moved = []
    if _migrated:
        return moved
    _migrated = True
    for old, new in _legacy_pairs():
        if not old.exists() or new.exists():
            continue
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
            moved.append(new)
            log(f"  moved {old.name} -> {new}")
        except OSError as e:
            log(f"  could not move {old}: {e}")
    return moved


def fetch_all(log=print) -> None:
    """Idempotent: each step tests for its own output first, and the hub
    cache makes a repeated download free."""
    migrate_legacy(log)
    if not WEIGHTS.exists():
        log("  weights ...")
        fetch_weights()
    if not corpus_complete():
        log("  corpus ...")
        fetch_corpus()
    if not INDEX.exists():
        log("  index (~30 s) ...")
        build_index()
