"""Tests for the shared fetch.

install.py and terminal.py both bootstrap. The risk this file guards is
drift: two fetchers naming two repos, or one of them pulling the whole model
repo and dragging stage 2's spare gigabyte along.
"""
import importlib.util
import sys
import types
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bootstrap", Path(__file__).resolve().parent / "bootstrap.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def test_cuda_wheel_only_where_there_is_a_cuda_card(monkeypatch):
    monkeypatch.setattr(B, "has_nvidia_gpu", lambda: True)
    assert "download.pytorch.org/whl/cu126" in " ".join(B.torch_install_args())


def test_no_card_means_the_default_wheel(monkeypatch):
    """The CUDA wheel is ~2.5 GB and useless without an NVIDIA card. Most
    EDITH answers never touch the model at all, so a CPU install is a real
    install, not a degraded one."""
    monkeypatch.setattr(B, "has_nvidia_gpu", lambda: False)
    args = B.torch_install_args()
    assert "cu126" not in " ".join(args)
    assert "torch" in args


def test_gpu_detection_is_false_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(B.shutil, "which", lambda name: None)
    assert B.has_nvidia_gpu() is False


def test_gpu_detection_is_false_when_nvidia_smi_times_out(monkeypatch):
    """When nvidia-smi exists but times out, choose the CPU wheel, not the
    2.5 GB CUDA wheel."""
    monkeypatch.setattr(B.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(B.subprocess, "run",
                       lambda *a, **kw: (_ for _ in ()).throw(B.subprocess.TimeoutExpired("test", 15)))
    assert B.has_nvidia_gpu() is False


def test_missing_lists_every_absent_artefact(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "checkpoints" / "model.safetensors")
    monkeypatch.setattr(B, "INDEX", tmp_path / "retrieve" / "index.pkl")
    monkeypatch.setattr(B, "CORPUS", tmp_path / "curated")
    assert B.missing() == [B.WEIGHTS, B.INDEX]

    B.WEIGHTS.parent.mkdir(parents=True)
    B.WEIGHTS.write_text("x")
    assert B.missing() == [B.INDEX]


def test_missing_reports_the_checkpoint_it_was_given(monkeypatch, tmp_path):
    """Terminal takes --ckpt, and infer/test_terminal.py:199 passes a custom
    one and expects that name back. Reading the module constant instead would
    report a file the caller never asked about."""
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "model.safetensors")
    monkeypatch.setattr(B, "INDEX", tmp_path / "index.pkl")
    B.WEIGHTS.write_text("x")
    B.INDEX.write_text("x")
    nope = tmp_path / "nope.pt"
    assert B.missing(nope) == [nope]
    assert B.missing() == []


def test_fetch_weights_takes_two_files_by_name_not_the_whole_repo(monkeypatch, tmp_path):
    """The model repo also holds stage 2. A snapshot pull would cost a spare
    gigabyte."""
    asked = []
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "model.safetensors")
    monkeypatch.setattr(B, "CONFIG_JSON", tmp_path / "config.json")

    def fake_download(repo_id, filename, local_dir, **kw):
        asked.append((repo_id, filename))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        return str(p)

    monkeypatch.setattr(B, "hf_hub_download", fake_download)
    B.fetch_weights()
    assert [f for _, f in asked] == ["model.safetensors", "config.json"]
    assert all(r == B.MODEL_REPO for r, _ in asked)


def test_partial_corpus_triggers_fetch(monkeypatch, tmp_path):
    """A partial corpus (directory exists but missing files) should trigger
    a re-fetch, not skip it. This test proves the bug: the old code only
    checked CORPUS.exists() and would skip the fetch forever."""
    called = []
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "model.safetensors")
    monkeypatch.setattr(B, "INDEX", tmp_path / "index.pkl")
    monkeypatch.setattr(B, "CORPUS", tmp_path / "curated")
    B.WEIGHTS.write_text("x")
    B.INDEX.write_text("x")
    # Create the directory but with only some files - incomplete corpus
    B.CORPUS.mkdir()
    (B.CORPUS / "characters.txt").write_text("x")
    monkeypatch.setattr(B, "fetch_weights", lambda: called.append("w"))
    monkeypatch.setattr(B, "fetch_corpus", lambda: called.append("c"))
    monkeypatch.setattr(B, "build_index", lambda: called.append("i"))
    B.fetch_all(log=lambda *a: None)
    # With the fix, fetch_corpus should be called because corpus is incomplete
    assert called == ["c"]


def test_download_mb_totals_weights_and_corpus_when_both_are_missing(monkeypatch, tmp_path):
    """Everything missing: the index carries no price of its own (it never
    downloads), so the total is exactly the two artefacts that do."""
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "checkpoints" / "model.safetensors")
    monkeypatch.setattr(B, "CORPUS", tmp_path / "curated")
    assert B.download_mb() == B.WEIGHTS_MB + B.CORPUS_MB


def test_download_mb_is_zero_when_only_the_index_is_missing(monkeypatch, tmp_path):
    """FIX ROUND 1: someone who already has the weights and a complete
    corpus but deleted their index used to be quoted a fixed 810 MB and
    asked to approve a download that would never happen - the index
    rebuilds locally from the corpus, never over the network."""
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "model.safetensors")
    monkeypatch.setattr(B, "CORPUS", tmp_path / "curated")
    B.WEIGHTS.write_text("x")
    B.CORPUS.mkdir()
    for name in B.CORPUS_FILES:
        (B.CORPUS / name).write_text("x")
    assert B.download_mb() == 0


def test_download_mb_counts_only_the_weights_when_only_they_are_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "model.safetensors")
    monkeypatch.setattr(B, "CORPUS", tmp_path / "curated")
    B.CORPUS.mkdir()
    for name in B.CORPUS_FILES:
        (B.CORPUS / name).write_text("x")
    assert B.download_mb() == B.WEIGHTS_MB


def test_download_mb_reports_the_checkpoint_it_was_given(monkeypatch, tmp_path):
    """Mirrors test_missing_reports_the_checkpoint_it_was_given: Terminal's
    --ckpt names a file, and the quoted size must be about THAT file, not
    the module's own WEIGHTS constant."""
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "model.safetensors")
    monkeypatch.setattr(B, "CORPUS", tmp_path / "curated")
    B.WEIGHTS.write_text("x")
    B.CORPUS.mkdir()
    for name in B.CORPUS_FILES:
        (B.CORPUS / name).write_text("x")
    nope = tmp_path / "nope.pt"
    assert B.download_mb(nope) == B.WEIGHTS_MB


def test_fetch_all_skips_what_is_already_there(monkeypatch, tmp_path):
    """When all required files exist, fetch_all skips all downloads."""
    called = []
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "model.safetensors")
    monkeypatch.setattr(B, "INDEX", tmp_path / "index.pkl")
    monkeypatch.setattr(B, "CORPUS", tmp_path / "curated")
    B.WEIGHTS.write_text("x")
    B.INDEX.write_text("x")
    # Create corpus with all 11 required files
    B.CORPUS.mkdir()
    for name in B.CORPUS_FILES:
        (B.CORPUS / name).write_text("x")
    monkeypatch.setattr(B, "fetch_weights", lambda: called.append("w"))
    monkeypatch.setattr(B, "fetch_corpus", lambda: called.append("c"))
    monkeypatch.setattr(B, "build_index", lambda: called.append("i"))
    B.fetch_all(log=lambda *a: None)
    assert called == []


def test_fetch_weights_recovers_when_hub_names_are_still_none(monkeypatch, tmp_path):
    """Reproduces the fresh-install crash.

    bootstrap.py imports hf_hub_download/snapshot_download at module level
    inside a try/except ImportError, binding both to None when
    huggingface_hub is absent - deliberately, so install.py can import
    bootstrap.py BEFORE pip has installed anything. install.py then
    pip-installs huggingface_hub and calls bootstrap.fetch_all() in the SAME
    process, where the names are still None: installing a package does not
    rebind a name that was already resolved to None at import time.

    Every other test in this file monkeypatches B.hf_hub_download with a
    fake callable directly, which can never observe this - the name is never
    actually None when they run. This test instead simulates the real
    pre-install state (name is None) and the real post-install state
    (huggingface_hub becomes importable), without installing anything for
    real: it patches sys.modules['huggingface_hub'] with a fake module.
    """
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "model.safetensors")
    monkeypatch.setattr(B, "CONFIG_JSON", tmp_path / "config.json")
    # The pre-install state: both names are None, exactly as they are bound
    # when the module-level `from huggingface_hub import ...` hit ImportError.
    monkeypatch.setattr(B, "hf_hub_download", None)
    monkeypatch.setattr(B, "snapshot_download", None)

    asked = []

    def fake_hf_hub_download(repo_id, filename, local_dir, **kw):
        asked.append((repo_id, filename))
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        return str(p)

    fake_module = types.SimpleNamespace(
        hf_hub_download=fake_hf_hub_download,
        snapshot_download=lambda *a, **kw: None,
    )
    # huggingface_hub "becomes" importable, as it would after install.py's
    # pip install - without actually installing or importing the real thing.
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    B.fetch_weights()  # must not raise TypeError: 'NoneType' object is not callable

    assert [f for _, f in asked] == ["model.safetensors", "config.json"]
    assert all(r == B.MODEL_REPO for r, _ in asked)


def test_ensure_hub_leaves_a_monkeypatched_callable_alone(monkeypatch):
    """_ensure_hub only rebinds when the name is still None, so a test (or
    caller) that has already monkeypatched hf_hub_download/snapshot_download
    with a fake callable is unaffected - it must not be clobbered by a real
    huggingface_hub import."""
    sentinel_dl = lambda *a, **kw: "sentinel-dl"
    sentinel_snap = lambda *a, **kw: "sentinel-snap"
    monkeypatch.setattr(B, "hf_hub_download", sentinel_dl)
    monkeypatch.setattr(B, "snapshot_download", sentinel_snap)

    B._ensure_hub()

    assert B.hf_hub_download is sentinel_dl
    assert B.snapshot_download is sentinel_snap


# ------------------------------------------------- the pre-data-dir layout

def _legacy(monkeypatch, tmp_path):
    """A clone with artefacts beside the code, and an empty data directory."""
    clone, data = tmp_path / "clone", tmp_path / "data"
    (clone / "checkpoints").mkdir(parents=True)
    (clone / "retrieve").mkdir(parents=True)
    (clone / "curated").mkdir(parents=True)
    (clone / "checkpoints" / "model.safetensors").write_text("weights")
    (clone / "checkpoints" / "config.json").write_text("cfg")
    (clone / "retrieve" / "index.pkl").write_text("index")
    (clone / "retrieve" / "names.pkl").write_text("names")
    for name in B.CORPUS_FILES:
        (clone / "curated" / name).write_text("x")
    monkeypatch.setattr(B, "ROOT", clone)
    monkeypatch.setattr(B, "WEIGHTS", data / "checkpoints" / "model.safetensors")
    monkeypatch.setattr(B, "CONFIG_JSON", data / "checkpoints" / "config.json")
    monkeypatch.setattr(B, "CORPUS", data / "curated")
    monkeypatch.setattr(B, "INDEX", data / "index.pkl")
    monkeypatch.setattr(B.paths, "NAMES", data / "names.pkl")
    monkeypatch.setattr(B, "_migrated", False)
    return clone, data


def test_migration_moves_an_old_clone_into_the_data_directory(monkeypatch, tmp_path):
    """The bug this guards: a `git pull` that relocates the artefacts would
    otherwise read as an 800 MB re-download, with the old copy orphaned."""
    clone, data = _legacy(monkeypatch, tmp_path)
    B.migrate_legacy()
    assert B.WEIGHTS.read_text() == "weights"
    assert B.INDEX.read_text() == "index"
    assert B.paths.NAMES.read_text() == "names"
    assert B.corpus_complete()
    assert not (clone / "checkpoints" / "model.safetensors").exists()


def test_migration_leaves_nothing_to_download(monkeypatch, tmp_path):
    _legacy(monkeypatch, tmp_path)
    assert B.download_mb() == B.WEIGHTS_MB + B.CORPUS_MB   # before
    B.migrate_legacy()
    assert B.missing() == []
    assert B.download_mb() == 0                            # after


def test_migration_never_overwrites_the_data_directory(monkeypatch, tmp_path):
    """A newer download in place must win over an older clone's copy."""
    clone, data = _legacy(monkeypatch, tmp_path)
    B.WEIGHTS.parent.mkdir(parents=True)
    B.WEIGHTS.write_text("newer")
    B.migrate_legacy()
    assert B.WEIGHTS.read_text() == "newer"
    assert (clone / "checkpoints" / "model.safetensors").exists()


def test_migration_is_a_noop_with_no_old_clone(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "ROOT", tmp_path / "empty")
    monkeypatch.setattr(B, "WEIGHTS", tmp_path / "d" / "model.safetensors")
    monkeypatch.setattr(B, "_migrated", False)
    assert B.migrate_legacy() == []


def test_migration_survives_a_move_that_fails(monkeypatch, tmp_path):
    """A half-finished move must not be fatal - the artefact stays put and
    gets re-fetched, which beats losing it."""
    clone, data = _legacy(monkeypatch, tmp_path)
    monkeypatch.setattr(B.shutil, "move",
                        lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    assert B.migrate_legacy() == []
    assert (clone / "checkpoints" / "model.safetensors").exists()


def test_migration_runs_once_per_process(monkeypatch, tmp_path):
    _legacy(monkeypatch, tmp_path)
    assert len(B.migrate_legacy()) == 5
    assert B.migrate_legacy() == []
