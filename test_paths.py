"""The two layouts EDITH ships as.

The same code is a development checkout here and an installed command in a
virtualenv there, and the difference is only ever WHERE the 1.3 GB goes.
Getting it wrong is not subtle: an installed EDITH that reads "beside me"
downloads half a gigabyte into site-packages, and a checkout that reads
~/.edith trains in one place and answers from another.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "edith_paths", Path(__file__).resolve().parent / "paths.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

ROOT = Path(__file__).resolve().parent


def test_a_checkout_keeps_its_artefacts_beside_the_code(tmp_path):
    root = tmp_path / "clone"
    (root / ".git").mkdir(parents=True)
    assert P.data_home(root, None, tmp_path / "home") == root


def test_a_worktree_counts_as_a_checkout(tmp_path):
    """Inside a git worktree `.git` is a FILE pointing at the real one."""
    root = tmp_path / "wt"
    root.mkdir()
    (root / ".git").write_text("gitdir: ../real/.git\n")
    assert P.data_home(root, None, tmp_path / "home") == root


def test_an_install_uses_the_data_directory(tmp_path):
    """No .git: this is site-packages, and beside-the-code is a folder the
    user has never heard of."""
    root = tmp_path / "site-packages" / "edith"
    root.mkdir(parents=True)
    home = tmp_path / "home"
    assert P.data_home(root, None, home) == home / ".edith"


def test_edith_home_wins_over_both(tmp_path):
    root = tmp_path / "clone"
    (root / ".git").mkdir(parents=True)
    assert P.data_home(root, str(tmp_path / "elsewhere"),
                       tmp_path / "home") == tmp_path / "elsewhere"


def test_this_repo_resolves_to_the_paths_it_has_always_used(tmp_path):
    """The whole point of the checkout branch: adopting paths.py must not
    move one file in this tree. These five are what CLAUDE.md, the harnesses
    and train/ all name."""
    assert P.CHECKOUT
    assert P.WEIGHTS == ROOT / "checkpoints" / "model.safetensors"
    assert P.CONFIG_JSON == ROOT / "checkpoints" / "config.json"
    assert P.CORPUS == ROOT / "curated"
    assert P.INDEX == ROOT / "retrieve" / "index.pkl"
    assert P.NAMES == ROOT / "retrieve" / "names.pkl"


def test_the_config_stays_out_of_the_tree(tmp_path):
    """A theme preference written into the repo is an untracked file every
    `git status` then shows."""
    assert P.CONFIG == Path.home() / ".edith" / "config.json"
