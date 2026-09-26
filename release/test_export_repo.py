"""Tests for the public-tree export.

This is the one operation whose mistakes are published to strangers, so the
copy is an allowlist and the scan runs before anything leaves.
"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "export_repo", Path(__file__).resolve().parent / "export_repo.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

# This file is scanned by the tool it tests, so the developer username
# sentinel is assembled at runtime to avoid the literal appearing here.
USERNAME_SENTINEL = "pola" + "c"
# Same reason: written out, either of these would be caught in this file by
# the scan they are here to test.
WIN_HOME = "C:" + "\\" + "Users" + "\\" + "you"      # the README placeholder
DEV_PATH = "B:" + "\\" + "learning"                  # a real one


def fake_tree(tmp_path):
    """Create a synthetic source tree with one file for every INCLUDE entry."""
    src = tmp_path / "src"

    # Core directories and files from INCLUDE
    (src / "infer").mkdir(parents=True)
    (src / "infer" / "terminal.py").write_text("print('hi')\n")
    (src / "infer" / "test_terminal.py").write_text("def test_x(): pass\n")
    (src / "infer" / "engine.py").write_text("# infer module\n")
    (src / "infer" / "test_engine.py").write_text("def test_x(): pass\n")

    (src / "retrieve").mkdir(parents=True)
    (src / "retrieve" / "search.py").write_text("# retrieve module\n")
    (src / "retrieve" / "test_search.py").write_text("def test_x(): pass\n")

    (src / "train").mkdir(parents=True)
    (src / "train" / "trainer.py").write_text("# train module\n")
    (src / "train" / "run.sh").write_text("#!/bin/bash\n")
    (src / "train" / "run.bat").write_text("@echo off\n")
    (src / "train" / "test_trainer.py").write_text("def test_x(): pass\n")

    (src / "crawl").mkdir(parents=True)
    (src / "crawl" / "crawl.py").write_text("# crawl module\n")
    (src / "crawl" / "test_crawl.py").write_text("def test_x(): pass\n")

    (src / "release").mkdir(parents=True)
    (src / "release" / "export_repo.py").write_text("# export module\n")

    (src / "tokenizer").mkdir(parents=True)
    (src / "tokenizer" / "marvel_bpe_50257.model").write_text("m\n")
    (src / "tokenizer" / "marvel_bpe_50257.vocab").write_text("vocab\n")
    (src / "tokenizer" / "marvel_bpe_32000.model").write_text("m\n")

    (src / "bootstrap.py").write_text("# bootstrap\n")
    (src / "paths.py").write_text("# paths\n")
    (src / "edith_cli.py").write_text("# cli\n")
    (src / "pyproject.toml").write_text("[project]\n")
    (src / "test_paths.py").write_text("def test_x(): pass\n")
    (src / "install.py").write_text("# install\n")
    (src / "install.ps1").write_text("# install ps1\n")
    (src / "test_bootstrap.py").write_text("def test_x(): pass\n")
    (src / "test_install.py").write_text("def test_x(): pass\n")
    (src / "edith").write_text("#!/bin/bash\n")
    (src / "edith.cmd").write_text("@echo off\n")
    (src / ".gitattributes").write_text("*.py text\n")
    (src / "README.md").write_text("public\n")
    (src / "MEASUREMENTS.md").write_text("measurements\n")
    (src / "RELEASE.md").write_text("release\n")
    (src / "LICENSE").write_text("license\n")
    (src / "LICENSE-DATA").write_text("license data\n")
    (src / "ATTRIBUTION.md").write_text("attribution\n")

    (src / "media").mkdir(parents=True)
    (src / "media" / "demo.txt").write_text("media\n")

    # Files that should not cross over
    (src / "ROADMAP.md").write_text("the notebook\n")
    (src / "CLAUDE.md").write_text("working context\n")
    (src / "docs" / "superpowers").mkdir(parents=True)
    (src / "docs" / "superpowers" / "plan.md").write_text("internal\n")

    return src


def test_the_allowlist_copies_code_and_tests(tmp_path):
    src = fake_tree(tmp_path)
    E.export(src, tmp_path / "out")
    assert (tmp_path / "out" / "infer" / "terminal.py").exists()
    assert (tmp_path / "out" / "infer" / "test_terminal.py").exists()
    assert (tmp_path / "out" / "README.md").exists()


def test_the_notebook_does_not_cross(tmp_path):
    src = fake_tree(tmp_path)
    E.export(src, tmp_path / "out")
    for gone in ("ROADMAP.md", "CLAUDE.md", "docs/superpowers/plan.md"):
        assert not (tmp_path / "out" / gone).exists(), gone


def test_only_the_tokenizer_actually_used_ships(tmp_path):
    """The 32k tokenizer was measured and rejected. Shipping it only raises
    the question of which one to use."""
    src = fake_tree(tmp_path)
    E.export(src, tmp_path / "out")
    assert (tmp_path / "out" / "tokenizer" / "marvel_bpe_50257.model").exists()
    assert not (tmp_path / "out" / "tokenizer" / "marvel_bpe_32000.model").exists()


def test_a_public_gitignore_is_written(tmp_path):
    src = fake_tree(tmp_path)
    E.export(src, tmp_path / "out")
    text = (tmp_path / "out" / ".gitignore").read_text()
    for line in ("checkpoints/", "curated/", "retrieve/*.pkl"):
        assert line in text


def test_a_machine_specific_path_fails_the_scan(tmp_path):
    src = fake_tree(tmp_path)
    (src / "infer" / "terminal.py").write_text(
        f"P = r'C:\\\\Users\\\\{USERNAME_SENTINEL}\\\\weights'\n")
    E.export(src, tmp_path / "out")
    hits = E.scan(tmp_path / "out")
    assert any("terminal.py" in h for h in hits)


def test_a_token_shaped_string_fails_the_scan(tmp_path):
    src = fake_tree(tmp_path)
    (src / "infer" / "terminal.py").write_text(
        "TOKEN = 'hf_" + "a" * 34 + "'\n")
    E.export(src, tmp_path / "out")
    assert E.scan(tmp_path / "out")


def test_a_clean_tree_scans_clean(tmp_path):
    src = fake_tree(tmp_path)
    E.export(src, tmp_path / "out")
    assert E.scan(tmp_path / "out") == []


def test_it_refuses_a_non_empty_output_directory(tmp_path):
    src = fake_tree(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "something.txt").write_text("mine\n")
    with pytest.raises(SystemExit):
        E.export(src, out)
    assert (out / "something.txt").exists()


def test_every_include_entry_arrives_in_output(tmp_path):
    """A typo in any INCLUDE glob passes all existing tests. This test catches
    it by verifying that one file from each pattern arrived in the output."""
    src = fake_tree(tmp_path)
    E.export(src, tmp_path / "out")
    out = tmp_path / "out"

    # Sample one file from each INCLUDE pattern to verify it arrived
    expected_files = [
        "infer/terminal.py",
        "retrieve/search.py",
        "train/trainer.py",
        "train/run.sh",
        "train/run.bat",
        "crawl/crawl.py",
        "release/export_repo.py",
        "tokenizer/marvel_bpe_50257.model",
        "tokenizer/marvel_bpe_50257.vocab",
        "bootstrap.py",
        "test_bootstrap.py",
        "edith",
        "edith.cmd",
        ".gitattributes",
        "README.md",
        "MEASUREMENTS.md",
        "LICENSE",
        "LICENSE-DATA",
        "ATTRIBUTION.md",
        "media/demo.txt",
    ]
    for f in expected_files:
        assert (out / f).exists(), f"Missing: {f}"


def test_missing_required_file_raises(tmp_path):
    """If a REQUIRED file is missing from the source, export raises SystemExit
    rather than shipping an incomplete repo."""
    src = tmp_path / "src"
    (src / "infer").mkdir(parents=True)
    (src / "infer" / "engine.py").write_text("# code\n")
    (src / "README.md").write_text("readme\n")
    # Missing: LICENSE, LICENSE-DATA, ATTRIBUTION.md, MEASUREMENTS.md
    with pytest.raises(SystemExit):
        E.export(src, tmp_path / "out")


def test_export_prefix_credential_is_caught(tmp_path):
    """The credential env assignment pattern should catch export-prefixed secrets."""
    src = fake_tree(tmp_path)
    (src / "train" / "setup.sh").write_text(
        "export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n")
    E.export(src, tmp_path / "out")
    hits = E.scan(tmp_path / "out")
    assert any("setup.sh" in h and "credential env assignment" in h for h in hits)


def test_bash_config_lines_do_not_match(tmp_path):
    """The credential env assignment pattern should not match legitimate bash config."""
    src = fake_tree(tmp_path)
    (src / "train" / "config.sh").write_text(
        "WORKDIR=/root/marvel-slm\n"
        "BUSY_PROCS='trainer.py|eval_ppl|strip_ckpt'\n"
        "CKPT=checkpoints/stage3/latest.pt\n")
    E.export(src, tmp_path / "out")
    hits = E.scan(tmp_path / "out")
    # These bash config lines should not produce any scan hits
    assert not any("config.sh" in h for h in hits)


def test_developer_username_isolated(tmp_path):
    """The developer username pattern should be pinned to its own test.
    A bare comment with the sentinel should match only the username pattern."""
    src = fake_tree(tmp_path)
    (src / "infer" / "notes.py").write_text(
        f"# maintainer: {USERNAME_SENTINEL}\n")
    E.export(src, tmp_path / "out")
    hits = E.scan(tmp_path / "out")
    # Should have exactly one hit from the username pattern
    username_hits = [h for h in hits if "developer username" in h and "notes.py" in h]
    assert len(username_hits) == 1
    assert "developer username" in username_hits[0]

def test_the_web_app_ships_with_its_page(tmp_path):
    """FOUND 2026-09-26, before it shipped: the allowlist reads `infer/*.py`,
    which takes infer/web.py and leaves infer/web/ behind. The public repo
    would have had the server and none of the page it serves, so `edith
    --web` would answer 404 for every file it asked for - and nothing in the
    export would have complained.
    """
    src = fake_tree(tmp_path)
    (src / "infer" / "web").mkdir(parents=True, exist_ok=True)
    for name in ("page.html", "app.css", "app.js"):
        (src / "infer" / "web" / name).write_text("x")
    out = tmp_path / "out"
    E.export(src, out)
    for name in ("page.html", "app.css", "app.js"):
        assert (out / "infer" / "web" / name).exists(), name


def test_the_sidecars_ship(tmp_path):
    """They are data, not code, so an allowlist written around *.py drops
    them silently. Without images.json.gz every answer loses its art;
    without legacy.json.gz the picker loses "others who have gone by this
    name" - both degrade quietly rather than failing, which is worse.
    """
    src = fake_tree(tmp_path)
    (src / "retrieve" / "images.json.gz").write_bytes(b"x")
    (src / "retrieve" / "legacy.json.gz").write_bytes(b"x")
    out = tmp_path / "out"
    E.export(src, out)
    assert (out / "retrieve" / "images.json.gz").exists()
    assert (out / "retrieve" / "legacy.json.gz").exists()


def test_the_packaging_layer_ships(tmp_path):
    """FOUND 2026-09-26 by exporting into a clone of the public repo and
    reading the diff before committing it: paths.py, edith_cli.py and
    pyproject.toml lived ONLY in the public repo, and bootstrap.py,
    terminal.py, search.py, resolve.py and build_names.py all load paths.py.
    Copying the private tree over them would have left an installed EDITH
    looking for its weights inside site-packages - and `uv tool install` with
    no pyproject.toml to read at all.
    """
    src = fake_tree(tmp_path)
    out = tmp_path / "out"
    E.export(src, out)
    for name in ("paths.py", "edith_cli.py", "pyproject.toml"):
        assert (out / name).exists(), name


def test_both_licence_files_begin_with_their_licence(tmp_path):
    """GitHub labels a licence tab by DETECTING the text - licensee matches a
    file that IS the licence, and a two-line preamble in front of "MIT
    License" drops it below the threshold. Both tabs on the public repo read
    a generic "License" because of one, and the repo showed no MIT badge at
    all. What the preambles said is in README's "Licenses and credit" and in
    ATTRIBUTION.md, which is where a reader looks anyway.
    """
    root = Path(__file__).resolve().parent.parent
    assert (root / "LICENSE").read_text(
        encoding="utf-8").startswith("MIT License")
    assert (root / "LICENSE-DATA").read_text(
        encoding="utf-8").startswith("Attribution-ShareAlike 4.0 International")


def test_the_dead_installer_does_not_come_back(tmp_path):
    """install.py, install.ps1 and test_install.py were DELETED publicly in
    e903ebc - installation runs through `uv` now and all three are dead
    paths. The allowlist still named them, so every export resurrected three
    files the public repo had deliberately removed.
    """
    src = fake_tree(tmp_path)
    out = tmp_path / "out"
    E.export(src, out)
    for dead in ("install.py", "install.ps1", "test_install.py"):
        assert not (out / dead).exists(), dead


def test_the_public_readme_survives_an_export(tmp_path):
    """The PUBLIC repo owns README.md. Its copy carries the banner, the demo
    gif and the uv instructions - about 80 lines the private one has never
    had - and the export used to overwrite it wholesale with the private
    copy, which is the developer's file, not the visitor's.
    """
    src = fake_tree(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "README.md").write_text("the public one\n")
    E.export(src, out, force=True)
    assert (out / "README.md").read_text() == "the public one\n"


def test_the_readmes_windows_placeholder_is_not_a_leak(tmp_path):
    """FOUND 2026-09-26 by the scan refusing a real export: the README says
    where EDITH keeps things, and the Windows half of that sentence is a
    drive path. It names no user, and it has been public since the repo was,
    so the scan must read it as documentation rather than as a leak.
    """
    src = fake_tree(tmp_path)
    (src / "MEASUREMENTS.md").write_text(
        f"All of that lands in `~/.edith` (`{WIN_HOME}\\.edith` on "
        "Windows) - one directory.\n")
    E.export(src, tmp_path / "out")
    assert not [h for h in E.scan(tmp_path / "out") if "MEASUREMENTS" in h]


def test_a_real_path_on_the_same_line_is_still_caught(tmp_path):
    """The placeholder is removed from the line, not the line from the scan."""
    src = fake_tree(tmp_path)
    (src / "MEASUREMENTS.md").write_text(
        f"`{WIN_HOME}\\.edith`, which on this machine is `{DEV_PATH}\\x`\n")
    E.export(src, tmp_path / "out")
    hits = E.scan(tmp_path / "out")
    assert any("MEASUREMENTS.md" in h and "absolute drive path" in h
               for h in hits)


def test_a_tree_with_no_readme_still_gets_one(tmp_path):
    """Owned by the public repo is not the same as never copied: a tree that
    has no README at all is a fresh export, not a repo with something to
    protect, and README.md is REQUIRED - without the fallback the export
    would refuse to finish.
    """
    src = fake_tree(tmp_path)
    out = tmp_path / "out"
    E.export(src, out)
    assert (out / "README.md").read_text() == "public\n"
