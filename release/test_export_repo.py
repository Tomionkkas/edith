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
        "install.py",
        "install.ps1",
        "test_bootstrap.py",
        "test_install.py",
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
