#!/usr/bin/env python3
"""Assemble the public EDITH tree from an allowlist, and scan it before it
leaves.

    py release/export_repo.py ../edith-public

An ALLOWLIST, not a denylist: a denylist misses files added after it was
written, and this is the one operation whose mistakes reach strangers.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Files that must be present in the output tree. If any is missing, export
# raises SystemExit rather than shipping an incomplete repo.
REQUIRED = (
    "README.md",
    # Without these three the published repo is not installable: no entry
    # point, and no file telling uv what to build.
    "pyproject.toml",
    "edith_cli.py",
    "paths.py",
    "LICENSE",
    "LICENSE-DATA",
    "ATTRIBUTION.md",
    "MEASUREMENTS.md",
)

# Everything that becomes public. Tests come with their modules - the suite is
# a large part of what makes the README's claims checkable.
INCLUDE = [
    "infer/*.py",
    # The page `edith --web` serves. `infer/*.py` takes web.py and leaves
    # this behind, which would ship the server without anything to serve -
    # every request a 404, and nothing in the export complaining.
    "infer/web/*",
    "retrieve/*.py",
    # Sidecars: data, not code, so an allowlist written around *.py drops
    # them silently. Without images.json.gz every answer loses its art;
    # without legacy.json.gz the picker loses "others who have gone by this
    # name". Both degrade quietly rather than failing, which is worse.
    "retrieve/images.json.gz",
    "retrieve/legacy.json.gz",
    "train/*.py", "train/*.sh", "train/*.bat",
    "crawl/*.py",
    "release/*.py",
    "tokenizer/marvel_bpe_50257.model",
    "tokenizer/marvel_bpe_50257.vocab",
    # install.py, install.ps1 and test_install.py are deliberately absent:
    # they were deleted publicly in e903ebc once installation ran through
    # `uv`, and an allowlist that still names them resurrects three dead
    # files on every export.
    "bootstrap.py",
    "test_bootstrap.py",
    # The packaging layer. paths.py is loaded by bootstrap.py, terminal.py,
    # search.py, resolve.py and build_names.py, and it is what keeps an
    # installed EDITH's 1.3 GB out of site-packages; pyproject.toml is what
    # `uv tool install` reads; edith_cli.py is the entry point it names.
    "paths.py", "test_paths.py",
    "edith_cli.py", "pyproject.toml",
    "edith", "edith.cmd",
    ".gitattributes",
    "README.md", "MEASUREMENTS.md",
    "LICENSE", "LICENSE-DATA", "ATTRIBUTION.md",
    "media/*",
]

# Files the PUBLIC repo owns. Its README carries the banner, the demo gif and
# the uv instructions - about 80 lines the private copy has never had, because
# the private one is the developer's file and this one is the visitor's. Copied
# only into a tree that has none, so a first export still produces a complete
# repo and every later one leaves the public copy alone.
PUBLIC_OWNED = ("README.md",)

PUBLIC_GITIGNORE = """\
# ---- fetched by bootstrap.py, never committed -------------------------------
checkpoints/
curated/
retrieve/*.pkl
data/
raw/

# ---- run artefacts ----------------------------------------------------------
out/
*.log
*.err
*.pt
*.bin
*.parquet
*.jsonl

# ---- python -----------------------------------------------------------------
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
dist/
*.whl
.pytest_cache/

# ---- editors / os -----------------------------------------------------------
.vscode/
.idea/
.DS_Store
Thumbs.db

# ---- agent scratch ----------------------------------------------------------
# Added to the public repo by hand once already ("Agent scratch notes do not
# belong in the public repo"). It belongs HERE, or the next export silently
# undoes it.
.superpowers/
.claude/
"""

# Generic shapes, never a literal secret - a scanner that contains the key it
# looks for has published it.
SCANS = [
    ("absolute drive path", r"\b[A-Za-z]:\\{1,2}[A-Za-z0-9_]"),
    ("developer username", r"\bpolac\b"),
    ("hardcoded credential", r"(?i)\b(api[_-]?key|secret|token|password|passwd|access[_-]?key)\b\s*[:=]\s*['\"][^'\"]{12,}['\"]"),
    ("aws access key id", r"\bAKIA[0-9A-Z]{16}\b"),
    ("credential env assignment", r"(?mi)^\s*(?:export\s+)?[A-Z0-9_]*(KEY|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*=\S{16,}$"),
    ("huggingface token", r"\bhf_[A-Za-z0-9]{30,}"),
    ("github token", r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


# Documented placeholders, removed from a line before it is scanned - not
# whole lines excused. `C:\Users\you\.edith` is the Windows half of the README
# sentence about where EDITH keeps things: it names no user, and a real path
# beside it is still caught, by this rule on the rest of the line and by the
# developer-username rule.
PLACEHOLDERS = (r"C:\Users\you",)


def export(src: Path, out: Path, force: bool = False) -> list[Path]:
    src, out = Path(src), Path(out)
    if out.exists() and any(out.iterdir()) and not force:
        raise SystemExit(
            f"{out} is not empty. Pass --force only if you mean to overwrite it.")

    written: list[Path] = []
    for pattern in INCLUDE:
        for path in sorted(src.glob(pattern)):
            if not path.is_file():
                continue
            target = out / path.relative_to(src)
            if (target.exists()
                    and path.relative_to(src).as_posix() in PUBLIC_OWNED):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            written.append(target)

    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text(PUBLIC_GITIGNORE, encoding="utf-8")
    written.append(out / ".gitignore")

    # Verify all required files arrived
    missing = [f for f in REQUIRED if not (out / f).exists()]
    if missing:
        raise SystemExit(
            f"Export incomplete - missing: {', '.join(missing)}")

    return written


def scan(tree: Path) -> list[str]:
    """Offending `path:line: label` strings. Empty means clean."""
    hits: list[str] = []
    for path in sorted(Path(tree).rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                      # binaries: the tokenizer model
        for n, line in enumerate(text.splitlines(), 1):
            for placeholder in PLACEHOLDERS:
                line = line.replace(placeholder, "")
            for label, pattern in SCANS:
                if re.search(pattern, line):
                    hits.append(f"{path.relative_to(tree)}:{n}: {label}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="directory to assemble the public tree in")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src = Path(__file__).resolve().parent.parent
    written = export(src, Path(a.out), a.force)
    print(f"{len(written)} files -> {a.out}")

    hits = scan(Path(a.out))
    if hits:
        print("\nSCAN FAILED - do not push:", file=sys.stderr)
        for h in hits:
            print(f"  {h}", file=sys.stderr)
        return 1
    print("scan clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
