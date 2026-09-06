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
    "LICENSE",
    "LICENSE-DATA",
    "ATTRIBUTION.md",
    "MEASUREMENTS.md",
)

# Everything that becomes public. Tests come with their modules - the suite is
# a large part of what makes the README's claims checkable.
INCLUDE = [
    "infer/*.py",
    "retrieve/*.py",
    "train/*.py", "train/*.sh", "train/*.bat",
    "crawl/*.py",
    "release/*.py",
    "tokenizer/marvel_bpe_50257.model",
    "tokenizer/marvel_bpe_50257.vocab",
    "bootstrap.py", "paths.py", "edith_cli.py", "pyproject.toml",
    "test_bootstrap.py",
    "edith", "edith.cmd",
    ".gitattributes",
    "README.md", "MEASUREMENTS.md",
    "LICENSE", "LICENSE-DATA", "ATTRIBUTION.md",
    "media/*",
]

PUBLIC_GITIGNORE = """\
# ---- fetched on first run, never committed ---------------------------------
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
.pytest_cache/

# ---- editors / os -----------------------------------------------------------
.vscode/
.idea/
.DS_Store
Thumbs.db

# ---- agent scratch ----------------------------------------------------------
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
