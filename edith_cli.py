"""The `edith` command.

`[project.scripts]` needs something importable to point at, and EDITH loads
its own modules by file path rather than by import name. This is the one shim
between the two: it hands control to infer/terminal.py's main().

The point is not tidiness. The old `edith` launcher was `exec python3 ...`,
which runs whatever `python3` happens to be first on PATH - and on a stock Mac
that is a 3.9 with no torch, so the first command in the README ended in a
ModuleNotFoundError traceback. A console entry point is written by the
installer with the absolute path of the interpreter that has the
dependencies baked into its shebang, so it cannot pick the wrong one. It also
arrives executable, and arrives as edith.exe on Windows, which is the other
two launcher bugs gone.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    """Run the terminal. Returns its exit code."""
    spec = importlib.util.spec_from_file_location(
        "terminal", ROOT / "infer" / "terminal.py")
    terminal = importlib.util.module_from_spec(spec)
    sys.modules["terminal"] = terminal
    spec.loader.exec_module(terminal)
    return terminal.main()


if __name__ == "__main__":
    raise SystemExit(main())
