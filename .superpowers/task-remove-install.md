# Remove the dead install.py path

## Files deleted
- `install.py`
- `install.ps1`
- `test_install.py`

(via `git rm`, staged in the working tree)

## Messages rewritten in `infer/terminal.py`

### DECLINED
Before:
```
DECLINED = """
No problem. When you want them:

    edith                  run it again and say yes
    python3 install.py     from a clone, with more output
"""
```
After:
```
DECLINED = """
No problem. When you want them:

    edith                  run it again and say yes
"""
```
Second line dropped, no replacement route added — running `edith` again is
the only route now.

### FETCH_FAILED
Before:
```
FETCH_FAILED = """
That did not finish: {error}

What already downloaded is kept - running EDITH again resumes rather than
starting over. From a clone, python3 install.py does the same job with more
output along the way.
"""
```
After:
```
FETCH_FAILED = """
That did not finish: {error}

What already downloaded is kept - running EDITH again resumes rather than
starting over.
"""
```
Trailing sentence about `install.py` removed; the true/useful part (partial
progress is kept, re-running resumes) is unchanged.

### STILL_MISSING
Before:
```
STILL_MISSING = """
Still missing after fetching:

  {missing}

Something did not arrive. Try again, or run python3 install.py from a clone
for more detail.
"""
```
After:
```
STILL_MISSING = """
Still missing after fetching:

  {missing}

Something did not arrive. Running edith again retries. Deleting the data
directory (~/.edith, or wherever EDITH_HOME points) forces a clean re-fetch.
"""
```
Replacement advice checked against `paths.py`: `HOME = Path(os.environ.get("EDITH_HOME") or Path.home() / ".edith")` — so "`~/.edith`, or wherever `EDITH_HOME` points" is accurate.

## `bootstrap.py`

### ImportError in `_ensure_hub()`
Before:
```
raise ImportError(
    "huggingface_hub is required to fetch EDITH's weights/corpus "
    "but is not installed. Run install.py again, or "
    "`py -m pip install huggingface_hub` yourself."
) from e
```
After:
```
raise ImportError(
    "huggingface_hub is required to fetch EDITH's weights/corpus "
    "but is not installed. Run `uv tool install --force "
    "git+https://github.com/Tomionkkas/edith`, or "
    "`py -m pip install huggingface_hub` yourself."
) from e
```
Kept the pip-install fallback for someone working from a clone in a venv.

### Docstrings updated (all install.py references removed, no explanation deleted)
- Module docstring (was line 4): "Imported by install.py and by infer/terminal.py's first-run prompt. One module on purpose - two fetchers would eventually name two different repos." → "Imported by infer/terminal.py's first-run prompt: Terminal.boot() calls fetch_all() when something is missing. One module on purpose - a second fetcher would eventually name a different repo."
- Data-directory load comment (was line 24): "...see install.py." → "...see infer/terminal.py's load of this module." (terminal.py's `_load("bootstrap", "bootstrap.py")` is the real example of the file-path-loading pattern now that install.py is gone.)
- try/except comment (was line 60): `# before install.py has run` → `# before huggingface_hub is installed`
- `_ensure_hub()` docstring (was lines 67-71, the "why lazy" explanation): rewrote to describe the real remaining scenario — a bare clone that has not set up its dependencies yet, and the same-process re-import race the test suite exercises (`test_bootstrap.py`'s fake-module monkeypatch) — instead of install.py's old pip-install-then-fetch-in-one-process behavior. The load-bearing reasoning (installing a package does not rebind an already-resolved name, so `_ensure_hub` re-resolves on first use) is unchanged.
- `migrate_legacy()` docstring (was line 209): "Called explicitly by install.py and by Terminal.boot()..." → "Called explicitly by Terminal.boot()'s first-run prompt and by fetch_all()..." (both are real callers: `offer_bootstrap()` calls `bootstrap.migrate_legacy()` directly, and `fetch_all()` also calls it.)

## `paths.py`
Docstring (was line 17): "...see install.py's load of bootstrap.py." → "...see infer/terminal.py's load of bootstrap.py." (same reasoning as above — citing a real, still-existing example of the cross-file, load-by-path pattern.)

## Five tests fixed in `infer/test_terminal.py`

All five previously asserted the literal substring `"install.py"` in captured stdout. Each now pins the actual recovery advice its scenario's message gives, so it still fails if that advice regresses or disappears — none were weakened to a trivially-true assertion.

1. **`test_a_missing_checkpoint_explains_itself`** (was line 209): scenario is `term.boot()` on a non-TTY with a missing checkpoint → prints `DECLINED`. Changed `assertIn("install.py", ...)` to `assertIn("run it again", ...)`, pinning the one remaining route text. Also tightened the docstring's stale "install command" phrase to "retry instruction" since it directly describes this assertion.
2. **`test_ctrl_c_on_the_prompt_is_a_decline_not_a_crash`** (was line 297): Ctrl-C at the prompt is treated as "n" → `DECLINED` prints. Changed to `assertIn("run it again", ...)`.
3. **`test_a_failed_fetch_explains_itself_and_stops`** (was line 316): `fetch_all()` raises `ConnectionError` → `FETCH_FAILED` prints. Changed to `assertIn("running EDITH again resumes", ...)`, pinning the "partial progress kept / re-run resumes" promise that survived the rewrite (this phrase sits entirely on one line of the message, so the multi-line format string can't accidentally split it).
4. **`test_declining_is_not_an_error`** (was line 344): explicit "n" answer → `DECLINED` prints. Changed to `assertIn("run it again", shown)`.
5. **`test_a_non_tty_is_never_prompted`** (was line 359): non-TTY, missing checkpoint → `DECLINED` prints without prompting. Changed to `assertIn("run it again", out.getvalue())`.

None of these five exercise `STILL_MISSING` (that message's own test, `test_a_fetch_that_leaves_something_missing_is_reported_not_trusted`, only ever asserted on `"model.safetensors"` and needed no change).

## Test suite results (both run in the foreground, no backgrounding)
- `py -m pytest -q --ignore=infer/test_terminal_session.py` → **1017 passed, 101 skipped**
- `py -m pytest -q infer/test_terminal_session.py` → **8 skipped**

Both green. No module in the repo was imported directly to check the edits — verification was via `git status`/`grep`/`py -m py_compile` (byte-compiles without executing module-level code) plus the two required pytest invocations.

## References to the deleted files found but left untouched (out of this task's explicit scope)

The task named exact files/lines to edit (`infer/terminal.py`'s three messages, `bootstrap.py` lines ~4/24/60/68/209, `paths.py` ~line 17, and the five named tests). The following also mention `install.py`/`install.ps1`/`test_install.py` but were not in that list, are not user-facing, and do not affect either test suite (verified: both suites pass as-is), so they were left alone:

- `infer/terminal.py:26-27` — the exact same "data directory loaded by file path... see install.py" comment pattern that was fixed in `bootstrap.py` and `paths.py`, but this instance in `terminal.py` was not named in the task.
- `retrieve/resolve.py:30-31`, `retrieve/search.py:31-32`, `retrieve/build_names.py:21-22` — three more copies of that same comment pattern.
- `test_bootstrap.py:3`, `:181-182`, `:214` — docstring/comment prose describing install.py's old pip-install-then-fetch-in-one-process behavior (the scenario this file's tests simulate). Not in the task's file list.
- `.gitignore:1` — comment header `# ---- fetched by install.py, never committed ----`.
- `MEASUREMENTS.md:251` — historical measurement-log entry naming `test_install.py` as a file added in an earlier phase; this is a record of the past, not current-state documentation.
- `release/export_repo.py:38,39,48` — the public-export `INCLUDE` allowlist still lists `"install.py"`, `"install.ps1"`, `"test_install.py"`, and the generated `PUBLIC_GITIGNORE` string still says "fetched by install.py". Since these are literal glob patterns, `src.glob("install.py")` now simply matches nothing (no error, nothing raised) — export silently stops shipping these three files rather than failing. Functionally harmless but a stale manifest.
- `release/test_export_repo.py:55,56,58,165-168` — this test's `fake_tree()` fixture creates its own synthetic `install.py`/`install.ps1`/`test_install.py` files in a temp directory and asserts they get copied; it never touches the real repo files, so deleting the real ones has no effect on this test either way.

None of the above caused a test failure or required a "fix" — they're stale references that fall outside the task's stated scope, flagged here per the instruction to report anything referencing the deleted files that wasn't cleanly addressed.

## Commit
Committed locally only (not pushed), on branch `packaging-and-terminal-fixes`.
