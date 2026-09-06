"""The comparison is only meaningful if the windows never move."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "eval_ppl", Path(__file__).resolve().parent / "eval_ppl.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def test_windows_are_identical_across_calls():
    a = E.window_starts(1_000_000, 1024, 200)
    b = E.window_starts(1_000_000, 1024, 200)
    assert np.array_equal(a, b)


def test_windows_are_int64():
    """numpy defaults to int32 on Windows; 2.94e9 tokens wraps negative."""
    assert E.window_starts(3_000_000_000, 1024, 200).dtype == np.int64


def test_no_window_reads_past_the_end():
    n, block = 5000, 1024
    starts = E.window_starts(n, block, 50)
    assert starts.max() + block + 1 <= n


def test_large_offsets_stay_positive():
    starts = E.window_starts(2_940_000_000, 1024, 200)
    assert starts.min() >= 0 and starts.max() > 2_000_000_000


def test_windows_span_the_whole_file():
    starts = E.window_starts(1_000_000, 1024, 10)
    assert starts[0] == 0
    assert starts[-1] == 1_000_000 - 1024 - 1


def test_more_windows_than_tokens_is_clamped():
    starts = E.window_starts(1030, 1024, 500)
    assert len(starts) <= 6 and starts.max() + 1025 <= 1030


def test_tiny_file_does_not_crash():
    starts = E.window_starts(100, 1024, 10)
    assert len(starts) == 1 and starts[0] == 0
