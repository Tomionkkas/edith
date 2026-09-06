"""A biased slice is invisible: training runs fine on shard 0 alone."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "slice_bin", Path(__file__).resolve().parent / "slice_bin.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def test_starts_are_int64():
    """2.94e9 tokens wraps int32, which numpy defaults to on Windows."""
    starts, _ = S.chunk_starts(2_941_946_794, 100_000_000, 100)
    assert starts.dtype == np.int64 and starts.min() >= 0


def test_chunks_span_the_whole_file():
    """A prefix would sample only the first of four concatenated shards."""
    starts, per = S.chunk_starts(1_000_000, 10_000, 10)
    assert starts[0] == 0
    assert starts[-1] + per <= 1_000_000
    assert starts[-1] > 900_000


def test_no_chunk_reads_past_the_end():
    starts, per = S.chunk_starts(5_000, 1_000, 10)
    assert starts.max() + per <= 5_000


def test_chunks_are_contiguous_runs(tmp_path):
    """The loader reads consecutive tokens; a shuffled slice would splice
    unrelated documents mid-sentence at every boundary."""
    src = tmp_path / "src.bin"
    np.arange(0, 50_000, dtype=np.uint16).tofile(src)
    dst = tmp_path / "dst.bin"
    S.slice_bin(src, dst, tokens=5_000, chunks=5)
    out = np.fromfile(dst, dtype=np.uint16)
    per = 1_000
    for c in range(5):
        run = out[c * per:(c + 1) * per].astype(np.int64)
        assert np.all(np.diff(run) == 1), "chunk is not a contiguous run"


def test_writes_the_requested_size(tmp_path):
    src = tmp_path / "src.bin"
    np.arange(0, 50_000, dtype=np.uint16).tofile(src)
    dst = tmp_path / "dst.bin"
    n = S.slice_bin(src, dst, tokens=5_000, chunks=5)
    assert n == 5_000
    assert dst.stat().st_size == 10_000


def test_dtype_survives(tmp_path):
    src = tmp_path / "src.bin"
    np.array([65535, 0, 12345], dtype=np.uint16).repeat(400).tofile(src)
    dst = tmp_path / "dst.bin"
    S.slice_bin(src, dst, tokens=600, chunks=3)
    assert np.fromfile(dst, dtype=np.uint16).max() == 65535
