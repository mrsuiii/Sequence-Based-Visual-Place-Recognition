"""Tests for descriptors.py's `cached`. `seqslam_descriptors`/`dino_descriptors` need real
frame data (smoke-tested against the dataset elsewhere, not here)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from routealign.descriptors import cached


def test_cached_computes_and_persists_on_a_miss(tmp_path) -> None:
    """A first call with no existing cache entry must call `fn`, and write both a `.npy` and a
    `.json` sidecar."""
    calls = []

    def fn():
        calls.append(1)
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = cached(fn, "k", tmp_path)

    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    assert len(calls) == 1
    assert (tmp_path / "k.npy").exists()
    assert (tmp_path / "k.json").exists()


def test_cached_hit_does_not_recompute(tmp_path) -> None:
    """A second call with the same key must load from disk, not call `fn` again."""
    calls = []

    def fn():
        calls.append(1)
        return np.array([42.0], dtype=np.float32)

    first = cached(fn, "k", tmp_path)
    second = cached(fn, "k", tmp_path)

    assert len(calls) == 1  # fn only ran once
    np.testing.assert_array_equal(first, second)


def test_cached_force_recomputes_even_on_a_hit(tmp_path) -> None:
    """`force=True` must recompute regardless of an existing cache entry."""
    calls = []

    def fn():
        calls.append(1)
        return np.array([len(calls)], dtype=np.float32)

    cached(fn, "k", tmp_path)
    result = cached(fn, "k", tmp_path, force=True)

    assert len(calls) == 2
    np.testing.assert_array_equal(result, [2.0])


def test_cached_sidecar_records_shape_dtype_and_extra_provenance(tmp_path) -> None:
    """The JSON sidecar must record the array's shape/dtype and whatever `extra` provenance the
    caller supplies (e.g. source SHA-256 prefixes, the parameters `fn` closed over)."""
    fn = lambda: np.zeros((5, 3), dtype=np.float64)  # noqa: E731
    cached(fn, "k", tmp_path, extra={"source_sha256": "abc123", "params": {"window": 50}})

    sidecar = json.loads((tmp_path / "k.json").read_text())
    assert sidecar["key"] == "k"
    assert sidecar["shape"] == [5, 3]
    assert sidecar["dtype"] == "float64"
    assert sidecar["source_sha256"] == "abc123"
    assert sidecar["params"] == {"window": 50}


def test_cached_different_keys_do_not_collide(tmp_path) -> None:
    """Two different keys in the same cache_dir must not overwrite each other."""
    a = cached(lambda: np.array([1.0]), "a", tmp_path)
    b = cached(lambda: np.array([2.0]), "b", tmp_path)

    np.testing.assert_array_equal(a, [1.0])
    np.testing.assert_array_equal(b, [2.0])
    assert (tmp_path / "a.npy").exists()
    assert (tmp_path / "b.npy").exists()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cached_round_trips_dtype(tmp_path, dtype) -> None:
    """`np.save`/`np.load` must preserve dtype exactly -- downstream code relies on this (e.g.
    descriptors staying float32 to keep the cache small)."""
    arr = np.array([1.0, 2.0], dtype=dtype)
    result = cached(lambda: arr, "k", tmp_path)
    assert result.dtype == dtype
