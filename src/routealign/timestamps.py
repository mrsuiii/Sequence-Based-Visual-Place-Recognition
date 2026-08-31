"""Pure functions on the int64-nanosecond timestamp arrays. No file I/O beyond reading the text
file, no video. Step 1 (plan.md §3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_timestamps(path: str | Path) -> NDArray[np.int64]:
    """One capture timestamp per line, nanoseconds since the Unix epoch. int64 throughout — never
    float (epoch-ns values are ~1.7e18, past float64's 2**53 exact-integer range; a naive float
    parse would silently lose precision in exactly the interval arithmetic Task 1 depends on).
    Fails loudly on a non-monotone or duplicate-timestamp file rather than coercing it."""
    with open(path) as f:
        values = [int(line) for line in f if line.strip()]
    if not values:
        raise ValueError(f"{path}: no timestamps found")
    ts = np.array(values, dtype=np.int64)
    diffs = np.diff(ts)
    if not np.all(diffs > 0):
        bad = int(np.argmin(diffs))
        raise ValueError(
            f"{path}: timestamps are not strictly increasing at line {bad + 1} "
            f"({ts[bad]} -> {ts[bad + 1]})"
        )
    return ts


def interval_stats(ts: NDArray[np.int64]) -> dict:
    """Δt distribution in milliseconds: median/mean/std, percentiles, outlier gaps with indices,
    lag-1 autocorrelation. Differences of int64 timestamps stay exact (values are ~1e8 ns, far
    under float64's precision limit) — only the *raw* epoch values are precision-sensitive, and
    this function never touches those directly, only their differences."""
    dt_ns = np.diff(ts)
    dt_ms = dt_ns.astype(np.float64) / 1e6

    gaps = [
        {"index": int(i), "dt_ms": float(dt_ms[i])}
        for i in np.where(dt_ms > 150.0)[0]
    ]
    lag1 = float(np.corrcoef(dt_ms[:-1], dt_ms[1:])[0, 1]) if dt_ms.size > 2 else None

    return {
        "n_intervals": int(dt_ms.size),
        "median_ms": float(np.median(dt_ms)),
        "mean_ms": float(np.mean(dt_ms)),
        "std_ms": float(np.std(dt_ms)),
        "min_ms": float(np.min(dt_ms)),
        "max_ms": float(np.max(dt_ms)),
        "p05_ms": float(np.percentile(dt_ms, 5)),
        "p95_ms": float(np.percentile(dt_ms, 95)),
        "n_outside_90_110ms": int(np.sum((dt_ms < 90.0) | (dt_ms > 110.0))),
        "gaps_over_150ms": gaps,
        "lag1_autocorr": lag1,
    }


def frame_times(ts: NDArray[np.int64], decoded_count: int) -> NDArray[np.int64]:
    """First `decoded_count` timestamps — the ones with an actual decoded video frame under H0
    (frame k == timestamp line k == decoded frame k; CLAUDE.md §5). Fails loudly if `decoded_count`
    exceeds the number of timestamp lines, since that would mean H0 itself is impossible."""
    if decoded_count > ts.size:
        raise ValueError(
            f"decoded_count={decoded_count} exceeds {ts.size} timestamp lines — H0 cannot hold"
        )
    return ts[:decoded_count]
