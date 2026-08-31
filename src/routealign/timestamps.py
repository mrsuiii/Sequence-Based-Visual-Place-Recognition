"""Pure functions on the int64-nanosecond timestamp arrays. No file I/O beyond reading the text
file, no video. Step 1 (plan.md §3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_timestamps(path: str | Path) -> NDArray[np.int64]:
    """Load a `.timestamps.txt` file: one capture timestamp per line, nanoseconds since epoch.

    Always int64 -- epoch-ns values (~1.7e18) exceed float64's 2**53 exact-integer range, so a
    float parse would silently lose precision in exactly the interval arithmetic this module does.

    Args:
        path: Timestamp file to load.

    Returns:
        int64[N] timestamps, one per line.

    Raises:
        ValueError: If the file is empty, or its timestamps are not strictly increasing.
    """
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
    """Compute the distribution of Δt (time between consecutive frames), in milliseconds.

    Args:
        ts: int64[N] timestamps, e.g. from `load_timestamps`.

    Returns:
        dict with keys `n_intervals`, `median_ms`, `mean_ms`, `std_ms`, `min_ms`, `max_ms`,
        `p05_ms`, `p95_ms`, `n_outside_90_110ms`, `gaps_over_150ms` (list of
        `{"index", "dt_ms"}`, one per interval over 150 ms) and `lag1_autocorr`.
    """
    dt_ns = np.diff(ts)
    dt_ms = dt_ns.astype(np.float64) / 1e6

    gaps = [{"index": int(i), "dt_ms": float(dt_ms[i])} for i in np.where(dt_ms > 150.0)[0]]
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
    """Slice out the timestamps that have a matching decoded video frame, under H0.

    H0 (CLAUDE.md §5): timestamp line `k` == decoded frame `k`. Holds only for
    `k < decoded_count`; later lines have no video.

    Args:
        ts: int64[N] timestamps, e.g. from `load_timestamps`.
        decoded_count: Number of frames actually decoded for this run/camera.

    Returns:
        int64[decoded_count] timestamps, the first `decoded_count` entries of `ts`.

    Raises:
        ValueError: If `decoded_count` exceeds `len(ts)` (H0 would be impossible).
    """
    if decoded_count > ts.size:
        raise ValueError(
            f"decoded_count={decoded_count} exceeds {ts.size} timestamp lines — H0 cannot hold"
        )
    return ts[:decoded_count]
