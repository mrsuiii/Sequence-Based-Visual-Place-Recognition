"""Pure functions on the int64-nanosecond timestamp arrays. No file I/O beyond reading the text
file, no video. Step 1 (plan.md §3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_timestamps(path: str | Path) -> NDArray[np.int64]:
    """One capture timestamp per line, nanoseconds since the Unix epoch. int64 throughout — never
    float (epoch-ns values exceed float64's 2**53 exact-integer range; a naive float parse would
    silently lose precision in exactly the interval arithmetic Task 1 depends on).
    Not yet implemented — Step 1."""
    raise NotImplementedError("load_timestamps: implement in Step 1 (plan.md §3)")


def interval_stats(ts: NDArray[np.int64]) -> dict:
    """Δt distribution: median/mean/std, percentiles, outlier gaps with indices, grid-fit residual,
    lag-1 autocorrelation. Not yet implemented — Step 1."""
    raise NotImplementedError("interval_stats: implement in Step 1 (plan.md §3)")


def frame_times(ts: NDArray[np.int64], decoded_count: int) -> NDArray[np.int64]:
    """First `decoded_count` timestamps — the ones with an actual decoded video frame under H0.
    Not yet implemented — Step 1."""
    raise NotImplementedError("frame_times: implement in Step 1 (plan.md §3)")
