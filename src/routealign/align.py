"""Monotone alignment: DTW with open ends, path -> mapping. Step 4 (plan.md §6).

v1 (`dtw_open_ends`) is the committed default. v2 (`nullstate_dp`) is cut by default (see
plan.md's revision note and config.yaml `align.nullstate_dp.enabled`) -- build it only in the
day-2 buffer, and only after v1 is shipped, tested and understood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def dtw_open_ends(cost: NDArray[np.float64], lam: float) -> NDArray[np.int32]:
    """Find the lowest-cost monotone path through a cost matrix, with open start/end columns.

    Allows free local slope and open ends (no fixed/linear offset), matching the brief's "same
    order, not a fixed offset" constraint. Fully specified, including the row-vectorised
    recurrence, in plan.md §6.

    Args:
        cost: float64[NA, NB] non-negative cost matrix, e.g. `3 - clip(similarity, -3, 3)`.
        lam: Penalty added per non-diagonal step.

    Returns:
        int32[L, 2] path as (i, j) pairs, monotone, with every `i` covered.
    """
    raise NotImplementedError(
        "dtw_open_ends: implement in Step 4 (plan.md §6); the algorithm is already drafted there"
    )


def nullstate_dp(
    cost: NDArray[np.float64], g_open: float, g_ext: float, rho: float
) -> NDArray[np.int32]:
    """NULL-state affine-gap alignment (v2): explicit match/skip states instead of a post-hoc
    no-match threshold.

    Cut by default (plan.md §6 revision note) -- build only in the day-2 buffer, and only once v1
    is shipped and understood.

    Args:
        cost: float64[NA, NB] non-negative cost matrix.
        g_open: Gap-open penalty.
        g_ext: Gap-extend penalty.
        rho: Penalty for a non-diagonal match step.

    Returns:
        int32[L, 2] path as (i, j) pairs.
    """
    raise NotImplementedError("nullstate_dp: cut by default (plan.md §6); day-2 buffer only")


def path_to_mapping(
    path: NDArray[np.int32],
    similarity: NDArray[np.float64],
    stationary_b: list[tuple[int, int]],
) -> pd.DataFrame:
    """Convert a DTW path into a per-runA-row mapping, widened to cover matching stationary runs.

    Args:
        path: int32[L, 2] DTW path, e.g. from `dtw_open_ends`.
        similarity: float64[NA, NB] similarity matrix the path was computed on.
        stationary_b: runB stationary segments, e.g. from `frames.stationary_segments`.

    Returns:
        One row per runA index, with columns `runB_frame`, `lo`, `hi`, `boundary_clamped`
        (see outputs/SCHEMA.md).
    """
    raise NotImplementedError("path_to_mapping: implement in Step 4 (plan.md §6)")


def local_slope(path: NDArray[np.int32], window: int) -> NDArray[np.float64]:
    """Estimate local relative speed (runB frames per runA frame) along an alignment path.

    Args:
        path: int32[L, 2] DTW path, e.g. from `dtw_open_ends`.
        window: Number of path points to fit the local slope over.

    Returns:
        float64[L] local slope at each path point.
    """
    raise NotImplementedError("local_slope: implement in Step 4 (plan.md §6)")
