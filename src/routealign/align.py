"""Monotone alignment: DTW with open ends, path -> mapping. Step 4 (plan.md §6).

v1 (`dtw_open_ends`) is the committed default. v2 (`nullstate_dp`) is cut by default (see
plan.md's revision note and config.yaml `align.nullstate_dp.enabled`) -- build it only in the
day-2 buffer, and only after v1 is shipped, tested and understood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _dtw_forward(cost: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
    """Forward DP pass: lowest cumulative cost to reach every (i, j).

    Three predecessors are allowed into (i, j): diagonal `D[i-1,j-1]` (free -- both runs advance
    together), vertical `D[i-1,j] + lam` (runA advances, runB holds -- runB was momentarily
    slower/stopped) and horizontal `D[i,j-1] + lam` (runB advances, runA holds). Row 0 starts at
    `cost[0]` everywhere (open begin: the path may start at any runB column, at no penalty for the
    columns before it). The horizontal term folds a same-row dependency into a single pass via a
    cumulative-min identity rather than an explicit inner loop over `j` -- see decisions.md if this
    needs re-deriving; `tests/test_align.py` checks it against a naive triple-loop line by line.

    Args:
        cost: float64[NA, NB] non-negative cost matrix.
        lam: Penalty added per non-diagonal (vertical or horizontal) step.

    Returns:
        float64[NA, NB] cumulative cost matrix.
    """
    n_a, n_b = cost.shape
    d = np.empty((n_a, n_b), dtype=np.float64)
    d[0] = cost[0]
    for i in range(1, n_a):
        prev = d[i - 1]
        diag = np.concatenate(([np.inf], prev[:-1]))  # diag[j] = D[i-1, j-1]; undefined at j=0
        v = np.minimum(prev + lam, diag)  # best of {vertical, diagonal}, per column
        c_cum = np.cumsum(cost[i] + lam)
        d[i] = c_cum + np.minimum.accumulate(v + cost[i] - c_cum)
    return d


def _dtw_backtrack(d: NDArray[np.float64], lam: float) -> NDArray[np.int32]:
    """Recover the lowest-cost path from a forward-pass cost matrix.

    Starts at whichever column minimises the last row (open end: the path may end at any runB
    column). Ties between the three predecessors favour the diagonal, then vertical, then
    horizontal, in that fixed order (`np.argmin` returns the first minimum).

    Args:
        d: float64[NA, NB] cumulative cost matrix, e.g. from `_dtw_forward`.
        lam: Penalty added per non-diagonal step; must match the value `d` was built with.

    Returns:
        int32[L, 2] path as (i, j) pairs, monotone, with every `i` covered at least once.
    """
    i, j = d.shape[0] - 1, int(np.argmin(d[-1]))
    path = [(i, j)]
    while i > 0:
        diag = d[i - 1, j - 1] if j else np.inf
        up = d[i - 1, j] + lam
        left = d[i, j - 1] + lam if j else np.inf
        k = int(np.argmin((diag, up, left)))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
        path.append((i, j))
    return np.array(path[::-1], dtype=np.int32)


def dtw_open_ends(cost: NDArray[np.float64], lam: float) -> NDArray[np.int32]:
    """Find the lowest-cost monotone path through a cost matrix, with open start/end columns.

    Allows free local slope and open ends (no fixed/linear offset), matching the brief's "same
    order, not a fixed offset" constraint.

    Args:
        cost: float64[NA, NB] non-negative cost matrix, e.g. `3 - clip(similarity, -3, 3)`.
        lam: Penalty added per non-diagonal step.

    Returns:
        int32[L, 2] path as (i, j) pairs, monotone, with every `i` covered.
    """
    return _dtw_backtrack(_dtw_forward(cost, lam), lam)


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

    A runA row visits one runB column per diagonal/vertical DTW step, but a horizontal step (runB
    advanced while runA held) makes one row visit several consecutive columns; `runB_frame` is
    whichever of those has the highest similarity. `lo`/`hi` start as that visited range, then
    widen to the full runB stationary segment (`frames.stationary_segments`) containing
    `runB_frame`, since every frame in that segment shows the same place and is therefore an
    equally valid match. `boundary_clamped` is set from the *un-widened* visited range, and flags
    rows the open-ends path pinned against column 0 or `NB - 1` -- typically a runA tail beyond
    runB's end, where the path has nowhere real to go.

    Args:
        path: int32[L, 2] DTW path, e.g. from `dtw_open_ends`.
        similarity: float64[NA, NB] similarity matrix the path was computed on.
        stationary_b: runB stationary segments as inclusive `(lo, hi)` frame-index pairs, e.g. from
            `frames.stationary_segments`.

    Returns:
        One row per runA index (`n_a = path[:, 0].max() + 1`), with columns `runA_frame`,
        `runB_frame`, `lo`, `hi`, `boundary_clamped` (see outputs/SCHEMA.md).
    """
    i_vals, j_vals = path[:, 0], path[:, 1]
    n_a = int(i_vals[-1]) + 1
    n_b = similarity.shape[1]

    # per-column lookup: which stationary segment (if any) a runB column belongs to
    seg_lo_of = np.full(n_b, -1, dtype=np.int64)
    seg_hi_of = np.full(n_b, -1, dtype=np.int64)
    for seg_lo, seg_hi in stationary_b:
        seg_lo_of[seg_lo : seg_hi + 1] = seg_lo
        seg_hi_of[seg_lo : seg_hi + 1] = seg_hi

    # path is sorted by i (non-decreasing); edges[i]:edges[i+1] is row i's contiguous slice
    edges = np.searchsorted(i_vals, np.arange(n_a + 1))

    runb_frame = np.empty(n_a, dtype=np.int64)
    lo_arr = np.empty(n_a, dtype=np.int64)
    hi_arr = np.empty(n_a, dtype=np.int64)
    boundary_clamped = np.empty(n_a, dtype=bool)
    for i in range(n_a):
        start, stop = edges[i], edges[i + 1]
        lo, hi = int(j_vals[start]), int(j_vals[stop - 1])  # j is contiguous within a row group
        boundary_clamped[i] = lo == 0 or hi == n_b - 1

        best_j = lo + int(np.argmax(similarity[i, lo : hi + 1]))
        if seg_lo_of[best_j] >= 0:  # widen to the runB stationary segment containing the best match
            lo = min(lo, int(seg_lo_of[best_j]))
            hi = max(hi, int(seg_hi_of[best_j]))
        runb_frame[i] = best_j
        lo_arr[i] = lo
        hi_arr[i] = hi

    return pd.DataFrame(
        {
            "runA_frame": np.arange(n_a, dtype=np.int64),
            "runB_frame": runb_frame,
            "lo": lo_arr,
            "hi": hi_arr,
            "boundary_clamped": boundary_clamped,
        }
    )


def local_slope(path: NDArray[np.int32], window: int) -> NDArray[np.float64]:
    """Estimate local relative speed (runB frames per runA frame) along an alignment path.

    At each path point, fits a least-squares slope of `j` (runB) against `i` (runA) over the
    `window` path points centred on it (clipped at the path's ends, so the window narrows near
    the open start/end). A slope near 1.0 means the two runs advance at the same pace; higher
    means runB is locally moving faster (e.g. runA idled while runB kept going), 0 means runA
    idled while runB held too. `stationary_A`/`stationary_B` (Step 5) explain *why* a low or
    infinite slope happened; this function only measures it.

    Args:
        path: int32[L, 2] DTW path, e.g. from `dtw_open_ends`.
        window: Number of path points to fit the local slope over.

    Returns:
        float64[L] local slope at each path point. `inf` where the window's `i` never changes
        (a horizontal run wider than `window`) but `j` did -- runB advanced with no runA motion
        at all, i.e. an unbounded local rate.
    """
    n = path.shape[0]
    half = window // 2
    i = path[:, 0].astype(np.float64)
    j = path[:, 1].astype(np.float64)

    slope = np.empty(n, dtype=np.float64)
    for k in range(n):
        lo, hi = max(0, k - half), min(n, k + half + 1)
        i_w, j_w = i[lo:hi], j[lo:hi]
        i_dev = i_w - i_w.mean()
        var_i = float(np.sum(i_dev * i_dev))
        if var_i > 0.0:
            slope[k] = float(np.sum(i_dev * (j_w - j_w.mean())) / var_i)
        else:
            slope[k] = np.inf if j_w[-1] != j_w[0] else 1.0
    return slope
