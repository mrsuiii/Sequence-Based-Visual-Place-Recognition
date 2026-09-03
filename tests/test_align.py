"""Tests for align.py's DTW. CLAUDE.md §4: every DP has an exactness test against a naive
reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from routealign.align import dtw_open_ends, local_slope, path_to_mapping


def _dtw_naive(cost: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation: explicit triple loop, no vectorisation tricks. Mirrors
    `align._dtw_forward` / `_dtw_backtrack`'s recurrence and tie-breaking line by line, so any
    disagreement points at the vectorised version's cumulative-min trick, not a differently-posed
    problem. Never call this outside tests -- it is O(NA*NB) with a much larger constant.

    Returns:
        (D, path): the full cumulative-cost matrix and the backtracked (i, j) path.
    """
    n_a, n_b = cost.shape
    d = np.full((n_a, n_b), np.inf, dtype=np.float64)
    d[0, :] = cost[0, :]
    for i in range(1, n_a):
        for j in range(n_b):
            best = d[i - 1, j] + lam  # vertical
            if j > 0:
                best = min(best, d[i - 1, j - 1])  # diagonal
                best = min(best, d[i, j - 1] + lam)  # horizontal (already-filled same row)
            d[i, j] = cost[i, j] + best

    i, j = n_a - 1, int(np.argmin(d[n_a - 1, :]))
    path = [(i, j)]
    while i > 0:
        diag = d[i - 1, j - 1] if j > 0 else np.inf
        up = d[i - 1, j] + lam
        left = d[i, j - 1] + lam if j > 0 else np.inf
        k = int(np.argmin((diag, up, left)))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
        path.append((i, j))
    path.reverse()
    return d, np.array(path, dtype=np.int32)


@pytest.mark.parametrize("n_a,n_b", [(1, 1), (1, 5), (5, 1), (5, 5), (30, 40), (40, 30)])
@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
def test_dtw_matches_naive_reference(n_a: int, n_b: int, lam: float) -> None:
    """The vectorised D matrix and backtracked path must exactly match the naive reference, on
    random cost matrices covering both aspect ratios and every candidate `lam` (0.1, 0.5, 1.0)."""
    rng = np.random.default_rng(seed=n_a * 1000 + n_b)
    cost = rng.uniform(0.0, 6.0, size=(n_a, n_b))

    from routealign.align import _dtw_backtrack, _dtw_forward

    d_fast = _dtw_forward(cost, lam)
    path_fast = _dtw_backtrack(d_fast, lam)
    d_naive, path_naive = _dtw_naive(cost, lam)

    np.testing.assert_allclose(d_fast, d_naive, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(path_fast, path_naive)


@pytest.mark.parametrize("n_a,n_b", [(20, 25), (25, 20)])
def test_dtw_path_is_monotone_and_covers_every_row(n_a: int, n_b: int) -> None:
    """Structural contract from the docstring: i is non-decreasing, j is non-decreasing, and
    every runA index from 0 to NA-1 appears at least once."""
    rng = np.random.default_rng(seed=123)
    cost = rng.uniform(0.0, 6.0, size=(n_a, n_b))
    path = dtw_open_ends(cost, lam=0.5)

    assert np.all(np.diff(path[:, 0]) >= 0), "i must be non-decreasing"
    assert np.all(np.diff(path[:, 1]) >= 0), "j must be non-decreasing"
    assert set(path[:, 0].tolist()) == set(range(n_a)), "every runA row must be covered"


def test_dtw_recovers_a_known_diagonal_with_a_plateau() -> None:
    """Known-answer test, independent of the naive reference (guards against a bug shared by both
    implementations): a cost matrix built so the true best path is known by construction -- a
    clean diagonal, except runA rows 10-14 all cheapest against the single runB column 10 (a
    'plateau', modelling a stop), after which the diagonal *resumes from the plateau's exit
    column* rather than jumping back to `i == j` -- jumping back would need an expensive multi-
    column catch-up that is not actually the cheapest path, which is exactly the mistake an
    earlier version of this test made (see git history: the naive-reference test above passed
    while this one failed, correctly pointing at the test, not `dtw_open_ends`)."""
    n_a, n_b = 30, 30
    cost = np.full((n_a, n_b), 6.0)  # 6.0 = worst possible cost (clip(-3,3) -> cost=6)
    expected_j = np.arange(n_a)
    expected_j[10:15] = 10  # plateau: rows 10-14 all map to column 10
    expected_j[15:] = np.arange(15, n_a) - 4  # resume the diagonal from column 10, not column 15
    for i, j in enumerate(expected_j):
        cost[i, j] = 0.0  # 0.0 = best possible cost

    path = dtw_open_ends(cost, lam=0.5)
    recovered_j = np.zeros(n_a, dtype=np.int32)
    for i, j in path:
        recovered_j[i] = j  # last-visited j per row, matching path_to_mapping's intended use

    np.testing.assert_array_equal(recovered_j, expected_j)


def test_path_to_mapping_diagonal_path_is_complete_and_monotone() -> None:
    """One row per runA index, `lo <= runB_frame <= hi`, and a plain diagonal path (no horizontal
    steps, no stationary segments) maps row i to column i with a degenerate lo == hi == i."""
    n_a = n_b = 5
    path = np.array([[i, i] for i in range(n_a)], dtype=np.int32)
    similarity = np.eye(n_a, n_b, dtype=np.float64)

    mapping = path_to_mapping(path, similarity, stationary_b=[])

    np.testing.assert_array_equal(mapping["runA_frame"].to_numpy(), np.arange(n_a))
    np.testing.assert_array_equal(mapping["runB_frame"].to_numpy(), np.arange(n_a))
    np.testing.assert_array_equal(mapping["lo"].to_numpy(), np.arange(n_a))
    np.testing.assert_array_equal(mapping["hi"].to_numpy(), np.arange(n_a))
    assert (mapping["lo"] <= mapping["runB_frame"]).all()
    assert (mapping["runB_frame"] <= mapping["hi"]).all()
    # only row 0 (lo == 0) and row n_a-1 (hi == n_b-1) touch a matrix boundary
    np.testing.assert_array_equal(
        mapping["boundary_clamped"].to_numpy(), [True, False, False, False, True]
    )


def test_path_to_mapping_picks_best_of_several_visited_columns() -> None:
    """A horizontal DTW step (runB advances, runA holds) makes one runA row visit several
    consecutive runB columns; `runB_frame` must be the one with the highest similarity, and
    `lo`/`hi` must span the full visited range."""
    n_a, n_b = 3, 6
    # row 0 -> col 0; row 1 visits cols 1-3 (a horizontal run); row 2 -> col 4
    path = np.array([[0, 0], [1, 1], [1, 2], [1, 3], [2, 4]], dtype=np.int32)
    similarity = np.zeros((n_a, n_b), dtype=np.float64)
    similarity[1, 3] = 1.0  # col 3 is clearly the best match within row 1's visited range

    mapping = path_to_mapping(path, similarity, stationary_b=[])
    row1 = mapping.loc[mapping["runA_frame"] == 1].iloc[0]

    assert row1["runB_frame"] == 3
    assert row1["lo"] == 1
    assert row1["hi"] == 3
    assert not row1["boundary_clamped"]


def test_path_to_mapping_widens_to_stationary_segment() -> None:
    """When the best-matching column falls inside a runB stationary segment, `lo`/`hi` widen to
    cover the whole segment (every frame in it shows the same place), but `boundary_clamped` is
    still judged from the un-widened visited range."""
    n_a, n_b = 1, 20
    path = np.array([[0, 6]], dtype=np.int32)  # row 0 visits only column 6
    similarity = np.zeros((n_a, n_b), dtype=np.float64)
    similarity[0, 6] = 1.0
    stationary_b = [(5, 8)]  # column 6 sits inside this stationary run

    mapping = path_to_mapping(path, similarity, stationary_b)
    row0 = mapping.iloc[0]

    assert row0["runB_frame"] == 6
    assert row0["lo"] == 5
    assert row0["hi"] == 8
    assert not row0["boundary_clamped"]  # raw visited range (6, 6) touches neither boundary


def test_local_slope_is_one_on_a_plain_diagonal() -> None:
    """i and j advance together 1-for-1 everywhere, so every window has slope exactly 1.0,
    including windows clipped at the open start/end."""
    path = np.array([[i, i] for i in range(20)], dtype=np.int32)
    slope = local_slope(path, window=5)
    np.testing.assert_allclose(slope, np.ones(20))


def test_local_slope_matches_a_known_constant_rate() -> None:
    """runB advances twice as fast as runA (j = 2*i): the interior slope must recover 2.0
    regardless of window size, since the true relationship is exactly linear."""
    path = np.array([[i, 2 * i] for i in range(20)], dtype=np.int32)
    slope = local_slope(path, window=7)
    np.testing.assert_allclose(slope[3:-3], 2.0)  # away from the clipped edges


def test_local_slope_is_inf_when_i_never_changes_in_the_window() -> None:
    """A horizontal run (runA idle) wider than `window`: every `i` in the window is identical
    but `j` changes, so the least-squares slope is undefined -- by convention this is `inf`
    (an unbounded runB-per-runA rate), not silently 0 or NaN."""
    path = np.array([[0, j] for j in range(10)], dtype=np.int32)  # i stuck at 0, j sweeps 0..9
    slope = local_slope(path, window=3)
    assert np.all(np.isinf(slope[1:-1]))  # interior points: full window is flat in i
