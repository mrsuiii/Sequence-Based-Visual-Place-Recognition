"""Tests for verify.py's `ridge_strength` and `path_disagreement`. `margin` is still a stub
(Step 5, user-led)."""

from __future__ import annotations

import numpy as np
import pytest

from routealign.verify import margin, path_disagreement, ridge_strength


def test_ridge_strength_is_perfect_on_a_clean_diagonal() -> None:
    """A path that exactly tracks a diagonal of 1.0s, surrounded by 0.0 elsewhere, must average
    to exactly 1.0 at every row and every window size -- every neighbouring row's own path
    column is also on the diagonal, so the window never touches a 0."""
    n_a = n_b = 20
    similarity = np.eye(n_a, n_b, dtype=np.float64)
    path = np.array([[i, i] for i in range(n_a)], dtype=np.int32)

    ridge = ridge_strength(similarity, path, window=5)

    assert ridge.shape == (n_a,)
    np.testing.assert_allclose(ridge, np.ones(n_a))


def test_ridge_strength_output_length_matches_n_a_not_path_length() -> None:
    """A path with a horizontal step is longer than `n_a`; the output must still have exactly
    one entry per runA row, not one per path point."""
    n_a, n_b = 3, 6
    path = np.array([[0, 0], [1, 1], [1, 2], [1, 3], [2, 4]], dtype=np.int32)  # len(path) = 5
    similarity = np.zeros((n_a, n_b), dtype=np.float64)

    ridge = ridge_strength(similarity, path, window=1)

    assert ridge.shape == (n_a,)


def test_ridge_strength_penalises_a_weak_neighbourhood() -> None:
    """One row sits on a locally strong match (its own column is 1.0), but its neighbours' own
    path columns are weak (0.0) -- the windowed mean must be pulled below the row's own value,
    catching a spike the optimiser was merely forced through rather than a genuine ridge."""
    n_a = n_b = 5
    similarity = np.zeros((n_a, n_b), dtype=np.float64)
    similarity[2, 2] = 1.0  # only row 2's own path point is a strong match
    path = np.array([[i, i] for i in range(n_a)], dtype=np.int32)

    ridge = ridge_strength(similarity, path, window=2)

    assert ridge[2] < 1.0
    assert ridge[2] == pytest.approx(1.0 / 5.0)  # 1 strong hit out of 5 rows in the full window


def test_margin_finds_a_far_runner_up_outside_the_band() -> None:
    """A row with a clear winner and a near-tying runner-up far outside the band must report the
    difference between them; a near-tying value just inside the band must be ignored."""
    n_a, n_b = 1, 100
    similarity = np.zeros((n_a, n_b), dtype=np.float64)
    similarity[0, 50] = 0.95  # the claimed match (path below)
    similarity[0, 55] = 0.93  # inside the band (|55-50|=5 <= band=10) -- must be ignored
    similarity[0, 90] = 0.80  # outside the band -- this is the true runner-up
    path = np.array([[0, 50]], dtype=np.int32)

    result = margin(similarity, path, band=10)

    np.testing.assert_allclose(result, [0.95 - 0.80])


def test_margin_handles_a_band_clipped_at_the_matrix_edge() -> None:
    """The claimed match near column 0: the band is clipped by the matrix boundary rather than
    needing special-case handling, and the runner-up search still only looks outside it."""
    n_a, n_b = 1, 50
    similarity = np.zeros((n_a, n_b), dtype=np.float64)
    similarity[0, 2] = 0.9  # claimed match, band [0, 12] after clipping at the left edge
    similarity[0, 20] = 0.4  # outside the clipped band -- the runner-up
    path = np.array([[0, 2]], dtype=np.int32)

    result = margin(similarity, path, band=10)

    np.testing.assert_allclose(result, [0.9 - 0.4])


def test_margin_output_length_matches_n_a_not_path_length() -> None:
    """Same length contract as `ridge_strength`: one entry per runA row, not per path point."""
    n_a, n_b = 3, 60
    path = np.array([[0, 0], [1, 10], [1, 11], [1, 12], [2, 20]], dtype=np.int32)
    similarity = np.zeros((n_a, n_b), dtype=np.float64)

    result = margin(similarity, path, band=5)

    assert result.shape == (n_a,)


def test_margin_rejects_a_band_that_could_exclude_every_column() -> None:
    """A band wide enough to swallow the whole row for a centred claimed match must fail loudly
    instead of crashing on an empty-array max or silently returning a meaningless value."""
    n_a, n_b = 1, 20
    similarity = np.zeros((n_a, n_b), dtype=np.float64)
    path = np.array([[0, 10]], dtype=np.int32)
    with pytest.raises(ValueError, match="no runner-up possible"):
        margin(similarity, path, band=10)  # 2*10+1 = 21 >= n_b=20


def test_path_disagreement_is_zero_for_identical_paths() -> None:
    """Two identical alignments must disagree by exactly 0 everywhere."""
    path = np.array([[i, i] for i in range(10)], dtype=np.int32)
    disagreement = path_disagreement(path, path.copy())
    np.testing.assert_array_equal(disagreement, np.zeros(10))


def test_path_disagreement_uses_the_last_visited_column_per_row() -> None:
    """A row with a horizontal DTW step visits several columns; disagreement must be computed
    against the last one visited, matching the convention `_path_to_row_j` documents."""
    # row 1 of path_a visits columns 1, 2, 3 (horizontal run) -- last-visited is 3
    path_a = np.array([[0, 0], [1, 1], [1, 2], [1, 3], [2, 4]], dtype=np.int32)
    path_b = np.array([[0, 0], [1, 5], [2, 6]], dtype=np.int32)

    disagreement = path_disagreement(path_a, path_b)

    np.testing.assert_array_equal(disagreement, [0.0, 2.0, 2.0])  # |3-5|=2, |4-6|=2


def test_path_disagreement_rejects_mismatched_row_counts() -> None:
    """Paths that cover a different number of runA rows cannot be compared row-for-row."""
    path_a = np.array([[i, i] for i in range(5)], dtype=np.int32)
    path_b = np.array([[i, i] for i in range(6)], dtype=np.int32)
    with pytest.raises(ValueError, match="different runA row counts"):
        path_disagreement(path_a, path_b)
