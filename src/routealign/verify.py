"""Independent agreement cues used as confidence evidence. Step 5 (plan.md §7).

`sift_verify` is cut by default (see plan.md's revision note and config.yaml
`verify.sift.enabled`) -- the other three cues do not depend on it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .frames import FrameStore


def _path_to_row_j(path: NDArray[np.int32], n_a: int) -> NDArray[np.int64]:
    """Reduce a DTW path to one representative runB column per runA row.

    A row with a horizontal DTW step (runB advanced while runA held) visits several consecutive
    columns; this takes the last one visited, i.e. the column the path was at when it moved on to
    the next runA row -- the same convention `tests/test_align.py` uses to check DTW's plateau
    handling.

    Args:
        path: int32[L, 2] DTW path, sorted by non-decreasing `i` (guaranteed by `dtw_open_ends`).
        n_a: Number of runA rows the path covers.

    Returns:
        int64[n_a] last-visited runB column per runA row.
    """
    i_vals, j_vals = path[:, 0], path[:, 1]
    edges = np.searchsorted(i_vals, np.arange(n_a + 1))  # edges[i]:edges[i+1] is row i's slice
    return j_vals[edges[1:] - 1].astype(np.int64)


def ridge_strength(
    similarity: NDArray[np.float64], path: NDArray[np.int32], window: int
) -> NDArray[np.float64]:
    """Compute the mean similarity along the path, averaged over a small row window.

    A genuine match sits on a "ridge": similarity stays high not just at one path point but
    along the path's own trajectory through its neighbouring rows. So for runA row `i`, this
    averages `similarity[i', row_j[i']]` over rows `i'` in `[i - window, i + window]` -- each
    neighbouring row's *own* path column, not a column held fixed at row `i`'s -- since the path
    is diagonal, not vertical, and a column fixed at row `i` says nothing about whether the path
    stayed on a ridge as it moved through nearby rows. A weak/noisy neighbourhood pulls the mean
    down even if row `i` itself looks like a strong single match, which is the point: it flags
    rows the optimiser was merely forced through.

    Args:
        similarity: float64[NA, NB] joint similarity matrix.
        path: int32[L, 2] DTW path.
        window: Number of rows above/below each path point to average over.

    Returns:
        float64[NA] ridge strength per runA row.
    """
    n_a = similarity.shape[0]
    row_j = _path_to_row_j(path, n_a)

    ridge = np.empty(n_a, dtype=np.float64)
    for i in range(n_a):
        lo, hi = max(0, i - window), min(n_a, i + window + 1)
        rows = np.arange(lo, hi)
        ridge[i] = np.mean(similarity[rows, row_j[lo:hi]])
    return ridge


def margin(
    similarity: NDArray[np.float64], path: NDArray[np.int32], band: int
) -> NDArray[np.float64]:
    """Compute how much better the path's similarity is than the best value outside a column band.

    Large margin means the match is unambiguous; small margin flags loop-closure aliasing or
    repeated facades: a second runB location, far from the claimed match, that looks nearly as
    similar. Columns within `band` of the claimed match are excluded from the "runner-up" search
    (not compared against) since neighbouring frames are expected to look similar -- only a
    far-away near-tie is suspicious.

    Args:
        similarity: float64[NA, NB] joint similarity matrix.
        path: int32[L, 2] DTW path.
        band: Half-width of the column band, centred on the claimed match, excluded from the
            runner-up search.

    Raises:
        ValueError: If `band` is large enough that the excluded band could cover every column
            for some claimed match near the middle of a row (`2 * band + 1 >= NB`), leaving no
            runner-up candidate.

    Returns:
        float64[NA] margin per runA row (claimed-match similarity minus the best similarity
        strictly outside the band).
    """
    n_a, n_b = similarity.shape
    if 2 * band + 1 >= n_b:
        raise ValueError(
            f"margin: band ({band}) can exclude every column (NB={n_b}); no runner-up possible"
        )

    row_j = _path_to_row_j(path, n_a)
    winner = similarity[np.arange(n_a), row_j]

    cols = np.arange(n_b)
    in_band = np.abs(cols[np.newaxis, :] - row_j[:, np.newaxis]) <= band
    runner_up = np.where(in_band, -np.inf, similarity).max(axis=1)
    return winner - runner_up


def path_disagreement(path_a: NDArray[np.int32], path_b: NDArray[np.int32]) -> NDArray[np.float64]:
    """Compute |j from one alignment - j from another| for two independently-computed paths, e.g.
    cam0-only vs. cam5-only, or SeqSLAM vs. DINOv2.

    Two independent alignments landing far apart on the same runA row is evidence the match is
    unreliable there, even if each alignment looks confident on its own -- this is one of the
    agreement cues `confidence.fuse_confidence` combines.

    Args:
        path_a: int32[L, 2] first DTW path.
        path_b: int32[L, 2] second DTW path, covering the same runA rows.

    Raises:
        ValueError: If the two paths cover a different number of runA rows.

    Returns:
        float64[NA] absolute disagreement per runA row.
    """
    n_a_a = int(path_a[:, 0].max()) + 1
    n_a_b = int(path_b[:, 0].max()) + 1
    if n_a_a != n_a_b:
        raise ValueError(
            f"path_disagreement: paths cover different runA row counts: {n_a_a} vs {n_a_b}"
        )

    j_a = _path_to_row_j(path_a, n_a_a)
    j_b = _path_to_row_j(path_b, n_a_a)
    return np.abs(j_a - j_b).astype(np.float64)


def sift_verify(store_a: FrameStore, store_b: FrameStore, i: int, j: int, ratio: float) -> int:
    """Count RootSIFT inlier matches (RANSAC/MAGSAC) between one claimed-matching frame pair.

    Cut by default (plan.md §7 revision note) -- build only in the day-2 buffer if genuinely
    ahead of schedule.

    Args:
        store_a: Frame source for runA.
        store_b: Frame source for runB.
        i: runA frame index.
        j: runB frame index.
        ratio: Lowe's ratio-test threshold, e.g. config.yaml `verify.sift.ratio_test`.

    Returns:
        Number of geometrically-verified inlier matches.
    """
    raise NotImplementedError("sift_verify: cut by default (plan.md §7); day-2 buffer only")
