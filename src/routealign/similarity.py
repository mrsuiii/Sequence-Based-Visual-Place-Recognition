"""Similarity matrices between two runs' descriptor sets. Step 3 (plan.md §5)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d

_EPS = 1e-6  # why: numerical floor against a near-zero norm/std, never a real design parameter


def cosine(desc_a: NDArray[np.float32], desc_b: NDArray[np.float32]) -> NDArray[np.float64]:
    """Compute the cosine similarity matrix between two sets of descriptors.

    Re-normalises both inputs internally rather than trusting the caller already L2-normalised
    them (both `descriptors.seqslam_descriptors` and `dino_descriptors` do, but this function
    should be correct regardless of that upstream contract).

    Args:
        desc_a: float32[NA, D] descriptors, e.g. runA.
        desc_b: float32[NB, D] descriptors, e.g. runB.

    Returns:
        float64[NA, NB] cosine similarity, one row per `desc_a` entry.
    """
    a = desc_a.astype(np.float64)
    b = desc_b.astype(np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), _EPS)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), _EPS)
    return a @ b.T


def local_contrast_norm(similarity: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Apply SeqSLAM-style local contrast normalisation along the runB axis.

    For each runA row, z-scores every value against the *local* mean/std of a `window`-wide
    neighbourhood along the runB axis (not the row's global mean/std) -- this is what makes the
    correct-match ridge stand out as a local peak even when the overall similarity level drifts
    across the row (e.g. a stretch of runB shot under different lighting). Local mean/variance are
    computed via two box-filter passes (`scipy.ndimage.uniform_filter1d` on `similarity` and
    `similarity ** 2`) rather than a per-column sliding-window loop, since `NA` and `NB` can each
    be in the thousands.

    Args:
        similarity: float64[NA, NB] similarity matrix, e.g. from `cosine`.
        window: Width in columns of the local window, e.g. one of plan.md §5's candidates
            `{20, 50, 100}` (final value tuned on synthetic warps, not by this function).

    Returns:
        float64[NA, NB] z-scored similarity matrix, clipped to [-3, 3].
    """
    local_mean = uniform_filter1d(similarity, size=window, axis=1, mode="nearest")
    local_mean_sq = uniform_filter1d(similarity**2, size=window, axis=1, mode="nearest")
    local_var = np.maximum(local_mean_sq - local_mean**2, 0.0)  # floor against float round-off
    local_std = np.sqrt(local_var)
    z = (similarity - local_mean) / (local_std + _EPS)
    return np.clip(z, -3.0, 3.0)


def fuse(matrices: list[NDArray[np.float64]], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine multiple z-scored similarity matrices (per camera / per descriptor) into one.

    Args:
        matrices: Same-shape float64[NA, NB] matrices to combine, e.g. cam0 and cam5.
        weights: float64[len(matrices)] weight per matrix; normalised internally, so callers
            need not pre-scale them to sum to 1.

    Returns:
        float64[NA, NB] weighted-mean similarity matrix.

    Raises:
        ValueError: If `matrices` is empty, the matrices' shapes disagree, or `weights` has a
            different length than `matrices`.
    """
    if not matrices:
        raise ValueError("fuse: matrices is empty")
    if len(weights) != len(matrices):
        raise ValueError(f"fuse: {len(weights)} weights for {len(matrices)} matrices")
    if any(m.shape != matrices[0].shape for m in matrices[1:]):
        shapes = [m.shape for m in matrices]
        raise ValueError(f"fuse: matrices have mismatched shapes: {shapes}")

    stacked = np.stack(matrices, axis=0)  # [K, NA, NB]
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return np.tensordot(w, stacked, axes=(0, 0))
