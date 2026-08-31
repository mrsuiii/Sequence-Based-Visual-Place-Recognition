"""Similarity matrices between two runs' descriptor sets. Step 3 (plan.md §5)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosine(desc_a: NDArray[np.float32], desc_b: NDArray[np.float32]) -> NDArray[np.float64]:
    """Compute the cosine similarity matrix between two sets of descriptors.

    Args:
        desc_a: float32[NA, D] descriptors, e.g. runA.
        desc_b: float32[NB, D] descriptors, e.g. runB.

    Returns:
        float64[NA, NB] cosine similarity, one row per `desc_a` entry.
    """
    raise NotImplementedError("cosine: implement in Step 3 (plan.md §5)")


def local_contrast_norm(similarity: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Apply SeqSLAM-style local contrast normalisation along the runB axis.

    Args:
        similarity: float64[NA, NB] similarity matrix, e.g. from `cosine`.
        window: Half-width of the local window used to compute the local mean/std, in columns.

    Returns:
        float64[NA, NB] z-scored similarity matrix.
    """
    raise NotImplementedError("local_contrast_norm: implement in Step 3 (plan.md §5)")


def fuse(matrices: list[NDArray[np.float64]], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Combine multiple z-scored similarity matrices (per camera / per descriptor) into one.

    Args:
        matrices: Same-shape float64[NA, NB] matrices to combine, e.g. cam0 and cam5.
        weights: float64[len(matrices)] weight per matrix; need not sum to 1.

    Returns:
        float64[NA, NB] weighted-mean similarity matrix.
    """
    raise NotImplementedError("fuse: implement in Step 3 (plan.md §5)")
