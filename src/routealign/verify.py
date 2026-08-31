"""Independent agreement cues used as confidence evidence. Step 5 (plan.md §7).

`sift_verify` is cut by default (see plan.md's revision note and config.yaml
`verify.sift.enabled`) -- the other three cues do not depend on it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .frames import FrameStore


def ridge_strength(
    similarity: NDArray[np.float64], path: NDArray[np.int32], window: int
) -> NDArray[np.float64]:
    """Compute the mean similarity along the path, averaged over a small row window.

    Args:
        similarity: float64[NA, NB] joint similarity matrix.
        path: int32[L, 2] DTW path.
        window: Number of rows above/below each path point to average over.

    Returns:
        float64[NA] ridge strength per runA row.
    """
    raise NotImplementedError("ridge_strength: implement in Step 5 (plan.md §7)")


def margin(
    similarity: NDArray[np.float64], path: NDArray[np.int32], band: int
) -> NDArray[np.float64]:
    """Compute how much better the path's similarity is than the best value outside a column band.

    Large margin means the match is unambiguous; small margin flags loop-closure aliasing or
    repeated facades.

    Args:
        similarity: float64[NA, NB] joint similarity matrix.
        path: int32[L, 2] DTW path.
        band: Half-width of the excluded column band around each path point.

    Returns:
        float64[NA] margin per runA row.
    """
    raise NotImplementedError("margin: implement in Step 5 (plan.md §7)")


def path_disagreement(path_a: NDArray[np.int32], path_b: NDArray[np.int32]) -> NDArray[np.float64]:
    """Compute |j from one alignment - j from another| for two independently-computed paths, e.g.
    cam0-only vs. cam5-only, or SeqSLAM vs. DINOv2.

    Args:
        path_a: int32[L, 2] first DTW path.
        path_b: int32[L, 2] second DTW path, covering the same runA rows.

    Returns:
        float64[NA] absolute disagreement per runA row.
    """
    raise NotImplementedError("path_disagreement: implement in Step 5 (plan.md §7)")


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
