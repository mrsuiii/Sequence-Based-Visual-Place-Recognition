"""Per-frame descriptors: SeqSLAM-style thumbnails (R1) and DINOv2 embeddings (R2). Step 3
(plan.md §5). R2 is a committed part of the default pipeline, fused with R1, not an optional extra.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .frames import FrameStore


def seqslam_descriptors(store: FrameStore, indices: NDArray[np.int64]) -> NDArray[np.float32]:
    """Compute SeqSLAM-style descriptors: downsample, patch-normalise, flatten, L2-normalise.

    Args:
        store: Frame source to read from.
        indices: int64[N] frame indices to describe.

    Returns:
        float32[N, 2560] L2-normalised descriptors (a flattened 64x40 image).
    """
    raise NotImplementedError("seqslam_descriptors: implement in Step 3 (plan.md §5)")


def dino_descriptors(store: FrameStore, indices: NDArray[np.int64]) -> NDArray[np.float32]:
    """Compute DINOv2 ViT-S/14 embeddings (CLS token concatenated with a GeM-pooled patch summary).

    Args:
        store: Frame source to read from.
        indices: int64[N] frame indices to describe.

    Returns:
        float32[N, 768] L2-normalised descriptors.
    """
    raise NotImplementedError("dino_descriptors: implement in Step 3 (plan.md §5)")


def cached(
    fn: Callable[[], NDArray], key: str, cache_dir: str | Path, force: bool = False
) -> NDArray:
    """Compute-or-load an array, writing a `.npy` plus a provenance `.json` sidecar (source
    SHA-256 prefix, parameters, package versions -- CLAUDE.md §4).

    Args:
        fn: Zero-argument callable that computes the array on a cache miss.
        key: Cache key; determines the output filename.
        cache_dir: Directory the `.npy`/`.json` pair is stored in.
        force (optional): Recompute even if a cache entry exists. Defaults to `False`.

    Returns:
        The array, either loaded from cache or freshly computed.
    """
    raise NotImplementedError("cached: implement in Step 3 (plan.md §5)")
