"""Similarity matrices between two runs' descriptor sets. Step 3 (plan.md §5)."""

from __future__ import annotations


def cosine(*args, **kwargs):
    """Cosine similarity matrix between two [N, D] descriptor arrays. Not yet implemented — Step 3."""
    raise NotImplementedError("cosine: implement in Step 3 (plan.md §5)")


def local_contrast_norm(*args, **kwargs):
    """SeqSLAM-style local contrast normalisation along the runB axis (window R). Not yet
    implemented — Step 3."""
    raise NotImplementedError("local_contrast_norm: implement in Step 3 (plan.md §5)")


def fuse(*args, **kwargs):
    """Weighted mean of per-camera / per-descriptor z-scored similarity into one joint matrix.
    Not yet implemented — Step 3."""
    raise NotImplementedError("fuse: implement in Step 3 (plan.md §5)")
