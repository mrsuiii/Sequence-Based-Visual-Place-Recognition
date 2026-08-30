"""Per-frame descriptors: SeqSLAM-style thumbnails (R1) and DINOv2 embeddings (R2). Step 3
(plan.md §5). R2 is the committed default alongside R1, fused — not an optional extra.
"""

from __future__ import annotations


def seqslam_descriptors(*args, **kwargs):
    """Downsample to 64x40 -> 8x8 patch-normalise -> flatten -> L2. Not yet implemented — Step 3."""
    raise NotImplementedError("seqslam_descriptors: implement in Step 3 (plan.md §5)")


def dino_descriptors(*args, **kwargs):
    """DINOv2 ViT-S/14 (torch.hub) CLS+GeM embedding per frame, 448x280 input. Not yet
    implemented — Step 3."""
    raise NotImplementedError("dino_descriptors: implement in Step 3 (plan.md §5)")


def cached(*args, **kwargs):
    """Disk-cache wrapper: .npy plus a provenance .json sidecar (source SHA-256 prefix,
    parameters, package versions) — CLAUDE.md §4. Not yet implemented — Step 3."""
    raise NotImplementedError("cached: implement in Step 3 (plan.md §5)")
