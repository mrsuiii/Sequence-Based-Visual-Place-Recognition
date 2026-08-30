"""Fuse agreement cues into an ordinal confidence tier, plus the no-match (abstention) rule.
Step 5 (plan.md §7).
"""

from __future__ import annotations


def fuse_confidence(*args, **kwargs):
    """Ordinal tier from ridge/margin/disagreement thresholds (config.yaml `confidence.tiers`) —
    explicitly not a calibrated probability (plan.md §13: n ~= 24 anchors). Not yet implemented."""
    raise NotImplementedError("fuse_confidence: implement in Step 5 (plan.md §7)")


def reliability_table(*args, **kwargs):
    """Measured accuracy per confidence tier on held-out (test) anchors. Not yet implemented —
    Step 6 (plan.md §8)."""
    raise NotImplementedError("reliability_table: implement in Step 6 (plan.md §8)")
