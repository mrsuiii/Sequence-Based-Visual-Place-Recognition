"""Independent agreement cues used as confidence evidence. Step 5 (plan.md §7).

sift_verify is cut by default (see plan.md's revision note and config.yaml `verify.sift.enabled`) —
the other three cues do not depend on it and already give multi-signal verification on their own.
"""

from __future__ import annotations


def ridge_strength(*args, **kwargs):
    """Mean joint similarity along the path over a small +/- row window. Not yet implemented — Step 5."""
    raise NotImplementedError("ridge_strength: implement in Step 5 (plan.md §7)")


def margin(*args, **kwargs):
    """Path value minus the best value outside a +/- column band (catches loop-closure aliasing
    and repeated facades). Not yet implemented — Step 5."""
    raise NotImplementedError("margin: implement in Step 5 (plan.md §7)")


def path_disagreement(*args, **kwargs):
    """|j from one independent alignment - j from another| (cam0-only vs cam5-only, or
    SeqSLAM vs DINOv2). Not yet implemented — Step 5."""
    raise NotImplementedError("path_disagreement: implement in Step 5 (plan.md §7)")


def sift_verify(*args, **kwargs):
    """RootSIFT (ratio 0.8) + MAGSAC inlier count. Cut by default (plan.md §7 revision note) —
    build only in the day-2 buffer if genuinely ahead of schedule."""
    raise NotImplementedError("sift_verify: cut by default (plan.md §7); day-2 buffer only")
