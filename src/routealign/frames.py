"""FrameStore: read-through access to the JPEG cache, crops, LRU. Motion energy and stationarity
detection. Step 2 (plan.md §4).
"""

from __future__ import annotations


class FrameStore:
    """Read-through access to data/cache/frames/{run}/{cam}/ with an LRU. Not yet implemented."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("FrameStore: implement in Step 2 (plan.md §4)")


def motion_energy(*args, **kwargs):
    """Mean |Δ| between consecutive gray frames (64x48 probe). Not yet implemented — Step 2."""
    raise NotImplementedError("motion_energy: implement in Step 2 (plan.md §4)")


def stationary_segments(*args, **kwargs):
    """Runs where BOTH cameras' motion_energy sits below their own median for >= 10 frames —
    single-camera low motion can be low texture, not a stop (plan.md §4). Not yet implemented."""
    raise NotImplementedError("stationary_segments: implement in Step 2 (plan.md §4)")
