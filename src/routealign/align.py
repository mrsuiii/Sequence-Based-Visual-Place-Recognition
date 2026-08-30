"""Monotone alignment: DTW with open ends, path -> mapping. Step 4 (plan.md §6).

v1 (dtw_open_ends) is the committed default — the sequence-order constraint the brief asks for,
with free local slope and open ends, no fixed/linear offset. v2 (nullstate_dp) is cut by default
(see plan.md's revision note and config.yaml `align.nullstate_dp.enabled`) — build it only in the
day-2 buffer, and only after v1 is shipped, tested and understood.
"""

from __future__ import annotations


def dtw_open_ends(*args, **kwargs):
    """Vectorised open-ended monotone DTW; cost = 3 - clip(S, -3, 3), penalty `lam` per
    non-diagonal step. Fully specified (including the row-vectorised recurrence) in plan.md §6 —
    not yet transcribed here. Not yet implemented — Step 4."""
    raise NotImplementedError(
        "dtw_open_ends: implement in Step 4 (plan.md §6); the algorithm is already drafted there"
    )


def nullstate_dp(*args, **kwargs):
    """NULL-state affine-gap DP (v2): explicit match/skip states instead of a post-hoc no-match
    threshold. Cut by default (plan.md §6 revision note) — day-2 buffer only, and only if v1 is
    already shipped and understood."""
    raise NotImplementedError("nullstate_dp: cut by default (plan.md §6); day-2 buffer only")


def path_to_mapping(*args, **kwargs):
    """DTW path -> per-runA-row (runB_frame, lo, hi, boundary_clamped), widened to the runB
    stationary segment containing the best match. Not yet implemented — Step 4."""
    raise NotImplementedError("path_to_mapping: implement in Step 4 (plan.md §6)")


def local_slope(*args, **kwargs):
    """Local relative speed (runB frames per runA frame) along the alignment path. Not yet
    implemented — Step 4."""
    raise NotImplementedError("local_slope: implement in Step 4 (plan.md §6)")
