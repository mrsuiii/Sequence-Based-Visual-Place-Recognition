"""Task 3: the keep/discard manifest and its schema. Step 7 (plan.md §9)."""

from __future__ import annotations


def build_manifest(*args, **kwargs):
    """Per-frame keep/discard decision plus attached metadata (weather, lighting, lens_occlusion,
    correspondence, split via place_id) so a downstream consumer cannot silently misuse a frame.
    Not yet implemented — Step 7."""
    raise NotImplementedError("build_manifest: implement in Step 7 (plan.md §9)")
