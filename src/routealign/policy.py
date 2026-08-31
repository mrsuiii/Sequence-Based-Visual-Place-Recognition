"""Task 3: the keep/discard manifest and its schema. Step 7 (plan.md §9)."""

from __future__ import annotations

import pandas as pd


def build_manifest(mapping: pd.DataFrame, inspection: dict) -> pd.DataFrame:
    """Build the per-frame keep/discard manifest with attached metadata (weather, lighting,
    lens_occlusion, correspondence, split via place_id), so a downstream consumer cannot silently
    misuse a frame.

    Args:
        mapping: `outputs/mapping.csv` contents.
        inspection: Task 1 per-camera records, e.g. loaded from `outputs/inspection/*.json`.

    Returns:
        One row per frame, per plan.md §9's manifest schema; also written to
        `outputs/keep_manifest.csv`.
    """
    raise NotImplementedError("build_manifest: implement in Step 7 (plan.md §9)")
