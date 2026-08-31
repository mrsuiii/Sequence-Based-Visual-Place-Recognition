"""Fuse agreement cues into an ordinal confidence tier, plus the no-match (abstention) rule.
Step 5 (plan.md §7).
"""

from __future__ import annotations

import pandas as pd


def fuse_confidence(cues: pd.DataFrame) -> pd.DataFrame:
    """Assign an ordinal confidence tier and a no-match flag from the agreement cues.

    Tier thresholds come from config.yaml `confidence.tiers`; this is an ordinal ranking, not a
    calibrated probability (plan.md §13: n ~= 24 anchors).

    Args:
        cues: One row per runA index, with columns `ridge_z`, `margin_z`, `cam_disagree`,
            `desc_disagree` (see `verify.py`).

    Returns:
        `cues` with `confidence` and `status` columns added.
    """
    raise NotImplementedError("fuse_confidence: implement in Step 5 (plan.md §7)")


def reliability_table(mapping: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Compute measured accuracy per confidence tier against held-out (test) anchors.

    Args:
        mapping: `outputs/mapping.csv` contents.
        labels: Test-split labelled anchors (see `groundtruth.load_labels`).

    Returns:
        One row per confidence tier, with columns for anchor count and measured accuracy.
    """
    raise NotImplementedError("reliability_table: implement in Step 6 (plan.md §8)")
