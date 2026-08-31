"""Model-blind labelling sheets, label loading/merge, evaluation, the deletion test.
Step 6 (plan.md §8).

Labelling is user-primary; Claude's pass afterward is a QA/consistency check, not an independent
co-labeller (plan.md §0.2, §13) -- the two share a vision backbone with the pipeline's own DINOv2
descriptor and are not statistically independent judges. `evaluate` must report user-vs-Claude
agreement as a sanity number, never as inter-rater reliability or a label-noise floor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .frames import FrameStore


def make_sheets(
    store_b: FrameStore, anchor_indices: NDArray[np.int64], out_dir: str | Path
) -> None:
    """Build model-blind labelling tiles: a coarse pass (every 20th runB frame) followed by a
    fine pass (+/-20 frames around the coarse pick), built only from the JPEG cache -- never from
    model output.

    Args:
        store_b: Frame source for runB.
        anchor_indices: int64[N] runA frame indices to build sheets for.
        out_dir: Directory to write the labelling tiles into.
    """
    raise NotImplementedError("make_sheets: implement in Step 6 (plan.md §8)")


def load_labels(gt_dir: str | Path) -> pd.DataFrame:
    """Load `gt/anchors_user.csv` and `gt/anchors_claude.csv`.

    Args:
        gt_dir: Directory containing the label CSVs.

    Returns:
        One row per (labeller, anchor), with columns `runA_frame`, `runB_best`, `lo`, `hi`,
        `quality`, `note`.
    """
    raise NotImplementedError("load_labels: implement in Step 6 (plan.md §8)")


def evaluate(mapping: pd.DataFrame, anchors: pd.DataFrame) -> dict:
    """Compute error, hit@k and Wilson-95%-CI metrics against merged ground-truth anchors.

    Args:
        mapping: `outputs/mapping.csv` contents.
        anchors: Merged test-split anchors, e.g. `gt/anchors_merged.csv`.

    Returns:
        dict with per-stratum and per-confidence-tier metric breakdowns.
    """
    raise NotImplementedError("evaluate: implement in Step 6 (plan.md §8)")


def deletion_test(runb_delete_range: tuple[int, int]) -> dict:
    """Delete a known runB frame span, rerun the pipeline, and check that the mapping abstains on
    the corresponding runA rows. Needs no manual labels -- a self-supervised check on the
    no-match rule.

    Args:
        runb_delete_range: `(start, end)` runB frame indices to delete (exclusive end).

    Returns:
        dict with precision/recall of the no-match rule over the deleted span.
    """
    raise NotImplementedError("deletion_test: implement in Step 6 (plan.md §8)")
