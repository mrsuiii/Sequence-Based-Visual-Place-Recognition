"""Fuse agreement cues into an ordinal confidence tier, plus the no-match (abstention) rule.
Step 5 (plan.md §7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED_CUE_COLUMNS = frozenset(
    {"ridge_z", "cam_disagree", "desc_disagree", "boundary_clamped", "ambiguous_range", "no_video"}
)


def fuse_confidence(
    cues: pd.DataFrame,
    tau: float = 0.75,  # why: config.yaml confidence.no_match_tau_default, from the deletion test
    ridge_high: float = 1.5,  # why: plan.md §7 tier-0.9 rule
    ridge_mid: float = 1.0,  # why: plan.md §7 tier-0.7 rule
    disagree_tight: float = 3.0,  # why: plan.md §7 tier-0.9 rule
    disagree_loose: float = 5.0,  # why: plan.md §7 tier-0.7 rule ("cam <= 5, desc <= 5")
    disagree_no_match: float = 30.0,  # why: plan.md §7 no-match rule
    boundary_ridge_min: float = 1.5,  # why: plan.md §7 no-match rule
) -> pd.DataFrame:
    """Assign an ordinal confidence tier and a no-match flag from the agreement cues.

    Implements plan.md §7's no-match rule and tier rules. SIFT verification is cut by default
    (plan.md §7 revision note), so where the spec's tier-0.7 rule asks for "two of {cam <=5,
    desc <=5, SIFT verified}", this treats a missing `sift_inliers` column as SIFT never voting --
    i.e. it degrades to "both of {cam <= 5, desc <= 5}" rather than raising, so the function works
    identically before and after SIFT is added.

    No-match (confidence set to NaN, `status="no_match"` or `"no_video"`) fires on: no video for
    the row; `ridge_z < tau`; `boundary_clamped` with `ridge_z < boundary_ridge_min`; or both
    `cam_disagree` and `desc_disagree` above `disagree_no_match`. This is an ordinal ranking
    tuned on synthetic warps, the deletion test and the 6 dev anchors only -- not a calibrated
    probability (plan.md §13).

    Args:
        cues: One row per runA index, with columns `ridge_z`, `cam_disagree`, `desc_disagree`,
            `boundary_clamped`, `ambiguous_range`, `no_video` (see `verify.py`), and optionally
            `sift_inliers` if SIFT verification ran.
        tau (float, optional): Minimum `ridge_z` for a row to be considered matched at all.
            Defaults to 0.75.
        ridge_high (float, optional): Minimum `ridge_z` for the 0.9 tier. Defaults to 1.5.
        ridge_mid (float, optional): Minimum `ridge_z` for the 0.7 tier. Defaults to 1.0.
        disagree_tight (float, optional): Maximum `cam_disagree`/`desc_disagree` for the 0.9
            tier. Defaults to 3.0.
        disagree_loose (float, optional): Maximum `cam_disagree`/`desc_disagree` counted as
            agreeing for the 0.7 tier's 2-of-{cam, desc, SIFT} vote. Defaults to 5.0.
        disagree_no_match (float, optional): `cam_disagree` and `desc_disagree` threshold above
            which a row is abstained regardless of `ridge_z`. Defaults to 30.0.
        boundary_ridge_min (float, optional): `ridge_z` a `boundary_clamped` row must clear to
            avoid being abstained. Defaults to 1.5.

    Raises:
        ValueError: If `cues` is missing any required column.

    Returns:
        `cues` with `confidence` (float, NaN where unmatched), `status`
        (`{matched, ambiguous_range, no_match, no_video}`) and `reason` (human-readable, empty
        for matched/ambiguous_range rows; the first no-match trigger that fired, in the priority
        order no_video > weak ridge > clamped boundary > mutual disagreement, otherwise) columns
        added.
    """
    missing = _REQUIRED_CUE_COLUMNS - set(cues.columns)
    if missing:
        raise ValueError(f"fuse_confidence: cues is missing required columns: {sorted(missing)}")

    ridge_z = cues["ridge_z"]
    cam_dis = cues["cam_disagree"]
    desc_dis = cues["desc_disagree"]
    sift_agrees = cues["sift_inliers"] > 0 if "sift_inliers" in cues.columns else False

    weak_ridge = ridge_z < tau
    clamped_weak = cues["boundary_clamped"] & (ridge_z < boundary_ridge_min)
    mutual_disagree = (cam_dis > disagree_no_match) & (desc_dis > disagree_no_match)
    no_match = cues["no_video"] | weak_ridge | clamped_weak | mutual_disagree

    tier_09 = (ridge_z >= ridge_high) & (cam_dis <= disagree_tight) & (desc_dis <= disagree_tight)
    agreement_votes = (
        (cam_dis <= disagree_loose).astype(int)
        + (desc_dis <= disagree_loose).astype(int)
        + np.asarray(sift_agrees, dtype=int)
    )
    tier_07 = ~tier_09 & (ridge_z >= ridge_mid) & (agreement_votes >= 2)
    tier_05 = ~tier_09 & ~tier_07 & cues["ambiguous_range"]

    confidence = np.full(len(cues), 0.4)  # default: matched but weak (plan.md §7)
    confidence[tier_05.to_numpy()] = 0.5
    confidence[tier_07.to_numpy()] = 0.7
    confidence[tier_09.to_numpy()] = 0.9
    confidence[no_match.to_numpy()] = np.nan

    status = np.where(cues["ambiguous_range"], "ambiguous_range", "matched")
    status = np.where(no_match.to_numpy(), "no_match", status)
    status = np.where(cues["no_video"].to_numpy(), "no_video", status)

    # priority order matches `no_match`'s own precedence: earliest-listed condition that could
    # have fired is reported, even if a later one also holds for the same row
    reason = np.select(
        [
            cues["no_video"].to_numpy(),
            weak_ridge.to_numpy(),
            clamped_weak.to_numpy(),
            mutual_disagree.to_numpy(),
        ],
        [
            "no video for this row",
            f"ridge_z below tau ({tau})",
            f"boundary_clamped with ridge_z below {boundary_ridge_min}",
            f"cam_disagree and desc_disagree both above {disagree_no_match}",
        ],
        default="",
    )

    out = cues.copy()
    out["confidence"] = confidence
    out["status"] = status
    out["reason"] = reason
    return out


def reliability_table(mapping: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Compute measured accuracy per confidence tier against held-out (test) anchors.

    Args:
        mapping: `outputs/mapping.csv` contents.
        labels: Test-split labelled anchors (see `groundtruth.load_labels`).

    Returns:
        One row per confidence tier, with columns for anchor count and measured accuracy.
    """
    raise NotImplementedError("reliability_table: implement in Step 6 (plan.md §8)")
