"""Tests for confidence.py's `fuse_confidence`. `reliability_table` is Step 6 (not yet built)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from routealign.confidence import fuse_confidence

_BASE_ROW = {
    "ridge_z": 2.0,
    "cam_disagree": 0.0,
    "desc_disagree": 0.0,
    "boundary_clamped": False,
    "ambiguous_range": False,
    "no_video": False,
}


def _cues(**overrides: object) -> pd.DataFrame:
    row = {**_BASE_ROW, **overrides}
    return pd.DataFrame([row])


def test_fuse_confidence_requires_all_cue_columns() -> None:
    """A cues frame missing a required column must fail loudly, not silently misfuse."""
    incomplete = pd.DataFrame([{"ridge_z": 2.0}])
    with pytest.raises(ValueError, match="missing required columns"):
        fuse_confidence(incomplete)


def test_fuse_confidence_top_tier_needs_high_ridge_and_tight_agreement() -> None:
    """High ridge strength and both disagreements within the tight threshold -> tier 0.9."""
    out = fuse_confidence(_cues(ridge_z=1.5, cam_disagree=3.0, desc_disagree=3.0))
    assert out["confidence"].iloc[0] == pytest.approx(0.9)
    assert out["status"].iloc[0] == "matched"


def test_fuse_confidence_mid_tier_needs_two_of_two_agreement_without_sift() -> None:
    """Without a `sift_inliers` column, the 0.7 tier's "2 of {cam, desc, SIFT}" vote degrades to
    "both of {cam, desc}" -- SIFT simply never contributes a vote."""
    out = fuse_confidence(_cues(ridge_z=1.0, cam_disagree=5.0, desc_disagree=5.0))
    assert out["confidence"].iloc[0] == pytest.approx(0.7)


def test_fuse_confidence_sift_inliers_can_supply_the_missing_vote() -> None:
    """One disagreement cue failing is recoverable via a SIFT-verified vote, matching the "2 of
    3" rule -- disagreement here (cam=10) fails the tight tier but SIFT plus a passing desc cue
    should still clear the 0.7 tier's 2-of-3 vote."""
    out = fuse_confidence(_cues(ridge_z=1.0, cam_disagree=10.0, desc_disagree=5.0, sift_inliers=12))
    assert out["confidence"].iloc[0] == pytest.approx(0.7)


def test_fuse_confidence_ambiguous_range_without_qualifying_for_higher_tiers() -> None:
    """A plateau row that fails both the 0.9 and 0.7 rules but is flagged `ambiguous_range`
    lands at tier 0.5, with status `ambiguous_range`."""
    out = fuse_confidence(
        _cues(ridge_z=0.8, cam_disagree=50.0, desc_disagree=1.0, ambiguous_range=True)
    )
    assert out["confidence"].iloc[0] == pytest.approx(0.5)
    assert out["status"].iloc[0] == "ambiguous_range"


def test_fuse_confidence_default_tier_is_matched_but_weak() -> None:
    """A row that clears no tier rule and isn't abstained falls to the 0.4 default, with an
    empty `reason` (it wasn't abstained, so there's nothing to explain)."""
    out = fuse_confidence(_cues(ridge_z=0.8, cam_disagree=50.0, desc_disagree=1.0))
    assert out["confidence"].iloc[0] == pytest.approx(0.4)
    assert out["status"].iloc[0] == "matched"
    assert out["reason"].iloc[0] == ""


@pytest.mark.parametrize(
    "overrides",
    [
        {"no_video": True},
        {"ridge_z": 0.5},  # below tau
        {"boundary_clamped": True, "ridge_z": 1.0},  # below boundary_ridge_min
        {"cam_disagree": 40.0, "desc_disagree": 40.0},  # both above disagree_no_match
    ],
)
def test_fuse_confidence_no_match_rule_fires(overrides: dict) -> None:
    """Each independent no-match trigger from plan.md §7 must abstain the row: NaN confidence,
    and status `no_video` for that specific case, `no_match` for the others -- with a non-empty
    `reason` explaining which trigger fired."""
    out = fuse_confidence(_cues(**overrides))
    assert np.isnan(out["confidence"].iloc[0])
    expected_status = "no_video" if overrides.get("no_video") else "no_match"
    assert out["status"].iloc[0] == expected_status
    assert out["reason"].iloc[0] != ""


def test_fuse_confidence_no_match_overrides_a_would_be_top_tier() -> None:
    """A row that would otherwise qualify for tier 0.9 but has no video must still be abstained --
    no-match takes priority over every tier rule."""
    out = fuse_confidence(_cues(ridge_z=3.0, no_video=True))
    assert np.isnan(out["confidence"].iloc[0])
    assert out["status"].iloc[0] == "no_video"
    assert out["reason"].iloc[0] == "no video for this row"


def test_fuse_confidence_reason_priority_prefers_no_video_over_weak_ridge() -> None:
    """When multiple no-match triggers hold at once, `reason` reports the earliest in the
    documented priority order (no_video > weak ridge > clamped boundary > mutual disagreement),
    matching `no_match`'s own precedence."""
    out = fuse_confidence(_cues(ridge_z=0.1, no_video=True))  # both no_video and weak-ridge fire
    assert out["reason"].iloc[0] == "no video for this row"
