"""Tests for policy.py's `build_manifest` and its private helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from routealign.policy import (
    RunCameraFacts,
    _apply_keep_rule,
    _assign_split,
    _attach_correspondence,
    _attach_motion,
    _attach_static,
    _attach_timing,
    _dedup_keep_mask,
    _frame_skeleton,
    _interpolate_missing_corr_frame,
    _place_id_for_runb_frame,
    build_manifest,
)


def _facts(run: str, cam: str, decoded_count: int, **overrides: object) -> RunCameraFacts:
    base = dict(
        run=run,
        cam=cam,
        decoded_count=decoded_count,
        src_file=f"{run}/{cam}.hevc",
        src_sha256="deadbeef",
        fps_declared=10.0,
        fps_measured=9.99,
        width=64,
        height=48,
        codec="hevc",
        idr_indices=(0,),
        weather="sunny",
        lighting="daylight",
        lens_occlusion="none",
    )
    return RunCameraFacts(**{**base, **overrides})


def _even_timestamps(n: int, gap_after: int | None = None) -> np.ndarray:
    """`n` timestamps 100ms apart, ns since epoch; optionally a 300ms gap after `gap_after`."""
    dt = np.full(n - 1, 100_000_000, dtype=np.int64)
    if gap_after is not None:
        dt[gap_after] = 300_000_000
    return np.concatenate([[1_700_000_000_000_000_000], np.cumsum(dt) + 1_700_000_000_000_000_000])


# -- _dedup_keep_mask -----------------------------------------------------------------------


def test_dedup_keep_mask_thins_a_stationary_run_to_the_target_rate() -> None:
    """10 fps, keep_hz=1 -> stride 10: only every 10th frame inside the segment survives."""
    is_stationary, dedup_keep = _dedup_keep_mask(n=25, segments=[(5, 25)], fps=10.0, keep_hz=1.0)
    assert is_stationary[:5].sum() == 0
    assert is_stationary[5:].all()
    assert dedup_keep[5:25].sum() == 2  # offsets 0 and 10 within the 20-frame segment
    assert dedup_keep[:5].all()  # nothing to thin outside a segment


# -- _place_id_for_runb_frame ------------------------------------------------------------------


def test_place_id_buckets_a_mid_route_frame_normally() -> None:
    place = _place_id_for_runb_frame(np.array([250.0]), bucket=100, depot_margin=20, runb_max=1000)
    assert place[0] == 2


def test_place_id_merges_start_and_end_depot() -> None:
    place = _place_id_for_runb_frame(
        np.array([5.0, 995.0]), bucket=100, depot_margin=20, runb_max=1000
    )
    assert place[0] == place[1] == -1


def test_place_id_is_unplaced_for_nan() -> None:
    place = _place_id_for_runb_frame(np.array([np.nan]), bucket=100, depot_margin=20, runb_max=1000)
    assert place[0] == -1


# -- _interpolate_missing_corr_frame ------------------------------------------------------------


def test_interpolate_missing_corr_frame_fills_from_neighbours() -> None:
    runa = np.array([0, 1, 2, 3, 4])
    corr = np.array([10.0, np.nan, np.nan, np.nan, 14.0])
    filled = _interpolate_missing_corr_frame(runa, corr)
    np.testing.assert_allclose(filled, [10.0, 11.0, 12.0, 13.0, 14.0])


# -- _assign_split ------------------------------------------------------------------------------


def test_assign_split_is_contiguous_and_leaves_a_buffer() -> None:
    """10 places, val=test=0.2 (2 each), buffer=1 -> train, buffer, val, buffer, test."""
    place_id = pd.Series(range(10))
    split = _assign_split(place_id, val_frac=0.2, test_frac=0.2, buffer_places=1)
    assert (split == "test").sum() == 2
    assert (split == "val").sum() == 2
    assert (split == "unassigned").sum() >= 2  # the two buffers
    # every train place must sort below every val place, which sorts below every test place
    train_max = place_id[split == "train"].max()
    val_min, val_max = place_id[split == "val"].min(), place_id[split == "val"].max()
    test_min = place_id[split == "test"].min()
    assert train_max < val_min <= val_max < test_min


def test_assign_split_puts_the_merged_depot_in_unassigned() -> None:
    place_id = pd.Series([-1, -1, 0, 1, 2, 3, 4, 5, 6, 7])
    split = _assign_split(place_id, val_frac=0.2, test_frac=0.2, buffer_places=0)
    assert (split[place_id == -1] == "unassigned").all()


# -- _attach_timing -----------------------------------------------------------------------------


def test_attach_timing_flags_the_neighbours_of_an_irregular_gap() -> None:
    """A single 300ms gap after index 3 must flag exactly rows 3 and 4 -- not the whole run."""
    facts = _facts("runA", "cam0", decoded_count=8)
    ts = _even_timestamps(8, gap_after=3)
    df = _attach_timing(
        _frame_skeleton(facts, 8), ts, tail_zone_lines=0, irregular_lo_ms=80, irregular_hi_ms=120
    )
    assert df.loc[df["interval_irregular"], "frame"].tolist() == [3, 4]


def test_attach_timing_flags_the_configured_tail_window() -> None:
    facts = _facts("runA", "cam0", decoded_count=8)
    ts = _even_timestamps(8)
    df = _attach_timing(
        _frame_skeleton(facts, 8), ts, tail_zone_lines=3, irregular_lo_ms=80, irregular_hi_ms=120
    )
    assert df.loc[df["tail_zone"], "frame"].tolist() == [5, 6, 7]
    assert df.loc[~df["tail_zone"], "ts_confidence"].eq("measured").all()
    assert df.loc[df["tail_zone"], "ts_confidence"].eq("assumed").all()


# -- _attach_motion -----------------------------------------------------------------------------


def test_attach_motion_does_not_crash_on_the_last_decoded_frame() -> None:
    """Regression test: `frames.motion_energy` returns `decoded_count - 1` values (one per
    transition), so the last decoded frame has no motion measurement -- must be NaN, not an
    IndexError (found running `manifest` on the real dataset, decoded_count=2674, motion size
    2673)."""
    facts = _facts("runA", "cam0", decoded_count=5)
    df = _attach_static(_frame_skeleton(facts, 5), facts)
    out = _attach_motion(df, facts, motion_energy=np.full(4, 5.0), segments=[], keep_hz=1.0)
    assert out.loc[4, "decode_ok"]  # the last row did decode...
    assert np.isnan(out.loc[4, "motion_energy"])  # ...but has no transition to measure
    np.testing.assert_array_equal(out.loc[:3, "motion_energy"].to_numpy(), [5.0, 5.0, 5.0, 5.0])


# -- _attach_correspondence ----------------------------------------------------------------------


def test_attach_correspondence_copies_mapping_for_runa_rows() -> None:
    facts = _facts("runA", "cam0", decoded_count=3)
    df = _frame_skeleton(facts, 3)
    mapping = pd.DataFrame(
        {
            "runA_frame": [0, 1, 2],
            "runB_frame": [10.0, np.nan, 12.0],
            "runB_frame_lo": [10.0, np.nan, 12.0],
            "runB_frame_hi": [10.0, np.nan, 12.0],
            "confidence": [0.9, np.nan, 0.7],
            "status": ["matched", "no_match", "matched"],
            "method": ["seqslam+dinov2"] * 3,
        }
    )
    out = _attach_correspondence(df, "runA", mapping)
    assert out.loc[0, "corr_frame"] == 10.0
    assert pd.isna(out.loc[1, "corr_frame"])
    assert out.loc[2, "corr_frame"] == 12.0
    assert out.loc[1, "corr_status"] == "no_match"


def test_attach_correspondence_marks_runb_rows_not_applicable() -> None:
    facts = _facts("runB", "cam0", decoded_count=3)
    df = _frame_skeleton(facts, 3)
    out = _attach_correspondence(df, "runB", pd.DataFrame())
    assert (out["corr_status"] == "not_applicable").all()
    assert out["corr_frame"].isna().all()


# -- _apply_keep_rule ----------------------------------------------------------------------------


def test_apply_keep_rule_discards_only_undecoded_rows() -> None:
    df = pd.DataFrame({"decode_ok": [True, True, False]})
    out = _apply_keep_rule(df)
    assert out["keep"].tolist() == [True, True, False]
    assert out.loc[2, "reject_reason"] == "no decoded frame for this row"
    assert out.loc[0, "reject_reason"] == ""


# -- build_manifest (integration) -----------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    """A tiny synthetic universe: runA (10 lines, cam5 two short) and runB (12 lines, both full),
    with a stationary depot at each end of runB and a `no_match` runA row to interpolate."""
    run_cameras = [
        _facts("runA", "cam0", decoded_count=10),
        _facts("runA", "cam5", decoded_count=8),
        _facts("runB", "cam0", decoded_count=12, idr_indices=(0, 6)),
        _facts("runB", "cam5", decoded_count=12, idr_indices=(0, 6), lens_occlusion="traffic"),
    ]
    timestamps_by_run = {
        "runA": _even_timestamps(10, gap_after=3),
        "runB": _even_timestamps(12),
    }
    # size decoded_count - 1 (frames.motion_energy is per-transition, k -> k+1 -- no entry for
    # the very last decoded frame; the real dataset first surfaced this as an IndexError)
    motion_by_run_cam = {
        ("runA", "cam0"): np.full(9, 5.0),
        ("runA", "cam5"): np.full(7, 5.0),
        ("runB", "cam0"): np.full(11, 5.0),
        ("runB", "cam5"): np.full(11, 5.0),
    }
    stationary_by_run = {"runA": [], "runB": [(0, 2), (10, 12)]}
    mapping = pd.DataFrame(
        {
            "runA_frame": list(range(10)),
            "runB_frame": [1.0, 3.0, 5.0, np.nan, 9.0, 10.0, 11.0, np.nan, np.nan, np.nan],
            "runB_frame_lo": [1.0, 3.0, 5.0, np.nan, 9.0, 10.0, 11.0, np.nan, np.nan, np.nan],
            "runB_frame_hi": [1.0, 3.0, 5.0, np.nan, 9.0, 10.0, 11.0, np.nan, np.nan, np.nan],
            "confidence": [0.9, 0.9, 0.9, np.nan, 0.7, 0.7, 0.7, np.nan, np.nan, np.nan],
            "status": ["matched"] * 3 + ["no_match"] + ["matched"] * 3 + ["no_match"] * 3,
            "method": ["seqslam+dinov2"] * 10,
        }
    )
    return build_manifest(
        run_cameras,
        timestamps_by_run,
        motion_by_run_cam,
        stationary_by_run,
        mapping,
        interval_irregular_lo_ms=80.0,
        interval_irregular_hi_ms=120.0,
        tail_zone_lines=2,
        dedup_keep_hz=1.0,
        place_id_bucket_frames=100,
        depot_runb_margin=2,
        split_val_frac=0.2,
        split_test_frac=0.2,
        split_buffer_places=0,
    )


def test_manifest_has_one_row_per_timestamp_line_per_camera(manifest: pd.DataFrame) -> None:
    assert len(manifest) == 10 + 10 + 12 + 12  # runA cam0/cam5 + runB cam0/cam5, each 1 row/line


def test_manifest_keep_matches_decode_ok_exactly(manifest: pd.DataFrame) -> None:
    assert (manifest["keep"] == manifest["decode_ok"]).all()
    a_cam5 = manifest[(manifest["run"] == "runA") & (manifest["cam"] == "cam5")]
    assert a_cam5["keep"].tolist() == [True] * 8 + [False] * 2


def test_manifest_depot_frames_share_one_place_id(manifest: pd.DataFrame) -> None:
    """runB frames 0-1 (start) and 10-11 (end, margin=2) merge into the same depot place_id."""
    runb0 = manifest[(manifest["run"] == "runB") & (manifest["cam"] == "cam0")]
    depot = runb0[runb0["frame"].isin([0, 1, 10, 11])]
    assert (depot["place_id"] == -1).all()
    assert (depot["split"] == "unassigned").all()


def test_manifest_runa_place_id_follows_its_correspondence(manifest: pd.DataFrame) -> None:
    """runA frame 2 (-> runB 5, mid-route) must land in a normal, non-depot bucket."""
    a_cam0 = manifest[(manifest["run"] == "runA") & (manifest["cam"] == "cam0")]
    row = a_cam0[a_cam0["frame"] == 2].iloc[0]
    assert row["place_id"] == 0  # floor(5 / 100)


def test_manifest_lens_occlusion_is_camera_specific(manifest: pd.DataFrame) -> None:
    b_cam5 = manifest[(manifest["run"] == "runB") & (manifest["cam"] == "cam5")]
    b_cam0 = manifest[(manifest["run"] == "runB") & (manifest["cam"] == "cam0")]
    assert (b_cam5["lens_occlusion"] == "traffic").all()
    assert (b_cam0["lens_occlusion"] == "none").all()
