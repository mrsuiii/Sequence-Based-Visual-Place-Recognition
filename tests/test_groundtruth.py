"""Tests for groundtruth.py's `stratum_s_indices`, `make_sheets`, `deletion_test`,
`load_labels`, `evaluate`."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from routealign.frames import FrameStore
from routealign.groundtruth import (
    deletion_test,
    evaluate,
    load_labels,
    make_sheets,
    stratum_s_indices,
)


def _make_store(
    tmp_path, run: str, cam: str, n: int, size: tuple[int, int] = (30, 20)
) -> FrameStore:
    """Write `n` tiny synthetic JPEGs and open a `FrameStore` over them."""
    cache_dir = tmp_path / "frames"
    frame_dir = cache_dir / run / cam
    frame_dir.mkdir(parents=True)
    h, w = size
    for k in range(n):
        img = np.full((h, w, 3), fill_value=(k * 7) % 256, dtype=np.uint8)
        cv2.imwrite(str(frame_dir / f"{k:05d}.jpg"), img)
    return FrameStore(cache_dir, run, cam, crop_row_frac=1.0)


def test_stratum_s_indices_is_deterministic() -> None:
    """The same seed must always produce the same anchors -- CLAUDE.md §4 determinism."""
    a = stratum_s_indices(seed=42)
    b = stratum_s_indices(seed=42)
    np.testing.assert_array_equal(a, b)


def test_stratum_s_indices_different_seeds_differ() -> None:
    """Sanity check that `seed` actually does something (not silently ignored)."""
    a = stratum_s_indices(seed=1)
    b = stratum_s_indices(seed=2)
    assert not np.array_equal(a, b)


def test_stratum_s_indices_has_the_documented_default_count() -> None:
    """18 sweep points + 4 fixed points = 22 (config.yaml `groundtruth.stratum_s`), assuming no
    jitter collision with a fixed point (astronomically unlikely at these seeds/ranges)."""
    idx = stratum_s_indices(seed=42)
    assert idx.size == 22


def test_stratum_s_indices_includes_the_fixed_positions() -> None:
    """The plateau (60, 150) and tail (2600, 2650) anchors must always be present regardless of
    seed -- they are fixed points of known interest, not part of the random sweep."""
    idx = set(stratum_s_indices(seed=7).tolist())
    assert {60, 150, 2600, 2650} <= idx


def test_stratum_s_indices_sweep_points_are_sorted_and_unique() -> None:
    idx = stratum_s_indices(seed=42)
    assert np.all(np.diff(idx) > 0)


def test_make_sheets_writes_anchor_refs_and_coarse_tiles(tmp_path) -> None:
    """Both outputs that don't need a coarse pick must always be written: one reference tile per
    anchor, and the shared coarse overview across all of runB."""
    store_a0 = _make_store(tmp_path, "runA", "cam0", n=10)
    store_a5 = _make_store(tmp_path, "runA", "cam5", n=10)
    store_b0 = _make_store(tmp_path, "runB", "cam0", n=50)
    store_b5 = _make_store(tmp_path, "runB", "cam5", n=50)
    out_dir = tmp_path / "sheets"

    make_sheets(store_a0, store_a5, store_b0, store_b5, np.array([2, 5]), out_dir, coarse_step=10)

    assert (out_dir / "anchor_refs" / "00002.jpg").exists()
    assert (out_dir / "anchor_refs" / "00005.jpg").exists()
    # coarse_step=10 over 50 frames -> indices 0,10,20,30,40
    for j in (0, 10, 20, 30, 40):
        assert (out_dir / "coarse" / f"{j:05d}.jpg").exists()


def test_make_sheets_handles_a_camera_shorter_than_the_other(tmp_path) -> None:
    """runB cam5 shorter than cam0 (like the real dataset, 2615 vs 2622): coarse indices beyond
    cam5's own length must still produce a tile (with a placeholder), not crash."""
    store_a0 = _make_store(tmp_path, "runA", "cam0", n=5)
    store_a5 = _make_store(tmp_path, "runA", "cam5", n=5)
    store_b0 = _make_store(tmp_path, "runB", "cam0", n=50)
    store_b5 = _make_store(tmp_path, "runB", "cam5", n=25)  # shorter than cam0
    out_dir = tmp_path / "sheets"

    make_sheets(store_a0, store_a5, store_b0, store_b5, np.array([0]), out_dir, coarse_step=10)

    # index 40 exists in cam0 (n=50) but not cam5 (n=25) -- must still write a tile
    assert (out_dir / "coarse" / "00040.jpg").exists()
    img = cv2.imread(str(out_dir / "coarse" / "00040.jpg"))
    assert img is not None


def test_make_sheets_writes_no_fine_sheet_without_coarse_picks(tmp_path) -> None:
    """Without `coarse_picks`, only the anchor refs + coarse overview are written."""
    store_a0 = _make_store(tmp_path, "runA", "cam0", n=5)
    store_a5 = _make_store(tmp_path, "runA", "cam5", n=5)
    store_b0 = _make_store(tmp_path, "runB", "cam0", n=20)
    store_b5 = _make_store(tmp_path, "runB", "cam5", n=20)
    out_dir = tmp_path / "sheets"

    make_sheets(store_a0, store_a5, store_b0, store_b5, np.array([0]), out_dir, coarse_step=5)

    assert not any(out_dir.glob("fine_*"))


def test_make_sheets_writes_a_fine_sheet_for_a_given_coarse_pick(tmp_path) -> None:
    """With a coarse pick for an anchor, a fine sheet covering +/- fine_radius around it must be
    written, bounded to the valid frame range."""
    store_a0 = _make_store(tmp_path, "runA", "cam0", n=5)
    store_a5 = _make_store(tmp_path, "runA", "cam5", n=5)
    store_b0 = _make_store(tmp_path, "runB", "cam0", n=100)
    store_b5 = _make_store(tmp_path, "runB", "cam5", n=100)
    out_dir = tmp_path / "sheets"

    make_sheets(
        store_a0,
        store_a5,
        store_b0,
        store_b5,
        np.array([0]),
        out_dir,
        fine_radius=3,
        coarse_picks={0: 50},
    )

    fine_dir = out_dir / "fine_00000"
    for j in range(47, 54):  # 50 +/- 3, inclusive
        assert (fine_dir / f"{j:05d}.jpg").exists()
    assert not (fine_dir / "00046.jpg").exists()
    assert not (fine_dir / "00054.jpg").exists()


def _diagonal_similarity(n: int) -> np.ndarray:
    """A z-scored-style similarity matrix with a clean diagonal ridge (3.0) and a flat, weak
    off-diagonal (-1.0, safely below any reasonable `tau`) -- so `dtw_open_ends` follows the
    diagonal exactly when nothing is deleted, and `ridge_strength` reads unambiguously high on
    it, low off it."""
    sim = np.full((n, n), -1.0, dtype=np.float64)
    np.fill_diagonal(sim, 3.0)
    return sim


def test_deletion_test_makes_affected_rows_abstain() -> None:
    """Deleting the columns a row's original match depended on must drop that row's ridge_z
    below tau; a row far from the deletion must be unaffected."""
    n = 50
    similarity = _diagonal_similarity(n)
    original_runb_frame = np.arange(n, dtype=np.int64)  # identity matches the clean diagonal

    result = deletion_test(
        similarity,
        original_runb_frame,
        runb_delete_range=(20, 30),
        lam=0.5,
        ridge_window=3,
        tau=0.75,
    )

    assert result["n_affected"] == 10  # rows 20..29
    assert result["recall"] > 0.5  # most of the affected rows must abstain
    assert 0.0 <= result["precision"] <= 1.0


def test_deletion_test_leaves_far_rows_matched() -> None:
    """A row whose original match is nowhere near the deleted span must not spuriously abstain
    just because DTW's global path shifted a little near the deletion."""
    n = 50
    similarity = _diagonal_similarity(n)
    original_runb_frame = np.arange(n, dtype=np.int64)

    result = deletion_test(
        similarity,
        original_runb_frame,
        runb_delete_range=(20, 30),
        lam=0.5,
        ridge_window=3,
        tau=0.75,
    )

    assert result["tn"] > 0  # at least some far-away rows correctly stayed matched


def test_deletion_test_handles_no_affected_rows() -> None:
    """Deleting a span nothing originally matched into must not crash -- precision/recall are
    undefined (NaN), not a ZeroDivisionError."""
    n = 20
    similarity = _diagonal_similarity(n)
    original_runb_frame = np.arange(n, dtype=np.int64)  # all matches are on [0, 20), never >= 100

    result = deletion_test(
        similarity,
        original_runb_frame,
        runb_delete_range=(100, 110),
        lam=0.5,
        ridge_window=3,
        tau=0.75,
    )

    assert result["n_affected"] == 0
    assert np.isnan(result["recall"])


def test_deletion_test_returns_the_documented_keys() -> None:
    n = 20
    similarity = _diagonal_similarity(n)
    original_runb_frame = np.arange(n, dtype=np.int64)

    result = deletion_test(
        similarity,
        original_runb_frame,
        runb_delete_range=(5, 10),
        lam=0.5,
        ridge_window=3,
        tau=0.75,
    )

    assert set(result.keys()) == {"n_affected", "tp", "fp", "fn", "tn", "precision", "recall"}


def test_make_sheets_fine_sheet_clips_at_the_start_of_runb(tmp_path) -> None:
    """A coarse pick near 0 must not try to read a negative frame index."""
    store_a0 = _make_store(tmp_path, "runA", "cam0", n=5)
    store_a5 = _make_store(tmp_path, "runA", "cam5", n=5)
    store_b0 = _make_store(tmp_path, "runB", "cam0", n=20)
    store_b5 = _make_store(tmp_path, "runB", "cam5", n=20)
    out_dir = tmp_path / "sheets"

    make_sheets(
        store_a0,
        store_a5,
        store_b0,
        store_b5,
        np.array([0]),
        out_dir,
        fine_radius=5,
        coarse_picks={0: 2},
    )

    fine_dir = out_dir / "fine_00000"
    assert (fine_dir / "00000.jpg").exists()
    assert len(list(fine_dir.glob("*.jpg"))) == 8  # 0..7 inclusive, clipped from -5..7


def test_load_labels_reads_user_only_when_claude_file_is_absent(tmp_path) -> None:
    """Only `anchors_user.csv` exists so far -- `load_labels` must not require
    `anchors_claude.csv` to also be present."""
    pd.DataFrame({"runA_frame": [1, 2], "runB_best": [10, 20]}).to_csv(
        tmp_path / "anchors_user.csv", index=False
    )

    labels = load_labels(tmp_path)

    assert set(labels["labeller"]) == {"user"}
    assert len(labels) == 2


def test_load_labels_combines_both_files_when_both_exist(tmp_path) -> None:
    pd.DataFrame({"runA_frame": [1], "runB_best": [10]}).to_csv(
        tmp_path / "anchors_user.csv", index=False
    )
    pd.DataFrame({"runA_frame": [1], "runB_best": [11]}).to_csv(
        tmp_path / "anchors_claude.csv", index=False
    )

    labels = load_labels(tmp_path)

    assert set(labels["labeller"]) == {"user", "claude"}
    assert len(labels) == 2


def test_load_labels_raises_if_neither_file_exists(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path)


def _anchors(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic anchors DataFrame, filling in `runB_best`/`note` (real
    `gt/anchors_user.csv` columns `evaluate`'s `per_anchor` now carries through) with harmless
    defaults where a test case doesn't care about them."""
    defaults = {"runB_best": None, "note": ""}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _mapping(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_evaluate_perfect_predictions_score_zero_error() -> None:
    anchors = _anchors(
        [
            {"runA_frame": 1, "lo": 100, "hi": 100, "quality": "sure"},
            {"runA_frame": 2, "lo": 200, "hi": 200, "quality": "sure"},
        ]
    )
    mapping = _mapping(
        [
            {"runA_frame": 1, "runB_frame": 100, "confidence": 0.9, "status": "matched"},
            {"runA_frame": 2, "runB_frame": 200, "confidence": 0.9, "status": "matched"},
        ]
    )

    result = evaluate(mapping, anchors)

    assert result["median_error"] == 0.0
    assert result["hit_at_2"] == 1.0
    assert result["n_scored"] == 2
    assert result["per_anchor"]["correct"].all()
    # column names are prefixed so prediction vs ground truth can never be confused when this
    # is read back from a saved CSV
    assert list(result["per_anchor"].columns) == [
        "runA_frame",
        "pipeline_runB_frame",
        "gt_runB_best",
        "gt_lo",
        "gt_hi",
        "correct",
        "error_frames",
        "abstain_outcome",
        "pipeline_confidence",
        "pipeline_status",
        "gt_quality",
        "gt_note",
    ]
    assert (result["per_anchor"]["abstain_outcome"] == "correctly_answered").all()


def test_evaluate_computes_distance_to_nearest_bound_when_outside_range() -> None:
    anchors = _anchors([{"runA_frame": 1, "lo": 100, "hi": 110, "quality": "sure"}])
    mapping = _mapping(
        [{"runA_frame": 1, "runB_frame": 120, "confidence": 0.4, "status": "matched"}]
    )

    result = evaluate(mapping, anchors)

    assert result["median_error"] == 10.0  # |120 - 110|
    assert result["per_anchor"]["error_frames"].iloc[0] == 10.0
    assert result["per_anchor"]["correct"].iloc[0] is np.False_


def test_evaluate_counts_correct_abstention_as_true_positive_not_position_error() -> None:
    """Ground truth says no_match, model also abstained -- correct, and must not appear in
    position-error scoring (there is no position to be right or wrong about)."""
    anchors = _anchors([{"runA_frame": 1, "lo": None, "hi": None, "quality": "no_match"}])
    mapping = _mapping(
        [{"runA_frame": 1, "runB_frame": None, "confidence": None, "status": "no_match"}]
    )

    result = evaluate(mapping, anchors)

    assert result["abstain_true_positive"] == 1
    assert result["n_scored"] == 0


def test_evaluate_flags_a_guess_where_ground_truth_says_no_match() -> None:
    """Ground truth says no_match but the model predicted something anyway -- a false negative
    on abstention (it should have abstained and didn't)."""
    anchors = _anchors([{"runA_frame": 1, "lo": None, "hi": None, "quality": "no_match"}])
    mapping = _mapping(
        [{"runA_frame": 1, "runB_frame": 500, "confidence": 0.4, "status": "matched"}]
    )

    result = evaluate(mapping, anchors)

    assert result["abstain_false_negative"] == 1
    assert result["n_scored"] == 0  # not scored as a position error either


def test_evaluate_flags_a_missed_match_where_ground_truth_has_one() -> None:
    """Ground truth has a real match but the model abstained -- a false positive on abstention
    (it abstained when it shouldn't have)."""
    anchors = _anchors([{"runA_frame": 1, "lo": 100, "hi": 100, "quality": "sure"}])
    mapping = _mapping(
        [{"runA_frame": 1, "runB_frame": None, "confidence": None, "status": "no_match"}]
    )

    result = evaluate(mapping, anchors)

    assert result["abstain_false_positive"] == 1
    row = result["per_anchor"].iloc[0]
    assert row["abstain_outcome"] == "wrongly_abstained"
    assert pd.isna(row["correct"])  # not applicable, not False -- there is no position to score
    assert pd.isna(row["error_frames"])


def test_evaluate_per_anchor_includes_every_anchor_not_just_scored_ones() -> None:
    """Regression test: abstention-category anchors (no position to score) must still get a row
    in `per_anchor` -- a real gap the user caught (the first version silently dropped them, so
    `outputs/evaluation_anchors.csv` looked like anchors 2600/2650/183-style rows had vanished
    rather than been abstained on)."""
    anchors = _anchors(
        [
            {"runA_frame": 1, "lo": 100, "hi": 100, "quality": "sure"},  # scored
            {"runA_frame": 2, "lo": None, "hi": None, "quality": "no_match"},  # gt no_match
        ]
    )
    mapping = _mapping(
        [
            {"runA_frame": 1, "runB_frame": 100, "confidence": 0.9, "status": "matched"},
            {"runA_frame": 2, "runB_frame": None, "confidence": None, "status": "no_match"},
        ]
    )

    result = evaluate(mapping, anchors)

    assert len(result["per_anchor"]) == 2  # not just the 1 scored anchor
    per_anchor = result["per_anchor"]
    outcomes = dict(zip(per_anchor["runA_frame"], per_anchor["abstain_outcome"], strict=True))
    assert outcomes == {1: "correctly_answered", 2: "correctly_abstained"}


def test_evaluate_by_confidence_tier_breakdown() -> None:
    anchors = _anchors(
        [
            {"runA_frame": 1, "lo": 100, "hi": 100, "quality": "sure"},
            {"runA_frame": 2, "lo": 200, "hi": 200, "quality": "sure"},
            {"runA_frame": 3, "lo": 300, "hi": 300, "quality": "sure"},
        ]
    )
    mapping = _mapping(
        [
            {"runA_frame": 1, "runB_frame": 100, "confidence": 0.9, "status": "matched"},  # correct
            {"runA_frame": 2, "runB_frame": 200, "confidence": 0.9, "status": "matched"},  # correct
            {"runA_frame": 3, "runB_frame": 999, "confidence": 0.4, "status": "matched"},  # wrong
        ]
    )

    result = evaluate(mapping, anchors)

    assert result["by_confidence_tier"][0.9] == {"n": 2, "n_correct": 2, "accuracy": 1.0}
    assert result["by_confidence_tier"][0.4]["accuracy"] == 0.0


def test_evaluate_a_row_with_missing_lo_hi_does_not_poison_the_whole_batch() -> None:
    """Regression test: a real bug found on real data -- one anchor had `quality="unsure"` (not
    `"no_match"`) but empty `lo`/`hi` (a labelling data-entry gap). `np.median`/`np.percentile`
    propagate a single NaN to their entire output, so that one row silently turned
    `median_error`/`p90_error` into NaN across all anchors, not just itself."""
    anchors = _anchors(
        [
            {"runA_frame": 1, "lo": 100, "hi": 100, "quality": "sure"},
            {"runA_frame": 2, "lo": None, "hi": None, "quality": "unsure"},  # the data-entry gap
            {"runA_frame": 3, "lo": 300, "hi": 300, "quality": "sure"},
        ]
    )
    mapping = _mapping(
        [
            {"runA_frame": 1, "runB_frame": 100, "confidence": 0.9, "status": "matched"},
            {"runA_frame": 2, "runB_frame": 250, "confidence": 0.4, "status": "matched"},
            {"runA_frame": 3, "runB_frame": 300, "confidence": 0.9, "status": "matched"},
        ]
    )

    result = evaluate(mapping, anchors)

    assert result["n_scored"] == 2  # anchor 2 excluded, not scored
    assert result["median_error"] == 0.0  # not NaN despite anchor 2's missing range
    assert not np.isnan(result["p90_error"])


def test_evaluate_wilson_ci_is_a_valid_bounded_interval() -> None:
    anchors = _anchors(
        [{"runA_frame": i, "lo": i * 10, "hi": i * 10, "quality": "sure"} for i in range(6)]
    )
    mapping = _mapping(
        [
            {"runA_frame": i, "runB_frame": i * 10, "confidence": 0.9, "status": "matched"}
            for i in range(6)
        ]
    )

    result = evaluate(mapping, anchors)
    lo, hi = result["hit_at_5_wilson_ci"]

    assert 0.0 <= lo <= hi <= 1.0
