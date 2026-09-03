"""Experiment: full mapping + evaluation for the all-4 early-fusion configuration.

Builds the same artifacts as the shipped `verify`/`evaluate` stages -- a public-format mapping
CSV and a per-anchor evaluation table -- but from the all-4-sources early-fusion similarity
(`full_joint_early_fusion_path`), with a `ridge_z`-only confidence: early fusion has no second,
independently-computed path, so `cam_disagree`/`desc_disagree` cannot exist and their columns are
dropped from the output. The confidence itself is the production `confidence.fuse_confidence`
run with both disagreement cues set to 0 -- "cue absent, never votes against" -- under which its
rules reduce exactly to ridge-only tiers (0.9 = ridge >= 1.5, 0.7 = ridge >= 1.0, no-match =
ridge < tau or clamped-weak), rather than a re-implementation that could drift from production.

Exploration only (CLAUDE.md's notebooks/ rule): writes to notebooks/results/, never outputs/ --
outputs/mapping.csv is the shipped deliverable and is not touched.

Run from the repo root: conda run -n phase2-vpr python notebooks/early_fusion_pipeline.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from early_fusion_ablation import full_joint_early_fusion_path  # noqa: E402

from routealign import align, confidence, frames, timestamps, verify  # noqa: E402
from routealign.config import Config, load_config  # noqa: E402
from routealign.frames import FrameStore  # noqa: E402
from routealign.groundtruth import evaluate, load_labels  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
METHOD = "all4-early-fusion+ridge-only"
EARLY_FUSION_WINDOW = 200  # same untested-compromise window as the ablation row it mirrors

# outputs/mapping.csv's public column list, minus cam_disagree/desc_disagree (not computable
# under early fusion -- there is no second path to disagree with)
_FINAL_COLS = [
    "runA_frame",
    "runB_frame",
    "confidence",
    "lo",
    "hi",
    "status",
    "reason",
    "runA_time_utc",
    "runB_time_utc",
    "ridge_z",
    "margin_z",
    "sift_inliers",
    "slope",
    "runA_has_cam0",
    "runA_has_cam5",
    "runB_has_cam0",
    "runB_has_cam5",
    "method",
]


def _iso_utc(ns: int) -> str:
    """Format an int64-nanosecond epoch timestamp as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat()


def _store(cfg: Config, run: str, cam: str) -> FrameStore:
    """Open a `FrameStore` for one run/camera, same crop setting as the CLI."""
    return FrameStore(
        cfg.path("cache_dir") / "frames", run, cam, crop_row_frac=cfg.raw["frames"]["crop_row_frac"]
    )


def _stationary_b(cfg: Config) -> list[tuple[int, int]]:
    """runB stationary segments as inclusive (lo, hi) pairs -- same code path as `align`."""
    store_b0, store_b5 = _store(cfg, "runB", "cam0"), _store(cfg, "runB", "cam5")
    w, h = cfg.raw["frames"]["motion_thumb_w"], cfg.raw["frames"]["motion_thumb_h"]
    motion_b0 = frames.motion_energy(store_b0, w, h)
    motion_b5 = frames.motion_energy(store_b5, w, h)
    n_common = min(motion_b0.size, motion_b5.size)
    exclusive = frames.stationary_segments(
        motion_b0[:n_common],
        motion_b5[:n_common],
        cfg.raw["frames"]["stationary_threshold_frac"],
        cfg.raw["frames"]["stationary_min_length"],
    )
    return [(s, e - 1) for s, e in exclusive]


def main() -> None:
    cfg = load_config("config.yaml")

    print("computing all-4 early-fusion similarity + DTW path ...")
    path, s = full_joint_early_fusion_path(window=EARLY_FUSION_WINDOW)
    n_a = s.shape[0]

    mapping = align.path_to_mapping(path, s, _stationary_b(cfg))

    # same cue computations as `verify`, on this configuration's own s/path
    mapping["ridge_z"] = verify.ridge_strength(s, path, window=cfg.raw["verify"]["ridge_window"])
    mapping["margin_z"] = verify.margin(s, path, band=cfg.raw["verify"]["margin_band"])
    slope_per_point = align.local_slope(path, window=cfg.raw["align"]["local_slope_window"])
    row_edges = np.searchsorted(path[:, 0], np.arange(n_a + 1))
    mapping["slope"] = slope_per_point[row_edges[1:] - 1]
    mapping["sift_inliers"] = np.nan
    mapping["ambiguity"] = mapping["hi"] - mapping["lo"]
    mapping["ambiguous_range"] = mapping["ambiguity"] > 0
    mapping["no_frame"] = False
    # cue absent -> never votes against: with both at 0, fuse_confidence's mutual_disagree can
    # never fire and its tier conditions on disagreement are always satisfied, reducing the
    # production rules exactly to ridge-only tiers. Dropped from the output columns below.
    mapping["cam_disagree"] = 0.0
    mapping["desc_disagree"] = 0.0

    fused = confidence.fuse_confidence(mapping, tau=cfg.raw["confidence"]["no_match_tau_default"])

    store_a0, store_a5 = _store(cfg, "runA", "cam0"), _store(cfg, "runA", "cam5")
    store_b0, store_b5 = _store(cfg, "runB", "cam0"), _store(cfg, "runB", "cam5")
    fused["runA_has_cam0"] = fused["runA_frame"] < store_a0.n_frames
    fused["runA_has_cam5"] = fused["runA_frame"] < store_a5.n_frames
    fused["runB_has_cam0"] = fused["runB_frame"] < store_b0.n_frames
    fused["runB_has_cam5"] = fused["runB_frame"] < store_b5.n_frames
    fused["method"] = METHOD

    ts_a = timestamps.load_timestamps(
        cfg.path("dataset_dir") / "runA" / "cam0_20_yuv420p_output.hevc.timestamps.txt"
    )
    ts_b = timestamps.load_timestamps(
        cfg.path("dataset_dir") / "runB" / "cam0_20_yuv420p_output.hevc.timestamps.txt"
    )
    fused["runA_time_utc"] = [_iso_utc(int(ts_a[k])) for k in fused["runA_frame"]]
    fused["runB_time_utc"] = [
        _iso_utc(int(ts_b[j])) if j < ts_b.size else None for j in fused["runB_frame"]
    ]

    # H0 tail rows, so the variant covers all 2695 timestamp lines like the shipped mapping
    full_decoded_a = store_a0.n_frames
    if n_a == full_decoded_a:
        tail_k = np.arange(full_decoded_a, ts_a.size)
        if tail_k.size:
            tail = pd.DataFrame(
                {
                    "runA_frame": tail_k,
                    "runB_frame": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                    "boundary_clamped": False,
                    "confidence": np.nan,
                    "status": "no_frame",
                    "reason": "no decoded frame for this row",
                    "runA_time_utc": [_iso_utc(int(ts_a[k])) for k in tail_k],
                    "runB_time_utc": None,
                    "ridge_z": np.nan,
                    "margin_z": np.nan,
                    "sift_inliers": np.nan,
                    "slope": np.nan,
                    "runA_has_cam0": False,
                    "runA_has_cam5": False,
                    "runB_has_cam0": False,
                    "runB_has_cam5": False,
                    "method": METHOD,
                }
            )
            fused = pd.concat([fused, tail], ignore_index=True)

    # public format: same renames and no-match blanking as outputs/mapping.csv
    final = fused[_FINAL_COLS].rename(columns={"lo": "runB_frame_lo", "hi": "runB_frame_hi"})
    unmatched = final["status"].isin(["no_match", "no_frame"])
    final.loc[unmatched, ["runB_frame", "runB_frame_lo", "runB_frame_hi", "runB_time_utc"]] = np.nan

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = RESULTS_DIR / "mapping_early_fusion.csv"
    final.to_csv(mapping_path, index=False)
    for status in ("matched", "ambiguous_range", "no_match", "no_frame"):
        print(f"  {status}: {int((final['status'] == status).sum())}")
    print(f"confidence tiers: {final['confidence'].value_counts().sort_index().to_dict()}")
    print(f"wrote {mapping_path} ({len(final)} rows)")

    labels = load_labels("gt")
    anchors = labels[labels["labeller"] == "user"].copy()
    result = evaluate(final, anchors)

    anchors_path = RESULTS_DIR / "evaluation_anchors_early_fusion.csv"
    result["per_anchor"].to_csv(anchors_path, index=False)
    summary = {k: v for k, v in result.items() if k != "per_anchor"}
    summary_path = RESULTS_DIR / "evaluation_summary_early_fusion.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    lo_ci, hi_ci = result["hit_at_5_wilson_ci"]
    print(f"\nn_scored={result['n_scored']}/{result['n_total_anchors']}")
    print(f"median_error={result['median_error']:.1f}  p90_error={result['p90_error']:.1f}")
    print(
        f"hit@2={result['hit_at_2']:.2f}  hit@5={result['hit_at_5']:.2f}  "
        f"hit@10={result['hit_at_10']:.2f}  (hit@5 95% CI: [{lo_ci:.2f}, {hi_ci:.2f}])"
    )
    print(
        f"abstain: true_positive={result['abstain_true_positive']} "
        f"false_negative={result['abstain_false_negative']} "
        f"false_positive={result['abstain_false_positive']}"
    )
    print("by_confidence_tier:")
    for tier, stats in sorted(result["by_confidence_tier"].items()):
        print(
            f"  {tier}: n={stats['n']}  correct={stats['n_correct']}  acc={stats['accuracy']:.2f}"
        )
    print(f"wrote {anchors_path} and {summary_path}")


if __name__ == "__main__":
    main()
