"""Ablation: early (descriptor-concat) vs. late (score-average) camera fusion.

Exploration only (CLAUDE.md's notebooks/ rule) -- nothing in outputs/ comes from this script.
Reads already-cached descriptors from data/cache/ (run `describe --feat seqslam` and
`describe --feat dinov2` first) and the final ground truth (gt/anchors_user.csv). Reproduces the
numbers reported in findings.md's "Step 6 -- ablation study" entry and decisions.md, and writes
them to notebooks/results/early_fusion_ablation.csv so the run is a persisted artifact, not just
terminal output -- see notebooks/results/README.md for how to read it.

Run: conda run -n phase2-vpr python notebooks/early_fusion_ablation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from routealign.align import dtw_open_ends
from routealign.groundtruth import load_labels
from routealign.similarity import cosine, local_contrast_norm
from routealign.verify import ridge_strength

RESULTS_PATH = Path(__file__).parent / "results" / "early_fusion_ablation.csv"
RIDGE_RESULTS_PATH = Path(__file__).parent / "results" / "ridge_only_gate.csv"

N_A = 2674
N_B_COMMON = 2615  # min(runB cam0=2622, cam5=2615) -- concatenation needs both cameras to exist
NO_MATCH_ANCHORS = (2600, 2650)  # gt quality="no_match", no lo/hi -- record what each
# configuration predicts here instead of a position error
RIDGE_WINDOW = 5  # matches config.yaml verify.ridge_window -- same window, for comparability
TAU = 0.75  # matches config.yaml confidence.no_match_tau_default -- reused as a single
# consistent yardstick across all 9 configurations, NOT re-calibrated per
# configuration (the deletion test that chose 0.75 was run once, against S_joint's
# own statistics); ridge_z is already a local z-score, which is what makes reusing
# one threshold across different similarity landscapes defensible rather than
# arbitrary, but it is still not each configuration's own optimal threshold


def _l2norm(x: np.ndarray) -> np.ndarray:
    """L2-normalise each row. Needed after concatenating two already-unit-norm descriptors --
    the concatenation has norm sqrt(2), not 1, and cosine similarity assumes unit vectors."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)


def _row_j(path: np.ndarray, n_a: int) -> np.ndarray:
    """Reduce a DTW path to one (last-visited) runB column per runA row -- same convention as
    verify._path_to_row_j."""
    i_vals, j_vals = path[:, 0], path[:, 1]
    edges = np.searchsorted(i_vals, np.arange(n_a + 1))
    return j_vals[edges[1:] - 1]


def _score(j_arr: np.ndarray, anchors, name: str) -> dict:
    """Compute median/p90/hit@5 error against the ground-truth anchors, plus what this
    configuration predicts at the two anchors ground truth has no position for at all.

    Every configuration here is a bare `dtw_open_ends` path with no threshold/confidence layer
    on top, so `pred_2600`/`pred_2650` are never "no match" -- they are always some runB column,
    right or wrong. Only the shipped pipeline (`fuse_confidence`, not reproduced in this script)
    can abstain; comparing against it is what these two columns are for.

    Returns:
        dict with `configuration`, `n`, `median_error`, `p90_error`, `hit_at_5`, `pred_2600`,
        `pred_2650` -- one row of the results table this script writes to `RESULTS_PATH`.
    """
    errs = []
    for _, row in anchors.iterrows():
        k = int(row["runA_frame"])
        if k >= len(j_arr):
            continue
        pred = j_arr[k]
        lo, hi = row["lo"], row["hi"]
        errs.append(0.0 if lo <= pred <= hi else min(abs(pred - lo), abs(pred - hi)))
    errs = np.array(errs)
    result = {
        "configuration": name,
        "n": len(errs),
        "median_error": float(np.median(errs)),
        "p90_error": float(np.percentile(errs, 90)),
        "hit_at_5": float(np.mean(errs <= 5)),
    }
    for k in NO_MATCH_ANCHORS:
        result[f"pred_{k}"] = int(j_arr[k]) if k < len(j_arr) else None
    print(
        f"{name:40s}: n={result['n']:2d}  median={result['median_error']:6.1f}  "
        f"p90={result['p90_error']:7.1f}  hit@5={result['hit_at_5']:.2f}  "
        f"pred@2600={result['pred_2600']}  pred@2650={result['pred_2650']}"
    )
    return result


def _load_desc(feat: str, run: str, cam: str, n_full: int) -> np.ndarray:
    tag = "w64_h40_p8" if feat == "seqslam" else "dinov2_vits14"
    return np.load(f"data/cache/desc_{feat}_{run}_{cam}_n{n_full}_{tag}.npy")


def early_fusion_path(feat: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate cam0+cam5 descriptors per frame (early fusion), then run the ordinary
    cosine -> local_contrast_norm -> dtw_open_ends pipeline unchanged.

    Returns:
        `(path, s)` -- the DTW path and the z-scored similarity matrix it was computed from
        (needed for `ridge_strength`, which otherwise only this function ever sees).
    """
    a0 = _load_desc(feat, "runA", "cam0", 2674)
    a5 = _load_desc(feat, "runA", "cam5", 2674)
    b0 = _load_desc(feat, "runB", "cam0", 2622)[:N_B_COMMON]
    b5 = _load_desc(feat, "runB", "cam5", 2615)[:N_B_COMMON]

    concat_a = _l2norm(np.concatenate([a0, a5], axis=1))
    concat_b = _l2norm(np.concatenate([b0, b5], axis=1))
    s = local_contrast_norm(cosine(concat_a, concat_b), window=window)
    cost = 3.0 - np.clip(s, -3.0, 3.0)
    return dtw_open_ends(cost, lam=0.5), s


def full_joint_early_fusion_path(window: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ALL FOUR sources (cam0/cam5 x SeqSLAM/DINOv2) into one vector per frame.

    Caveat (see decisions.md): SeqSLAM contributes 2560+2560=5120 of the combined 6656
    dimensions (77%) -- a naive concatenation lets it dominate purely by dimension count, not
    demonstrated informativeness. `window` here is an untested compromise between SeqSLAM's
    tuned 50 and DINOv2's tuned 500 -- unlike those two, this value was never swept/validated.

    Returns:
        `(path, s)`, see `early_fusion_path`.
    """
    seq_a0 = _load_desc("seqslam", "runA", "cam0", 2674)
    seq_a5 = _load_desc("seqslam", "runA", "cam5", 2674)
    seq_b0 = _load_desc("seqslam", "runB", "cam0", 2622)[:N_B_COMMON]
    seq_b5 = _load_desc("seqslam", "runB", "cam5", 2615)[:N_B_COMMON]
    dino_a0 = _load_desc("dinov2", "runA", "cam0", 2674)
    dino_a5 = _load_desc("dinov2", "runA", "cam5", 2674)
    dino_b0 = _load_desc("dinov2", "runB", "cam0", 2622)[:N_B_COMMON]
    dino_b5 = _load_desc("dinov2", "runB", "cam5", 2615)[:N_B_COMMON]

    concat_a = _l2norm(np.concatenate([seq_a0, seq_a5, dino_a0, dino_a5], axis=1))
    concat_b = _l2norm(np.concatenate([seq_b0, seq_b5, dino_b0, dino_b5], axis=1))
    s = local_contrast_norm(cosine(concat_a, concat_b), window=window)
    cost = 3.0 - np.clip(s, -3.0, 3.0)
    return dtw_open_ends(cost, lam=0.5), s


def _ridge_gate(s: np.ndarray, path: np.ndarray, anchors, name: str) -> dict:
    """Apply a `ridge_z`-only no-match gate (no `cam_disagree`/`desc_disagree` -- those need a
    second, independently-computed path to compare against, which most of these configurations
    don't have) to one configuration, at `TAU`.

    Reports the gate's cost (how often it would incorrectly abstain on one of the 20 anchors
    that do have a real position) against its benefit (whether it catches the 2 anchors that
    genuinely have no position at all) -- the trade-off a from-scratch confidence system for
    each configuration would have to justify.

    Returns:
        dict with `configuration`, `false_abstain_n`, `false_abstain_rate` (of the 20 scored
        anchors), `ridge_2600`, `caught_2600`, `ridge_2650`, `caught_2650`.
    """
    ridge = ridge_strength(s, path, window=RIDGE_WINDOW)
    false_abstains = sum(1 for _, row in anchors.iterrows() if ridge[int(row["runA_frame"])] < TAU)
    n = len(anchors)
    ridge_at = {k: float(ridge[k]) for k in NO_MATCH_ANCHORS}
    result = {
        "configuration": name,
        "false_abstain_n": false_abstains,
        "false_abstain_rate": false_abstains / n,
        "ridge_2600": ridge_at[2600],
        "caught_2600": bool(ridge_at[2600] < TAU),
        "ridge_2650": ridge_at[2650],
        "caught_2650": bool(ridge_at[2650] < TAU),
    }
    print(
        f"{name:40s}: false_abstain={false_abstains:2d}/{n} ({result['false_abstain_rate']:.2f})"
        f"  ridge@2600={ridge_at[2600]:6.2f} (caught={result['caught_2600']})"
        f"  ridge@2650={ridge_at[2650]:6.2f} (caught={result['caught_2650']})"
    )
    return result


def main() -> None:
    labels = load_labels("gt")
    anchors = labels[labels["labeller"] == "user"].copy()
    anchors = anchors[
        (anchors["quality"] != "no_match") & anchors["lo"].notna() & anchors["hi"].notna()
    ]

    seqslam_early_path, seqslam_early_s = early_fusion_path("seqslam", window=50)
    dinov2_early_path, dinov2_early_s = early_fusion_path("dinov2", window=500)
    full_joint_path, full_joint_s = full_joint_early_fusion_path(window=200)
    s_joint = np.load("data/cache/S_joint.npy")
    argmax_j = np.argmax(s_joint, axis=1)
    argmax_path = np.column_stack([np.arange(N_A), argmax_j])

    # (name, similarity matrix, path) for every configuration -- one list so `_score` and
    # `_ridge_gate` run over exactly the same 9 configurations, never drifting out of sync.
    configs = [
        ("SeqSLAM early-fusion (concat)", seqslam_early_s, seqslam_early_path),
        ("DINOv2 early-fusion (concat)", dinov2_early_s, dinov2_early_path),
        ("Full joint early-fusion", full_joint_s, full_joint_path),
        (
            "SeqSLAM late-fusion (current)",
            np.load("data/cache/S_seqslam.npy"),
            np.load("data/cache/path_seqslam.npy"),
        ),
        (
            "DINOv2 late-fusion (current)",
            np.load("data/cache/S_dinov2.npy"),
            np.load("data/cache/path_dinov2.npy"),
        ),
        (
            "cam0-only (both desc, late-fused)",
            np.load("data/cache/S_cam0.npy"),
            np.load("data/cache/path_cam0.npy"),
        ),
        (
            "cam5-only (both desc, late-fused)",
            np.load("data/cache/S_cam5.npy"),
            np.load("data/cache/path_cam5.npy"),
        ),
        ("Joint late-fusion -- SHIPPED", s_joint, np.load("data/cache/path_joint.npy")),
        ("argmax-only on S_joint (no DTW)", s_joint, argmax_path),
    ]

    print("=== position error + no-match-anchor prediction (n=20 of 22) ===")
    score_results = [_score(_row_j(path, N_A), anchors, name) for name, _, path in configs]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_results).to_csv(RESULTS_PATH, index=False)
    print(f"wrote {len(score_results)} rows -> {RESULTS_PATH}")

    print()
    print(f"=== ridge_z-only no-match gate (tau={TAU}, window={RIDGE_WINDOW}, no disagreement) ===")
    ridge_results = [_ridge_gate(s, path, anchors, name) for name, s, path in configs]
    RIDGE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ridge_results).to_csv(RIDGE_RESULTS_PATH, index=False)
    print(f"wrote {len(ridge_results)} rows -> {RIDGE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
