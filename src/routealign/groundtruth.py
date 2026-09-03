"""Model-blind labelling sheets, label loading/merge, evaluation, the deletion test.

Labelling is user-primary; Claude's pass afterward is a QA/consistency check, not an independent
co-labeller -- the two share a vision backbone with the pipeline's own DINOv2 descriptor and are
not statistically independent judges. `evaluate` must report user-vs-Claude agreement as a
sanity number, never as inter-rater reliability or a label-noise floor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import align, verify
from .frames import FrameStore

log = logging.getLogger(__name__)


def stratum_s_indices(
    seed: int,
    lo: int = 200,
    hi: int = 2500,
    n_sweep: int = 18,
    jitter: int = 20,
    fixed: tuple[int, ...] = (60, 150, 2600, 2650),
) -> NDArray[np.int64]:
    """Pick stratum S's 22 runA anchor indices: an evenly-spaced, seeded-jitter sweep plus a few
    fixed positions of known interest (the plateau at 60/150, the loop-closure tail at
    2600/2650).

    Args:
        seed: RNG seed -- determinism (CLAUDE.md §4): the same seed always gives the same anchors.
        lo (optional): Start of the sweep range. Defaults to `200`.
        hi (optional): End of the sweep range (inclusive). Defaults to `2500`.
        n_sweep (optional): Number of evenly-spaced sweep points before jitter. Defaults to `18`.
        jitter (optional): Max +/- random offset applied to each sweep point. Defaults to `20`.
        fixed (optional): Extra fixed positions always included (plateau frames 60, 150; tail
            frames 2600, 2650). Defaults to `(60, 150, 2600, 2650)`.

    Returns:
        int64[n_sweep + len(fixed)] sorted, unique runA frame indices (22 for the defaults --
        `config.yaml` `groundtruth.stratum_s`).
    """
    rng = np.random.default_rng(seed)
    sweep = np.linspace(lo, hi, n_sweep, dtype=np.int64)
    sweep = sweep + rng.integers(-jitter, jitter + 1, size=n_sweep)
    all_idx = np.concatenate([sweep, np.array(fixed, dtype=np.int64)])
    return np.unique(all_idx)


def _label_tile(
    img0: NDArray[np.uint8], img5: NDArray[np.uint8] | None, text: str
) -> NDArray[np.uint8]:
    """Combine one runB (or runA) frame's cam0 + cam5 views into one labelled tile.

    Args:
        img0: uint8[H, W, 3] RGB cam0 frame.
        img5: uint8[H, W, 3] RGB cam5 frame, or `None` if this index has no cam5 frame (e.g.
            beyond cam5's own decoded count).
        text: Caption burned into the top-left corner, e.g. `"runB 1840"`.

    Returns:
        uint8[H, 2*W + 4, 3] side-by-side tile (cam0 | thin separator | cam5), BGR (ready for
        `cv2.imwrite`), with `text` burned in.
    """
    if img5 is None:
        img5 = np.zeros_like(img0)
        cv2.putText(img5, "no frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    sep = np.full((img0.shape[0], 4, 3), 255, dtype=np.uint8)
    tile = np.concatenate([img0, sep, img5], axis=1)
    tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
    cv2.putText(tile_bgr, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return tile_bgr


def make_sheets(
    store_a0: FrameStore,
    store_a5: FrameStore,
    store_b0: FrameStore,
    store_b5: FrameStore,
    anchor_indices: NDArray[np.int64],
    out_dir: str | Path,
    coarse_step: int = 20,
    fine_radius: int = 20,
    coarse_picks: dict[int, int] | None = None,
) -> None:
    """Build model-blind labelling tiles, built only from the JPEG cache -- never from model
    output (blind by construction).

    Always writes: one reference tile per anchor (`anchor_refs/{runA_frame:05d}.jpg`, both runA
    cameras side by side, so the labeller knows what they are looking for) and one shared coarse
    overview covering the whole of runB (`coarse/{j:05d}.jpg`, every `coarse_step`-th frame, both
    runB cameras side by side) -- the same coarse overview is reused for every anchor, since it
    does not depend on which anchor is being labelled.

    If `coarse_picks` is given (`{runA_frame: coarse runB estimate}`, from a labeller's first
    pass over the coarse overview), also writes a fine sheet per anchor with a pick
    (`fine_{runA_frame:05d}/{j:05d}.jpg`, every runB frame in `[pick - fine_radius, pick +
    fine_radius]`) for precise frame-level labelling.

    Args:
        store_a0: Frame source for runA cam0.
        store_a5: Frame source for runA cam5.
        store_b0: Frame source for runB cam0.
        store_b5: Frame source for runB cam5.
        anchor_indices: int64[N] runA frame indices to build sheets for.
        out_dir: Directory to write the labelling tiles into.
        coarse_step (optional): Frame stride for the coarse overview. Defaults to `20`.
        fine_radius (optional): +/- frames around each coarse pick for its fine sheet. Defaults
            to `20`.
        coarse_picks (optional): `{runA_frame: coarse runB frame}` to build fine sheets for.
            Defaults to `None` (coarse overview + anchor refs only, no fine sheets yet).
    """
    out_dir = Path(out_dir)

    ref_dir = out_dir / "anchor_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for k in anchor_indices:
        k = int(k)
        img0 = store_a0.get_rgb(k)
        img5 = store_a5.get_rgb(k) if k < store_a5.n_frames else None
        tile = _label_tile(img0, img5, f"runA {k}")
        cv2.imwrite(str(ref_dir / f"{k:05d}.jpg"), tile)
    log.info("wrote %d anchor reference tiles -> %s", len(anchor_indices), ref_dir)

    coarse_dir = out_dir / "coarse"
    coarse_dir.mkdir(parents=True, exist_ok=True)
    coarse_idx = range(0, store_b0.n_frames, coarse_step)
    for j in coarse_idx:
        img0 = store_b0.get_rgb(j)
        img5 = store_b5.get_rgb(j) if j < store_b5.n_frames else None
        tile = _label_tile(img0, img5, f"runB {j}")
        cv2.imwrite(str(coarse_dir / f"{j:05d}.jpg"), tile)
    log.info("wrote %d coarse overview tiles -> %s", len(list(coarse_idx)), coarse_dir)

    if not coarse_picks:
        return
    for runa_frame, pick in coarse_picks.items():
        fine_dir = out_dir / f"fine_{runa_frame:05d}"
        fine_dir.mkdir(parents=True, exist_ok=True)
        lo = max(0, pick - fine_radius)
        hi = min(store_b0.n_frames - 1, pick + fine_radius)
        for j in range(lo, hi + 1):
            img0 = store_b0.get_rgb(j)
            img5 = store_b5.get_rgb(j) if j < store_b5.n_frames else None
            tile = _label_tile(img0, img5, f"runB {j}")
            cv2.imwrite(str(fine_dir / f"{j:05d}.jpg"), tile)
        log.info("wrote fine sheet for runA %d (%d-%d) -> %s", runa_frame, lo, hi, fine_dir)


def load_labels(gt_dir: str | Path) -> pd.DataFrame:
    """Load `gt/anchors_user.csv` and `gt/anchors_claude.csv`.

    Loads whichever of the two files exist -- only `anchors_user.csv` exists so far, from the
    user's own frame-by-frame verification pass; a separate Claude QA pass was never split out
    into its own `anchors_claude.csv` file.

    Args:
        gt_dir: Directory containing the label CSVs.

    Raises:
        FileNotFoundError: If neither label file exists.

    Returns:
        One row per (labeller, anchor), with columns `labeller`, `runA_frame`, `runB_best`,
        `lo`, `hi`, `quality`, `note`.
    """
    gt_dir = Path(gt_dir)
    frames = []
    for labeller, filename in (("user", "anchors_user.csv"), ("claude", "anchors_claude.csv")):
        path = gt_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "labeller", labeller)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"{gt_dir}: no anchors_user.csv or anchors_claude.csv found")
    return pd.concat(frames, ignore_index=True)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion -- safer than a normal-approximation CI
    at the small sample sizes (fewer than 25 anchors) this project's anchor set has.

    Args:
        k: Number of successes.
        n: Number of trials.
        z (optional): Confidence-level z-score. Defaults to `1.96` (95%).

    Returns:
        `(lo, hi)` bounds in `[0, 1]`, or `(nan, nan)` if `n == 0`.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def evaluate(mapping: pd.DataFrame, anchors: pd.DataFrame) -> dict:
    """Compute error, hit@k and Wilson-95%-CI metrics against merged ground-truth anchors.

    Position error (`error = 0` if the prediction lies in `[lo, hi]`, else distance to the
    nearest end) is scored only over anchors where *both* sides have a real
    answer: ground truth `quality != "no_match"` and the pipeline actually predicted a
    `runB_frame` (didn't abstain). Anchors where one side has an answer and the other doesn't
    are counted separately, as abstention (dis)agreement, not folded into the position-error
    numbers where they'd be meaningless.

    Args:
        mapping: `outputs/mapping.csv` contents (or `mapping_full.csv` -- only `runA_frame`,
            `runB_frame`, `confidence`, `status` are used).
        anchors: Ground-truth anchors, one row per `runA_frame`, with columns `runB_best`, `lo`,
            `hi`, `quality` (e.g. from `load_labels`, already reduced to one row per anchor).

    Returns:
        dict with `n_scored`, `n_total_anchors`, `median_error`, `p90_error`, `hit_at_2/5/10`,
        `hit_at_5_wilson_ci`, `abstain_true_positive/false_negative/false_positive`,
        `by_confidence_tier` (`{tier: {n, n_correct, accuracy}}`), and `per_anchor` -- prediction
        vs. ground truth side by side, columns prefixed by which side they come from so the two
        are never confused (this is meant to be saved as a real, inspectable artifact, not just
        aggregated away). **Every** anchor gets a row, not only the `n_scored` ones -- `runA_frame`,
        `pipeline_runB_frame` (the prediction, empty if the pipeline abstained),
        `gt_runB_best`/`gt_lo`/`gt_hi`/`gt_quality`/`gt_note` (the labeller's answer and its own
        context, empty if ground truth is `no_match`), `error_frames` (0 if
        `pipeline_runB_frame` falls in `[gt_lo, gt_hi]`, else distance to the nearest end;
        empty for the anchors position error doesn't apply to), `correct` (nullable bool,
        `error_frames == 0` -- the direct right/wrong verdict, `<NA>` where not applicable),
        `abstain_outcome` (`correctly_answered` -- the usual `n_scored` case, ground truth has a
        match and the pipeline gave one; `correctly_abstained`; `wrongly_guessed` -- ground truth
        `no_match` but the pipeline answered anyway; `wrongly_abstained` -- ground truth has a
        real match but the pipeline abstained, the abstention rule's false-positive case).
        `pipeline_status`/`pipeline_confidence` describe whether the pipeline *abstained*, which
        is a different question from whether it was *correct* -- never read
        `pipeline_status == "matched"` as "this row was right", and never read a `no_match` row
        as simply missing from the table.
    """
    merged = anchors.merge(mapping, on="runA_frame", how="left")

    has_gt_match = merged["quality"] != "no_match"
    has_prediction = merged["runB_frame"].notna()
    # a row can claim quality != "no_match" yet still be missing lo/hi (a labelling data-entry
    # gap, not a real "no match" -- e.g. a caveated guess that was never filled in as numbers).
    # Such a row cannot be scored for position error either way, and must not be allowed to
    # poison the whole batch: np.median/np.percentile propagate a single NaN to every output.
    has_gt_range = merged["lo"].notna() & merged["hi"].notna()
    is_scored = has_gt_match & has_gt_range & has_prediction

    scored = merged[is_scored].copy()
    pred = scored["runB_frame"].to_numpy(dtype=np.float64)
    lo = scored["lo"].to_numpy(dtype=np.float64)
    hi = scored["hi"].to_numpy(dtype=np.float64)
    inside = (pred >= lo) & (pred <= hi)
    error = np.where(inside, 0.0, np.minimum(np.abs(pred - lo), np.abs(pred - hi)))
    scored["error"] = error

    hit_at = {k: float(np.mean(error <= k)) for k in (2, 5, 10)} if error.size else {}
    n_hit5 = int(np.sum(error <= 5)) if error.size else 0
    ci_hit5 = _wilson_ci(n_hit5, error.size)

    # abstention (dis)agreement: does the pipeline correctly know when it doesn't know? Every
    # anchor falls into exactly one of these 4 buckets (a full confusion matrix on "should this
    # row have a position answer at all"), independent of whether it happens to also be missing
    # lo/hi -- that's a separate, data-quality axis, not folded into this one.
    should_abstain = ~has_gt_match
    did_abstain = ~has_prediction
    abstain_tp = should_abstain & did_abstain  # correctly abstained
    abstain_fn = should_abstain & ~did_abstain  # gt no_match, pipeline guessed anyway
    abstain_fp = ~should_abstain & did_abstain  # gt has a match, pipeline abstained
    abstain_tn = ~should_abstain & ~did_abstain  # gt has a match, pipeline answered (may or may
    #                                               not end up in `is_scored`, if lo/hi is missing)

    by_tier: dict[float, dict[str, float]] = {}
    for tier, group in scored.groupby("confidence"):
        n = len(group)
        n_correct = int((group["error"] == 0.0).sum())
        by_tier[float(tier)] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n if n else float("nan"),
        }

    # full per-anchor table: EVERY anchor (all 22), not just the `is_scored` subset -- the 3
    # abstention-category rows are exactly as informative as the 19 scored ones (arguably more
    # so: they are where the no-match rule's own precision/recall lives) and must not silently
    # disappear from a saved artifact just because position error doesn't apply to them.
    full = merged.copy()
    full["error"] = np.nan
    full.loc[scored.index, "error"] = error
    full["correct"] = pd.array([pd.NA] * len(full), dtype="boolean")
    full.loc[scored.index, "correct"] = full.loc[scored.index, "error"] == 0.0
    full["abstain_outcome"] = np.select(
        [abstain_tp, abstain_fn, abstain_fp, abstain_tn],
        ["correctly_abstained", "wrongly_guessed", "wrongly_abstained", "correctly_answered"],
        default="unreachable",  # the 4 conditions are exhaustive (should_abstain x did_abstain);
    )  # np.select still requires a same-dtype default -- this should never actually appear

    return {
        "n_scored": int(scored.shape[0]),
        "n_total_anchors": int(merged.shape[0]),
        "median_error": float(np.median(error)) if error.size else float("nan"),
        "p90_error": float(np.percentile(error, 90)) if error.size else float("nan"),
        "hit_at_2": hit_at.get(2, float("nan")),
        "hit_at_5": hit_at.get(5, float("nan")),
        "hit_at_10": hit_at.get(10, float("nan")),
        "hit_at_5_wilson_ci": ci_hit5,
        "abstain_true_positive": int(abstain_tp.sum()),
        "abstain_false_negative": int(abstain_fn.sum()),
        "abstain_false_positive": int(abstain_fp.sum()),
        "by_confidence_tier": by_tier,
        "per_anchor": full[
            [
                "runA_frame",
                "runB_frame",
                "runB_best",
                "lo",
                "hi",
                "correct",
                "error",
                "abstain_outcome",
                "confidence",
                "status",
                "quality",
                "note",
            ]
        ].rename(
            columns={
                "runB_frame": "pipeline_runB_frame",
                "runB_best": "gt_runB_best",
                "lo": "gt_lo",
                "hi": "gt_hi",
                "error": "error_frames",
                "confidence": "pipeline_confidence",
                "status": "pipeline_status",
                "quality": "gt_quality",
                "note": "gt_note",
            }
        ),
    }


def deletion_test(
    similarity: NDArray[np.float64],
    original_runb_frame: NDArray[np.int64],
    runb_delete_range: tuple[int, int],
    lam: float,
    ridge_window: int,
    tau: float,
) -> dict:
    """Delete a known runB frame span, rerun DTW, and check that ridge-based abstention fires on
    the runA rows whose original match fell inside it. Needs no manual labels -- a
    self-supervised check on the no-match rule's most fundamental trigger (`ridge_z < tau`;
    `cam_disagree`/`desc_disagree` need a second camera/descriptor path each and are not part of
    this check).

    Deleted columns are set to the z-score clip floor (`-3.0`), the same "no evidence" value
    `_cmd_align` already uses for a camera's missing columns -- not removed and re-indexed, so
    every row index still means the same runA row before and after.

    Args:
        similarity: float64[NA, NB] the joint similarity matrix from the real (undeleted) run.
        original_runb_frame: int64[NA] `runB_frame` per runA row from the undeleted mapping --
            defines which rows are "affected" (their original match falls inside the deleted span).
        runb_delete_range: `(start, end)` runB frame indices to delete (exclusive end).
        lam: DTW non-diagonal penalty; must match the original run's, so any change in behaviour
            comes from the deletion, not a different alignment parameter.
        ridge_window: `ridge_strength` window; must match the original run's.
        tau: `ridge_z` threshold below which a row abstains (`confidence.fuse_confidence`'s
            `tau`).

    Returns:
        dict with `n_affected`, `tp`, `fp`, `fn`, `tn`, `precision`, `recall` (`nan` if the
        relevant denominator is 0 -- e.g. no rows were affected).
    """
    start, end = runb_delete_range
    masked = similarity.copy()
    masked[:, start:end] = -3.0

    cost = 3.0 - np.clip(masked, -3.0, 3.0)
    path = align.dtw_open_ends(cost, lam=lam)
    ridge_z = verify.ridge_strength(masked, path, window=ridge_window)
    abstained = ridge_z < tau

    affected = (original_runb_frame >= start) & (original_runb_frame < end)

    tp = int(np.sum(affected & abstained))
    fn = int(np.sum(affected & ~abstained))
    fp = int(np.sum(~affected & abstained))
    tn = int(np.sum(~affected & ~abstained))

    return {
        "n_affected": int(affected.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
    }
