# outputs/ schema

Generated files (git-ignored — regenerate with `python -m routealign align` then `... verify`).
Enforced by `tests/test_mapping_csv.py`.

## `mapping.csv` — the deliverable (2695 rows: one per runA timestamp line)

| column | type | meaning |
|---|---|---|
| `runA_frame` | int | runA row index, `0..2694`, every value exactly once, in order |
| `runB_frame` | int or empty | Best-match runB frame index. **Empty iff `status ∈ {no_match, no_video}`** — an abstained row never reports a value it has already said it doesn't trust (this is the no-match rule's actual output contract, plan.md §7). Monotone non-decreasing across rows where it is present. |
| `confidence` | float in `[0,1]` or empty | Ordinal tier from `confidence.fuse_confidence` (`{0.9, 0.7, 0.5, 0.4}`) — a ranking on held-out anchors, *not* a calibrated probability (plan.md §13). Empty iff `runB_frame` is empty. |
| `runB_frame_lo`, `runB_frame_hi` | int or empty | Candidate range `runB_frame` was chosen from, widened to cover a runB stationary segment if the match falls inside one (`align.path_to_mapping`). `lo <= runB_frame <= hi` whenever present. Empty alongside `runB_frame`. |
| `status` | `{matched, ambiguous_range, no_match, no_video}` | `matched`: normal. `ambiguous_range`: `hi > lo` (inside a plateau/stop — real match, but not frame-exact). `no_match`: abstained (see `reason`). `no_video`: this runA timestamp line has no decoded frame at all (H0 tail, `runA_frame >= 2674` in this dataset). |
| `reason` | string or empty | Which no-match trigger fired, for `no_match`/`no_video` rows only; empty otherwise. One of: `"no video for this row"`, `"ridge_z below tau (...)"`, `"boundary_clamped with ridge_z below ..."`, `"cam_disagree and desc_disagree both above ..."`. |
| `runA_time_utc`, `runB_time_utc` | ISO-8601 UTC string | From each run's own timestamp file. `runB_time_utc` empty alongside `runB_frame`. |
| `ridge_z` | float | Mean similarity along the path's own diagonal trajectory, ±`verify.ridge_window` rows (`verify.ridge_strength`). |
| `margin_z` | float | Claimed-match similarity minus the best similarity outside a ±`verify.margin_band` column window (`verify.margin`). |
| `cam_disagree` | float | `\|cam0-only DTW column − cam5-only DTW column\|` for this row (`verify.path_disagreement`). |
| `desc_disagree` | float | `\|SeqSLAM-only DTW column − DINOv2-only DTW column\|` for this row (`verify.path_disagreement`). Computed for real as of 2026-08-31, but **not yet trustworthy** — see "Known limitation" below. |
| `sift_inliers` | float, currently always empty | SIFT verification is cut by default (plan.md §7 revision note). |
| `slope` | float | Local relative speed (runB frames per runA frame) at this row (`align.local_slope`). |
| `runA_has_cam0`, `runA_has_cam5`, `runB_has_cam0`, `runB_has_cam5` | bool | Whether that specific camera actually decoded a frame at the relevant index — matters because runB's cam0/cam5 decoded counts differ (2622 vs. 2615, measured). |
| `method` | string | Descriptor(s) used for this row; currently always `"seqslam+dinov2"`. |

## `mapping_full.csv` — everything `mapping.csv` has, plus:

- `boundary_clamped` (bool): the *un-widened* DTW-visited range touched column 0 or the last
  runB column — typically a runA tail beyond runB's own coverage. Computed from the raw path,
  independent of the stationary-segment widening (`align.path_to_mapping`).
- `ambiguity` (int): `runB_frame_hi - runB_frame_lo`, before the `mapping.csv` column rename.
- Unlike `mapping.csv`, `runB_frame`/`lo`/`hi`/`runB_time_utc` are **not** blanked for
  `no_match` rows here — this file keeps the raw computed value so a `no_match` row's reasoning
  can be inspected (what the pipeline *would* have said, and why it didn't trust it).

## `mapping_draft.csv` — `align`'s intermediate output, before agreement cues/confidence

Columns: `runA_frame, runB_frame, lo, hi, boundary_clamped` — the direct output of
`align.path_to_mapping` on the fused cam0+cam5 SeqSLAM DTW path. Superseded by
`mapping_full.csv`/`mapping.csv` once `verify` has run; kept for debugging `align` in isolation.

## Known limitation of the current run

`desc_disagree` is computed for real (SeqSLAM + DINOv2, both descriptors run end to end) and is
now a sane signal: an earlier run found DINOv2's similarity landscape needed a much wider
`local_contrast_norm` window than SeqSLAM's tuned value (500 vs. 50) to avoid collapsing onto a
single attractor column — fixed, verified by re-running the DTW path at several window values and
confirming DINOv2's path now tracks the diagonal and agrees closely with SeqSLAM's own
independent path (`decisions.md` D9, D10). `no_match` is 62 (down from a broken 667, and better
than the pre-DINOv2 baseline of 106); confidence tiers 0.7/0.9 are populated for the first time
(1104 / 926 rows).

**Validated (2026-09-01)**: the user completed a real, frame-by-frame independent verification
pass over all 22 `gt/anchors_user.csv` anchors (decisions.md D8's 2026-09-01 update). Running
`groundtruth.evaluate` against it now gives a clean, monotonic reliability table — tier 0.4 → 50%
(n=4), 0.7 → 71% (n=7), **0.9 → 100%** (n=8) — resolving the earlier inversion seen against the
unverified labels, confirming that inversion was label noise, not a real calibration problem.
`median_error=0.0` frames, `p90_error=4.2` frames, `hit@2/5/10=89.5%` (Wilson 95% CI on hit@5:
`[68.6%, 97.1%]`, n=19 — still worth quoting the interval, not just the point estimate, at this
sample size). See `findings.md` for the full trace. Not yet done: dev/test split, stratum H,
ablation study, per-stratum breakdown, seconds-via-timestamps.
