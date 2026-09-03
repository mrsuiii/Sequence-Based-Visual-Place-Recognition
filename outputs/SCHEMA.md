# outputs/ schema

Generated files (git-ignored — regenerate with `python -m routealign align` then `... verify`).
Enforced by `tests/test_mapping_csv.py`.

## `mapping.csv` — the deliverable (2695 rows: one per runA timestamp line)

| column | type | meaning |
|---|---|---|
| `runA_frame` | int | runA row index, `0..2694`, every value exactly once, in order |
| `runB_frame` | int or empty | Best-match runB frame index. **Empty iff `status ∈ {no_match, no_frame}`** — an abstained row never reports a value it has already said it doesn't trust (this is the no-match rule's actual output contract). Monotone non-decreasing across rows where it is present. |
| `confidence` | float in `[0,1]` or empty | Ordinal tier from `confidence.fuse_confidence` (`{0.9, 0.7, 0.5, 0.4}`) — a ranking on held-out anchors, *not* a calibrated probability. Empty iff `runB_frame` is empty. |
| `runB_frame_lo`, `runB_frame_hi` | int or empty | Candidate range `runB_frame` was chosen from, widened to cover a runB stationary segment if the match falls inside one (`align.path_to_mapping`). `lo <= runB_frame <= hi` whenever present. Empty alongside `runB_frame`. |
| `status` | `{matched, ambiguous_range, no_match, no_frame}` | `matched`: normal. `ambiguous_range`: `hi > lo` (inside a plateau/stop — real match, but not frame-exact). `no_match`: abstained (see `reason`). `no_frame`: this runA timestamp line has no decoded frame at all (H0 tail, `runA_frame >= 2674` in this dataset). |
| `reason` | string or empty | Which no-match trigger fired, for `no_match`/`no_frame` rows only; empty otherwise. One of: `"no decoded frame for this row"`, `"ridge_z below tau (...)"`, `"boundary_clamped with ridge_z below ..."`, `"cam_disagree and desc_disagree both above ..."`. |
| `runA_time_utc`, `runB_time_utc` | ISO-8601 UTC string | From each run's own timestamp file. `runB_time_utc` empty alongside `runB_frame`. |
| `ridge_z` | float | Mean similarity along the path's own diagonal trajectory, ±`verify.ridge_window` rows (`verify.ridge_strength`). |
| `margin_z` | float | Claimed-match similarity minus the best similarity outside a ±`verify.margin_band` column window (`verify.margin`). |
| `cam_disagree` | float | `\|cam0-only DTW column − cam5-only DTW column\|` for this row (`verify.path_disagreement`). |
| `desc_disagree` | float | `\|SeqSLAM-only DTW column − DINOv2-only DTW column\|` for this row (`verify.path_disagreement`). Computed for real as of 2026-08-31, but **not yet trustworthy** — see "Known limitation" below. |
| `sift_inliers` | float, currently always empty | SIFT geometric verification was scoped in but not built. |
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
independent path. `no_match` is 62 (down from a broken 667, and better
than the pre-DINOv2 baseline of 106); confidence tiers 0.7/0.9 are populated for the first time
(1104 / 926 rows).

**Validated (2026-09-01)**: the user completed a real, frame-by-frame independent verification
pass over all 22 `gt/anchors_user.csv` anchors. Running
`groundtruth.evaluate` against it now gives a clean, monotonic reliability table — tier 0.4 → 50%
(n=4), 0.7 → 71% (n=7), **0.9 → 100%** (n=8) — resolving the earlier inversion seen against the
unverified labels, confirming that inversion was label noise, not a real calibration problem.
`median_error=0.0` frames, `p90_error=4.2` frames, `hit@2/5/10=89.5%` (Wilson 95% CI on hit@5:
`[68.6%, 97.1%]`, n=19 — still worth quoting the interval, not just the point estimate, at this
sample size). A formal dev/test split was considered and deliberately not done -- 22 anchors is
too few to split further and still get a usable confidence interval on either half; the deletion
test already covers threshold sanity, which the split was for. Not yet done: stratum H,
per-stratum breakdown, seconds-via-timestamps.

**The 3 abstention-category anchors** (excluded from `n_scored` but present as rows in
`evaluation_anchors.csv` with `abstain_outcome` set, per the fix above): `runA_frame` 183 is
`wrongly_abstained` — ground truth is a confident `[102, 104]` but the pipeline's own raw
candidate (column 100, from `mapping_full.csv`, before the no-match rule blanked it) sat only
2 frames outside that range, and was suppressed only because `ridge_z=0.6899` fell just under
`tau=0.75`; `runA_frame` 2600 is `correctly_abstained` — both sides say no_match, and the
pipeline's own signals support that (`ridge_z=0.36`, `cam_disagree=15.0`, well outside normal
range); `runA_frame` 2650 is `wrongly_guessed` — ground truth calls this a loop-closure no_match
(multiple plausible frames near the route's return leg) but the pipeline passed `tau`
(`ridge_z=0.79`) and reported `runB_frame=2600` at the lowest confidence tier (0.4). All three are
consistent with the reliability table above: the no-match rule is a genuine trade-off,
not free — it correctly suppresses 2600, and had it been slightly more permissive it might have
rescued 183, but 2650 shows the opposite failure is also live at the boundary tau sits near.

## `evaluation_anchors.csv` / `evaluation_summary.json` — `evaluate` subcommand output

Written by `python -m routealign evaluate` (`groundtruth.evaluate` against `outputs/mapping.csv`
x `gt/anchors_user.csv`) — the real prediction-vs-ground-truth comparison behind every accuracy
number quoted above, persisted as an actual artifact rather than only ever appearing in a
terminal.

`evaluation_anchors.csv` — **one row per anchor, all 22, every time** — not just the ones a
position error can be computed for. An earlier version of this file silently dropped the 3
abstention-category anchors (no `[lo, hi]` to score against, on either the ground-truth or the
pipeline side), which made a real, informative case (`runA_frame` 183) look like it had vanished
rather than been deliberately excluded — see "Known limitation" below. Columns are prefixed by
which side they come from, so prediction and ground truth are never confused at a glance:

| column | meaning |
|---|---|
| `runA_frame` | the anchor's runA index |
| `pipeline_runB_frame` | the pipeline's prediction (`outputs/mapping.csv`'s `runB_frame`); empty iff the pipeline abstained (`pipeline_status="no_match"`) |
| `gt_runB_best`, `gt_lo`, `gt_hi` | the labeller's answer and its range; empty iff `gt_quality="no_match"` |
| `correct` | **the direct right/wrong verdict** — `True` iff `pipeline_runB_frame` falls inside `[gt_lo, gt_hi]`, `False` if it's outside, and `<NA>` (nullable boolean, not `False`) for the 3 rows where no position comparison is possible (either side abstained) — see `abstain_outcome` for what happened on those. Read this column, not `pipeline_status`, to ask "was this row right" |
| `error_frames` | 0 if `correct`, else distance from `pipeline_runB_frame` to the nearest end of `[gt_lo, gt_hi]`; `NaN` alongside `correct=<NA>` |
| `abstain_outcome` | one of `correctly_answered` (both sides gave an answer — the normal, scored case, n=19), `correctly_abstained` (both sides said no_match), `wrongly_abstained` (ground truth has an answer but the pipeline abstained), `wrongly_guessed` (ground truth says no_match but the pipeline reported a match). The 3 non-`correctly_answered` rows are exactly the ones `correct`/`error_frames` can't score. |
| `pipeline_confidence`, `pipeline_status` | the pipeline's *own* opinion of the row (did it abstain, what tier) — **not** a correctness signal; a row can be `pipeline_status="matched"` and still `correct=False` (the pipeline was confident and wrong, e.g. `runA_frame` 60/150 in the current run) |
| `gt_quality`, `gt_note` | the labeller's own confidence and free-text reasoning for that anchor |

`evaluation_summary.json` — the aggregate metrics (`median_error`, `p90_error`,
`hit_at_2/5/10`, `hit_at_5_wilson_ci`, abstention TP/FN/FP, `by_confidence_tier`) — computed only
over the 19 `correctly_answered` rows (`n_scored`), since the other 3 have no position to average
in; `n_total_anchors=22` is reported alongside `n_scored` so the gap is visible in the summary
itself, not just in the per-anchor file.

Regenerate: `python -m routealign evaluate` (needs `align`/`verify` already run for
`outputs/mapping.csv`, and `gt/anchors_user.csv` to exist).

## `keep_manifest.csv` — Task 3 deliverable (10716 rows: one per `(run, cam, frame)`)

Written by `python -m routealign manifest` (`policy.build_manifest`) — one row per
timestamp line **per camera** (not per run), including frames that never decoded, so nothing is
silently dropped: `2695 runA lines x 2 cams + 2663 runB lines x 2 cams = 10716`. Real run
(2026-09-01): `keep=10585 discard=131` (131 exactly matches the independently-measured
`no_frame` gap: 21 runA + 41 runB/cam0 + 48 runB/cam5, each x1 since `no_frame` is per-camera).
Only one rule is a hard discard; everything else is a flag a downstream consumer filters on
explicitly.

| column | meaning |
|---|---|
| `run`, `cam`, `frame` | Identity. `frame` is the timestamp-line index (`0..len(timestamps)-1`), shared by both cameras of a run under H0. |
| `decoded_index` | `frame` if decoded, else `-1`. |
| `src_file`, `src_sha256` | Source `.hevc` path (relative to `dataset/`) and its SHA-256 (`outputs/inspection/integrity.json`) — traceable back to the exact byte content. |
| `t_capture_utc_ns`, `t_capture_iso`, `t_rel_s` | Capture time: raw ns, ISO-8601, and seconds since this run's first timestamp line. Present even for `no_frame` rows (the timestamp *line* exists even when the video doesn't). |
| `ts_source`, `ts_confidence` | Always `"capture_timestamp_file"`; confidence is `"assumed"` inside `tail_zone`, `"measured"` otherwise. |
| `dt_prev_ms`, `dt_next_ms`, `interval_irregular` | Interval to the neighbouring lines; flagged outside `[80,120]`ms (`policy.interval_irregular_lo/hi_ms`) — **not** the tighter `[90,110]`ms band `outputs/inspection/*.json` uses for T1 characterisation, a different, stricter purpose. |
| `fps_nominal_declared`, `fps_measured`, `width`, `height`, `codec`, `colour_range` | Per-file constants from `outputs/inspection/*.json` (T1), copied onto every row of that `(run, cam)`. `width`/`height` are the source video's cropped resolution (1440x1080), not the 640x480 JPEG working cache. |
| `is_keyframe` | `frame` is an IDR picture (`outputs/inspection/*.json` `gop.idr_indices`); always `False` for `no_frame` rows. |
| `is_stationary`, `stationary_run_id`, `dedup_keep` | From `frames.stationary_segments` on that run's own motion energy (both cameras). `stationary_run_id` is `-1` for "not in any segment" (`int\|null` encoded as a sentinel, not a separate nullable dtype). `dedup_keep=True` outside a segment (nothing to thin) or on every `round(fps/dedup_keep_hz)`-th frame inside one. |
| `motion_energy` | Mean abs pixel diff to the *next* frame (`frames.motion_energy`); `NaN` for the last decoded frame of each `(run, cam)` (no transition to measure) as well as for `no_frame` rows. |
| `weather`, `lighting`, `lens_occlusion` | Qualitative tags: `weather`/`lighting` per run (runA sunny/hard-shadow, runB overcast/light-rain, a one-time visual read), `lens_occlusion` per camera (`cam5="intermittent_traffic"`, structurally more exposed than kerb-facing `cam0`). Constants, not a per-frame detector. |
| `tail_zone` | Within `policy.tail_zone_lines` (50) of the run's own timestamp-file tail. |
| `decode_ok` | `frame < decoded_count` for this `(run, cam)`. The only input to `keep`. |
| `corr_run`, `corr_frame`, `corr_lo`, `corr_hi`, `corr_conf`, `corr_status`, `corr_method` | Task 2's correspondence, runA rows only (direct copy of `mapping.csv`) — `corr_status="not_applicable"` for every runB row (T2 never computed or evaluated a reverse runB->runA mapping). |
| `place_id` | `floor(runB_frame / 100)` for runB rows; for runA rows, the same formula applied to `corr_frame` (linearly interpolated from the nearest matched runA neighbours when `corr_frame` is itself missing — `corr_frame` in the output stays honestly `NaN`, only the derived `place_id` uses the interpolated value). `-1` for the merged start/end depot (any side within `policy.depot_runb_margin`, 50 frames, of either end of runB's route) and for rows with no derivable position at all. |
| `split` | `{train, val, test, unassigned}`, contiguous blocks of `place_id` with a 1-place buffer between them (never by frame index -- both runs' frames of one place must always share a split). The merged depot and any unplaced row are always `"unassigned"`. |
| `keep`, `reject_reason` | The one hard rule: `keep = decode_ok`; `reject_reason` is `"no decoded frame for this row"` or an empty string (not `None`/NaN — reads back as NaN after a CSV round-trip like every other empty string column in this project, e.g. `mapping.csv`'s `reason`). |

**Checked, not just assumed**: `keep = decode_ok` is the only encoded hard-discard condition,
but the two other candidate conditions (corrupt decode, exact duplicates) were checked for real
rather than left an open question — re-decoding all 4 streams at a stricter ffmpeg verbosity than
production found zero warnings, and `motion_energy` was exactly `0.0` for zero consecutive-frame
pairs across the whole dataset. No rows exist that these would additionally discard, so no extra
column was needed for them.

Regenerate: `python -m routealign manifest` (needs `align`/`verify` already run for
`outputs/mapping.csv`).
