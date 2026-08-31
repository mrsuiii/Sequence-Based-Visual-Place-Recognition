# plan.md — Aligning two traversals of the same route

Execution plan for the take-home. Checkbox legend: `[ ]` todo · `[~]` in progress · `[x]` done — keep it current.
Code rules live in `CLAUDE.md`; numbers go to `findings.md` with their provenance; choices go to `decisions.md`.

**Revision (2026-08-30, post-review):** cut-order inverted for S4 (SIFT) and v2 (NULL-state DP) — both are now
*add-if-ahead* in the day-2 buffer, not *cut-if-behind* defaults (§6, §7, §11, §12); they were scoped for someone
already fluent in DTW/MAGSAC, not a first build. Ground truth reframed: the user labels, Claude does a blind QA
pass afterward, not a claimed-independent second labeller (§0.2, §8, §13) — the two are not statistically
independent judges. A dedicated rehearse/defensibility slot added to day 2 (§12) — being able to explain what
shipped matters more than shipping more.

## 0. The brief, the weights, the decisions

### 0.1 Brief (as received)

> **Task: align two traversals of the same route.** runA and runB are two recordings of the same route, made on
> different days. Each contains footage from two cameras (cam0 and cam5) and, for each camera, a file of frame
> capture timestamps in nanoseconds since the Unix epoch, one per line. These are unmodified exports from our
> recording stack.
>
> **1. Characterise the recordings.** For each run and each camera, establish and report: frame count, resolution,
> and file size; nominal frame rate and the actual frame rate achieved; total route time, and when the run was
> recorded; the distribution of intervals between consecutive frames, including any irregularities. State how you
> determined each figure. Where a tool reports a value, say whether that value is measured or merely declared, and
> treat the two differently.
>
> **2. Match the frames between runA and runB — the main task.** Produce a frame-level correspondence between the
> two runs: for a frame in runA, which frame in runB shows the same place? The vehicle was not driven identically on
> the two occasions, so this is not a fixed offset and the mapping is not linear. Parts of the route may have no
> usable correspondence at all; say so where that happens rather than forcing a match. Roughly, you will need to
> decide four things: how to represent a frame so that two views of the same place are comparable; how to search
> for the match; what constraint to impose on the result, given the two runs traverse the route in the same order;
> and how to tell a correct match from a confident wrong one. Deliver: a mapping file, CSV, with at least
> `runA_frame,runB_frame,confidence`, including every runA frame, using an empty runB_frame where you assert no
> match; your method, and why you chose it over the alternatives you considered; an honest accuracy estimate — how
> did you measure it, on what ground truth, and where is the mapping least reliable? A convincing account of your
> failure cases is worth more than a high number you cannot defend.
>
> **3. Recommend a policy.** If you were preparing this data for model testing or training, which frames would you
> keep, which would you discard? What would you attach to each surviving frame so that a downstream consumer cannot
> silently misuse it? Be concrete.
>
> **What to send:** a report of 2–4 pages (PDF) listing your approach, thought process and results; it is discussed
> at the second interview. Weights: Task 2 — 40 %; Tasks 1 and 3 — 20 %; clarity of reasoning and how well you
> separate what you proved from what you assume — 20 %; code quality and robustness — 20 %. Use of AI tooling is
> expected and encouraged; understand and be able to defend everything you submit.

### 0.2 Decisions already made

- Time budget ≤ 2 days (§12). Docs, code and report in English.
- Method: classical SeqSLAM-style baseline **and** a pretrained deep descriptor (DINOv2), fused; global monotone
  alignment (DTW) with explicit abstention. Recommendation and alternatives in §11.
- Ground truth: ~30 anchors labelled by the user (primary; sheet tool is blind to model output by construction).
  Claude does a second pass afterward as a **QA/consistency check, not an independent co-labeller** — user and
  Claude are not statistically independent judges (Claude's visual read can share failure modes with the DINOv2
  descriptor used in the pipeline itself). User's label is authoritative; Claude's flags get re-checked, not
  averaged in. See §13.
- Index convention **H0**: frame `k` = timestamp line `k` = decoded frame `k` of each camera (evidence in §1);
  `mapping.csv` has one row per runA timestamp line (2695), `status=no_video` for lines without a decoded frame.

## 1. What is already established (read-only survey, 2026-08-30; re-verify inside the pipeline and cite in findings.md)

Data & integrity
- `dataset/{runA,runB}/{cam0,cam5}_20_yuv420p_output.hevc` + `.timestamps.txt`; all 8 SHA-256 match
  `~/Downloads/SHA256SUMS.txt` (copy it to `dataset/`). `requirements.txt` from an earlier session: ffmpeg CLI is
  the only video I/O path; PyAV must not be installed. Reference reading: `~/Downloads/vprtutorial.pdf`
  (Schubert et al. 2023, "Visual Place Recognition: A Tutorial").

Bitstream (declared unless marked measured)
- Raw Annex-B HEVC, no container. SPS: 1440×1088 coded, `conf_win_bottom_offset=4` ⇒ 1440×1080, square pixels,
  Main, level 4.0, 8-bit 4:2:0, `sps_max_num_reorder_pics=2`; VUI `time_scale/num_units_in_tick = 10/1` ⇒
  **declares 10 fps**; `video_signal_type_present_flag=0` ⇒ colour range is *not* signalled (ffprobe's `tv` is a
  default). ffprobe `avg_frame_rate=25/1` is the raw-demuxer default (meaningless); `nb_frames`, `duration`,
  `bit_rate` = N/A. Filename token `_20_` is a third, unexplained declaration.
- Measured: VPS/SPS/PPS/SEI/IDR at byte 0; 11 IDR per file (GOP 240 = 24 s ⇒ decode sequentially); one slice per
  picture; per-picture prefix SEI = `pic_timing` HRD delays only (no wall-clock). Decoded frames
  (`ffprobe -count_frames`, packets == frames) = coded pictures (NAL parse of `first_slice_segment_in_pic_flag`):
  **runA cam0 2674, cam5 2674; runB cam0 2622, cam5 2615**. All four sizes are exact multiples of 262 144 B
  (256 KiB) and end mid-slice ⇒ block-buffered writer stopped without a flush. That explains ~5–10 lost frames per
  file; the remaining 11–43 are *hypothesised* encoder/pipeline latency at shutdown (assumption, flag it).

Timestamps (measured)
- cam0 and cam5 files are byte-identical within each run. runA 2695 lines (21 more than frames), runB 2663
  (41 / 48 more). Integers, strictly increasing, no duplicates. Intervals sit on an exact 100.000 ms grid
  (median 99.95 / 100.02 ms; residual σ 2.1 / 2.3 ms; lag-1 autocorrelation −0.42 / −0.46 ⇒ read-out jitter around
  a stable trigger grid). Expected frames at 10 Hz over the span vs lines: 2697 vs 2695, 2666 vs 2663 ⇒ the only
  missing triggers are one start-up gap each (runA idx 4→5 = 300.8 ms, runB idx 5→6 = 396.9 ms, both while
  stationary). Spans 269.59 s / 266.49 s. First timestamps: runA 2025-02-28 06:24:09 UTC (Fri),
  runB 2024-04-13 09:12:41 UTC (Sat) — runB is 321 days *earlier*. Scene banners ("Sayangi Petaling Jaya") ⇒
  Malaysia ⇒ local 14:24 / 17:12 at UTC+8 (inference from image content; recorder clock unverifiable).

Footage (measured from contact sheets and 64×48 probes)
- runA: sunny, hard shadows, clean lens, mean gray 65–83. runB: overcast, raining, **water droplets on the glass**,
  mean gray 30–38 (⇒ roughly half the bitrate: ~3.9 vs ~2.1 Mbit/s, derived). Both cameras side-facing wide-angle
  (left-hand traffic ⇒ cam0 = left/kerb side: façades, parked cars; cam5 = right/road side: traffic-dominated,
  weaker). Wing mirror + vignetting occupy the bottom ~20 % ⇒ crop rows [0, 0.8·H). The route is a closed loop
  and both runs start and end at the same depot ⇒ the similarity matrix has bright blocks in all four corners.
- Stationary: runA 0–173 (cam0) / 0–176 (cam5) and ~2616–end; runB 0–46. No mid-route stop ≥ 1 s found.
  Low motion-energy stretches on one camera (runA ~1560–1700 cam0, runB ~1450–1650) are low texture, not stops
  (the other camera shows motion) ⇒ stop detection must use both cameras. Ego-motion stop at runA ~2620 appears in
  both cameras within ±1 frame; runB onset agrees within ±3 ⇒ consistent with index sync (H0). Raw motion-energy
  cross-correlation is *not* a usable sync test (windowed lags scatter ±30).
- SeqSLAM-lite probe (no sequence constraint): cam0 row-argmax within ±30 of a running median for 94 % of runA
  rows, cam5 77 %. runA's stationary start maps many-to-one onto *moving* runB ~82–100 (runB parked ~80–100 frames
  earlier along the same street). runB ends about where runA frame ~2588 is ⇒ **runA ~2590–2673 is a genuine
  no-match region**, and the runA tail aliases to runB's start (loop closure).

Tooling
- Apple M4 (10 cores), 16 GB RAM, 93 GB free, macOS 26. ffmpeg 8.1.1 with libx265: software HEVC decode ≈ 3 s per
  file; `-hwaccel videotoolbox` fails on these files. Conda env `phase2-vpr` (Python 3.11.16): numpy 2.4,
  scipy 1.17, pandas 3.0, opencv 5.0, pillow, matplotlib, tqdm. To install: pytest, ruff, pyyaml, torch (no hub
  cache yet; internet available). TeX Live 2026 + latexmk, pandoc, git, make available.

## 2. Step 0 — Setup & integrity (1:00)

- [x] `git init`; `.gitignore` for `data/cache/`, `outputs/*.npy`, `report/*.aux|log|out`, `__pycache__`, `.pytest_cache`.
- [x] Copy `~/Downloads/SHA256SUMS.txt` → `dataset/SHA256SUMS.txt`.
- [~] `pip install pytest ruff pyyaml` (done); `pip install torch torchvision` started in the background, unpinned
      (let pip resolve current stable — see note below), still running as of this checkbox.
- [x] Package skeleton (`src/routealign/` modules per `CLAUDE.md` §3), `config.yaml`, `Makefile`, `pyproject.toml`
      (ruff config, `python -m routealign` entry point); editable-installed (`pip install -e .`).
- [x] `make verify-data` → `outputs/inspection/integrity.json` (sha256 of all 8 inputs vs manifest): 8/8 OK.
- [x] `make decode`: one sequential ffmpeg pass per file → `data/cache/frames/{run}/{cam}/{k:05d}.jpg` (640×480,
      q=3, `-fps_mode passthrough`, 0-indexed); JPEG count asserted == `count_decoded` for all 4 files (matched
      exactly: 2674/2674/2622/2615); `ffmpeg -version` recorded in `findings.md`.
- Output: repo skeleton, verified inputs (8/8), JPEG cache (323 MB measured, smaller than the ~640 MB estimate),
  first entries in `findings.md` and `decisions.md`. Note: `pyproject.toml` declares no `[project.dependencies]`
  — installs still go through `requirements.txt`; revisit if that split becomes awkward.

## 3. Step 1 — Task 1 characterisation (1:30)

`make characterise` → `outputs/inspection/{run}_{cam}.json`, `outputs/inspection/task1_table.md`, figures. One row
per requested figure, each citing the exact command / function:

| figure | declared (source) | measured (command / code) |
|---|---|---|
| frame count | none — `nb_frames=N/A`; timestamp lines 2695 / 2663 (`wc -l`) | `ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames` → 2674/2674/2622/2615; `scan_nals` coded-picture count (identical); JPEG count after decode |
| resolution | SPS 1440×1088 + `conf_win_bottom_offset=4` ⇒ 1440×1080, SAR 1:1 (`ffmpeg -bsf:v trace_headers`) | decoded frame shape |
| file size | — | `os.path.getsize`; multiples of 256 KiB; last NAL cut mid-slice |
| nominal fps | filename `20` (unresolved), VUI `10/1`, ffprobe `avg_frame_rate 25/1` (demuxer default) | none — an elementary stream carries no per-frame timing; SEI is HRD only |
| actual fps | — | median Δt, mean rate `(N−1)/span`, 100.000 ms grid fit ⇒ 9.99 fps |
| route time | — | timestamp span (269.59 / 266.49 s); video-covered span under H0 (`ts[N_dec−1] − ts[0]`); stationary time at start/end |
| when recorded | — | first timestamp → UTC; local time = UTC+8 by scene inference; clock correctness unverifiable; file mtimes are export times |
| Δt distribution | — | histogram (log-y), percentiles, count outside [90, 110] ms, gaps > 150 ms with indices, lag-1 autocorrelation, grid residual |
| bit rate | none (`N/A`) | size × 8 / covered span |
| codec / colour | Main, L4.0, reorder 2, DPB 6, HRD present; colour range **unsignalled** | GOP 240, one slice per picture, TRAIL_N/TRAIL_R pattern |
| count discrepancy | — | lines − decoded = 21/21/41/48; tail-loss evidence; H0 stated as an assumption with its consequence if wrong (a constant time shift; the visual alignment is unaffected) |

- [~] `timestamps.interval_stats`, `io_video.probe/scan_nals/sps_vui` implemented and run for real on
      all 4 files (`make characterise` → `outputs/inspection/{run}_{cam}.json` + `task1_table.md`).
      `viz`: Δt histogram + Δt-vs-index done for all 4 files. Not done: frame strip around each gap,
      last decoded frame at full resolution (truncation check) — see `findings.md`'s "not yet done" note.
      `scan_nals`/`sps_vui` are ffprobe/ffmpeg-`trace_headers`-based (picture/GOP-level and SPS/VUI
      text parsing), not a hand-rolled byte-level Annex-B NAL scanner — see `decisions.md`.
- [ ] Write the T1 section of the report skeleton immediately (numbers are final).

**Step 1 result, 2026-08-30:** all core Task 1 figures measured and cross-checked (see
`findings.md`). Two numbers from the prior exploratory session's informal claims did **not**
reproduce (GOP size 250 not 240; Δt lag-1 autocorrelation −0.137 not −0.42/−0.46 on runA/cam0) —
this repo's measured values are treated as authoritative since they have runnable code behind
them; flagged rather than silently overwritten.

## 4. Step 2 — Preprocessing & sync (1:00)

- [ ] `FrameStore` over the JPEG cache; gray/RGB crops of rows [0, 384) of 640×480 (drops mirror/vignette band).
- [ ] `motion_energy` per camera (mean |Δ| on 64×48 gray); `stationary_segments` = *both* cameras below 15 % of
      their median for ≥ 10 frames. Static-mask sanity figure (per-pixel temporal std).
- [ ] Sync test (records evidence for H0): stop/onset event indices per camera; cam0-only vs cam5-only alignment
      paths (§6) must agree — this bounds the *differential* cam0/cam5 offset (a common offset would cancel; say so).

## 5. Step 3 — Representation (1:30)

- [ ] R1 SeqSLAM: crop → 64×40 (`INTER_AREA`), 8×8 non-overlapping patches, `(p − mean) / (std + 1)` in 0..255
      units, clip ±3, flatten (2560-d), L2. `make describe FEAT=seqslam`.
- [ ] R2 DINOv2 ViT-S/14: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)`
      (fallback: `timm` `vit_small_patch14_dinov2.lvd142m`); crop → 448×280 (multiples of 14), ImageNet mean/std,
      fp32, `no_grad`, batch 32 on `mps` (cpu fallback); descriptor = L2(concat(L2(CLS), L2(GeM₃(patch tokens))))
      768-d; CLS-only is the cut-first fallback. Record model/commit in the cache sidecar. 30-minute cap on
      install/hub trouble, then continue with R1 only and say so.
- [ ] Similarity per camera and descriptor: cosine → SeqSLAM local contrast normalisation along runB per runA row
      (window R ∈ {20, 50, 100} tuned on synthetic warps; `scipy.ndimage.uniform_filter1d` on S and S²), clip ±3 ⇒
      z-units comparable across descriptors/cameras. `Ŝ_joint = weighted mean` (cam0 ≥ cam5; weights from the
      synthetic tests) — only after the sync test passes.
- [ ] Diagnostics figure: matrix + unconstrained argmax per camera/descriptor (corner aliasing visible).
- Why these two: R1 is dependency-free and already finds the ridge; R2 is the strongest zero-training feature under
  appearance change (backbone of AnyLoc/SALAD). Not chosen: NetVLAD/CosPlace/EigenPlaces (extra weights, marginal
  gain expected here), CLIP (weaker geometry), training anything (no labels, no time).

## 6. Step 4 — Search & monotonic constraint (2:00)

Constraint: both runs traverse the route in the same order ⇒ the mapping is a monotone non-decreasing warp with
free local slope (stops = horizontal/vertical runs), open ends (runB starts earlier, runA ends later), and possible
gaps (no correspondence). Loop closure (start ≈ end) is exactly what a single-frame argmax cannot resolve.

- [ ] v1 DTW with open ends — cost `c = 3 − clip(Ŝ_joint, −3, 3)` ∈ [0, 6] plus `lam` per non-diagonal step
      (`lam` ∈ {0.1, 0.25, 0.5, 1.0}, default 0.5); exact row-vectorised recurrence, float64, < 1 s for 2674×2622:

```python
def dtw_open_ends(c: NDArray, lam: float) -> NDArray:
    """c: float64[NA, NB] non-negative cost. Returns path int32[L, 2] of (i, j), monotone, every i covered."""
    NA, NB = c.shape
    D = np.empty((NA, NB), np.float64)
    D[0] = c[0]                                            # open begin on the runB axis
    for i in range(1, NA):
        prev = D[i - 1]
        diag = np.concatenate(([np.inf], prev[:-1]))       # D[i-1, j-1]
        V = np.minimum(prev + lam, diag)                   # best of up (penalised) / diagonal
        C = np.cumsum(c[i] + lam)                          # C[j] = sum_{l<=j} (c[i, l] + lam)
        D[i] = C + np.minimum.accumulate(V + c[i] - C)     # = c[i, j] + min(V[j], D[i, j-1] + lam)
    i, j = NA - 1, int(np.argmin(D[-1]))                   # open end
    path = [(i, j)]
    while i > 0:                                           # backtrack by recomputing the argmin; ties -> diagonal
        cand = (D[i-1, j-1] if j else np.inf, D[i-1, j] + lam, D[i, j-1] + lam if j else np.inf)
        k = int(np.argmin(cand))
        i, j = (i-1, j-1) if k == 0 else (i-1, j) if k == 1 else (i, j-1)
        path.append((i, j))
    return np.array(path[::-1], np.int32)
```

- [ ] `path_to_mapping`: per runA row, `runB_frame = argmax Ŝ_joint` over the visited columns `[lo, hi]`; widen
      `[lo, hi]` to the runB stationary segment containing `runB_frame` (all those frames show the same place);
      flag `boundary_clamped` for runs stuck at column 0 or NB−1 (the runA tail beyond runB's end).
- [ ] v2 (**cut by default** — build only in the day-2 buffer, and only if v1 is shipped, tested, and understood;
      do not start this while v1 is still unverified) NULL-state affine-gap DP: score `m = Ŝ_joint − s0` (s0 = 1.0),
      states M (match), X (runA frame unmatched), Y (runB frame skipped), penalties `g_open 3.0`, `g_ext 0.5`,
      `rho 0.5` for non-diagonal match steps; row-vectorised with `cumsum`/`maximum.accumulate` like v1; must
      reproduce v1 on gap-free synthetic warps and pass the deletion test (§8) before it ships. Rows in state X get
      an empty `runB_frame` from the optimiser instead of from post-hoc thresholds.
- Not chosen: fixed/linear offset (violates the brief), nearest neighbour only (loop closure + uniform stretches
  alias), HMM with hand-tuned transition matrix (equivalent to v2 with more knobs), full SLAM (no time, no need).

## 7. Step 5 — Verification & confidence (2:00; SIFT is cut by default — see below and §12)

Independent evidence per runA row (all become columns of `outputs/mapping_full.csv`):

| feature | meaning | why it catches a confident wrong match |
|---|---|---|
| `ridge_z` | mean `Ŝ_joint` along the path over ±5 rows | weak ridge ⇒ the optimiser was merely forced through |
| `margin_z` | path value minus best value outside a ±30-column band, min over cameras | a second place that looks the same (loop closure, repeated façades) |
| `cam_disagree` | \|j from cam0-only DTW − j from cam5-only DTW\| | two views of the same moment must agree |
| `desc_disagree` | \|j from SeqSLAM DTW − j from DINOv2 DTW\| | two representations must agree |
| `argmax_disagree` | \|path j − unconstrained argmax j\| | the path is corroborated by single-frame evidence |
| `sift_inliers` | RootSIFT (ratio 0.8) + `cv2.USAC_MAGSAC` fundamental matrix on 640×384 gray, assigned pair ±2 | geometric consistency independent of global descriptors |
| `slope`, `stationary_A/B`, `boundary_clamped`, `ambiguity = hi − lo` | context flags | plateaus and tails are where errors live |

- [ ] SIFT calibration — **cut by default**, not part of the core day-1/day-2 path. Add only in the day-2 buffer
      (§12) if genuinely ahead: inlier distribution for correct anchors vs pairs offset by 60–200 frames; "verified"
      threshold = value at which offset pairs pass ≤ 5 %. Under rain many correct pairs will fail ⇒ use only as
      positive evidence; if it does not separate, drop it and say so in the report. `cam_disagree`, `desc_disagree`
      and `argmax_disagree` already give multi-signal verification without it — SIFT is upside, not a dependency.
- [ ] No-match rule (empty `runB_frame`): no video for the row; `ridge_z < tau` (tau from the deletion test,
      default 0.75); `boundary_clamped ∧ ridge_z < 1.5`; `cam_disagree > 30 ∧ desc_disagree > 30`; or DP gap (v2).
- [ ] Confidence tiers (ordinal; meaning = the reliability table on held-out anchors, *not* a calibrated
      probability): 0.9 if `ridge_z ≥ 1.5` and both disagreements ≤ 3 · 0.7 if `ridge_z ≥ 1.0` and two of
      {cam ≤ 5, desc ≤ 5, SIFT verified} · 0.5 for `ambiguous_range` rows inside plateaus · 0.4 otherwise (matched
      but weak). Thresholds tuned on synthetic warps + deletion test + the 6 dev anchors only.
- [ ] `outputs/mapping.csv` columns: `runA_frame, runB_frame, confidence, runB_frame_lo, runB_frame_hi,
      status{matched, ambiguous_range, no_match, no_video}, reason, runA_time_utc, runB_time_utc, ridge_z,
      margin_z, cam_disagree, desc_disagree, sift_inliers, slope, runA_has_cam0, runA_has_cam5, runB_has_cam0,
      runB_has_cam5, method`. Schema documented in `outputs/SCHEMA.md` and enforced by `tests/test_mapping_csv.py`.

## 8. Step 6 — Ground truth & evaluation (user labels 1.5 h blind; Claude QA pass ~0:30 after, not in parallel; analysis 1:30)

- [ ] `gt/labelling_protocol.md`: "same place" = within ±0.5 s of runB travel (±5 frames at 10 Hz); lane offset
      ignored. The sheet tool reads only the frame cache — it never sees model output (blind by construction):
      coarse sheet = every 20th runB frame for both cameras (131 tiles, indices burnt in) → fine sheet = every frame
      in ±20 of the coarse pick (41 tiles). Record `runA_frame, runB_best, lo, hi, quality{sure, unsure, no_match},
      note`. Scanning sequentially from the previous anchor's runB position is allowed (route order is a property
      of the data, not of the model).
- [ ] Anchors (30, labelled by the user; Claude QA pass after — see §0.2): stratum S (22): runA 200 … 2500 every
      ~110 frames with ±20 jitter (fixed seed) + plateau frames 60, 150 + tail frames 2600, 2650. Stratum H
      (8, reported separately): 2 in the low-texture block, 2 with passing traffic on cam5, 4 at the largest
      SeqSLAM/DINOv2 path disagreement (positions model-informed, labels still blind).
- [ ] Merge: the user's label is the consensus (authoritative). Claude's QA pass flags rows where its read differs
      by more than `max(3, half the union width)` for the user to re-check by hand; only the user's post-recheck
      label ships. Report user-vs-Claude agreement as a **QA sanity number**, explicitly not inter-rater
      reliability or a label-noise floor — they are not independent judges (§13). A true noise floor would need a
      second independent human on a subset; optional, not required — note its absence in the report if skipped.
- [ ] Split: 6 dev anchors (threshold sanity only), ~24 test anchors reported with Wilson 95 % CIs
      (e.g. 22/24 → [74 %, 98 %]).
- [ ] Metrics: error = 0 if the prediction lies in `[lo, hi]`, else distance to the nearest end; hit@2/5/10 frames
      (and seconds via timestamps); median and p90 error; per stratum (plateau / mid-route / tail); per confidence
      tier (reliability table); no-match precision/recall from the **deletion test** (delete runB [j0, j0+150),
      rerun, the mapping must abstain on the corresponding runA rows) plus the labelled runA tail.
- [ ] Ablation on the same anchors: argmax-only vs DTW; cam0 / cam5 / both; SeqSLAM / DINOv2 / both;
      with / without local contrast normalisation.
- [ ] Failure taxonomy with side-by-side figures: start plateau (many-to-one onto moving runB), runA tail beyond
      runB's truncated end, low-texture block, cam5 passing traffic, rain-degraded stretch, loop-closure corners.

## 9. Step 7 — Task 3 policy (1:00) → `outputs/keep_manifest.csv`, `outputs/SCHEMA.md`, dataset-card section

Keep/discard rules (per frame, per camera):
1. Discard rows without an image (timestamp lines beyond the decoded count), failed decode, exact duplicates.
2. Flag `interval_irregular` for neighbours of Δt ∉ [80, 120] ms — discard for temporal-window consumers, keep
   for single-frame consumers.
3. Stationary runs: keep one frame per second (`dedup_keep=true`), flag the rest with `stationary_run_id` so
   near-duplicates neither dominate training nor inflate test numbers.
4. `tail_zone=true`, `ts_confidence=assumed` for the last 50 frames of each run.
5. Run-level `weather`, `lighting`, `lens_occlusion` (runB: rain, water droplets) — consumers filter explicitly.
6. cam5 traffic occlusion: expose `motion_energy` (and an occlusion score if implemented); never drop silently.
7. Correspondence attached as `corr_frame, corr_lo, corr_hi, corr_conf, corr_status, corr_method`; frames with
   `corr_status=no_match` remain usable for single-run tasks.
8. **Split-leakage rule**: `place_id = floor(runB_frame / 100)` (≈ 10 s of route) shared by both runs — runB frames
   by their own index, runA frames via their correspondence (interpolated from neighbours where `no_match`); the
   depot at start and end gets one merged `place_id`; splits are contiguous blocks of `place_id` with 5-s buffers;
   both runs' frames of one place always share a split; never split by random frame index.

Manifest fields (typed): `run:str, cam:str, frame:int, decoded_index:int, src_file:str, src_sha256:str,
t_capture_utc_ns:int64, t_capture_iso:str, t_rel_s:float, ts_source:str, ts_confidence:{measured, assumed},
dt_prev_ms:float|null, dt_next_ms:float|null, interval_irregular:bool, fps_nominal_declared:float (10.0, SPS VUI),
fps_measured:float, width:int, height:int, codec:str, colour_range:"unsignalled", is_keyframe:bool,
is_stationary:bool, stationary_run_id:int|null, dedup_keep:bool, motion_energy:float, weather:str, lighting:str,
lens_occlusion:str, tail_zone:bool, decode_ok:bool, corr_run:str, corr_frame:int|null, corr_lo:int|null,
corr_hi:int|null, corr_conf:float|null, corr_status:enum, corr_method:str, place_id:int|null,
split:{train, val, test, unassigned}, keep:bool, reject_reason:str|null`. Export frames as
`{run}_{cam}_{frame:05d}_{t_utc_ns}.png` so the timestamp travels with the pixels; ship CSV + JSON schema + a
dataset card stating nominal-vs-measured fps, the H0 assumption and the split rule.

## 10. Step 8 — Tests, report, submission (1:00 + 3:00 + 1:00)

- [ ] Tests (< 60 s, synthetic fixtures): `test_timestamps` (10 Hz grid + jitter + one 300 ms gap → gap index and
      expected-vs-lines = 2; duplicates / non-monotone raise; `frame_times` tail count), `test_similarity`
      (contrast normalisation removes a constant row offset and a bright block), `test_align` (vectorised DTW ==
      naive triple-loop reference exactly, incl. the path, on random 30×40 costs; synthetic monotone warp with a
      plateau and padding → path within ±1 for ≥ 95 % rows, correct open ends; NULL-DP gap precision/recall ≥ 0.9
      and equality with DTW when gap-free; `path_to_mapping` complete, monotone, lo ≤ best ≤ hi), `test_io`
      (20 frames of `lavfi testsrc` encoded with libx265 to a raw `.hevc` → `count_decoded` = 20, NAL pictures = 20,
      `sps_vui` parses `time_scale`), `test_mapping_csv` (schema: every runA frame once, confidence ∈ [0, 1], empty
      `runB_frame` iff status ∈ {no_match, no_video}, monotone over matched rows), `test_cli_smoke`
      (`all --limit 60` on real data if present, else skip).
- [ ] Report (LaTeX via the `latex-document-skill`, 4 pages): T1 table + Δt figure (1 p); T2 method, alternatives,
      results, ablation, reliability table, failure cases (2 p); T3 (0.5 p); reproducibility + proved-vs-assumed
      ledger (0.5 p). Figures: Δt histograms; similarity matrix with the final path, rejected spans and the corner
      aliasing; anchor error plot coloured by tier; 2–3 failure-case thumbnails.
- [ ] README: `make all` from a clean cache with timings, `ffmpeg -version`, pinned versions, the H0 statement.
- [ ] Final: `make lint test`, `make all` twice (byte-identical CSVs), archive `code + outputs + report.pdf`.

## 11. Candidate solutions and the recommendation

| id | representation | search / constraint | verification | pros | cons | effort |
|---|---|---|---|---|---|---|
| S1 SeqSLAM classic | patch-normalised 64×40 thumbnails | global monotone DTW | agreement cues | no dependencies, seconds; probe: 94 % of cam0 rows already near the ridge | cam5 77 %; rain/droplets; no notion of abstention | 0.5 d |
| **S2 deep global + DTW** (recommended) | DINOv2 CLS+GeM fused with S1 | z-scored cosine + DTW v1 (NULL-state DP v2) | margin, cam/desc agreement, deletion test, reliability table | condition-robust, explainable, fast, ablatable | needs a torch install; global descriptors stay ambiguous on uniform stretches | 1 d |
| S3 VPR-trained descriptor | EigenPlaces / CosPlace ResNet-50 | as S2 | as S2 | best condition invariance in the literature | extra weights; marginal gain expected here | +2 h (cut) |
| S4 local-feature refinement | SIFT / LightGlue | argmax inliers within ±k of the S2 path | inliers | frame-accurate when moving; doubles as verifier | rain, droplets, repeated façades; slow for all pairs | +3 h (lite: verification only) |
| S5 odometry / arc-length prior | optical-flow pseudo-distance | align in the distance domain | — | native handling of stops and speed changes | monocular scale drift; complexity | stretch |

Recommendation: **S2 (v1 DTW only) as the committed default.** S4-lite (SIFT) and v2 (NULL-state DP) are
day-2-buffer upside, not dependencies (§12) — the probe shows the sequence constraint already does most of the
work, the deep descriptor buys robustness on cam5 and the rain-degraded stretches, and `cam_disagree` /
`desc_disagree` / `argmax_disagree` give multi-signal verification on their own. Abstention and the reliability
table are what make the result defensible, not the extra verifiers.

## 12. Two-day schedule (≈ 9.5 h per day) and cut order

Day 1
- 08:30 skeleton, config, Makefile, `io_video`, `timestamps`, `make verify-data`; JPEG decode in the background.
- 09:30 Task 1 `characterise` (NAL scan, VUI, Δt analysis, plots, the declared-vs-measured table); write the T1
  report section.
- 11:00 SeqSLAM descriptors → similarity → contrast normalisation → DTW → mapping v0; similarity figure; sync test.
- 13:00 torch + DINOv2 (30-minute cap) → `S_dino` → DTW → compare paths.
- 14:00 labelling sheet tool (30 min); user labels the 30 anchors (blind to predictions) while
  `groundtruth.evaluate` is coded; Claude's QA pass happens after, in its own sitting, not alongside.
- 16:30 synthetic warp + deletion tests (pytest), tune `lam`, `tau`, `R`; first accuracy numbers and ablation table.
- 18:00 confidence features, no-match rule v1, mapping v1, commit.

Day 2
- 08:30 Claude QA labelling pass (blind, after the user's — §8); reconcile, reliability table, ablation table on
  the full anchor set.
- 09:30 hard-case anchors (stratum H), failure taxonomy with side-by-side figures.
- 10:30 Task 3 manifest + schema + dataset card.
- 11:30 finish tests (`test_align`, `test_mapping_csv`, `test_io`, …), `make all` from a clean cache, timings,
  README.
- 12:30 **buffer / catch-up**, absorbs day-1 overrun first. Only if genuinely ahead of schedule: SIFT calibration
  (§7) then v2 NULL-state DP (§6), in that order — neither is a dependency for anything downstream.
- 14:00 **rehearse**: read the shipped code and report draft end to end; for every number and every §14 answer,
  confirm you can explain it without notes. Cut anything you can't explain — a simpler thing you understand beats
  a fancier thing you don't, at interview.
- 15:00 report (LaTeX, 4 pages).
- 17:30 final pytest, archive, buffer.

**Add only if ahead** (day-2 buffer, in this order): SIFT verification → NULL-state DP → GeM+CLS ensembling polish.
**Cut first if behind** (within the default scope): stratum-H anchors beyond 4 → anchors 30 → 20. The rehearse
slot shrinks last, not first — an answer you can't defend at interview costs more than a smaller anchor set. Never
cut: the T1 table, DTW v1 with abstention, the deletion test, the reliability table, `test_align.py`, the
ground-truth-independence caveat in the report.

## 13. Risks and the proved-vs-assumed ledger

Risks: torch/MPS install trouble (fallback: CPU, ViT-S only, or S1 alone); rain defeats SIFT (say so; rely on
agreement cues); H1 head-offset instead of tail truncation (only shifts time labels; the differential cam0/cam5
test bounds it); label noise on plateaus (report ranges); loop-closure corners (sequence constraint + margin band).

| claim | status | evidence |
|---|---|---|
| actual rate ≈ 9.99 fps | proved | timestamp arithmetic, 100.000 ms grid |
| the filename's "20" is the frame rate | rejected | contradicted by VUI and timestamps; semantics unknown |
| frame k ↔ timestamp line k (H0) | assumed, evidenced | intact head, mid-slice tail at 256 KiB, synchronous stop/onset, cam-only path agreement |
| cam0/cam5 index-synchronous | tested | stop/onset events; differential path test |
| local time UTC+8 | inferred | scene content (Petaling Jaya banners) |
| runA 2590–2673 has no runB counterpart | evidenced, to be labelled | probe + tail anchors |
| confidence tiers are calibrated probabilities | not claimed | n ≈ 24 anchors; ordinal tiers + reliability table instead |
| user and Claude are independent ground-truth labellers | rejected | Claude's visual read can share failure modes with the DINOv2 descriptor in the pipeline; reframed as user-primary + Claude QA pass, agreement reported as a sanity check, not inter-rater reliability |

## 14. Interview questions to prepare (one-line answers)

1. Why 10 fps when the filename says 20? — Timestamps measure 100.000 ms intervals and the SPS VUI declares 10; the
   "20" has no verifiable meaning in the data; I report the measured 9.99 and list all three declarations.
2. How do you know timestamp line k is decoded frame k? — H0 with evidence: intact headers at byte 0, tail cut
   mid-slice on a 256 KiB boundary, synchronous stop events in both cameras, cam0-only vs cam5-only path agreement;
   residual uncertainty flagged as `ts_confidence=assumed` on the tail.
3. Why not nearest-neighbour retrieval? — Loop closure and uniform stretches make single-frame retrieval alias (the
   corner blocks); the order constraint turns a global ambiguity into a local one; the ablation quantifies it.
4. Doesn't DTW assume a fixed or linear offset? — No: a non-parametric monotone warp with free slope, open ends and
   explicit stationary steps.
5. DTW forces a match everywhere — how do you abstain? — Post-hoc rejection on ridge strength and agreement,
   thresholds set by the deletion test (v1, shipped, measured with precision/recall). NULL-state DP with affine
   gaps (v2) is a cleaner formulation I scoped out given the time budget — future work, not shipped.
6. How does a confident wrong match arise and how do you catch it? — Repeated façades, hedges, and the depot at
   both ends; caught by the margin outside the path band, cam0/cam5 and SeqSLAM/DINOv2 disagreement, optionally
   geometric verification; residual failures are shown, not hidden.
7. What does confidence 0.9 mean? — An ordinal tier whose measured accuracy on held-out anchors is in the
   reliability table; not a calibrated probability at n ≈ 24.
8. Why DINOv2 and not a VPR-trained model? — One-line pretrained model with the strongest generic features under
   appearance change; the ablation shows what it adds over SeqSLAM; EigenPlaces/AnyLoc-VLAD are the next step.
9. Why two cameras, and why do side cameras matter? — Independent views of the same position give an agreement cue;
   side views discriminate position finely but suffer lane offset and, on cam5, traffic; fused only after the sync test.
10. What is your ground truth and how noisy is it? — 30 anchors I labelled blind to model output, with ranges;
    Claude did a second blind pass as a QA check afterward — I report that agreement as a sanity number, not
    inter-rater reliability, since we're not independent judges; Wilson CIs on the accuracy estimate itself.
11. Where is the mapping least reliable? — Start plateau (many-to-one onto moving runB), runA tail beyond runB's
    truncated end, low-texture stretches, rain-degraded frames, cam5 during passing traffic.
12. What can be wrong in the Task-1 numbers? — Times depend on the recorder clock; counts depend on how this ffmpeg
    version handles a truncated tail (coded == decoded verified); colour range is unsignalled, not "tv".
13. How would you split this data for training? — By `place_id` derived from the correspondence, both runs
    together, contiguous blocks with buffers; never random frames.
14. B-frames and ordering? — ffmpeg emits display order; counts via `-count_frames`; decode order is irrelevant to
    the mapping.
15. With another week? — NULL-state affine-gap DP (cut for time, §6), odometry/GNSS as a prior, local-feature
    re-ranking (LightGlue/SIFT, cut for time, §7), more anchors and a genuinely independent second human labeller,
    a per-camera occlusion detector, banded/coarse-to-fine DTW for hour-long logs.
