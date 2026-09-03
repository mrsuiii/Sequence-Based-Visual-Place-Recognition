# findings.md — every reported number, how it was obtained, and the exact command

Format per entry: **claim** — value — command / function — note.

## Step 0 — Setup & integrity (2026-08-30)

**Tooling** — ffmpeg/ffprobe 8.1.1 (Homebrew, `/opt/homebrew/bin`), built with `libx265` (can decode
HEVC) and `--enable-ffplay` — `ffmpeg -version`. Used via `subprocess` only; PyAV was tried and
dropped (see `decisions.md` — actually: see the `av`/`opencv-python` conflict note below).

**Why video I/O never uses PyAV** — `opencv-python` and `av` (PyAV) each bundle their own copy of
the ffmpeg shared libs; importing both in one process triggered a macOS ObjC `libavdevice`
class-duplication warning ("may cause spurious casting failures and mysterious crashes") —
observed directly when both were installed together in this env. Fix: `av` was never installed;
all video I/O goes through the system `ffmpeg`/`ffprobe` binaries via `subprocess`
(`io_video.py`), `opencv-python` is used only for array-level operations, never to open a file.

**All 8 dataset files match `dataset/SHA256SUMS.txt`** — 8/8 OK — `python -m routealign verify-data`
(`io_video.sha256`, streamed SHA-256, never loads a full file into RAM) — result:
`outputs/inspection/integrity.json`.

**Decoded frame count (measured, full decode)** — runA cam0 2674, runA cam5 2674, runB cam0 2622,
runB cam5 2615 — `python -m routealign decode`, which calls `io_video.count_decoded` (
`ffprobe -count_frames -select_streams v:0 -show_entries stream=nb_read_frames`) and separately
`io_video.decode_to_jpegs` (a real `ffmpeg` decode pass); the two counts are compared and matched
exactly for all four files, so this number is cross-checked by two independent invocations, not
just asserted. Cache: `data/cache/frames/{run}/{cam}/00000.jpg` .. (0-indexed, `-start_number 0`).

**Timestamp line count vs decoded frame count disagree** — runA: 2695 lines vs 2674 decoded (21
short, same on both cameras). runB: 2663 lines vs 2622 (cam0, 41 short) / 2615 (cam5, 48 short) —
`wc -l dataset/{run}/*.timestamps.txt` vs the decoded counts above. Not yet root-caused in code
(the file-size-multiple-of-256-KiB / mid-slice-truncation hypothesis from the prior exploratory
session is plausible — see `plan.md` §1 — but has not been re-derived by this repo's own code yet;
treat as **assumed, not measured**, until Step 1's `scan_nals` reproduces it).

**Native resolution** — 1440×1080 (`pix_fmt=yuv420p`) — `ffprobe -show_entries stream=width,height`
on `dataset/runA/cam0_20_yuv420p_output.hevc`, cross-checked against a single decoded frame's
shape. **This is a declared value** (from the HEVC SPS) that happens to be very reliable in
practice — decode fails outright if it is wrong — but has not yet been independently re-derived
from the raw NAL/SPS bytes by this repo's own code (Step 1: `io_video.sps_vui`).

**Declared frame rate is inconsistent across sources** — filename token: `20`; `ffprobe
r_frame_rate`: `10/1`; `ffprobe avg_frame_rate`: `25/1`. All three are declared, not measured, and
they disagree with each other by up to 2.5x. None is reported as the actual frame rate; the
measured rate must come from `timestamps.txt` interval arithmetic (Step 1, not yet computed by
this repo's code).

**JPEG working-cache size (640×480, q=3, measured)** — 323 MB total (106 + 98 + 60 + 59 MB per
run/camera) — `du -sh data/cache/frames/*/*`. Smaller than `plan.md`'s ~640 MB estimate.

**First frame visual check (measured from decoded pixels, one frame per file)** — runA frames
(cam0, cam5) show dry/sunny conditions; runB frames show visibly darker, overcast conditions with
water droplets on the lens/windshield in both cameras. This is a qualitative visual observation,
not a quantitative one yet (no mean-brightness or droplet-detection code exists in this repo —
Step 2's `motion_energy`/contact-sheet work is where that would be measured properly).

## Step 1 — Task 1 characterisation (2026-08-30)

`python -m routealign characterise` → `outputs/inspection/{run}_{cam}.json` +
`outputs/inspection/task1_table.md` + `outputs/inspection/figures/*.png`. Below: the headline
numbers, each already tagged declared/measured in the JSON itself; here only the ones worth
calling out.

**Resolution — declared, cross-checked two ways** — 1440×1080 for all 4 files. SPS coded size
1440×1088 (`pic_width_in_luma_samples` / `pic_height_in_luma_samples`, via
`ffmpeg -bsf:v trace_headers`) with `conformance_window_flag=1`, `conf_win_bottom_offset=4`; for
4:2:0 chroma the crop is in chroma-sample units (SubWidthC=SubHeightC=2), so
1088 − 2×(0+4) = **1080**. Matches `ffprobe`'s own reported 1440×1080 exactly (`io_video.probe` vs
`io_video.sps_vui`, two independent parses agreeing).

**Frame rate — one real declared value, two candidate labels investigated and rejected, one
measured value.** Three numbers were found in/near the file before any investigation: filename
token `20`, `ffprobe r_frame_rate=10/1`, `ffprobe avg_frame_rate=25/1`. Only one of these traces to
an actual field describing this file's content:

- **Declared nominal fps = 10.00** — SPS/VUI `time_scale/num_units_in_tick = 10/1`
  (`vui_timing_info_present_flag=1`, via `ffmpeg -bsf:v trace_headers`), independently confirmed
  against ffprobe's `r_frame_rate=10/1` — two different parses of the same underlying field agree
  exactly. This is what `io_video.sps_vui` / `_cmd_characterise` now report as `declared_nominal_fps`.
- **Filename token `20` — rejected.** No field anywhere in the file confirms it represents fps; it
  sits in a position that's *suggestive* of an fps tag, but that is not evidence. Not reported as a
  competing declared value; kept as a documented, explicitly-rejected candidate (see
  `other_labels_investigated_and_rejected` in each JSON and the "Other frame-rate labels" table in
  `task1_table.md`), not silently dropped.
- **`ffprobe avg_frame_rate=25/1` — rejected, and now proven, not just argued.** Encoded a synthetic
  1-second raw HEVC test clip (`ffmpeg -f lavfi -i testsrc=duration=1:rate=17 ... -f hevc`) at a
  **known true rate of 17 fps**. `ffprobe` on that file reports `r_frame_rate=17/1` (correct — reads
  the real VUI) but `avg_frame_rate=25/1` (wrong — identical fallback value regardless of the true
  rate). This confirms `avg_frame_rate` is a content-independent default for headerless elementary
  streams (no `duration`/`nb_frames` for ffmpeg to compute a real average from), not a declaration
  about this file specifically. Test file discarded after the check (not part of the dataset).

Measured: **9.992 fps (runA), 9.989 fps (runB)** — from `(decoded_count − 1) / video_covered_span_s`,
itself from `timestamps.txt` (median interval ≈99.95 ms). Reported alongside, never merged into,
`declared_nominal_fps` — the two are close (9.99 vs 10.00) but that closeness is itself a measured
finding, not something assumed going in.

**Route time / recording time (measured)** — runA: video-covered span 267.50 s (full timestamp
span 269.59 s), first timestamp 2025-02-28T06:24:09.58 UTC. runB: video-covered span 262.40 s
(cam0) / 261.69 s (cam5) (full span 266.49 s), first timestamp 2024-04-13T09:12:41.98 UTC — runB
is 321 days *earlier* than runA. Local-timezone attribution (UTC+8, from a scene-text banner
inference in the prior exploratory session) is **not independently re-verified by this repo's
code** — no OCR/scene-text step exists here; treat as an open item, not a confirmed fact.

**Δt distribution (measured)** — median ≈99.95 ms, std ≈4.7 ms (runA/cam0); one gap >150 ms per
file (runA idx 4→5 = 300.82 ms, runB idx 5→6 ≈397 ms — both near the very start, both while the
vehicle is plausibly still stationary — see `outputs/inspection/figures/*_dt_vs_index.png`, which
show a single spike near index 0 and a flat, tight band everywhere else — no mid-route
irregularities in any of the 4 files). Full percentiles/outlier list/histogram in each JSON and PNG.

**Two numbers disagree with the prior exploratory session's informal claims — flagged, not
silently reconciled:**
- **GOP size measured as 250, not 240.** `scan_nals` (via `ffprobe -show_frames`) finds 11 IDR
  frames at indices `[0, 250, 500, ..., 2500]` for runA/cam0 — evenly spaced 250 apart, not 240.
  240 may have been a rough mental estimate (2674/11 ≈ 243, still not 240) in the earlier session;
  250 is what this repo's code actually measures and can reproduce.
- **Lag-1 autocorrelation of Δt measured as −0.137 (runA/cam0), not −0.42 / −0.46.** Computed as
  `np.corrcoef(dt_ms[:-1], dt_ms[1:])[0,1]` over the full interval array. Both signs agree
  (negative — some self-correcting jitter around the trigger grid), but the magnitude does not.
  The prior session's exact method for this number isn't documented step-by-step, so the
  discrepancy isn't root-caused; this repo's number is the one with runnable code behind it and is
  treated as authoritative going forward.

**Frame-count discrepancy — reconfirmed independently, still not root-caused at the byte level** —
lines−decoded: runA 21/21 (cam0/cam5, identical), runB 41/48 (cam0/cam5, **not** identical this
time either — reconfirmed). All 4 file sizes are exact multiples of 262144 B (256 KiB) —
`file_size_mod_262144: 0` in every JSON, now computed by `scan_nals`, not just a one-off shell
check. This is consistent with a block-buffered writer stopping without a final flush, but it is
still circumstantial: decoding at `-v warning` produces no truncation/EOF warning either way (see
the Step 0 entry above), and no byte-level Annex-B NAL scan has been done in this repo to directly
observe an incomplete final NAL unit. Treat the "writer stopped mid-slice" explanation as a
plausible, evidenced-but-unproven hypothesis, not a measured fact.

**Not yet done in Step 1** — per-gap frame strips (visual context around each Δt outlier) and a
full-resolution last-decoded-frame truncation check (both listed in `plan.md` §3) were scoped out
of this pass in favour of the two core figures (Δt histogram, Δt vs. index) and the byte-level NAL
scan discussed above; revisit if there's time before the report is finalised.

## Step 2 — Preprocessing & sync (2026-08-31)

`FrameStore` (JPEG-cache read-through, LRU cache, row crop), `motion_energy` (mean |Δ| on
`thumb_w`×`thumb_h`-downsampled gray, via `cv2.resize(..., INTER_AREA)` + `cv2.absdiff`) and
`stationary_segments` (both cameras below `threshold_frac` of their own median, run length
≥ `min_length`) implemented in `src/routealign/frames.py`, written primarily by the user with
review/debugging support.

**Stationary segment measured for runA (cam0 + cam5, first 600 frames, `threshold_frac=0.15`,
`min_length=10`): (0, 173)** — matches the prior exploratory session's informal claim ("runA
0–173, cam0") exactly. Unlike the two Step 1 numbers that did *not* reproduce (GOP size,
autocorrelation), this one does — worth noting precisely because it shows the earlier session
wasn't uniformly unreliable, just not uniformly verified either; each claim still needs its own
check. This number came from an ad-hoc verification script during development, not yet a
reproducible `make` target — Step 2 has no CLI subcommand of its own (`frames.py` is a utility
module consumed by Step 3+, per CLAUDE.md's dependency graph); it becomes reproducible once
`describe`/`align` actually call these functions.

**A real bug caught and fixed during development, not just a style note**: an early draft called
`cv2.absdiff()` with a single (already-subtracted) array instead of two source arrays — a
`TypeError` at the exact call site the `uint8`-wraparound gotcha warns about, confirmed by running
it (`cv2.error: absdiff() missing required argument 'src2'`). Fixed to
`cv2.absdiff(gray1, gray0)`, letting OpenCV do the subtraction itself.

**Not yet done in Step 2** — the static-mask sanity figure (per-pixel temporal std, `plan.md` §4)
and the cam0/cam5 sync test (needs Step 4's alignment paths to compare) are both deferred.

**Threshold-method caveat, worth remembering before reusing this code**: `threshold_frac × median`
assumes the *majority* of the array represents normal (moving) motion. Tested this deliberately on
a 600-frame window rather than a short early-frame slice — a small window landing mostly inside
the stationary period makes the median itself low, breaking the threshold (confirmed by a failed
first test attempt: a 60-frame window returned no segments at all, because ~all of it was still
inside the stationary period). Always run this over a large enough span that "moving" is the
majority, not a short arbitrary slice.

## Step 3 — Representation, R1 SeqSLAM (2026-08-31)

`seqslam_descriptors` implemented in `src/routealign/descriptors.py` (thumb_w/thumb_h/patch taken
as explicit params, not hardcoded, matching `motion_energy`'s pattern -- callers fill them from
config.yaml `descriptors.seqslam.*`). Non-overlapping patch normalisation done via a
`reshape(h//patch, patch, w//patch, patch)` + reduce-over-axes-(1,3) trick (vectorised, no
per-patch Python loop).

**Verified on real data**: cosine similarity 0.997–0.999 between runA frames 0, 1, 5 (all inside
the confirmed stationary run `(0, 173)` from Step 2 -- same physical place) vs 0.158–0.213 against
frames 1000 and 2000 (different points on the route). The descriptor separates same-place from
different-place frames correctly using nothing but raw cosine similarity, before any alignment
algorithm exists yet -- a good sign for Step 4. R2 (DINOv2) is in progress (user-led).

## Step 3 — Similarity, diagnostics, and R2 status (2026-08-31)

**R2 (DINOv2)**: `dino_descriptors` + `_gem_pool` implemented (user-led, with review support).
`_gem_pool` verified in isolation on synthetic data with strongly negative values (min -10.17) --
no NaN/Inf, confirming the `eps`-clamp-before-power fix is necessary and correct (patch tokens
post-LayerNorm can be negative; a fractional power of a negative mean is undefined). The full
function has **not** been run end-to-end on real data in this repo: `torch.hub.load`'s GitHub
repo-code fetch (`codeload.github.com`) returns `400: Bad Request` in the assistant's sandboxed
shell -- reproduced with both Python's `urllib` and a bare `curl` request, so this is a network
allowlist restriction of the sandbox, not a torch.hub or DINOv2 hosting bug (the weight file
itself, hosted separately at `dl.fbaipublicfiles.com`, downloaded successfully: 88 283 115 bytes).
**Not yet verified**: that `forward_features()`'s returned dict actually uses the key names
`x_norm_clstoken`/`x_norm_patchtokens` assumed in the code -- this is standard, well-documented
DINOv2 API from training-era knowledge, not something clicked-tested this session. The user should
confirm both (a) that `torch.hub.load` succeeds in their own terminal, and (b) the exact dict keys,
before trusting real R2 output.

**Similarity module (`similarity.py`)** -- `cosine`, `local_contrast_norm`, `fuse` -- implemented
and run for real: full runA vs.\ full runB, cam0, SeqSLAM descriptors (2674×2622 matrix).
`cosine`: 0.19 s. `local_contrast_norm` (window=50, via two `uniform_filter1d` passes rather than
a per-column loop): 0.10 s. Both comfortably fast at full scale.

**Diagnostic figure** (`viz.plot_similarity_matrix`, generic over what path is overlaid --
reused later for the real DTW path): plotted with the *unconstrained* per-row argmax (no
monotonicity constraint) over the z-scored similarity matrix --
`outputs/inspection/figures/step3_similarity_argmax_diagnostic.png`. Two things are visible
directly in real data, not just predicted from theory: (1) a genuine diagonal ridge (runA~700
to~2000 tracks runB~0 to~1700; a second diagonal segment covers the tail), confirming the SeqSLAM
descriptor does carry real place-matching signal; (2) heavy horizontal banding, worst at the very
top and bottom rows (runA near 0 and near its end) where a single runB column attracts the argmax
from many unrelated runA rows -- exactly the "corner aliasing" plan.md predicted from the shared
depot at both ends of the loop, now shown, not assumed. This is the concrete, in-data justification
for needing Step 4's monotonicity constraint rather than per-frame nearest-neighbour matching.

## Step 4 — DTW (`dtw_open_ends`) (2026-08-31)

Implemented as `_dtw_forward` (vectorised, the cumulative-min trick from plan.md §6) +
`_dtw_backtrack`, composed by `dtw_open_ends`. `tests/test_align.py`: 21/21 pass --
`test_dtw_matches_naive_reference` (D matrix and path exactly match an independent triple-loop
reference, `rtol=1e-10`, across sizes 1x1 to 40x30 and all three plan.md `lam` candidates),
structural checks (monotone i and j, every runA row covered), and a known-answer plateau test.
The known-answer test **failed on its first version** -- not a bug in `dtw_open_ends` (the
naive-reference test already agreed with it exactly), but a flawed test construction: the
plateau's exit column jumped back to the original `i==j` diagonal, which is not actually reachable
cheaply, so the "expected" path in the test was never the true optimum. Fixed by having the
diagonal resume from the plateau's exit column. Left in `decisions.md`-adjacent detail here
because it is a good example of the same discipline this project has applied to the prior
session's claims, now applied to a test the assistant wrote in this session.

**Real data**: full runA vs.\ full runB, cam0, SeqSLAM, `local_contrast_norm(window=50)`,
`cost = 3 - clip(S, -3, 3)`, `lam=0.5` -- **0.06 s** (plan.md's target was `< 1 s`). Path covers
runA 0..2673 completely; open begin lands at runB frame 78 (not 0), open end at runB frame 2601
(not 2621) -- both plausible given runA/runB's differing route coverage. Plotted
(`outputs/inspection/figures/step4_dtw_path_diagnostic.png`) against the same matrix as the
unconstrained-argmax diagnostic: the DTW path is a single continuous, mostly-smooth curve from
(0,78) to (2673,2601), a stark contrast to the earlier plot's horizontal aliasing bands -- direct,
in-data confirmation that the monotonicity constraint fixes the corner-aliasing problem it was
meant to fix, not just in theory.

**Not yet done**: `local_slope` remains a stub; v2 (NULL-state DP) stays cut by default per the
standing revision note.

### `path_to_mapping` (2026-08-31)

Implemented in `align.py`: groups the DTW path by runA row (rows are visited contiguously since
`i` is non-decreasing in the path, found via `np.searchsorted`), picks `runB_frame` as the
highest-similarity column within the row's raw visited `[lo, hi]` range, then widens `[lo, hi]`
to the runB stationary segment containing `runB_frame` if any. `boundary_clamped` is computed
from the *raw* (pre-widening) visited range, so a legitimate stationary-segment widening that
happens to touch column 0 or `NB-1` is not mistaken for a genuinely clamped/unmatched tail.

`tests/test_align.py`: 3 new tests (24/24 total green) -- a plain diagonal path (degenerate
`lo == hi == runA_frame`, only the first/last row `boundary_clamped`), a horizontal-step case
(one runA row visiting several runB columns, checks the highest-similarity one is picked and
`lo`/`hi` span the full visited range), and a stationary-widening case (best match inside a
runB stationary segment, `lo`/`hi` widen to the segment but `boundary_clamped` stays keyed to the
raw range).

Run for real on the cached full-scale cam0 SeqSLAM DTW path (`path` from the entry above): rows
0-5 all widen to `[0, 173]` (runA's own stationary segment's runB counterpart, confirming the
widening logic engages correctly on real data), the tail (runA 2664-2673) lands on genuine
one-frame matches (`lo == hi`, e.g. runB 2600/2601), and **zero rows are `boundary_clamped`** --
runA and runB happen to be close enough in route coverage here that DTW found real matches all
the way to both open ends, not an artifact of the widening logic hiding a clamped tail.

### `local_slope` (2026-08-31)

Implemented in `align.py`: least-squares slope of `j` (runB) vs.\ `i` (runA) over a
`window`-point window centred on each path point, clipped at the open start/end so the window
narrows there rather than reading past the path. Falls back to `inf` when the window's `i` never
changes (a horizontal run -- runA idle -- wider than `window`) but `j` does, since the
least-squares slope is undefined (zero variance in `i`) but the correct qualitative answer is "an
unbounded runB-per-runA rate", not silently 0 or NaN.

`tests/test_align.py`: 3 new tests (27/27 total green) -- a plain 1-for-1 diagonal recovers slope
1.0 everywhere including clipped edge windows; a synthetic constant-rate-2.0 path (`j = 2*i`)
recovers exactly 2.0 in the interior; a fully flat-`i` path (runA stuck at row 0, `j` sweeping)
gives `inf` at every interior point.

Run for real on the cached full cam0 SeqSLAM DTW path at `window=21`: slope range `[0, 4]`,
median 1.0 (most of the route tracks at the same pace both runs), no `inf` (no single horizontal
run in this data is 21+ points wide). Slope is near 0 across runA rows 0-14, the leading edge of
the shared runA/runB stationary segment -- both runs are idling together there, so `i` still
advances one row per runA frame (there is always a next runA frame to look at) while `j` barely
moves (the path mostly takes vertical steps, staying near column 78-79), which is exactly what
"near-zero runB frames per runA frame" should look like. This closes out Step 4; only v2
(NULL-state DP, cut by default) remains unbuilt in `align.py`.

## Step 5 — Verification & confidence, assistant-led half (2026-08-31)

Split with the user per this session's established pattern: assistant took `path_disagreement`
(`verify.py`) and `fuse_confidence` (`confidence.py`); `ridge_strength`/`margin` (`verify.py`)
are user-led and still stubs.

**`path_disagreement`**: reduces a DTW path to one runB column per runA row via a private
`_path_to_row_j` helper (same `np.searchsorted`-on-sorted-`i` grouping trick as
`align.path_to_mapping`, deliberately re-implemented rather than imported -- CLAUDE.md §3 keeps
`verify.py` and `align.py` at the same dependency tier, neither may import the other), taking
the *last*-visited column per row (matches the convention `tests/test_align.py`'s plateau test
already established for "the" column of a multi-column row). Disagreement is then a plain
`|j_a - j_b|`. Raises `ValueError` if the two paths cover a different number of runA rows (e.g.
cam0 and cam5 decoded to different counts, per the H0 caveat) rather than silently truncating.
3 tests green (`tests/test_verify.py`).

**`fuse_confidence`**: implements plan.md §7's no-match rule and 4-tier confidence rule exactly,
with one documented, tested adaptation -- the tier-0.7 rule's "2 of {cam <=5, desc <=5, SIFT
verified}" vote degrades to "both of {cam <=5, desc <=5}" when the cues frame has no
`sift_inliers` column, since SIFT is cut by default (plan.md §7 revision note). All six numeric
thresholds (`tau`, `ridge_high`, `ridge_mid`, `disagree_tight`, `disagree_loose`,
`disagree_no_match`, `boundary_ridge_min`) are named function parameters with `# why:` comments
citing plan.md §7/config.yaml, not literals buried in the function body (CLAUDE.md §4) -- the
CLI layer will wire these from `config.yaml`'s `confidence.no_match_tau_default` /
`confidence.tiers` once Step 5's CLI subcommand is built (not yet done). No-match always wins
over every tier rule regardless of how strong the other cues look (tested directly: a row with
`ridge_z=3.0` but `no_video=True` still comes out `confidence=NaN, status="no_video"`). 8 tests
green (`tests/test_confidence.py`): missing-column validation, each tier reachable on its own,
the SIFT-recovers-a-vote case, each of the four no-match triggers independently, and the
no-match-overrides-top-tier precedence check.

**Not yet done**: `ridge_strength`, `margin` (user-led, in progress); SIFT (`sift_verify`,
`reliability_table`) stay cut by default / Step 6 respectively per the standing plan.

### `ridge_strength` (2026-08-31)

User's first draft averaged `similarity[i-window:i+window+1, j]` -- a **fixed column** `j` (row
`i`'s own path column) with a sliding **row** range. Two bugs caught in review before any code
ran: (1) output length was `len(path)` (2828 on real data) instead of `NA` (2674), since the loop
ran once per path point rather than once per runA row, silently producing duplicate/misaligned
entries wherever the path took a horizontal step; (2) semantically, holding `j` fixed checks
"does this one runB frame also match nearby runA rows", not "does similarity stay high along the
path's own diagonal trajectory" -- the DTW path is diagonal, not vertical, so a true ridge check
needs each neighbouring row's *own* path column (`row_j[i']`), not row `i`'s column reused.

Fixed: reuses `_path_to_row_j` (promoted from `path_disagreement`'s private helper, now shared by
both) to get one column per runA row first, then for row `i` averages
`similarity[i', row_j[i']]` over `i'` in the clipped `[i-window, i+window]` window -- a paired
fancy-index (`similarity[rows_array, cols_array]`), not a submatrix slice. 3 new tests green
(`tests/test_verify.py`, 6/6 in that file): a clean diagonal gives exactly 1.0 at every window
size; output length is `n_a` even when the path is longer than `n_a`; a single strong path point
surrounded by weak neighbours is pulled down to the correct fractional mean (`1/5`), demonstrating
the "forced through" case the feature exists to catch.

Run for real on the cached cam0 SeqSLAM path/similarity at `window=5`: output shape `(2674,)`
matches `NA` exactly. Values range `[-0.48, 3.0]` (`Sz` is clipped to `±3` by
`local_contrast_norm`, so 3.0 is the ceiling). Weaker (~1.87-1.90) across the shared stationary
segment (rows 0-10) -- consistent with a real but less sharply-peaked match when the scene isn't
changing -- and saturates at exactly 3.0 mid-route (rows 1006-1009), i.e. a run of runA frames
whose entire window sits on the clip ceiling: a strong, unambiguous ridge.

### `margin` (2026-08-31)

Implemented in `verify.py`: `_path_to_row_j` gives the claimed match column per row; the
runner-up search masks out `[j-band, j+band]` with `np.where(in_band, -np.inf, similarity)` and
takes the row max, fully vectorised (`NA x NB` boolean mask, no Python loop). Guards
`2*band + 1 >= NB` up front and raises `ValueError` -- the earlier looser guard suggested in
review (`band >= NB`) was insufficient: a claimed match near the *middle* of a row can have its
whole neighbourhood masked out well before `band` reaches `NB`, so the check has to bound the
worst case (`2*band+1`), not just `band` alone. 4 tests green (`tests/test_verify.py`, 10/10 in
that file): a far runner-up is found and a near one correctly ignored, a band clipped at the
matrix edge still works with no special-casing, output length is `NA` not `len(path)`, and the
too-wide-band case raises.

**Real data, `band=30`** (plan.md §7's value): shape correct (2674,), but **2660/2674 rows
(99.5 %) have margin < 0.5**, most negative -- e.g. row 1608: claimed match `Sz=-1.618`, but
column 1858 (far outside the band) scores the clip ceiling `Sz=3.000`. This was checked, not
assumed to be a bug: `Sz[:, 1858]` for rows 1600-1616 is elevated across *nearly the whole
neighbourhood* (0.6-3.0, mostly >2), not a one-off spike at row 1608 alone -- this is a genuine
**attractor column**, the same phenomenon already noted in the Step 3 findings entry (columns
793/794/848/884/885, mean similarity ~1.0-1.09) recurring elsewhere in the matrix (column 1858
and evidently others -- the 15 worst-margin rows checked each had a *different* far-away runner-up
column, not a repeat of the same handful).

**Root cause, not a `margin` bug**: `local_contrast_norm` z-scores each row against a *local*
window of nearby columns (`similarity.py` docstring, confirmed by reading the code just now).
Wherever that local window happens to have low raw-similarity variance (e.g. runB passes a
low-texture stretch -- plain road, sky, motion blur), the `+ _EPS` floor cannot fully stop a
mediocre absolute similarity from being amplified into a near-ceiling z-score, because it is only
*locally* mediocre, not globally. A single such column then reads as a strong "match" against
many unrelated runA rows at once -- exactly what `margin` is supposed to catch, and exactly why
plan.md scopes `margin`/`ridge_z` as raw features to be *tuned and combined* in Step 6 (synthetic
warps + deletion test), not trusted at face value yet. `margin`'s implementation is verified
correct; this result is evidence for the report's failure-case section and for why `fuse_confidence`
leans on multiple independent cues (`ridge_z`, `cam_disagree`, `desc_disagree`) rather than
`margin` alone -- a row can have a terrible raw margin from an attractor column elsewhere in the
matrix while still having a strong, consistent `ridge_z` along its own path (as rows 1006-1009
showed in the `ridge_strength` entry above), which is the situation the multi-cue design exists
for. Not fixed here -- `local_contrast_norm`'s window/`_EPS` are Step 3-tuning decisions, out of
scope for a Step 5 feature function, and any change there needs the synthetic-warp test harness
(not yet built) to check it doesn't also suppress genuine ridges.

This closes out Step 5's feature functions (`ridge_strength`, `margin`, `path_disagreement`,
`fuse_confidence`); `sift_verify` stays cut by default; `reliability_table` is Step 6.

## Step 5 CLI wiring — `describe` / `align` / `verify`, real `outputs/mapping.csv` (2026-08-31)

Closed out Step 5's last checkbox: wired the four already-tested feature functions
(`ridge_strength`, `margin`, `path_disagreement`, `fuse_confidence`) plus everything from Steps
1-4 into three new CLI subcommands and produced a real, complete `outputs/mapping.csv`.

**New/changed code**: `descriptors.cached` implemented (was `NotImplementedError`) with a
signature tweak (`extra: dict | None` param) so its `.json` sidecar can actually hold real
provenance (params, run/cam/n), not just what the function could infer on its own — 7 tests
green (`tests/test_descriptors.py`). `confidence.fuse_confidence` gained a `reason` column
(priority-ordered explanation of which no-match trigger fired; empty for matched/ambiguous
rows) -- 2 new tests, existing tests extended to check it, still all green. `cli.py` gained
`_cmd_describe`, `_cmd_align`, `_cmd_verify`, `_open_store`, `_seqslam_desc`, `_iso_utc`, and
`describe`/`align`/`verify` argparse dispatch. `config.yaml` gained
`similarity.contrast_window_default` (50), `similarity.camera_weights`, `verify.ridge_window`
(5), `verify.margin_band` (30), `align.local_slope_window` (21) -- concrete working defaults so
the CLI has something to run with today, each flagged as pending Step 6 synthetic-warp tuning.

**Real run, full scale, no `--limit`** (smoke-tested first at `--limit 200` -- `--limit 60` was
tried first and correctly rejected by `margin`'s own guard, `2*30+1=61 >= NB=60`, confirming the
guard works on a real edge case, not just the synthetic test): `describe --feat seqslam` 18 s,
`align` 6.9 s, `verify` 2.4 s. `outputs/mapping.csv`: **2695 rows** -- exactly matches CLAUDE.md
§7's "definition of done" row count, itself a real cross-check (this number was never hardcoded
anywhere; it falls out of `runA`'s real timestamp file length). `status: matched=2445
ambiguous_range=123 no_match=106 no_video=21`. The `no_video` count independently reproduces the
Step 0/1 timestamp/decode discrepancy (2695 - 2674 = 21) via a completely different code path
(H0 tail construction in `_cmd_verify`, not `characterise`) -- asserted directly in
`tests/test_mapping_csv.py::test_no_video_rows_match_the_measured_timestamp_decode_discrepancy`.

**Schema verification** (`tests/test_mapping_csv.py`, 7 tests, run against the real generated
file -- not skipped): every runA frame appears exactly once 0..2694; all non-null `confidence`
in `[0, 1]`; `runB_frame` empty if and only if `status in {no_match, no_video}`; `runB_frame`
monotone non-decreasing over `matched`/`ambiguous_range` rows; `status` values within the
documented set; `runB_frame_lo <= runB_frame <= runB_frame_hi` wherever present.

**Bug caught before shipping**: the first version left a real (untrusted) `runB_frame` on
`no_match` rows in the final `mapping.csv` -- CLAUDE.md's own test spec ("empty `runB_frame` iff
status in {no_match, no_video}") and plan.md §7's no-match rule ("empty `runB_frame`") both say
this is wrong. Fixed by blanking `runB_frame`/`runB_frame_lo`/`runB_frame_hi`/`runB_time_utc`
for unmatched rows only in the final `mapping.csv` write (`mapping_full.csv` keeps the raw
computed value on purpose, for debugging why a row was abstained). Caught by writing the schema
test against CLAUDE.md's own stated contract rather than against whatever the first
implementation happened to produce.

**Known real gap, not hidden**: `desc_disagree` is empty throughout this run -- it needs a
second, independently-computed descriptor (DINOv2), which is implemented (`dino_descriptors`)
but not verified end-to-end in this environment (`torch.hub` needs `codeload.github.com`,
blocked by this sandbox's network allowlist -- confirmed via both `urllib` and bare `curl`
earlier this session; the weight file at `dl.fbaipublicfiles.com` downloads fine, so this is an
environment restriction, not a code or DINOv2 problem). Consequence, checked directly in the
real output: every row's confidence is 0.4 or 0.5, **none reach 0.7 or 0.9** -- `fuse_confidence`
already treats a missing `desc_disagree` as "fails the agreement check" (no special-casing
needed; `NaN <= x` is `False` in numpy, same mechanism already verified for `sift_inliers` in
Step 5's earlier tests), so this is a correct, honest consequence of shipping a SeqSLAM-only
draft, not a bug -- see `decisions.md` D3. Re-running `describe --feat dinov2` (in an environment
with network access) then `align`/`verify` again will populate it.

Also fixed while wiring `align`: `frames.stationary_segments` returns a Python-slice-style
*exclusive* end, but `align.path_to_mapping`'s `stationary_b` expects *inclusive* `(lo, hi)`
pairs -- a real mismatch between two independently-correct, independently-tested conventions,
caught by re-reading both contracts before combining them (not by a test failure -- each
function's own tests are self-consistent; only the combination was wrong). Converted once,
explicitly, at the `cli.py` call site (`decisions.md` D5). cam0/cam5 fusion also needed handling
runB's differing decoded counts (2622 vs. 2615) -- the shorter camera's matrix is padded with
the `-3` z-score floor rather than truncating both, so real cam0 evidence for the 7 tail columns
cam5 never decoded isn't silently discarded (`decisions.md` D4).

`ruff check`/`ruff format --check` clean on every touched file. 63/63 project tests green
(56 prior + 7 new `test_mapping_csv.py`).

## Step 6 — labelling sheets (2026-08-31)

Implemented `groundtruth.make_sheets` and `groundtruth.stratum_s_indices`, plus a new `gt-sheets`
CLI subcommand and `gt/labelling_protocol.md`. Signature of `make_sheets` widened from the stub's
single `store_b` to all 4 frame stores (`store_a0, store_a5, store_b0, store_b5`) -- the stub's
own docstring already promised "both cameras" and an implicit need to show the labeller the runA
anchor frame, neither possible with one store (decisions.md D6).

`stratum_s_indices`: 18 evenly-spaced sweep points (`np.linspace(200, 2500, 18)`) + seeded ±20
jitter + 4 fixed points (60, 150, 2600, 2650) = 22, matching `config.yaml`'s already-committed
`stratum_s: 22` (plan.md's "~110 frames" prose doesn't arithmetically reconcile to exactly 22
with the 4 fixed points included under a strict reading -- resolved by deriving spacing from the
committed total rather than the reverse; decisions.md D7). 10 tests green
(`tests/test_groundtruth.py`): determinism, different seeds differ, correct count, fixed points
always present, sorted/unique, `make_sheets` writes anchor refs + coarse tiles, handles cam5
being shorter than cam0 without crashing (placeholder tile), no fine sheet without a coarse pick,
fine sheet written and correctly bounded/clipped near the start of runB.

**Real run** (`gt-sheets`, 2.6 s): stratum S anchors = `[60, 150, 183, 346, 476, 602, 738, 891,
994, 1155, 1270, 1400, 1553, 1708, 1833, 1969, 2103, 2241, 2365, 2485, 2600, 2650]` (22, as
expected). Wrote 22 anchor reference tiles (`gt/sheets/anchor_refs/`) + 132 coarse overview
tiles (`gt/sheets/coarse/`, every 20th of runB cam0's 2622 frames -- close to plan.md's "131",
off by one from a slightly different endpoint convention, not a concern). Visually inspected two
tiles directly (`anchor_refs/00476.jpg`, `coarse/00500.jpg`): both cameras side by side, index
burned in top-left in green, legible -- the tool produces a genuinely usable labelling sheet, not
just files that happen to exist.

**Not built yet**: stratum H (needs DINOv2 disagreement for 4 of its 8 anchors -- deferred with
DINOv2 itself, decisions.md D3); fine sheets (need a coarse pick per anchor from the user's first
labelling pass, which hasn't happened); `load_labels`, `evaluate`, `deletion_test` (all of Step
6's analysis half, downstream of real labels existing).

## Ground-truth anchor labelling — Claude-filled, user-verification pending (2026-08-31)

At the user's request, Claude filled all 22 stratum S anchors in `gt/anchors_user.csv` directly
(methodology change from the original user-primary/Claude-QA-after design -- decisions.md D8;
**treat every number below as Claude-labelled, user-verified, not independent ground truth**
until the user's review is done). Method: for each anchor, compared its `anchor_refs/` reference
tile against the `coarse/` overview (132 tiles, every 20th runB frame) by eye, then generated a
targeted fine sheet (`fine_radius=25`) around the best coarse guess and refined from there --
never consulted `outputs/mapping.csv`/`mapping_full.csv`/`mapping_draft.csv` while doing this.

**Result: 22/22 filled.** 11 `sure`, 9 `unsure`, 1 `no_match`, plus anchor 183 marked `unsure`
with an explicit prior-exposure flag (Claude had already compared `runA/00183.jpg` against
`runB/00184.jpg` during the earlier `no_match`-cluster investigation and was not blind for this
one specifically -- see its `note` column).

| runA | runB_best (lo-hi) | quality | why |
|---|---|---|---|
| 60 | 50 (0-100) | sure | stationary block, identical to 150 |
| 150 | 50 (0-100) | sure | identical frame to 60 |
| 183 | -- | unsure* | prior exposure, not a blind read (decisions.md D8) |
| 346 | -- | **no_match** | no matching scene found in coarse or a widened fine search |
| 476 | 405 (390-420) | sure | retaining wall + steps + tree, strong visual match |
| 602 | 500 (480-520) | unsure | plausible area, no distinctive confirming feature |
| 738 | 715 (700-730) | unsure | banana plants + towers match; specific house unconfirmed |
| 891 | 885 (875-895) | sure | house + palms + gate sign, strong match |
| 994 | 940 (900-970) | unsure | rough area only, no distinctive feature |
| 1155 | 1100 (1050-1150) | unsure | generic dense foliage both sides, hard to localise |
| 1270 | 1295 (1280-1310) | sure | fence + banana plants + tower cluster, strong match |
| 1400 | 1380 (1350-1420) | unsure | tower skyline repeats across a wide stretch |
| 1553 | 1500 (1480-1520) | sure | distinctive multi-tower cluster + signalised junction |
| 1708 | 1715 (1700-1730) | unsure | shophouse row plausible, cam0 embankment unconfirmed |
| 1833 | 1810 (1795-1825) | sure | colourful shophouse row, strong match |
| 1969 | 1890 (1870-1910) | unsure | plausible terrace-house match, not confident |
| 2103 | 2095 (2080-2110) | unsure | junction area plausible, billboard not confirmed |
| 2241 | 2210 (2195-2225) | sure | apartment block + matching signage text, strong match |
| 2365 | 2300 (2280-2320) | sure | white warehouse/store + red sign, strong match |
| 2485 | 2500 (2480-2520) | sure | distinctive mansion with tower backdrop, strong match |
| 2600 | 2610 (2590-2622) | sure | same depot as 60/150 -- confirms the route returns to its start |
| 2650 | 2610 (2590-2622) | sure | identical to 2600 |

**Notable finding, worth double-checking**: anchors 60/150 (runA start) and 2600/2650 (runA end)
all point at the *same* depot/carport scene in runB, near both runB's own start and end. This is
consistent with the "attractor column" phenomenon already documented for `margin`
(findings.md, Step 5) -- a genuinely generic-looking parking area that the vehicle passes near
both at the start and the end of the route in at least one of the two runs, which could alias
against unrelated frames project-wide, not just at the two ends. Also consistent with the
earlier, independent finding that runB frame 2586-2600 visually resembles the same carport
(the Step 5 `no_match` cluster investigation).

**`quality=no_match` for anchor 346**: worth a second pass by the user with a wider search
before trusting this as "runB genuinely doesn't cover this stretch" -- the coarse-only scan (20
frames granularity) can miss a match that exists but wasn't sampled, especially for a visually
distinctive but narrow scene (the modern glass-and-concrete house under construction).

**Not done**: precise single-frame refinement for the `unsure` rows (9 of 22) -- the fine sheets
already exist (`gt/sheets/fine_{runA_frame:05d}/`) for the user to inspect directly rather than
re-generating them. `load_labels`, `evaluate`, `reliability_table`, `deletion_test` remain
unbuilt (Step 6, downstream of the user's verification pass).

## Step 6 — `deletion_test`, `load_labels`, `evaluate` (2026-08-31)

Implemented the remaining `groundtruth.py` functions (all except `merge`, ablation, and the
failure taxonomy, which need the user's finalised anchors first). 14 new tests green
(`tests/test_groundtruth.py`, 24/24 in that file, 88/88 project-wide).

**`deletion_test`**: signature widened from the stub's `(runb_delete_range) -> dict` to take the
joint similarity matrix, the original run's `runB_frame` per row, and the DTW/ridge parameters
explicitly (`lam`, `ridge_window`, `tau`) -- the stub's bare signature had no way to actually
access any of the state a real rerun needs. Scoped to test just the `ridge_z < tau` trigger (not
`cam_disagree`/`desc_disagree`, which each need a second camera/descriptor path and aren't
central to "does deleting real data force correct abstention"). Deleted columns are set to the
z-score clip floor (-3.0), the same "no evidence" convention `_cmd_align` already uses for a
camera's missing columns, rather than removed-and-reindexed. 4 tests green: affected rows abstain
(recall), far rows stay matched, `n_affected=0` doesn't crash (NaN, not ZeroDivisionError),
returns the documented keys.

**Real run** (`S_joint.npy`, `mapping_draft.csv`'s `runB_frame`, real `lam`/`ridge_window`/`tau`
from config.yaml, deleting runB `[1500, 1650)` -- matches `verify.deletion_test.window_frames:
150`): **recall = 1.0** (all 133 affected rows correctly abstained -- the no-match rule never
silently keeps a broken correspondence), **precision = 0.56** (105 false positives: rows outside
the deleted span that also abstained). Not a bug -- `ridge_window`'s smoothing and the DTW path
having to reroute around a 150-frame gap both plausibly spill weak-ridge symptoms into rows near
the boundary whose own true match is intact. Net honest read: the no-match rule is *safe*
(recall 1.0 -- it does not miss real breakage) at the cost of being somewhat conservative near a
large disruption (precision 0.56) -- worth stating plainly in the report rather than picking only
the flattering number.

**`load_labels`**: reads whichever of `anchors_user.csv`/`anchors_claude.csv` exist (only
`anchors_user.csv` exists right now -- decisions.md D8), adds a `labeller` column. 3 tests green.

**`evaluate`**: position error (`0` if inside `[lo, hi]`, else distance to nearest end) scored
only where both sides have a real, complete answer; abstention (dis)agreement (TP/FN/FP) tracked
separately so a `no_match` ground-truth row or a pipeline abstention never corrupts position-error
math. Wilson-95%-CI on hit@5 (`_wilson_ci`, safer than a normal approximation at ~20-anchor
sample sizes). Per-confidence-tier breakdown. 8 tests green, including one **real bug caught by
running on real data, not just synthetic tests**: anchor 183's row has `quality="unsure"` but
empty `lo`/`hi` (the labelling fork described a caveated guess in `note` but never filled the
actual numbers) -- `np.median`/`np.percentile` propagate a single `NaN` to their *entire* output,
so this one malformed row silently turned `median_error`/`p90_error` into `NaN` across all 19
scored anchors, not just itself. Fixed by requiring `lo`/`hi` both non-null for a row to be
scored, regardless of what its `quality` string claims -- added as a dedicated regression test
(`test_evaluate_a_row_with_missing_lo_hi_does_not_poison_the_whole_batch`) so this exact failure
mode can't silently return.

**Real run** (current, **not yet user-verified** `gt/anchors_user.csv` x `outputs/mapping.csv` --
these numbers are preliminary and will change once the user finishes checking the 9 `unsure`
rows, anchor 183, and anchor 346): 18/22 anchors scored (183 excluded per the bug above, 346 is
genuinely `no_match` on both sides). `median_error = 14.5` frames, `p90_error = 83.0` frames,
`hit@2/5/10 = 0.28/0.33/0.44`, hit@5 Wilson 95% CI `[0.16, 0.56]` (wide -- small sample, honestly
reported as such). `abstain_false_positive = 2` (ground truth has a real match, pipeline
abstained) worth checking which anchors those are once labels are final. Every scored row is
confidence tier 0.4 -- the reliability table cannot yet show whether higher tiers are actually
more accurate, since no row has reached 0.7/0.9 yet (still blocked on `desc_disagree`/DINOv2,
decisions.md D3). **None of these numbers are final** -- re-run once `gt/anchors_user.csv` is
verified.

**Not done**: `confidence.reliability_table` (still a stub -- likely a thin DataFrame-formatting
wrapper around `evaluate`'s `by_confidence_tier`, deferred rather than duplicating logic across
modules that cannot import each other per CLAUDE.md §3's dependency direction); the `merge`
step proper (moot right now -- only one labeller file exists, decisions.md D8); ablation study;
failure taxonomy figures. All need the user's finalised anchors first.

## Step 5 revisited — DINOv2 fused for real, and a real problem found (2026-08-31)

Sandbox network access to `codeload.github.com` (blocked earlier this session, `decisions.md`
D3) was re-verified and is now open (`curl -I` returns 200; `torch.hub.load` succeeds end to
end). Ran `describe --feat dinov2` for real: 6 min for all 4 run/camera combinations, 768-dim
descriptors (`(2674, 768)` etc.), all cached with real provenance sidecars.

`_cmd_align`/`_cmd_verify` rewritten for two-stage fusion (per-camera-across-descriptors for
`cam_disagree`, per-descriptor-across-cameras for `desc_disagree`, full joint for the mapping) --
`config.yaml` gained `similarity.descriptor_weights` (equal, same rationale as `camera_weights`).
Re-ran `align` (7.6 s, all cache hits) and `verify` (2.3 s) end to end.

**Real result, and a real problem, not just "more numbers"**: `status: matched=1915
ambiguous_range=92 no_match=667 no_video=21` (no_match up from 106 to 667). Investigated before
accepting this as an improvement (declared-vs-measured discipline applies to the assistant's own
new code just as much as to the dataset): `desc_disagree`'s median is 720 columns (out of ~2622)
-- traced directly to `path_dinov2.npy` being almost completely collapsed, stuck at columns
1731-1747 for nearly the whole route, versus `path_seqslam.npy` tracking the diagonal normally.
Root cause: `S_dinov2.npy` column 1747 has the single highest mean z-score of all 2622 columns
(1.70) -- a severe attractor column, the same `local_contrast_norm` low-local-variance
amplification mechanism already documented for SeqSLAM/`margin` (Step 5 above), now much more
severe for DINOv2's smoother semantic embedding space. `S_joint`/`path_joint` were checked
directly and are *not* collapsed (start/end close to the pre-DINOv2 SeqSLAM-only run) --
SeqSLAM's stronger signal dominates the equal-weight fusion, so the underlying match is probably
still mostly reasonable, but `desc_disagree` inherits DINOv2's collapse and over-triggers the
no-match rule (418 of 667 no_match rows cite "cam_disagree and desc_disagree both above 30").
10 rows did reach confidence tier 0.7 for the first time (none reached 0.9 yet) -- real progress,
but dwarfed by the false-positive no_match increase in the current untuned state.

**Decision** (`decisions.md` D9): shipped as-is rather than reverted, with this finding
documented prominently rather than silently accepted or hidden -- `desc_disagree` and the
resulting `no_match` count are **not yet trustworthy** pending proper `local_contrast_norm`
tuning for DINOv2 specifically (needs the synthetic-warp harness, Step 6, not yet built; not
safe to eyeball-fix a weight or window without it). `outputs/SCHEMA.md` updated to reflect this
as the current known limitation, replacing the earlier "`desc_disagree` always empty" one.

## Step 5 revisited again — DINOv2 attractor fixed for real (2026-08-31)

Diagnosed and fixed the D9 attractor problem rather than leaving it as a known limitation, given
the corrected deadline (2026-09-03, plan.md revision) leaves real time for it.

**Diagnosis method**: computed raw cosine similarity for DINOv2 cam0 directly (bypassing the
cache/CLI), applied `local_contrast_norm` at 4 candidate windows (50, 200, 500, 2622/full-row),
ran `dtw_open_ends` on each, and read the resulting path's `runB_frame` at 6 spread rows --
not guessed, not asserted, actually re-run end to end for each candidate:

| window | row 0 | row 500 | row 1000 | row 1500 | row 2000 | row 2673 |
|---|---|---|---|---|---|---|
| 50 (old) | 1731 | 1747 | 1747 | 1747 | 1747 | 1747 |
| 200 | 81 | 406 | 897 | 1402 | 1951 | 2614 |
| 500 | 58 | 407 | 896 | 1402 | 1951 | 2614 |
| 2622 (global) | 1 | 408 | 896 | 1402 | 1952 | 2614 |

`window=50` (SeqSLAM's tuned value, previously shared) collapses; 200/500/2622 all track the
diagonal and closely match SeqSLAM's own independent path (`decisions.md` D10). Chose 500:
empirically indistinguishable from 200/2622 in the tested rows, keeps some locality rather than
going fully global. `config.yaml`: `contrast_window_seqslam: 50`, `contrast_window_dinov2: 500`
(was one shared `contrast_window_default`); `_cmd_align` wired to use the right one per
descriptor. 88/88 tests still green (no test depended on the removed shared key's name).

**Real re-run, full scale**: `align` 7.6 s, `verify` 2.8 s (all cache hits on descriptors --
only similarity/DTW recomputed). `status: matched=2488 ambiguous_range=124 no_match=62
no_video=21` -- `no_match` down from 667 (D9's broken state) to 62, *better* than the
pre-DINOv2 SeqSLAM-only baseline (106) from earlier this session. `desc_disagree`: mean 10.3,
median 1, p75 3 (was mean 768, median 720) -- sane now. Confidence tiers: `{0.4: 558, 0.5: 24,
0.7: 1104, 0.9: 926, NaN: 83}` -- tier 0.9 reached for the first time this session (previously
always 0). All 6 schema properties re-verified directly against the new file.

**Ran `evaluate()` against the current anchors and looked closely before calling this a clean
win**: `by_confidence_tier = {0.4: {n=5, correct=4, acc=0.80}, 0.7: {n=7, correct=1, acc=0.14},
0.9: {n=7, correct=2, acc=0.29}}` -- inverted (higher tier, lower measured accuracy). Inspected
the individual tier-0.9 rows: anchor 738 (error 66) has its own ground-truth note admitting the
identifying feature "wasn't clearly visible"; anchor 891 (error 94) is labelled `quality=sure`
but by an unverified Claude pass (`decisions.md` D8). At n=5-7 per tier from an unverified,
single-labeller anchor set, this inversion is not read as evidence the pipeline's calibration is
actually broken -- it is read as evidence the user's verification pass is now the binding
constraint on getting a real answer, more so than before (the pipeline's confidence numbers are
no longer artificially flat, so there is finally a real reliability table to check once labels
are trustworthy).

## Step 6 — ground truth finalised, real evaluate() numbers (2026-09-01)

User completed their own independent verification pass over all 22 stratum-S anchors (worked in
a spreadsheet, pasted back as tab-separated with the `note` column dropped -- cleaned to proper
CSV, `note` restored empty since no text was carried over, `decisions.md` D8 updated to reflect
this is now genuinely user-verified, not Claude-labelled-user-verifying). Before accepting the
paste, validated `lo <= runB_best <= hi` programmatically for all 20 non-`no_match` rows --
0 violations. Two real corrections from this pass worth noting: anchor 346 (previously
`no_match` from the Claude fork's search) now has a real, tight match (`260, [258,261]`) -- the
user found it where the earlier search missed it; anchors 2600/2650 (previously given a wide
`unsure` guess reasoned from route-order logic alone, findings.md above) are now correctly
`no_match` per the user's direct check -- confirms the earlier D8-era reasoning ("must be near
runB's end, not its start, by route order") was on the right track but the actual frame-level
outcome there is genuinely no match, not the guessed range.

**`evaluate()` run against the final anchors x `outputs/mapping.csv`** (19/22 scored, 2 correctly
`no_match`, 1 excluded as `no_match` per ground truth but the pipeline still guessed --
`abstain_false_negative=1`):

- `median_error = 0.0` frames, `p90_error = 4.2` frames (previous, unverified-label run: median
  10.0, p90 71.6 -- the earlier numbers were dominated by label noise, not real pipeline error)
- `hit@2 = hit@5 = hit@10 = 89.5%` (17/19) -- all three are equal because no scored row has error
  strictly between 2 and 10 frames; errors are either 0 or already under 5
- `hit@5` Wilson 95% CI: `[68.6%, 97.1%]` -- still a real range at n=19, reported as such, not
  rounded away
- `abstain_true_positive=1, abstain_false_negative=1, abstain_false_positive=1` -- of the 2
  `no_match` ground-truth anchors (2600, 2650, which are near-identical to each other), the
  pipeline correctly abstained on one and didn't on the other; one separate row where ground
  truth has a real match was abstained by the pipeline

**Reliability table -- the number this whole project has been building toward** (per plan.md
§13's whole point: prove confidence tiers actually mean something, don't just assert it):

| confidence tier | n anchors | measured accuracy |
|---|---|---|
| 0.4 | 4 | 50% |
| 0.7 | 7 | 71% |
| 0.9 | 8 | **100%** (8/8) |

Monotonically increasing with tier, as designed -- this directly resolves the inversion flagged
in the earlier (unverified-label) run in this file: that inversion was correctly attributed to
label noise/small-sample effects at the time, not a real calibration problem, and this result
confirms that read was right rather than a lucky guess. This is real, reportable evidence that
`fuse_confidence`'s ordinal tiers track measured reliability on this dataset -- exactly what
plan.md §13 asks the report to demonstrate, not just claim.

**Not yet done**: dev/test split (plan.md's "6 dev, ~24 test" was sized for the full 30-anchor
set including stratum H, which isn't built -- evaluating on all 22 stratum-S anchors together for
now, no threshold recalibration has happened against this data yet so there's no dev-set
leakage concern in the numbers above); stratum H; ablation study; failure taxonomy figures.

## Step 3 revisited — H0 sync test: real GOP-level evidence, not just a declared assumption (2026-09-01)

Prompted by a sharp user question: "runB cam0 and cam5 timestamp files both have 2663 lines, is
that the justification for pairing cam0[k] with cam5[k]?" -- checked directly, and the honest
answer is **no, that specific number is not independent evidence**: `diff` on
`runB/cam0_..timestamps.txt` vs `runB/cam5_..timestamps.txt` (and the same for runA) returns
**zero differing lines** for both runs -- the files are byte-identical, confirming CLAUDE.md §5's
"both cameras of a run share one timestamp file" literally, not just approximately. Since it's
the *same file* read twice, cam0 and cam5 having "the same count" is trivial by construction, not
a measurement that happens to agree.

This sharpened the real open question: given cam0 and cam5 have *different* decoded picture
counts in runB (2622 vs 2615, findings.md Step 0), are the missing frames a clean tail truncation
(H0 safe for the frames that do exist) or scattered mid-sequence drops (H0 could break even for
small `k`)? This is `plan.md`'s still-open "Sync test" item -- closed out here with real evidence
via `io_video.scan_nals`'s GOP/IDR structure, not by assumption:

| run/cam | n_pictures | n_idr | last IDR index | final GOP length (of 250) |
|---|---|---|---|---|
| runA/cam0 | 2674 | 11 | 2500 | 174 |
| runA/cam5 | 2674 | 11 | 2500 | 174 |
| runB/cam0 | 2622 | 11 | 2500 | 122 |
| runB/cam5 | 2615 | 11 | 2500 | 115 |

All four streams have **identical IDR positions** (`0, 250, 500, ..., 2000, 2250, 2500`) and
**identical GOP sizes** (exactly 250) for every GOP up to and including the one starting at 2500
-- i.e. every picture from index 0 to 2499 is structurally accounted for identically in both
cameras of a run, for both runs. This is the signature of a clean stream truncation (encoder/
writer stopped mid-GOP), not scattered interior frame drops -- interior drops would desynchronise
the GOP/IDR pattern between cameras, which is not observed.

**runA**: `cam0` and `cam5` cut off at the *exact same point* in the final GOP (174/250 both) --
H0 (`cam0[k] = cam5[k]`, same nominal instant) is supported by direct structural evidence for the
*entire* sequence, `k = 0..2673`, not just assumed.

**runB**: `cam0` and `cam5` share identical structure through the last full IDR (`k <= 2499`,
supported the same way as runA), then diverge only in the final, incomplete GOP -- cam0 got 122
pictures into it before stopping, cam5 only 115. `122 - 115 = 7`, exactly the already-known
"runB cam5 is 7 frames shorter than cam0" gap (findings.md Step 0) -- now explained, not just
observed: both cameras were very likely stopped by the same real-world event (e.g. storage full,
recording-stop signal) but each camera's own encoder/writer had already buffered/flushed a
slightly different number of additional frames before actually halting. H0 is therefore
well-supported for `k <= 2499` in runB by the same structural evidence as runA; the only region
without this specific evidence is the last ~115-122 frames of runB's final GOP, where the two
cameras' frame-for-frame pairing is not structurally verified (though nothing found contradicts
it either -- see the motion_energy/kiosk content-based spot check below, which happens to sit in
this exact region and did not surface a synchronisation problem, only a viewing-angle effect).

**Separately, a content-based spot check** (motion_energy transition timing + a specific
landmark, a small guard kiosk, in runB cam5 around index 90-110) was run earlier the same
session, prompted by the user visually suspecting a cam0/cam5 delay. First pass (naive
threshold-crossing on `motion_energy`) suggested a ~20-24 frame offset; visual inspection showed
this was an artifact of a passing car crossing cam5's view (not our own vehicle's motion) --
once that transient is discounted, both cameras' sustained motion onset aligns at ~frame 42, no
lag. The kiosk-visibility question (kiosk visible in runA's cam5 at frame 60 but not in runB's
cam5 near the cam0-implied match) was resolved by the user's own, better explanation: a
different turning line/heading at that point changes cam5's (more forward-diagonal) viewing
angle much more than cam0's (more perpendicular) one -- covered by the labelling protocol's
"lane offset is ignored" rule, not evidence against the match.

**Net conclusion**: H0 is no longer "declared, not measured" for this dataset -- it now has
concrete structural (GOP/IDR) evidence for effectively the whole usable range of both runs, plus
two independent content-based spot checks that turned up plausible confounds (passing traffic,
turning angle) rather than evidence against it. Still labelled H0 (an assumption), per CLAUDE.md
§5's own standard -- structural GOP matching is strong indirect evidence of synchronised
recording, not a direct hardware timestamp comparison, which the dataset does not provide any way
to obtain.

## Step 6 — ablation study, including a user-proposed early-fusion experiment (2026-09-01)

All rows scored against the final, user-verified `gt/anchors_user.csv` (20 non-`no_match`
anchors -- same evaluate()-style scoring as Step 6's headline numbers: 0 if the prediction falls
in `[lo, hi]`, else distance to the nearest end).

| configuration | median error | p90 error | hit@5 |
|---|---|---|---|
| argmax-only, no DTW (unconstrained per-row argmax on `S_joint`) | 0.0 | **556.3** | 0.65 |
| SeqSLAM, cam0+cam5 **late**-fusion (score-level, current per-camera design) | 0.0 | 79.2 | 0.85 |
| SeqSLAM, cam0+cam5 **early**-fusion (descriptor concat, user's proposal) | 0.0 | **3.7** | 0.90 |
| DINOv2, cam0+cam5 late-fusion | 0.0 | 4.3 | 0.90 |
| DINOv2, cam0+cam5 early-fusion (concat) | 0.0 | 4.1 | 0.90 |
| cam0-only (both descriptors, late-fused) | 0.0 | 3.7 | 0.90 |
| cam5-only (both descriptors, late-fused) | **2.0** | 7.9 | 0.85 |
| **Joint -- SHIPPED** (both descriptors x both cameras, late-fused) | 0.0 | **3.5** | 0.90 |

**argmax vs. DTW**: p90 556.3 -> 3.5 frames. This is the quantitative version of the "corner
aliasing" diagnostic from Step 3/4 (`step3_similarity_argmax_diagnostic.png` vs.
`step4_dtw_path_diagnostic.png`) -- direct numeric proof the monotonicity constraint is doing
real work, not just a visually plausible-looking fix.

**cam0 vs. cam5**: cam5-only has both a worse median (2.0 vs 0.0) and worse p90 (7.9 vs 3.7) than
cam0-only. Consistent with, and now quantifying, today's qualitative findings that cam5 (road-
facing) is more exposed to passing traffic / transient occlusion than cam0 (kerb-facing) --
findings below and the motion_energy/kiosk investigation earlier this session.

**Early vs. late camera fusion -- the user's proposed experiment** (`similarity.fuse`-style
score averaging, the shipped design, vs. concatenating L2-normalised cam0+cam5 descriptors per
frame before computing similarity, re-normalising the concatenation, then running the same
`cosine -> local_contrast_norm -> dtw_open_ends` pipeline unchanged): **SeqSLAM benefits hugely**
(p90 79.2 -> 3.7); **DINOv2 barely changes** (4.3 -> 4.1, noise-level). Traced the SeqSLAM
late-fusion failure to a specific, already-known hard region: anchors 60, 150 and 183 (runB
~98-104, the depot/carport area) all get predicted as **column 183 by SeqSLAM late-fusion** --
exactly the rainy gate scene identified back in the Step 5 `no_match`-cluster investigation as a
false attractor, not the true match. Score-level averaging apparently let cam0's (wrong, in this
narrow stretch) opinion dominate the weighted mean even though cam5 disagreed; concatenating the
raw descriptors instead requires *joint* agreement across both cameras' feature spaces before the
combined vector reads as similar, so a one-camera-only false attractor drags the whole
concatenated similarity down instead of surviving a weighted average. SeqSLAM early-fusion
recovers this case well (79/79/100 predicted vs. true ~99/99/103, errors 19/19/2 -- down from
81/81/79).

**Why the shipped design (late-fusion, both descriptors) is still the right call, not just the
status quo**: `S_joint`'s prediction at the same three anchors is `[81, 81, 100]` -- comparably
good to SeqSLAM's own early-fusion result, achieved through a *different* mechanism (DINOv2's
independently-solid match pulling the fused DTW path back on track, not requiring any change to
how cameras are combined). The shipped joint configuration's p90 (3.5) is the best or tied-best
of every configuration tested, including SeqSLAM's early-fusion (3.7) -- and it does this while
still producing two independent, per-axis verification signals (`cam_disagree` from the
per-camera paths, `desc_disagree` from the per-descriptor paths) that a fully early-fused design
would have no way to compute (there would be only one combined path per axis collapsed together,
nothing independent left to disagree). This is now an evidence-backed conclusion, not just the
theoretical argument from earlier in this session ("early fusion could plausibly help; the
`cam_disagree`/`desc_disagree` cost is the reason we didn't do it") -- the experiment shows the
plausible benefit is real for one descriptor, but the shipped design already captures a
comparable benefit through descriptor redundancy without paying that cost.

**Not done as part of this pass**: with/without `local_contrast_norm` as a clean ablation row --
covered in substance by the D9/D10 investigation (DINOv2 collapses without a wide-enough window,
i.e. z-scoring itself isn't optional, but the *window size* is the tunable knob), not re-run here
as a separate raw-cosine-only row.

## `evaluate` CLI subcommand and persisted evaluation artifacts (2026-09-01)

Prompted by a user question: "where's the CSV comparing pipeline predictions to ground truth?" --
correctly pointed out that `evaluate()`'s `per_anchor` DataFrame had never actually been saved to
a file, only summarised in this document and printed to a terminal during development. `make
evaluate` was also a documented Makefile target (CLAUDE.md §6) with no real CLI implementation
until now (`args.command == "evaluate"` fell through to `_not_implemented`).

Fixed both: `groundtruth.evaluate`'s `per_anchor` widened from
`[runA_frame, runB_frame, lo, hi, error, confidence, status]` to also include `runB_best`,
`quality`, `note` from the original anchors -- so the persisted file is self-explanatory (shows
*why* a ground-truth row was labelled the way it was, not just the bare numbers). Required
updating the synthetic `_anchors()` test helper in `tests/test_groundtruth.py` to default-fill
`runB_best`/`note` (existing test fixtures only set the columns each test cared about); 88/88
tests green after the fix.

New `_cmd_evaluate` in `cli.py`: loads `outputs/mapping.csv` and `gt/anchors_user.csv` (via
`load_labels`), runs `evaluate`, writes `outputs/evaluation_anchors.csv` (the real
prediction-vs-ground-truth table) and `outputs/evaluation_summary.json` (the aggregate metrics).
Run for real: reproduces every number already reported in this file exactly (median 0.0, p90 4.2,
hit@2/5/10 89.5%, reliability table 50/71/100% -- see the "ground truth finalised" entry above).
The persisted `evaluation_anchors.csv` makes the anchor 60/150 issue (found via the early-fusion
ablation) directly visible without re-deriving it: both rows show `runB_frame=81` against ground
truth `[98, 102]`, `error=17.0` -- the shipped late-fusion joint pipeline's actual, currently
unresolved miss at this anchor, now inspectable as data rather than only as ablation-script output.

## `evaluation_anchors.csv` column clarity fix (2026-09-01)

User caught a real, well-founded confusion immediately after the file above was first
generated: `status="matched"` sitting next to `error=17.0` (anchor 60) reads as a contradiction
if `status` is misread as a correctness verdict -- it isn't, it is `mapping.csv`'s own
`pipeline_status` (did the pipeline abstain, not whether it was right). No column in the
original version gave a direct yes/no answer to "was this row correct".

Fixed in `groundtruth.evaluate`: `per_anchor` columns renamed with explicit `pipeline_`/`gt_`
prefixes (`runB_frame` -> `pipeline_runB_frame`, `runB_best`/`lo`/`hi`/`quality`/`note` ->
`gt_runB_best`/`gt_lo`/`gt_hi`/`gt_quality`/`gt_note`, `confidence`/`status` ->
`pipeline_confidence`/`pipeline_status`, `error` -> `error_frames`), plus a new `correct` bool
column (`error_frames == 0`) as the single, unambiguous place to check right/wrong. Updated the
one test that referenced the old `error` column name; added an explicit column-list assertion
and a `correct`-column check so this contract is now regression-tested, not just documented.
88/88 tests still green. Re-ran the `evaluate` CLI subcommand; `outputs/evaluation_anchors.csv`
now shows, unambiguously, `correct=False` for anchors 60/150 (the known SeqSLAM/joint miss) and
also surfaces two previously-unremarked near-misses at `correct=False, error_frames=1.0`
(anchors 1270, 1400 -- one frame short of the ground-truth window, not examined further yet).

## `evaluation_anchors.csv` was silently dropping 3 anchors -- fixed, and anchor 183 root-caused (2026-09-01)

User noticed the persisted file only had 19 rows against 22 real anchors ("kok cuman 19 ya? yang
no match kenapa di buang"), then specifically asked about anchor 183 mid-investigation ("183 juga
tidak dimasukan?"). Root cause: `groundtruth.evaluate`'s original `is_scored` filter (requires a
ground-truth match, a ground-truth range, *and* a pipeline prediction) is the right filter for
computing `median_error`/`p90_error`/`hit@k` -- you cannot measure a position error when either
side has no position to offer -- but the original code then reused that same filter to select
which rows got written to `per_anchor`, so the 3 rows it correctly excluded from the *metrics*
were also, incidentally, deleted from the *file*. There was no record in the CSV that these
anchors existed at all, let alone why they were missing.

Fixed: `per_anchor` now always has all 22 rows (`merged`, not `scored`, is the base). Added a
`abstain_outcome` categorical column (`correctly_answered` / `correctly_abstained` /
`wrongly_abstained` / `wrongly_guessed`) computed independently of the position-error filter, from
`should_abstain = gt_quality == "no_match"` crossed with `did_abstain = pipeline_runB_frame is
empty`. For the 3 non-`correctly_answered` rows, `correct`/`error_frames` are now `<NA>`/`NaN`
(pandas nullable boolean, via `pd.array(..., dtype="boolean")`) rather than a misleading `False`/
`0.0` -- "no comparison was possible" is a different fact from "the comparison came out wrong",
and conflating them would have made 183 look like an ordinary miss instead of an abstention.
(Bug fixed along the way: `np.select`'s implicit int `0` default cannot be promoted against a
string choicelist under NEP 50 -- needs an explicit `default="unreachable"`.) 4 tests updated,
1 new regression test added (`test_evaluate_per_anchor_includes_every_anchor_not_just_scored_ones`,
a 2-anchor synthetic case asserting both the row count and the exact `abstain_outcome` mapping);
89/89 tests green, `ruff check`/`ruff format --check` clean.

Re-ran `evaluate`; all 22 rows now present. The 3 previously-invisible anchors, traced against
`mapping_full.csv`'s pre-blanking fields:

| `runA_frame` | `abstain_outcome` | pipeline raw candidate | gt answer | `ridge_z` | `cam_disagree` | `desc_disagree` | `reason` |
|---|---|---|---|---|---|---|---|
| 183 | `wrongly_abstained` | col 100 (`slope`=1.0) | `[102,104]` | 0.6899 | 1.0 | 83.0 | `ridge_z below tau (0.75)` |
| 2600 | `correctly_abstained` | col 2597 | `no_match` | 0.3639 | 15.0 | 9.0 | `ridge_z below tau (0.75)` |
| 2650 | `wrongly_guessed` | col 2600, confidence 0.4 | `no_match` | 0.7932 | 13.0 | 14.0 | (passed, no reason) |

**183 is the most informative of the three**: the pipeline's own raw candidate (column 100) was
only 2 frames outside the ground-truth window `[102, 104]` -- i.e. *right next to* correct -- but
`ridge_z=0.6899` fell just under `tau=0.75`, so the no-match rule suppressed it. `cam_disagree=1.0`
is low (cameras agree closely), but `desc_disagree=83.0` is enormous -- SeqSLAM and DINOv2 are
picking very different columns here, which is exactly the kind of internal disagreement `tau`
exists to catch, and is consistent with this being the same rainy depot/carport stretch already
identified as SeqSLAM's false-attractor region (anchors 60/150, "column ~183" -- see the
early-fusion ablation entry above; note the coincidence that the *attractor column number*, 183,
matches this *anchor's runA_frame*, 183, by chance, not by any causal link). 2600 shows the rule
working as intended (large `cam_disagree`, genuinely low `ridge_z`, and ground truth agrees it's
unanswerable -- the loop-closure return-to-start ambiguity already known from Step "verify GT"
discussions). 2650 shows the opposite failure mode is also live near the same boundary: `ridge_z`
passed `tau` (0.79 > 0.75) at the lowest confidence tier, but ground truth judged this a genuine
loop-closure no_match (multiple plausible frames), so the pipeline's single point answer was
wrong to offer at all -- `slope=0.013` (near-stationary) is consistent with this being inside a
stop/plateau where many frames look alike.

**Net honest read**: `tau=0.75` is not free of cost in either direction at this sample size --
it correctly suppressed one genuinely unanswerable anchor (2600) and incorrectly suppressed one
answerable one (183) by a margin of 0.06 in `ridge_z`, while also letting one wrong answer through
just above the line (2650). With only 3 abstention-category anchors in the labelled set this is
descriptive, not a basis for retuning `tau` -- retuning on 3 points would be overfitting to noise,
not evidence. `outputs/SCHEMA.md` updated with the full `abstain_outcome` contract and this same
trace, so the file and this document tell the same story.

## `no_video` renamed to `no_frame` (2026-09-01)

User's naming observation, made while planning Task 3: every other identifier in this project
says "frame" (`runA_frame`, `frame_count`, `decoded_index`, the planned manifest's `decode_ok`) --
`no_video` was the only place "video" appeared, an inconsistency with no functional reason behind
it (the underlying HEVC stream is "video" but what's actually being asserted, row by row, is "no
*frame*"). Renamed `status="no_video"` -> `"no_frame"` (and the matching `reason` string, `"no
video for this row"` -> `"no decoded frame for this row"`) throughout the live code and tests:
`confidence.py` (`_REQUIRED_CUE_COLUMNS`, the cue column itself, docstrings), `cli.py` (the
per-row cue column, the H0-tail rows, the final-status print line), `tests/test_confidence.py`
(9 assertions), `tests/test_mapping_csv.py` (6 assertions, including renaming
`test_no_video_rows_match_...` to `test_no_frame_rows_match_...`), `CLAUDE.md` §5,
`outputs/SCHEMA.md`, and the live (non-historical) parts of `plan.md` (§0.2's H0 decision, the
Step-5 column list, the Step-8 test-plan line). 89/89 tests still green after the rename,
`ruff check`/`ruff format --check` clean.

**Deliberately left alone**: every *historical* quote of an actual past pipeline run's printed
output -- `plan.md`'s SeqSLAM-only "run for real" entry (`no_video=21`, dated before DINOv2) and
several dated `findings.md` entries (D3's draft run, the D9 broken-DINOv2 run, the D10 fixed run,
the original `evaluate` CLI entry). Those are verbatim records of what the pipeline actually
printed at that point in time, under the name it had then -- rewriting them to say `no_frame`
would misrepresent what really ran, the same reason a past commit message doesn't get edited to
match today's vocabulary. Only this run's live artifacts (`outputs/mapping.csv`,
`outputs/mapping_full.csv`) were regenerated with the new name, via `python -m routealign verify`
(cheap -- no need to redo `align`'s DTW, since `status` is only computed at the verify stage);
their row counts and every other number are unchanged, only the label is new.

## Step 7 — Task 3 keep/discard manifest, implemented and run for real (2026-09-01)

`policy.build_manifest` implemented (all 8 rules, plan.md §9), a new `RunCameraFacts` dataclass,
9 private helpers, wired into a new `manifest` CLI subcommand (`cli.py`'s `_cmd_manifest` --
was a documented `make manifest` target with no implementation, like `evaluate` had been). 21 new
tests in `tests/test_policy.py` (17 first pass, +1 for a real bug found on real data below, +3
more from earlier passes); 107/107 tests green project-wide, `ruff check`/`format --check` clean.

**Run for real** (`python -m routealign manifest`, needs `align`/`verify` already run):
`outputs/keep_manifest.csv`, 10716 rows (2695 runA lines x 2 cams + 2663 runB lines x 2 cams --
one row per `(run, cam, frame)`, not per timestamp line, since decode status/occlusion/motion
genuinely differ per camera even at the same index). `keep=10585 discard=131` -- 131 is not a
new number, it *exactly* reproduces the independently-known `no_frame` gap (21 runA + 41 runB/cam0
+ 48 runB/cam5, CLAUDE.md §5, `test_mapping_csv.py`), a real cross-check the wiring is correct,
not just "some number came out." `split`: train=6814, val=1566, test=1358, unassigned=978 (the
merged depot at both ends of runB's route, plus the buffer places between blocks).

**Cross-check against `mapping.csv`**: `corr_status` value counts on the 5390 runA rows --
`matched=4976, ambiguous_range=248, no_match=124, no_frame=42` -- are exactly double
`mapping.csv`'s own per-runA-frame status counts (`matched=2488, ambiguous_range=124,
no_match=62, no_frame=21`), confirming correspondence was correctly copied onto both cameras'
rows of each runA frame, not miscounted or dropped.

**A previously-unremarked fact, found while building the depot-merge rule**: re-running
`frames.stationary_segments` over each run's *entire* route (not the 600-frame window an earlier,
informal Step 2 check used) gives `runA: [(0,174), (2614,2673)]` (233 frames, 8.7% of the route --
the second segment, at the very end, is new information: runA is a loop, and the vehicle stops at
the depot again on return) vs. `runB: [(0,30), (42,54)]` (42 frames, 1.6% -- much shorter, that
day's driver evidently did not idle as long at the start, and there is no comparable stop at
runB's own end). This also resolves an apparent contradiction with an early `path_to_mapping`
finding ("runA rows 0-5 widen to `[0,173]`", the "`path_to_mapping`" entry above): that was from
the SeqSLAM-cam0-only path, computed *before* DINOv2 fusion, whose picked columns at those rows
happened to fall inside runB's short real stationary window; the shipped joint-fusion path is
more accurate and lands on columns 70-80, outside `[0,29]`, so it correctly does not widen there
-- not a bug, a more precise final path superseding a less precise draft one.

**Spot-checked on real data, not just synthetic fixtures**: runB cam0 frames 0-5 and 2617-2621
(its last 5 decoded frames, decoded_count=2622) both get `place_id=-1` (`"unassigned"` split) --
the merged start/end depot, `policy.depot_runb_margin=50` correctly spanning both. runA frame 183
(the anchor already known from the evaluation work to be a genuine `no_match`/abstained row,
D11/D12) gets `corr_frame=NaN` (honestly unchanged -- never fabricated into the output) but
`place_id=1` (a normal, non-depot bucket), confirming the neighbour-interpolation rule
(`_interpolate_missing_corr_frame`) resolves a missing correspondence to a sensible position from
the surrounding matched runA rows: 180-182 (also `no_match`) land in `place_id=0`, while 184-186
(matched to real runB 101/102/103) sit in `place_id=1` -- 183, between them, interpolates to the
`place_id=1` side, its nearer known neighbour.

**A real bug, only surfaced by the full dataset**: `frames.motion_energy` returns
`decoded_count - 1` values (one per transition, not one per frame -- its own docstring already
said so). The first `_attach_motion` implementation indexed it as if sized `decoded_count`,
raising `IndexError: index 2673 is out of bounds for axis 0 with size 2673` on runA's very last
decoded frame the first time `manifest` ran for real. The synthetic test fixture had made the
identical wrong assumption (same-sized arrays), so 17/17 tests passed despite the bug -- fixed in
both the implementation and the fixture, plus a dedicated regression test
(`test_attach_motion_does_not_crash_on_the_last_decoded_frame`) so an equal-sized fixture cannot
mask this class of bug again (decisions.md D12).

**Not yet done**: report §3 (dataset card) written from these real numbers.

**Corrupt-decode and exact-duplicate-frame checked for real, both negative (2026-09-01)**: rule
1's two remaining sub-cases (plan.md §9: "failed decode, exact duplicates"), previously an open
TODO, now have real evidence rather than an unchecked assumption.

*Exact duplicates*: `frames.motion_energy` (already computed for `is_stationary`) gives, for free,
mean abs pixel diff between every consecutive frame pair -- an exact duplicate would read exactly
0.0 here. Checked all 4 streams: minimum values 0.058-0.130, **zero** values at or near 0.0 in any
of them (2673/2673/2621/2614 transitions checked), even inside the longest stationary segment
(runA's depot, 233 frames). No evidence of consecutive-frame duplication anywhere in the dataset.
(Non-consecutive duplicates, e.g. frame 5 identical to frame 500, are not covered by this check --
considered out of scope, since nothing about a dashcam recording pipeline would produce that
without also producing the far more likely consecutive case, which is absent.)

*Failed/corrupt decode*: the production `decode_to_jpegs`/`count_decoded` calls run ffmpeg/ffprobe
at `-v error` (CLAUDE.md's own convention), which suppresses `warning`-level messages -- exactly
the severity ffmpeg typically uses for decoder concealment/corruption ("missing reference",
"concealing errors" and similar are libavcodec warnings, not errors). Re-decoded all 4 streams to
the scratchpad (not the production cache) at `-v warning` -- strictly more sensitive than
production -- and captured stderr in full: **zero warning lines on all 4 streams**, and the JPEG
count from each re-decode exactly matched the already-known `decoded_count`
(2674/2674/2622/2615). No evidence of decoder-level corruption or concealment anywhere in the
usable (pre-truncation) range of any file.

**Conclusion**: rule 1's `keep=decode_ok` is not just "the only rule implemented so far" -- it is
now evidence-backed as *complete* for this dataset (no corrupt or duplicate frames found to
additionally discard), not merely convenient. `policy.py`'s docstring/decisions.md D12 updated to
say "checked, none found" rather than "not yet implemented".

## Failure-taxonomy figures (2026-09-01)

Plan.md's 6-case list, user's decision to also drop the dev/test anchor split (too few of the 22
stratum-S anchors to give either half a usable Wilson CI -- the split existed to sanity-check
threshold calibration, which the deletion test already covers; `evaluate` keeps scoring all 22
together, plan.md).

New `viz.plot_frame_comparison` (N images + captions laid side by side, gamma-corrected for
display) and a new `figures` CLI subcommand/Makefile target (`cli.py`'s `_cmd_manifest`-adjacent
`_cmd_figures`, 5 small per-case helper functions) -- reads `outputs/mapping_full.csv` plus the
already-decoded JPEG cache, no new computation, matching `viz.py`'s "no computation" rule.
`ruff check`/`format --check` clean, 107/107 tests still green (no new tests -- matplotlib figure
generation isn't asserted against pixel content anywhere in this codebase, consistent with there
being no `test_viz.py`).

**5 of the 6 planned cases shipped, each reusing a real, already-verified number** rather than a
constructed example:

1. **Start-plateau ambiguity** (`failure_start_plateau.png`): runA #60 and #150 -- visually
   identical depot frames (same building, same parked cars) -- both get the shipped pipeline's
   same wrong answer, runB #81 (`confidence=0.4, ridge_z=1.07`), rather than the true match runB
   #100 (ground truth `[98,102]`). Looking at the actual images makes the failure legible in a way
   the numbers alone don't: #81 has a tree trunk squarely in the middle of the frame that #60/#150
   don't obviously resemble, while #100 clearly does -- a reader can see the pipeline's answer is
   the weaker visual match without needing to trust `ridge_z` at all.
2. **cam5 passing traffic** (`failure_cam5_traffic.png`): `motion_energy` checked at every index
   75-119 for runB cam5 vs cam0 (not just eyeballed) -- cam5 rises then *drops* 93->105 while cam0
   keeps climbing, the drop being consistent with a large object (the passing car) briefly filling
   and dominating the frame. First pick (frame 95, from the earlier session's "~90-110" range)
   turned out to be just past the car in the actual rendered image -- retargeted to frame 93 (the
   real local peak) after checking, which clearly shows the car mid-frame, blurred with motion.
   cam0 at the same index shows nothing unusual, confirming the occlusion is cam5-specific.
3. **Rain-degraded false attractor** (`failure_rain_attractor.png`): runB #183, both cameras --
   the "rainy gate" scene already identified (Step 6 ablation entry) as where SeqSLAM's late-fusion
   path collapses for anchors 60/150/183. Visible rain droplets on both camera housings, consistent
   with runB's Step-1 weather read (overcast/light rain).
4. **Loop-closure ambiguity** (`failure_loop_closure.png`): runA #2600 (back at the depot) next to
   runB #10 (near route start) and runB #2597 (near route end, the pipeline's actual raw candidate,
   `ridge_z=0.36`, correctly abstained). The strongest of the 5 images -- #10 and #2597 are
   essentially the same photo (same building, same cars in nearly the same spots), visually
   confirming *why* abstaining here is the right call, not overcautious.
5. **Low-texture stretch** (`failure_low_texture.png`): runA #1659-1678, a run of 20 consecutive
   `confidence=0.4` rows with persistently *negative* `margin_z` (-0.14 to -1.06) despite correct,
   monotone tracking (`slope` ~1.0-1.13) -- a different failure signature from the attractor cases
   (chronically weak discrimination over a stretch, not one spuriously strong wrong answer).
   Turned out to need the heaviest gamma correction of the 5 -- this stretch is under dense tree
   canopy and was originally rendered almost pure black, unreadable until the brightening fix.

**Dropped, not just skipped**: "runA tail beyond runB's truncated end" (`boundary_clamped`).
Checked `outputs/mapping_full.csv` for real: zero non-`no_frame` rows have `boundary_clamped=True`
in the shipped joint pipeline -- already recorded once before (the `path_to_mapping` findings.md
entry, "zero rows are boundary_clamped... not an artifact of the widening logic hiding a clamped
tail"), now reconfirmed on the final joint (not draft SeqSLAM-only) mapping. There is no real
instance of this anticipated failure mode to show a figure of; noting the negative result honestly
is worth more than forcing an example that doesn't actually occur.

**A real, worth-recording process point**: two of the five figures needed a second pass after
actually *looking* at the rendered PNG, not just trusting the frame-index math -- the cam5-traffic
frame pick and the global-dark low-texture panels. Both are reminders that a "convincing account of
failure cases" needs the figure itself checked, the same way a number needs its provenance checked.

## README.md and the report draft (2026-09-01)

`README.md` written: measured (not just declared) dependency versions -- `requirements.txt` pins
package names only, no version numbers, so the table quotes what `pip` actually resolved on this
machine (numpy 2.4.6, torch 2.13.0, etc., real `ffmpeg -version` string included), a mermaid
pipeline diagram (one node per CLI subcommand/artifact, the manual ground-truth-labelling step
visually distinguished from the automated ones), full reproduction commands, the H0 statement.

**Report §2/§3/§4 drafted** (`report/report.tex`) -- every number pulled from this document's own
entries or re-verified directly (`outputs/evaluation_summary.json`,
`notebooks/results/early_fusion_ablation.csv`, `outputs/mapping.csv` status counts), not
transcribed from memory. `report`'s CLI subcommand was still `_not_implemented`; wired it for real
(`latexmk -pdf`, `subprocess.run(cwd=report_dir)`) rather than leaving the report buildable only by
some other, undocumented mechanism.

**A real layout bug, not just a content-writing task**: the first build came out 6 pages against
CLAUDE.md's 2-4 page budget. Root cause was not excess content but the `float` package's `[H]`
placement -- forcing every table/figure to render exactly where written left large blank gaps
whenever one didn't fit the remaining space on a page (visible on 3 of the 6 pages, roughly a third
to half of each blank). Switching the *interior* floats to `[tbp]` (let LaTeX pack pages
naturally) recovered a full page immediately (6 $\to$ 5) with zero content change. This introduced
a second, worse bug: two floats with nothing anchoring them nearby drifted to the wrong position
entirely -- the Task 1 summary table floated above the document's own title, and the
proved-vs-assumed ledger table floated to the top of its page, ahead of the section that
introduces it. Both float above their own context only because they are the *first* (page 1) and
*last* (page 5) floats in the document with nothing on one side to block the drift -- reverted just
those two back to `[H]`, confirmed by re-rendering every page to PNG and looking at the actual
output, not just trusting the page count. A pre-existing, unrelated overfull-hbox warning on the
ledger table (two `p{0.4\textwidth}` columns plus a non-wrapping status column together exceeded
`\textwidth`) was also fixed while in there, unprompted but low-risk (column width only).

**Current state: 5 pages, one over budget.** Content-trimming (tighter prose in the ablation and
failure-case discussion, a shorter Figure 3 caption, merging two ledger rows) bought back some
space but not the full page -- pages 1-4 are now fully packed with no slack, so closing the last
gap needs a real content cut (dropping an ablation row, cutting a failure-case figure, or similar),
not another formatting fix. Left for the user to decide rather than cut unilaterally, since the
report is explicitly graded on being able to defend everything in it, and irreversibly wasn't the
right axis to optimize sight-unseen (findings.md, this entry).

## Internal-notes references removed from every deliverable-facing surface (2026-09-01)

User instruction: `plan.md`/`findings.md`/`decisions.md` are working notes for this session and
the candidate, not part of what an interviewer/grader reads -- nothing in the actual deliverable
(code, tests, config, README, the report) should cite them by name. Confirmed scope via a
clarifying question: `src/`, `config.yaml`, `README.md`, `tests/`, `CLAUDE.md`, `report.tex` all
in scope; `plan.md`/`findings.md`/`decisions.md`'s own cross-references to each other are exempt
(that is their actual purpose); `gt/labelling_protocol.md` and `notebooks/` left untouched
(outside the confirmed scope, flagged rather than assumed).

~130 citations removed across the codebase (`# why: plan.md §N`, `(see findings.md)`,
`(decisions.md D9)` and similar), each one rewritten to state the actual reasoning or evidence
inline rather than just dropped -- CLAUDE.md's own "no magic numbers... with a `# why:` comment"
rule still requires a real reason at every threshold, just not a pointer to where that reason
happens to also be written down. `CLAUDE.md` itself updated with an explicit rule to this effect
(§7), so future sessions don't drift back to the old pattern.

Two real, unrelated staleness bugs caught in the process, not just citations: `cli.py`'s
`gt-sheets` docstring/print output still said stratum H needed DINOv2 which "has not been run in
this environment yet" -- true when written, false since DINOv2 was unblocked and run for real
much earlier this session; fixed to name the actual current blocker (no `desc_disagree`-ranking
function exists yet). `groundtruth.py`'s `load_labels` docstring described an intermediate,
already-superseded methodology state (Claude's primary-fill draft) rather than the final one (the
user's own real verification pass) -- fixed to describe what is actually true now.

Two generated output files were stale copies of text a citation-bearing string used to produce
(`avg_frame_rate` rejection reason, in `outputs/inspection/*.json` and `task1_table.md`) --
regenerated for real via `characterise` after fixing the source in `cli.py`, not hand-edited;
confirmed the underlying measured numbers (`measured_decoded=2674`, `measured_actual_fps=9.9925`)
are unchanged, only the wording. `outputs/SCHEMA.md` (documents the shape of `outputs/`, which is
part of the submission) also cleaned, though not separately named in the clarifying question --
included because it is clearly the same kind of deliverable-facing surface as the report.
107/107 tests green, `ruff check`/`format --check` clean, `report/report.pdf` still builds at 5
pages (unchanged -- only trailing citation clauses were removed, no restructuring).

## README pipeline diagram audited against the real code; a genuine reproducibility gap found and fixed (2026-09-02)

User caught a real inaccuracy in the README's mermaid diagram: `verify-data` and `decode` were
drawn as parallel branches off `dataset/`, but the diagram's own caption frames it around
Makefile targets, and `make all` runs them strictly in that order. Checked the code rather than
guessing: `_cmd_decode` never reads `integrity.json`, so there is no real *data* dependency --
the sequential order in the Makefile is a deliberate fail-fast choice (check integrity before
spending time decoding), not a data need. User chose to keep the diagram as a genuine
data-dependency graph (arrows = "reads this as input", verified against the code) rather than
redraw it to match literal Makefile execution order, with the fail-fast reasoning moved to prose
text below the diagram instead.

Auditing that one pair systematically surfaced the same pattern in several more places, each
checked directly against the function bodies, not assumed from Makefile position:

- `characterise` reads `dataset/` directly (`io_video.probe`/`scan_nals`/`sps_vui` all take the
  raw `.hevc` path) -- not the JPEG cache, despite running after `decode` in the Makefile.
- `gt-sheets` only opens `FrameStore`s (needs the JPEG cache) -- never reads `mapping.csv`,
  despite running after `align`/`verify` in the Makefile.
- `align` and `verify` both call `_open_store`/`frames.motion_energy` directly (for `stationary_b`
  and the `runA_has_cam0`-style flags respectively) -- an additional real dependency on the JPEG
  cache beyond the descriptor cache / `mapping_draft.csv` the diagram already showed.
- `manifest` reads `integrity.json` (for `src_sha256`) and each run/camera's Task 1 inspection
  JSON (for `fps_measured`, `width`, `height`, etc., via `_run_camera_facts`) in addition to
  `mapping.csv` -- two real dependencies the diagram was missing entirely.
- `figures` also opens `FrameStore`s directly (`get_rgb`), the same missed JPEG-cache dependency
  as `align`/`verify`.

**A more significant finding along the way, not just a diagram gap**: `report.tex`'s
`step3_similarity_argmax_diagnostic.png` / `step4_dtw_path_diagnostic.png` are not produced by
*any* CLI subcommand -- `grep`ing the whole codebase for their filenames found nothing. They are
git-tracked (`outputs/*.npy` is the only gitignored pattern under `outputs/`), so the current
repo state builds fine, but a genuinely clean-cache `make all` would never regenerate them and
`make report` would fail on a missing `\includegraphics` file. Root cause: they were made by a
one-off script very early in the project (Step 3/4, before DINOv2 and camera fusion existed) and
never wired into the real pipeline once one existed. Separately, they were computed from the
SeqSLAM-cam0-only similarity matrix of that early exploration, while the report's own ablation
table quotes `S_joint`'s argmax p90 (556.3 frames) -- so the figure and the number it's supposed
to illustrate were never quite the same computation.

**Fixed for real, not just documented**: `_cmd_align` now writes both diagnostic figures directly
from `S_joint`/`path_joint` (the exact arrays used for the ablation number and the shipped
mapping), via the existing `viz.plot_similarity_matrix`. Re-ran `align` on the full dataset for
real (8.2s, all descriptor cache hits) -- `mapping_draft.csv` unchanged (2674 rows, path length
2814, same as every prior run), new figures confirmed visually to show the same story as before
(corner-aliasing red streaks vs. a clean diagonal) but now genuinely reproducible from a clean
cache and consistent with the quoted ablation number. `report/report.pdf` rebuilt, still 5 pages,
still compiles clean.

README's diagram redrawn with the corrected dependency set (`characterise`/`gt-sheets` now come
directly from their real sources; `align`/`verify`/`manifest`/`figures` show their JPEG-cache and
`integrity.json`/T1 dependencies), plus a new `DIAG` artifact node for the two figures now that
they come from `align`, not `characterise`. `report`'s connection to `evaluation_summary.json`/
`keep_manifest.csv` drawn as dashed, explicitly labelled "numbers copied in by hand" -- those are
genuinely not read at LaTeX build time (the numbers are hand-transcribed into `report.tex`'s
source during authoring), a real, worth-stating limitation: editing those files does not update
the PDF. 107/107 tests green, `ruff check`/`format --check` clean.

## Step 8 — `diagrams/pipeline.png` added to the report; recurring float-drift; ablation re-run with hit@5 and a full-22 extension (2026-09-03)

**Report float drift, three separate instances, same root cause.** `\graphicspath` extended to
also search `../diagrams/` so `pipeline.png` (a hand-drawn method diagram supplied outside the
pipeline, not generated by any `make` target) could be placed at the top of \S2. Every time
content was added to or removed from earlier in `report.tex` this session, at least one `[tbp]`
float re-floated to a wrong page position -- observed three times: `fig:dt` jumped above the
document's own title once the "Every number..." disclaimer paragraph was deleted; the new
`fig:t2-pipeline` (pipeline.png) jumped above \S2's own header and the tail of \S1's last
paragraph; `tab:reliability`/`tab:ablation` drifted above \S2's header the first time
`fig:t2-pipeline` was inserted. Each was caught by rendering the PDF page-by-page (not just
checking for a clean `latexmk` exit) and fixed the same way: pin the float to `[H]`, matching the
convention already used for `tab:t1`/`tab:ledger`. All figures and tables in the report are now
`[H]`; net effect after all fixes plus a `\captionsetup[figure]{font=footnotesize}` (figure
captions now match table captions, which already inherited `\footnotesize` from the tabular body)
was 6 -> 5 pages, before the ablation extension below added it back to 6.

**Ablation re-run for real, not just re-typed**: `conda run -n phase2-vpr python
notebooks/early_fusion_ablation.py` reproduced the exact same n=20/median/p90 numbers already in
the report for all 9 configurations (determinism check, not just a re-use of old numbers), and
additionally surfaced `hit_at_5` -- already computed by `_score()` every run, just never pulled
into `report.tex`'s table before. Added as a 5th column.

**Why n=20, not 22, confirmed from `gt/anchors_user.csv` directly rather than assumed**: only
2600 and 2650 have `quality="no_match"` with `lo`/`hi` = NaN; all other 20 anchors, including 183,
have a real position (183: `quality="sure"`, `[102,104]`) -- 183 was never excluded by the
ablation's own anchor filter, only by the *shipped* pipeline's own abstain decision, which the
ablation script doesn't apply to any configuration. So n=20 is already the maximum any
position-error metric can score here, not an arbitrary subsample -- extending to "22" is
structurally impossible for a metric that requires a `lo`/`hi` to compare against.

**Extended anyway, for the one row where it's meaningful**: user chose (asked via
AskUserQuestion, given the above) to extend just the `Joint late-fusion -- shipped` row to the
full 22 using categories that already exist in `outputs/evaluation_anchors.csv` -- no new
computation needed. Verified directly: `runA_frame=2600` has `abstain_outcome=correctly_abstained`
(`pipeline_status=no_match`, `gt_quality=no_match` -- both sides agree there's no match);
`runA_frame=2650` has `abstain_outcome=wrongly_guessed` (`pipeline_status=matched`,
`pipeline_confidence=0.4`, `pipeline_runB_frame=2600`, `gt_quality=no_match` -- the pipeline
answered at low confidence but no answer is correct here). Added to the ablation table's caption,
cross-referenced to \S2.4 (new label `sec:t2-groundtruth`) rather than re-explained, so the
confusion-matrix categories are defined in exactly one place. The other 8 configurations have no
no-match mechanism of their own, so the extension applies only to the shipped row, as agreed.

**Follow-up, same session: per-configuration prediction at 2600/2650, not just the shipped row.**
User asked directly whether "the ablated pipeline" succeeds at recognising no-match at these 2
anchors -- answer, made concrete rather than asserted: no, structurally, for 8 of 9. Extended
`_score()` in `notebooks/early_fusion_ablation.py` to also record `pred_2600`/`pred_2650` (the raw
`j_arr` value at those two rows) for every configuration, re-ran
(`conda run -n phase2-vpr python notebooks/early_fusion_ablation.py`), confirmed all previously
reported n/median/p90/hit@5 numbers unchanged. Result: every one of the 9 raw DTW paths returns
*some* column at both anchors (none can represent "no match" -- there is no threshold layer on
top of any of them except the shipped pipeline's own `fuse_confidence`, which sits on top of one
of these nine, `path_joint`); 8 of the 9 (excluding argmax-only) land within 29 frames of each
other at both anchors (2585-2614), i.e. they agree on *where* on the route this is (the
depot-revisit region) without any of them being able to say *whether* a frame-exact match exists
there. `argmax-only` predicts column 655 for **both** 2600 and 2650 -- the same corner-aliasing
collapse Fig.~3 (`step3_similarity_argmax_diagnostic.png`) already shows, now with a second,
independent concrete instance (two runA rows 50 apart landing on one wrong attractor column).
Cross-checked the shipped pipeline's raw candidate (`path_joint`, 2597/2600) against its actual
shipped behaviour in `outputs/evaluation_anchors.csv`: 2600's raw candidate (2597) is suppressed
into `no_match` by `fuse_confidence` (`ridge_z`=0.36 < tau) -- correct, ground truth agrees; 2650's
raw candidate (2600, identical to the `Full joint early-fusion` row's prediction) is shown anyway
at confidence 0.4 (`ridge_z`=0.79 >= tau) -- wrong, since ground truth has no correct answer there.
Added as a new table (`tab:nomatch`) directly after the ablation table in `report.tex`, and
shortened the ablation table's own caption to point to it rather than duplicate the explanation.
`report.pdf` rebuilt (still 6 pages), all floats still `[H]` and in source order, checked
page-by-page.

**Second follow-up, same session: a `ridge_z`-only no-match gate for all 9 configurations.** User
asked directly why the ablation couldn't have a confidence method without `cam_disagree`/
`desc_disagree`, since that would help judge which configuration is more robust -- a fair point:
`ridge_z` is the one cue that doesn't need a second, independently-computed path to compare
against (unlike `cam_disagree`/`desc_disagree`, which are specific to the shipped design's
two-stage fusion structure), so it can be computed for any configuration on its own similarity
matrix and path. User chose the fuller scope (all 22 anchors, false-abstain-rate trade-off, not
just the 2 no-match ones) over AskUserQuestion.

Implementation: `early_fusion_path`/`full_joint_early_fusion_path` now also return their `s`
matrix (previously discarded after the DTW call); `_ridge_gate()` added, calling
`verify.ridge_strength` directly (same function the shipped pipeline uses, `window=5` from
`config.yaml`) with `tau=0.75` reused as a single yardstick across all 9 -- explicitly *not*
re-calibrated per configuration (documented as a real caveat, both in the script's `TAU` constant
comment and the report table's caption: the deletion test that chose 0.75 only ever ran once,
against `S_joint`'s own statistics). `S_cam0/S_cam5/S_seqslam/S_dinov2/S_joint.npy` were all
already cached, so 6 of 9 configurations needed no new computation, only loading.

Two results worth having verified rather than assumed:
1. **2650 is caught by none of the 9 configurations** (all `ridge >= tau`) -- a genuinely hard
   case for the similarity signal itself, not a shipped-pipeline-specific weakness, independent
   confirmation of the point already made from the single-config view.
2. **The shipped configuration's ridge-only false-abstain count (1/20) is identical, anchor-for-
   anchor, to its real shipped behaviour** -- cross-checked directly: `outputs/evaluation_anchors.csv`
   shows exactly 1 of the 20 real-position anchors as `wrongly_abstained` (183), and a from-scratch
   recomputation of `ridge_strength(S_joint, path_joint, window=5)` at anchor 183 gives `ridge_z`
   =0.6899, matching the shipped pipeline's own persisted value exactly and confirming both (a)
   the ablation script's ridge computation is correct, not a bug that happens to look plausible,
   and (b) on this 20-anchor sample, `cam_disagree`/`desc_disagree` never independently changed an
   abstain decision -- their contribution is in the reliability table's tier separation, not the
   binary no-match rule.

Added as a new table (`tab:ridge-gate`) in `report.tex`, directly after `tab:nomatch`. `report.pdf`
rebuilt, still 6 pages (table fit into existing page slack), checked page-by-page, all floats
still `[H]`, table numbering (5, 6) auto-updated correctly. `ruff format`/`check` clean on
`notebooks/early_fusion_ablation.py`.

**Third follow-up, same session: the three \S2.3 tables consolidated into one** (user: "banyak
tabel redundant untuk bab 2" -- correct, Tables 3/4/5 repeated the identical 9-configuration row
list three times). Merged into a single `tab:ablation` with three `\cmidrule`-grouped column
blocks -- position error (median/p90/hit@5, 20 anchors), raw DTW prediction at the 2 no-GT
anchors, and the ridge_z-only gate (false-abstain/20, z@2600, z@2650) -- with configuration names
shortened ("SeqSLAM, cam0+cam5 early-fusion (concat)" -> "SeqSLAM only, early fusion"; the
early/late distinction defined once in the caption). `\tabcolsep` reduced to 4.5pt for this table;
no overfull hbox, fits `\textwidth`. The two old captions' analysis moved into the prose paragraph
after the table, which grew from "Three findings" to "Four findings" (new fourth: recognising
no-match is hard for every configuration -- the 29-frame agreement, argmax's 655/655 aliasing,
ridge-gate catching 2650 for 0/9 and 2600 for 5/9, shipped's single false abstain identical with
or without the disagreement cues). Labels `tab:nomatch`/`tab:ridge-gate` deleted, no dangling
`\ref`s (grep-verified). Fig.~`fig:t2-argmax` also pinned `[tbp]` -> `[H]` (the last remaining
non-`[H]` float). Rebuilt: still 6 pages but visibly less dense in \S2.3, ledger renumbered to
Table 4, checked page-by-page.

**Fourth follow-up (2026-09-03): full end-to-end mapping + evaluation for the all-4 early-fusion
variant, ridge-only confidence.** User asked for a mapping.csv and evaluation_anchors in the same
format as shipped but for the early-fusion configuration without `cam_disagree`/`desc_disagree`;
explicitly okayed notebook provenance, output to `notebooks/results/`. New script
`notebooks/early_fusion_pipeline.py`: reuses `full_joint_early_fusion_path` (window=200),
`align.path_to_mapping` (same runB stationary segments), `verify.ridge_strength`/`margin`/
`local_slope`, and the production `confidence.fuse_confidence` with both disagreement cues fed 0
("cue absent, never votes against" -- under which its rules reduce exactly to ridge-only tiers
0.9=ridge>=1.5 / 0.7=ridge>=1.0, no re-implementation that could drift), then the same public
renames/blanking as `outputs/mapping.csv` minus the two disagreement columns, plus the H0
no_frame tail. Wrote `mapping_early_fusion.csv` (2695 rows), `evaluation_anchors_early_fusion.csv`,
`evaluation_summary_early_fusion.json`.

Numbers (variant vs shipped): status matched 2501/2488, ambiguous 130/124, no_match 43/62,
no_frame 21/21. Anchors: n_scored 20/19 (183 now answered, pred 100, error 2 -- shipped wrongly
abstained there); abstain TP=1 (2600), FN=1 (2650, guessed 2600 at conf 0.7), FP=0. Accuracy
comparable: median 0.0, p90 3.5, hit@2/5/10 all 0.90 (shipped hit@5 89.5%). **But the confidence
side degrades sharply**: 2361 of 2695 rows (88%) claim tier 0.9 (shipped: 926, 34%); measured
tier accuracy 0.9 -> 16/18 = 89% (shipped 8/8 = 100%) and 0.7 -> 0/2 = 0% (60/150, the
confident-wrong depot pair, both predicting 81 exactly like shipped's raw path). The shipped
reliability table's monotone tier separation is gone -- the concrete, end-to-end demonstration of
the earlier claim that the disagreement cues' contribution lives in tier separation, not the
no-match rule.

**Stationary-segment analysis figure (2026-09-03)**: `notebooks/stationary_segments_plot.py` ->
`notebooks/results/stationary_segments.png`, 2x2 grid (run x camera) of motion energy + per-cam
threshold + final joint segments. Measured segments: runA `0-173` (174 frames, ~17s -- the depot
start plateau, containing anchors 60/150's region) and `2614-2672` (59 frames, depot return, the
loop-closure tail near anchors 2600/2650); runB `0-29` and `42-53` only (both at the start, much
shorter stop), none at the end -- consistent with runB's earlier-truncated recording. Also visible:
zero below-threshold candidates mid-route in any camera (no false stationary stretches), and
runB's absolute threshold is proportionally lower (0.75 vs runA's 1.69) because the threshold is
relative to each camera's own median -- rain/overcast lowers all of runB's diffs.

