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
