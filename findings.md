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
