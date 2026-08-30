# decisions.md — ADR-lite: choice, alternatives considered, why, evidence

## D0 — Video I/O via ffmpeg/ffprobe subprocess only; never PyAV

**Choice:** all video decode and metadata extraction goes through the `ffmpeg`/`ffprobe` CLI
binaries via `subprocess` (`io_video.py`). `opencv-python` is used only for array-level operations
on already-decoded frames, never to open a video file. PyAV (`av`) is not a dependency.

**Alternatives considered:** (a) PyAV for frame-accurate decode + metadata access in pure Python;
(b) `cv2.VideoCapture` for decoding.

**Why:** (a) was tried first. Importing `av` alongside `opencv-python` in the same process produced
a macOS ObjC `libavdevice` class-duplication warning — each package bundles its own copy of the
ffmpeg shared libs at a different version, and the warning explicitly states this "may cause
spurious casting failures and mysterious crashes." Given these are raw HEVC elementary streams with
no container-level metadata (already an edge case for most video bindings), the risk wasn't worth
it when the system `ffmpeg` 8.1.1 (with `libx265`) already does everything needed. (b) was not
tried — `ffmpeg` subprocess gives more direct, inspectable control over exactly which flags are
used (`-fps_mode passthrough`, `-start_number 0`), which matters for the H0 index convention.

**Evidence:** the import-time warning, reproduced when both `opencv-python` and `av` were installed
together in `phase2-vpr` (see `findings.md`). `av` was uninstalled; not reinstalled since.

## D1 — SIFT verification and NULL-state DP are cut by default, not "cut if behind"

**Choice:** S4 (SIFT + MAGSAC verification, `verify.sift_verify`) and v2 (NULL-state affine-gap DP,
`align.nullstate_dp`) are not built unless the day-2 buffer slot is reached with genuine time to
spare (`plan.md` §12, `config.yaml` `verify.sift.enabled` / `align.nullstate_dp.enabled` both
`false`). The originally-drafted plan treated them as default scope, cut only if the schedule slipped.

**Alternatives considered:** (a) keep them as default scope, cut first if behind — the original
plan; (b) drop them entirely, never revisit — loses real upside (SIFT verification is a
well-understood confidence signal; NULL-state DP is a cleaner abstention mechanism than a post-hoc
threshold).

**Why:** the original schedule assumed roughly 9.5 h/day of friction-free execution by someone
already fluent in DTW and MAGSAC. Discovering at hour 30 of 48 that these don't fit is worse than
deciding upfront not to depend on them. `verify.ridge_strength` / `path_disagreement` (cam0 vs
cam5, SeqSLAM vs DINOv2) already give multi-signal verification without SIFT, and v1's post-hoc
no-match threshold (calibrated by the deletion test) already gives abstention without v2.

**Evidence:** none yet — a scoping decision, not an empirical one. Revisit only if day 1 finishes
with slack (`plan.md` §12's "add only if ahead" order: SIFT, then NULL-state DP).

## D2 — Ground truth is user-primary with a Claude QA pass, not two independent labellers

**Choice:** the user labels all anchors, blind to model output by construction (`groundtruth.
make_sheets` reads only the JPEG cache). Claude does a second blind pass afterward that flags rows
to re-check; the user's (possibly revised) label ships as the consensus. User-vs-Claude agreement
is reported as a QA sanity number, never as inter-rater reliability or a label-noise floor.

**Alternatives considered:** (a) treat the user and Claude as two independent co-labellers and
merge/average their labels, as originally planned; (b) skip the Claude pass entirely.

**Why:** Claude's visual judgement is not independent of the DINOv2 descriptor used inside the
pipeline being evaluated — both are vision-model pattern matching over the same frames and can
share failure modes (e.g. the same loop-closure aliasing, the same rain-degraded stretch). Two
non-independent judges agreeing does not measure accuracy the way two independent ones would; (a)
would silently overstate confidence in exactly the report section (Task 2's honest accuracy
estimate) that carries the most weight. (b) is available but throws away a genuinely useful QA
pass at no real cost.

**Evidence:** none yet — a methodology decision made before any labelling happened, recorded so it
cannot be forgotten once real numbers exist. `plan.md` §13's ledger records the rejected version of
this assumption explicitly ("user and Claude are independent ground-truth labellers" — rejected).
