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

## D3 — `outputs/mapping.csv` ships as a SeqSLAM-only draft; DINOv2 fusion is deferred, not faked

**Choice:** the first real, complete `outputs/mapping.csv` (2695 rows, all schema checks green,
`findings.md`) was produced using SeqSLAM descriptors only. `desc_disagree` (needs a second,
independently-computed descriptor per plan.md §7) is left empty rather than filled with a
placeholder. `fuse_confidence` already treats an empty `desc_disagree` as "does not satisfy the
agreement check" (never special-cased for this — it falls out of ordinary `<=` comparisons on a
missing value), so every row in the current file is structurally capped at tier 0.4/0.5 — no row
can reach 0.7 or 0.9 yet, and `outputs/SCHEMA.md` says so explicitly.

**Alternatives considered:** (a) set `desc_disagree = 0.0` (perfect agreement) so more rows reach
the top tiers; (b) skip shipping `mapping.csv` at all until DINOv2 is verified; (c) this session's
choice — ship a complete, correct, honestly-capped draft now, add DINOv2 as a follow-up.

**Why:** (a) is a false-confidence bug wearing the shape of a convenience — reporting "two
descriptors agree" when only one exists directly contradicts CLAUDE.md §1's top priority (honesty
of the matching evaluation). (b) throws away a real, working, fully-tested pipeline over one
missing input, and violates plan.md §12's own schedule, which treats "mapping v0" (SeqSLAM-only
draft) as an explicit, ordinary milestone before DINOv2 comparison. `dino_descriptors` is fully
implemented and unit-testable, just not verified end-to-end here: `torch.hub.load` needs to fetch
the DINOv2 repo from `codeload.github.com`, which this sandbox's network allowlist blocks (`HTTP
Error 400`, reproduced via both `urllib` and bare `curl`; the weight file itself, hosted at
`dl.fbaipublicfiles.com`, downloads fine — so this is an environment restriction, not a bug in the
code or a DINOv2 problem). The user's own terminal is not subject to this sandbox and should be
able to run `describe --feat dinov2` directly.

**Evidence:** `outputs/mapping.csv`/`mapping_full.csv`, generated for real (`findings.md` Step 5
CLI entry); `tests/test_confidence.py`'s SIFT-degradation tests already established the same
missing-signal-degrades-gracefully pattern before this decision reused it for `desc_disagree`.

## D4 — cam0/cam5 fusion pads the shorter camera's columns, does not truncate both

**Choice:** runB's cam0 (2622 decoded frames) and cam5 (2615) differ in length, so their
z-scored similarity matrices have different `NB` and cannot be fused directly (`similarity.fuse`
requires matching shapes). The shorter camera's matrix is padded on the right with `-3.0` (the
z-score clip floor) up to the wider camera's width, then fused with equal weight, rather than
truncating both matrices to `min(2622, 2615) = 2615` columns.

**Alternatives considered:** (a) truncate both to the shorter camera's width; (b) pad the shorter
camera with the clip floor (chosen).

**Why:** cam0 and cam5 within one run share a single timestamp file (CLAUDE.md §5's H0
convention), so column `j` means the same nominal instant in both matrices wherever both exist —
truncating would silently discard 7 real, valid columns of cam0 evidence (columns 2615-2621) for
no reason other than cam5 not having decoded that far. Padding with the worst possible z-score
means those columns simply get no vote from cam5 (as if cam5 "abstains"), while cam0's real signal
there is preserved — the fused matrix's rightmost 7 columns are effectively cam0-only, which is
honest (that's genuinely all the evidence that exists there) rather than lossy.

**Evidence:** measured decoded counts (`findings.md`, Step 0); confirmed no shape-mismatch error
and no schema violation across the real full-scale `align`/`verify` run.

## D5 — `stationary_segments`' exclusive end vs. `path_to_mapping`'s inclusive `(lo, hi)`

**Choice:** `frames.stationary_segments` returns `(start, end)` with `end` **exclusive** (ordinary
Python slice convention: `[start:end]`, already tested that way in `tests/test_frames.py`-adjacent
work from Step 2). `align.path_to_mapping`'s `stationary_b` parameter expects **inclusive**
`(lo, hi)` pairs (its docstring, and `tests/test_align.py`'s stationary-widening test). These are
genuinely different, and were not reconciled until wiring the `align` CLI subcommand: the
conversion `[(s, e - 1) for s, e in stationary_b_exclusive]` now happens explicitly, once, at the
call site in `cli.py`, with a comment pointing at this entry.

**Alternatives considered:** (a) change `stationary_segments` to return an inclusive end, matching
`path_to_mapping`; (b) change `path_to_mapping` to expect an exclusive end, matching
`stationary_segments`; (c) leave both as-is (each is independently the more natural convention for
its own use — Python slicing for `stationary_segments`, human-readable inclusive frame ranges for
`path_to_mapping`'s public output columns), convert explicitly at the one call site that bridges
them (chosen).

**Why:** (a)/(b) would each require re-deriving/re-verifying an already-tested function and
touching already-reported numbers (`findings.md`'s "(0, 173)"-style stationary segment reports)
for a difference that is not actually a bug in either function — it is two reasonable conventions
meeting at a module boundary. Making the conversion explicit, commented, and localised to one spot
is safer than silently picking one convention as "correct" and quietly breaking the other's
existing tests/reports.

**Evidence:** caught by re-reading both functions' actual contracts before wiring `_cmd_align`,
not by a test failure — worth noting as a case where reading the code carefully before using it
prevented an off-by-one that unit tests in isolation could not have caught (each function's own
tests are self-consistent; only the *combination* was wrong).

## D6 — `make_sheets` takes all 4 frame stores, not the single `store_b` the stub declared

**Choice:** `groundtruth.make_sheets`'s signature changed from `(store_b, anchor_indices,
out_dir)` to `(store_a0, store_a5, store_b0, store_b5, anchor_indices, out_dir, coarse_step,
fine_radius, coarse_picks)`.

**Why:** the stub's own docstring already promised things a single `FrameStore` cannot produce:
"every 20th runB frame **for both cameras**" needs `store_b0` and `store_b5` both; a usable
labelling tool also has to show the labeller *what they are looking for* -- the runA anchor's own
frame -- which needs `store_a0`/`store_a5` too. Discovered by re-reading the stub's docstring
carefully before implementing it (same discipline as D5), not by a test failure.

**Evidence:** real run, `gt-sheets`: 22 anchor reference tiles + 132 coarse overview tiles
(`gt/sheets/`), visually confirmed legible (both cameras side by side, index burned in, readable)
by inspecting `anchor_refs/00476.jpg` and `coarse/00500.jpg` directly.

## D7 — stratum S's "~110 frame" spacing is approximate prose, not a literal formula

**Choice:** `groundtruth.stratum_s_indices` builds stratum S as 18 evenly-spaced sweep points
(`np.linspace(200, 2500, 18)`) plus jitter (±20, seeded) plus 4 fixed points (60, 150, 2600,
2650) = 22 total, matching `config.yaml`'s already-committed `stratum_s: 22`. The resulting sweep
spacing is ~135 frames, not exactly the "~110" in plan.md's prose.

**Why:** plan.md's "runA 200 … 2500 every ~110 frames ... + plateau frames 60, 150 + tail frames
2600, 2650" does not arithmetically reconcile to exactly 22 total anchors for any clean literal
reading of "every 110 frames" (a strict `arange(200, 2501, 110)` gives 21 sweep points -> 25
total with the 4 fixed points, not 22). Rather than guess which part of the prose was the typo,
the total anchor count (22) was treated as the load-bearing, already-committed number (used
downstream for the dev/test split and CI reporting), and the spacing was derived from it, not
the reverse. "~110" was written before this reconciliation happened.

**Evidence:** none needed beyond the arithmetic itself; `tests/test_groundtruth.py` locks the
count (22) and the fixed points down so this can't silently drift.

## D8 — Claude filled `gt/anchors_user.csv` directly, inverting D2's user-primary design

**Choice:** at the user's explicit request ("sekarang kamu isi semuanya dong, nanti saya
verifikasi"), Claude produced the primary fill of all 22 stratum S entries in
`gt/anchors_user.csv`, with the user reviewing/correcting afterward, rather than the user
labelling first (blind) and Claude doing a QA pass second — the design D2 committed to.

**Alternatives considered:** (a) refuse/insist on the original order; (b) do it as requested,
document the trade-off plainly (chosen); (c) do it silently without flagging the methodology
change.

**Why:** the user is time-constrained (48 h total) and made an informed trade-off after this
session had already explained, at length, why Claude's read is not independent of the pipeline
(D2: shared DINOv2 backbone, correlated failure modes) — including declining to label anchor
183 earlier in the same session specifically because of direct prior exposure to that anchor's
matched-frame pair. (a) would be unhelpfully rigid given the user already understands and
accepts the cost. (c) would violate this project's core "declared vs measured" / "ship nothing
you cannot explain" ethos — a silent methodology change is exactly the kind of thing that fails
at interview when asked "who labelled your ground truth?".

**Consequence, stated plainly:** `gt/anchors_user.csv` as it stands is **Claude-labelled,
user-verified**, not independent human ground truth. Any accuracy number computed against it
(Step 6's `evaluate`) must be reported with that caveat attached, not presented as if the user
had labelled blind. It only becomes closer to the originally-intended methodology if the user's
verification pass substantively re-derives labels (actually re-checking the coarse/fine sheets
against Claude's picks) rather than skimming for obviously-wrong entries. Two entries carry an
extra explicit flag beyond the general caveat: anchor 183 (Claude was not blind for this one
specifically, said so in the `note` column, and gave a heavily-caveated best guess rather than
skipping it, per the user's instruction to fill in "everything"); anchor 346 (`no_match` — no
matching scene found in either the coarse overview or a widened fine search around the initial
estimate; may need a fresh, wider search rather than being a genuine absence).

**Evidence:** `gt/anchors_user.csv` (22 rows, all filled); the visual search process is not
separately logged beyond the CSV's `note` column and this entry — unlike the pipeline's own
numbers, this labelling pass has no independent re-derivation path other than the user's own
review, which is exactly why the caveat above must travel with any number derived from it.

**Update (2026-09-01) — superseded by a real re-derivation pass.** The user completed their own
independent, frame-by-frame verification of all 22 anchors (not a skim — every `lo`/`hi` range
tightened substantially from Claude's original wide/uncertain guesses to precise few-frame
windows, one anchor corrected from `no_match` to a real find (346), two corrected from a
plausible-but-unverified guess to genuine `no_match` (2600, 2650)). This satisfies the "closer to
the originally-intended methodology" condition stated above. `gt/anchors_user.csv` is now treated
as real, user-derived ground truth for `evaluate()` purposes — the "Claude-labelled" caveat no
longer applies to the *values*, though the historical fact that Claude produced the first-pass
draft (and that no second, independent human labeller exists) is still worth disclosing in the
report for full transparency about the process. Validated the pasted-back values programmatically
(`lo <= runB_best <= hi` for all 20 non-`no_match` rows, 0 violations) before accepting them
rather than trusting the spreadsheet round-trip blindly. Real `evaluate()` result against this
now-trustworthy ground truth: reliability table monotonic and clean (tier 0.4 → 50%, 0.7 → 71%,
0.9 → 100%, n=4/7/8) — see `findings.md` for the full numbers.

## D9 — DINOv2-only similarity has a severe attractor column; `desc_disagree` is not yet trustworthy

**Choice:** ship the SeqSLAM+DINOv2 fused `outputs/mapping.csv` as-is (real, reproducible), but
flag prominently (here, `findings.md`, `outputs/SCHEMA.md`) that `desc_disagree` and the
resulting jump in `no_match` rows (106 -> 688) are currently **not a trustworthy signal** and
should not be read as "the pipeline got more honest." Do not silently accept the new number.

**What was found:** once DINOv2 network access unblocked (2026-08-31, see the plan.md deadline
revision), `describe --feat dinov2` and a re-run of `align`/`verify` were done for real.
`desc_disagree`'s median jumped to **720 columns** (out of ~2622) -- an order of magnitude beyond
anything SeqSLAM ever showed. Traced directly: `path_dinov2.npy` (the DINOv2-only DTW path) is
almost completely collapsed -- stuck at columns 1731-1747 for nearly the entire route (`runA`
rows 0 to 2673), instead of tracking the diagonal the way SeqSLAM's own path does. `S_dinov2.npy`
column 1747 has the single highest mean z-score of all 2622 columns (1.70) -- a severe attractor
column, the same phenomenon already root-caused for SeqSLAM/`margin` in Step 5 (`local_contrast_norm`
amplifying a low-local-variance region into a spurious near-ceiling z-score), but far more severe
here. Plausible reason DINOv2 is more susceptible: its 768-dim semantic embedding likely varies
*less* locally across nearby runB columns than SeqSLAM's raw-patch descriptor does in some
stretches, so `local_contrast_norm`'s per-row local-window z-scoring divides by an even smaller
local std there, amplifying more.

**Why the joint mapping is probably still mostly fine, but confidence isn't**: `S_joint`/`path_joint`
were checked directly and are *not* collapsed (start/end (0,77)->(2673,2587), close to the
SeqSLAM-only run's (0,78)->(2673,2601)) -- SeqSLAM's stronger, better-behaved signal dominates the
equal-weight (1.0/1.0) fusion. But `desc_disagree` (which compares the *DINOv2-only* path against
the SeqSLAM-only path) inherits DINOv2's collapse directly, so it now massively over-triggers the
no-match rule's "`cam_disagree` and `desc_disagree` both above 30" condition (418 of the 667
new-total no_match rows) -- this is an artifact of a broken per-descriptor signal poisoning a
verification cue, not evidence the underlying match quality regressed by nearly that much.

**Alternatives considered:** (a) revert to the SeqSLAM-only `mapping.csv` and treat this as an
unshipped experiment; (b) ship as-is with prominent caveats (chosen); (c) ad-hoc lower DINOv2's
fusion weight or change its `local_contrast_norm` window right now to "fix" the numbers.

**Why:** (a) throws away a real, reproducible finding and reverts to a *known-incomplete* state
(desc_disagree permanently NaN) instead of a *known-imperfect-but-diagnosed* one -- worse for
defensibility, not better ("why is desc_disagree empty" is a harder question to answer well than
"here's a specific, traced attractor-column problem and here's the plan to fix it"). (c) would be
exactly the kind of untested magic-number tweak CLAUDE.md §4 exists to prevent -- any reweighting
or window change needs the synthetic-warp/deletion-test harness (Step 6, not yet built) to justify
it, not eyeballing one bad run. (b) keeps the real artifact, keeps the finding honest and traceable,
and leaves the actual fix for proper calibration.

**Next step, not yet done**: investigate whether a different `local_contrast_norm` window suits
DINOv2 specifically (its candidate list in config.yaml, `{20, 50, 100}`, was chosen without DINOv2
in mind), or whether per-descriptor windows are needed at all, once the synthetic-warp test
harness exists to validate a change rather than guess at one.

**Evidence:** `data/cache/S_dinov2.npy`, `data/cache/path_dinov2.npy`, `data/cache/path_seqslam.npy`,
`data/cache/S_joint.npy`, `data/cache/path_joint.npy`, `outputs/mapping.csv`/`mapping_full.csv`
(2026-08-31 run) -- all inspected directly, not inferred.

## D10 — DINOv2 needs its own (much wider) `local_contrast_norm` window than SeqSLAM

**Choice:** `similarity.contrast_window_default` (shared, 50) split into
`contrast_window_seqslam: 50` and `contrast_window_dinov2: 500` (`config.yaml`), wired
per-descriptor in `_cmd_align` (D9's follow-up, now done).

**Evidence, not a guess**: tested 4 window values (50, 200, 500, 2622/full-row) by actually
rerunning `cosine -> local_contrast_norm -> dtw_open_ends` for DINOv2-only end to end and reading
the resulting path's `runB_frame` at 6 spread-out rows. `window=50`: path collapsed (stuck at
column ~1747 for nearly the whole route -- D9). `window=200/500/2622`: path tracked the diagonal
cleanly and closely matched SeqSLAM's own independently-computed path (e.g. row 1000: DINOv2
896-897 vs SeqSLAM 898) -- genuine corroboration, not noise. 500 chosen over 200/2622: empirically
indistinguishable from either in the mid-route rows tested, but keeps *some* locality (illumination
drift correction) rather than going fully global, without the cost `window=50` had.

**Real-run impact** (`findings.md`): `desc_disagree` median dropped from 720 to 1; `no_match` from
667 down to **62** (better than the pre-DINOv2 SeqSLAM-only baseline of 106); confidence tier 0.9
went from 0 rows to **926**, tier 0.7 to **1104** -- DINOv2 now functions as intended, a second
descriptor that corroborates rather than a broken signal that poisons verification.

**Caveat, checked before declaring victory**: ran `evaluate()` against the anchors in
`gt/anchors_user.csv` -- still Claude-labelled, not yet user-verified (decisions.md D8) -- and
the resulting reliability breakdown is *inverted* at this sample size: tier 0.9 shows 28.6% (2/7)
accuracy, tier 0.4 shows 80% (4/5). This is **not** being reported as "DINOv2 broke calibration" --
sample sizes are tiny (5-7 anchors per tier) and two of the tier-0.9 "misses" have ground-truth
labels that are themselves shaky (anchor 738's own note admits its identifying feature "wasn't
clearly visible"; anchor 891 is `quality=sure` but from an unverified labeller). A real read on
calibration needs the user's finished verification pass first -- this makes that verification more
urgent, not less, now that the pipeline's confidence numbers are no longer artificially capped.

**Evidence**: window sweep output, real `align`/`verify` re-run, `evaluate()` output -- all
reproduced directly, not asserted from memory (`findings.md`).
