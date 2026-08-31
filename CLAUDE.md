# CLAUDE.md — route-alignment take-home

Standing instructions for every Claude Code session in this repository. Read `plan.md` next.

## 1. What this repo is

Phase-2 take-home test for an autonomous-driving software company. Two recordings of the same route made on
different days (`runA`, `runB`), each with two side-facing wide-angle cameras (`cam0` = left/kerb side,
`cam5` = right/road side), delivered as raw HEVC elementary streams plus one capture-timestamp file per camera
(nanoseconds since the Unix epoch, one line per frame).

Tasks: (1) characterise the recordings, separating **measured** from **declared** values; (2) build a frame-level
correspondence runA → runB (`outputs/mapping.csv`) with a confidence per frame and an honest accuracy estimate —
40 % of the grade; (3) recommend a keep/discard + per-frame metadata policy. Deliverables: a 2–4 page PDF + code.
Other weights: T1+T3 20 %, clarity and proved-vs-assumed separation 20 %, code quality and robustness 20 %.

Priorities in every session, in order: matching quality and honesty of its evaluation → correctness of T1/T3
findings → clarity → polish. The user must be able to defend every line in an interview.

## 2. Layout

```
CLAUDE.md            this file
plan.md              step-by-step plan, candidate methods, schedule (keep its checkboxes current)
findings.md          every reported number + how it was obtained + the exact command
decisions.md         ADR-lite: choice, alternatives considered, why, evidence
README.md            how to reproduce (make all), environment, pinned versions
config.yaml          the single source of parameters (paths, sizes, windows, thresholds)
requirements.txt     pinned dependencies (the ffmpeg -version string is recorded in README)
Makefile             one target per pipeline stage
dataset/             raw inputs — READ ONLY (runA/, runB/, SHA256SUMS.txt)
data/cache/          generated, git-ignored: frames/{run}/{cam}/00000.jpg, desc_*.npy(+.json), S_*.npy, path_*.npy, motion_*.npy
src/routealign/      the package (see §3)
tests/               pytest, one file per module
gt/                  labelling_protocol.md, anchors_user.csv, anchors_claude.csv, anchors_merged.csv
outputs/             inspection/, figures/, mapping.csv, mapping_full.csv, keep_manifest.csv, SCHEMA.md
report/              report.tex → report.pdf (build files git-ignored)
notebooks/           exploration only; nothing in outputs/ may come from a notebook
```

## 3. Package structure — one responsibility per module

| module | responsibility (one sentence) |
|---|---|
| `config.py` | `Config` frozen dataclass + `load_config(path, overrides)`; nothing else reads `config.yaml` |
| `io_video.py` | ffprobe/ffmpeg subprocess wrappers: `probe`, `count_decoded`, `scan_nals`, `sps_vui`, `decode_to_jpegs`, `sha256` |
| `timestamps.py` | `load_timestamps`, `interval_stats`, `frame_times` — pure functions on int64 arrays |
| `frames.py` | `FrameStore` (JPEG-cache access, crops, LRU), `motion_energy`, `stationary_segments` |
| `descriptors.py` | `seqslam_descriptors`, `dino_descriptors`, `cached` — arrays in, arrays out |
| `similarity.py` | `cosine`, `local_contrast_norm`, `fuse` |
| `align.py` | `dtw_open_ends`, `nullstate_dp`, `path_to_mapping`, `local_slope` |
| `verify.py` | `ridge_strength`, `margin`, `path_disagreement`, `sift_verify` |
| `confidence.py` | `fuse_confidence` (tiers + no-match rule), `reliability_table` |
| `groundtruth.py` | `make_sheets` (model-blind), `load_labels`, `evaluate`, `deletion_test` |
| `policy.py` | `build_manifest` + schema export |
| `viz.py` | every figure in `outputs/figures/`; no computation |
| `cli.py` | argparse subcommands; the only place with `print` and `sys.exit` |

Dependency direction is one-way and must stay so:

```
cli → (policy, groundtruth, confidence, viz) → (verify, align) → (similarity, descriptors)
    → (frames, timestamps, io_video) → config
```

No module imports upward; no circular imports; a module that starts doing two things gets split.

## 4. Code style rules (each one checkable)

- Python 3.11, `from __future__ import annotations`, full type hints on every public function. Arrays are
  annotated with `numpy.typing.NDArray` and the docstring states shape and dtype, e.g. `float32[NA, NB]`.
- Functions ≤ ~40 lines, one job each, pure where possible (inputs → outputs, no hidden state). I/O lives at
  the edges (`io_video`, `frames`, `cli`); the maths in the middle (`similarity`, `align`, `verify`) never
  touches the disk.
- Naming: `snake_case` functions/variables, `PascalCase` classes and dataclasses, `UPPER_CASE` module constants.
  Reserved index names: `i` = runA row, `j` = runB column, `k` = frame index within a run — never reused for
  anything else. Run and camera ids are the literal strings `"runA" | "runB"` and `"cam0" | "cam5"` everywhere.
- No magic numbers in module bodies: every threshold, window, size and penalty lives in `Config` (loaded from
  `config.yaml`) with a `# why:` comment giving the reason or the test that chose it.
- Data crossing a module boundary is a small frozen `@dataclass` (e.g. `NalStats`, `AlignmentPath`) or a
  pandas DataFrame whose columns are documented in `outputs/SCHEMA.md`. No ad-hoc dicts or tuples across
  boundaries.
- Validate at the boundary and fail loudly: raise `ValueError` with the offending value. A non-monotone
  timestamp file or a frame-count mismatch stops the pipeline; it is never silently coerced.
- Logging: `log = logging.getLogger(__name__)`; `print` only in `cli.py`; `tqdm` only for loops longer than ~5 s.
- Caching: `cached(fn, key)` writes `.npy` plus a `.json` sidecar (source SHA-256 prefix, parameters, package
  version, `ffmpeg -version` line). Cache hits are logged; `--force` recomputes. Never cache full-resolution frames.
- Determinism: fixed seeds, sorted file iteration, no wall-clock values in outputs; running `make all` twice must
  produce byte-identical CSVs.
- Formatting and lint: `ruff format` and `ruff check` (line length 100) clean before every commit. Google-style
  docstrings on public functions. Comments say *why*, not *what*.
- Tests: `pytest` in `tests/`, one file per module, synthetic fixtures; every DP and parser has an exactness test
  against a naive reference implementation; one skip-if-missing smoke test on the real data (`--limit 60`).
- CLI: `python -m routealign <subcommand> --config config.yaml [--limit N] [--force]`; each subcommand equals
  one Makefile target; subcommands compose and never duplicate each other's logic.
- Commits: small, imperative subject line, body says what and why. Raw data, caches and `outputs/*.npy` are
  never committed.

## 5. Data conventions every piece of code must respect

- **Index convention (H0, head-aligned):** frame index `k` = line `k` of the run's timestamp file = decoded
  display-order frame `k` of each camera. Both cameras of a run share one timestamp file (byte-identical).
  Video frames exist only for `k < decoded_count(run, cam)`; the remaining timestamp lines are `no_video`.
  H0 is a documented assumption with evidence (see `findings.md`), not a fact — keep it labelled as such.
- **Declared vs measured:** every number carries its provenance. Declared = filename token `_20_`, SPS/VUI
  (`10/1`), ffprobe defaults (`avg_frame_rate 25/1`, `color_range tv`), the timestamp files. Measured = full
  decode counts, NAL parse, timestamp arithmetic, image content. Never report a default as a fact.
- Never assume `len(timestamps) == decoded frames` (they differ: 2695 vs 2674 in runA, 2663 vs 2622/2615 in
  runB) or that cam0 and cam5 are index-synchronous; both are tested and the results recorded.
- Video I/O only through the `ffmpeg` / `ffprobe` CLI via `subprocess`, always with `-fps_mode passthrough`.
  Never `-hwaccel videotoolbox` (fails on these streams). Never install or import PyAV (conflicts with OpenCV's
  bundled ffmpeg on macOS). Never hold a full-resolution run in RAM (16 GB machine) — downscale in the filter graph.
- Time is `int64` nanoseconds; derived seconds are `float64`. Local time (UTC+8) is an inference from scene
  content and is labelled as such wherever it appears.
- Test anchors are never used for tuning; dev anchors, synthetic warps and the deletion test are.

## 6. Commands

```
conda activate phase2-vpr                 # Python 3.11 env: numpy, scipy, pandas, opencv, matplotlib, tqdm
make verify-data                          # SHA-256 of all inputs vs dataset/SHA256SUMS.txt
make characterise                         # Task 1 → outputs/inspection/, figures, findings.md numbers
make decode                               # one-pass JPEG cache (640×480) per run/camera
make describe FEAT=seqslam|dinov2         # descriptors → data/cache
make align                                # similarity + DTW → mapping (draft)
make verify                               # agreement cues, SIFT, confidence → outputs/mapping.csv
make gt-sheets                            # model-blind labelling sheets
make evaluate                             # metrics vs gt/anchors_merged.csv, reliability table, deletion test
make manifest                             # Task 3 → outputs/keep_manifest.csv + SCHEMA.md
make report                               # LaTeX → report/report.pdf
make lint test                            # ruff + pytest
make all                                  # everything from a clean cache; prints per-stage timings
```

`--limit N` on any subcommand processes only the first N frames (smoke runs). Missing tools to install before
the first run: `pytest`, `ruff`, `pyyaml`, and `torch` (macOS arm64 wheel) for the DINOv2 descriptor.

## 7. Working agreement with Claude

- Do not run pipeline stages, install packages, decode video, or run git commands unless the user asks for it
  in that session. Propose the exact command, then wait. Anything expected to take > 10 minutes gets a heads-up.
- Keep `plan.md` checkboxes, `findings.md` and `decisions.md` current as work progresses; a number without its
  provenance is not finished.
- Ship nothing the user cannot explain: non-obvious choices get a `# why:` comment and an entry in `decisions.md`
  with the alternatives that were considered.
- Definition of done: `outputs/mapping.csv` (2695 runA rows, schema test green), `report/report.pdf`
  (2–4 pages), README reproduction, `make lint test` green, `make all` from a clean cache, findings/decisions
  current, submission archive built.




## 8. Docstring Standard

1. Follow Google Python Style Guide for docstrings.
2. Explain concisely what a class and/or function does.
3. List the arguments and public attributes and their data types, exceptions that are triggered by a function, and what the return type is. If any of these are empty or None, omit the section.
4. If there are class methods that are marked with the `@property` decorator, list them as an attribute. The type should be the property getter method's return type.
5. No need to list the methods.
6. No need to list protected or private attributes.
7. If a function argument has a default, clearly indicate it as optional (e.g. `arg (type, optional)`). In the last sentence of the argument's explanation, state what the default value is (e.g. "Defaults to None").
8. The above rules apply for both public and protected methods.
9. For the constructor, no need to put its `Args` section in the class docstring. Only put them inside the `__init__` method's docstring.