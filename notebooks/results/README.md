# notebooks/results/

Persisted output from `notebooks/*.py` exploration scripts — regeneratable, not hand-edited.
Full narrative interpretation of every number here lives in `findings.md` / `decisions.md`
(search for the matching filename); this directory only holds the raw table each script writes.

## `early_fusion_ablation.csv`

Written by `notebooks/early_fusion_ablation.py` — run it again (`conda run -n phase2-vpr python
notebooks/early_fusion_ablation.py`) to regenerate this file from the cached descriptors in
`data/cache/` and the ground truth in `gt/anchors_user.csv`.

Columns: `configuration` (which fusion strategy / baseline), `n` (anchors scored — some of the
22 are excluded: `no_match` ground truth or missing `lo`/`hi`, see `groundtruth.evaluate`'s
docstring), `median_error` / `p90_error` (frames, 0 if the prediction falls inside `[lo, hi]`),
`hit_at_5` (fraction of anchors with error ≤ 5 frames).

Interpretation, root-causing of the SeqSLAM late-fusion outlier (anchors 60/150/183), and the
early-fusion-vs-verification-signal trade-off discussion: `findings.md`, "Step 6 — ablation
study, including a user-proposed early-fusion experiment"; design rationale: `decisions.md`.

## `ridge_only_gate.csv`

Also written by `notebooks/early_fusion_ablation.py`, same run. A `ridge_z`-only no-match gate
(production `tau=0.75`/window 5, no `cam_disagree`/`desc_disagree`) applied to every
configuration. Columns: `false_abstain_n`/`false_abstain_rate` (of the 20 real-position anchors,
how many the gate would wrongly abstain on), `ridge_2600`/`caught_2600` and
`ridge_2650`/`caught_2650` (ridge value at the 2 no-ground-truth anchors and whether it falls
below tau there). Discussion: `findings.md`, "Second follow-up" under the Step 8 entry.

## `stationary_segments.png`

Written by `notebooks/stationary_segments_plot.py`. 2x2 grid (run x camera): motion-energy curve,
each camera's own threshold line (`stationary_threshold_frac` x its median -- relative, so runB's
rain-dimmed contrast gets a proportionally lower bar), a light band where that camera alone is
below threshold, and green spans for the final run-level segments (both cameras below threshold
for >= `stationary_min_length` frames). Segments found: runA 0-173 (174f, depot start) and
2614-2672 (59f, depot return); runB 0-29 (30f) and 42-53 (12f), both at the start, none at the
end (runB's recording is truncated earlier). Analysis only.

## `mapping_early_fusion.csv`, `evaluation_anchors_early_fusion.csv`, `evaluation_summary_early_fusion.json`

Written by `notebooks/early_fusion_pipeline.py` (`conda run -n phase2-vpr python
notebooks/early_fusion_pipeline.py`, from the repo root). A full end-to-end mapping + evaluation
for the all-4 early-fusion configuration with a ridge-only confidence — same formats as
`outputs/mapping.csv` / `outputs/evaluation_anchors.csv` / `outputs/evaluation_summary.json`,
minus the `cam_disagree`/`desc_disagree` columns (not computable under early fusion: there is no
second, independently-computed path to disagree with). Experiment only — the shipped deliverable
in `outputs/` is untouched. Discussion: `findings.md`, "Fourth follow-up" under the Step 8 entry.
