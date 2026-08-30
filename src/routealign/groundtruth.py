"""Model-blind labelling sheets, label loading/merge, evaluation, the deletion test.
Step 6 (plan.md §8).

Labelling is user-primary; Claude's pass afterward is a QA/consistency check, not an independent
co-labeller (plan.md §0.2, §13, revision note at the top) — the two share a vision backbone with
the pipeline's own DINOv2 descriptor and are not statistically independent judges. `evaluate` must
report user-vs-Claude agreement as a sanity number, never as inter-rater reliability or a
label-noise floor.
"""

from __future__ import annotations


def make_sheets(*args, **kwargs):
    """Coarse (every 20th frame) then fine (+/-20 around the coarse pick) labelling tiles, built
    only from the JPEG cache — never from model output. Not yet implemented — Step 6."""
    raise NotImplementedError("make_sheets: implement in Step 6 (plan.md §8)")


def load_labels(*args, **kwargs):
    """Read gt/anchors_user.csv and gt/anchors_claude.csv. Not yet implemented — Step 6."""
    raise NotImplementedError("load_labels: implement in Step 6 (plan.md §8)")


def evaluate(*args, **kwargs):
    """Error / hit@k / Wilson-95%-CI metrics vs gt/anchors_merged.csv, split dev/test, per-stratum
    and per-confidence-tier breakdowns. Not yet implemented — Step 6."""
    raise NotImplementedError("evaluate: implement in Step 6 (plan.md §8)")


def deletion_test(*args, **kwargs):
    """Delete a known runB span, rerun, check the mapping abstains on the corresponding runA rows.
    Needs no manual labels — a self-supervised check on the no-match rule. Not yet implemented."""
    raise NotImplementedError("deletion_test: implement in Step 6 (plan.md §8)")
