"""Schema tests for outputs/mapping.csv (CLAUDE.md §10). Skipped if the file doesn't exist --
it is generated from the real dataset (`python -m routealign align && ... verify`), not shipped
in git.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_MAPPING_PATH = Path(__file__).resolve().parent.parent / "outputs" / "mapping.csv"

pytestmark = pytest.mark.skipif(
    not _MAPPING_PATH.exists(),
    reason="outputs/mapping.csv not generated -- run `align` then `verify` first",
)


@pytest.fixture(scope="module")
def mapping() -> pd.DataFrame:
    """Load the real, generated `outputs/mapping.csv` once per test module run."""
    return pd.read_csv(_MAPPING_PATH)


def test_every_runa_frame_appears_exactly_once(mapping: pd.DataFrame) -> None:
    """One row per runA timestamp line, in order, no gaps or duplicates."""
    assert mapping["runA_frame"].tolist() == list(range(len(mapping)))


def test_confidence_is_in_unit_range_or_empty(mapping: pd.DataFrame) -> None:
    """`confidence` is an ordinal tier in [0, 1] wherever it isn't empty (unmatched rows)."""
    conf = mapping["confidence"].dropna()
    assert ((conf >= 0) & (conf <= 1)).all()


def test_runb_frame_is_empty_iff_status_is_unmatched(mapping: pd.DataFrame) -> None:
    """The no-match rule's actual contract: an abstained row reports no `runB_frame` at all,
    not a value the pipeline has already said it doesn't trust."""
    unmatched = mapping["status"].isin(["no_match", "no_video"])
    assert (unmatched == mapping["runB_frame"].isna()).all()


def test_runb_frame_is_monotone_over_matched_and_ambiguous_rows(mapping: pd.DataFrame) -> None:
    """Both runs traverse the route in the same order (plan.md §6): `runB_frame` must never
    decrease across rows that actually report one."""
    core = mapping[mapping["status"].isin(["matched", "ambiguous_range"])]
    assert (core["runB_frame"].diff().dropna() >= 0).all()


def test_status_values_are_within_the_documented_set(mapping: pd.DataFrame) -> None:
    """No stray/typo'd status strings snuck in."""
    documented = {"matched", "ambiguous_range", "no_match", "no_video"}
    assert set(mapping["status"].unique()) <= documented


def test_runb_frame_lo_hi_bracket_runb_frame(mapping: pd.DataFrame) -> None:
    """The reported best match is always within its own candidate range."""
    sub = mapping.dropna(subset=["runB_frame"])
    assert (sub["runB_frame_lo"] <= sub["runB_frame"]).all()
    assert (sub["runB_frame"] <= sub["runB_frame_hi"]).all()


def test_no_video_rows_match_the_measured_timestamp_decode_discrepancy(
    mapping: pd.DataFrame,
) -> None:
    """Cross-check against an independent measurement: Step 0/1 found runA has 2695 timestamp
    lines but only 2674 decoded frames (findings.md) -- `no_video` rows should equal that gap,
    not just "some positive number"."""
    n_no_video = int((mapping["status"] == "no_video").sum())
    assert n_no_video == len(mapping) - 2674
