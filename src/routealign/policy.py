"""Task 3: the keep/discard manifest and its schema.

`build_manifest` attaches Task 1's per-file facts and Task 2's correspondence to every frame
(one row per `(run, cam, frame)`, including frames that never decoded), then applies the
keep/discard and split rules. All maths; no file I/O -- callers (`cli.py`) open the JPEG
caches, load raw timestamps and read the inspection JSON, and pass arrays/facts in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from routealign.timestamps import ns_to_iso

_DEPOT_PLACE_ID = -1  # why: the shared start/end depot gets one id, outside the normal
#                        floor(frame / bucket) numbering, so it can never collide with one


@dataclass(frozen=True)
class RunCameraFacts:
    """Static, file-level facts about one `(run, cam)` recording, already measured by Task 1.

    Attributes:
        run: `"runA"` or `"runB"`.
        cam: `"cam0"` or `"cam5"`.
        decoded_count: Number of frames that actually decoded (`io_video.count_decoded`).
        src_file: Path to the source `.hevc` file, relative to `paths.dataset_dir`.
        src_sha256: SHA-256 of `src_file` (`outputs/inspection/integrity.json`).
        fps_declared: Nominal fps from the SPS/VUI (`outputs/inspection/*.json`).
        fps_measured: Measured fps from timestamp arithmetic (same source).
        width: Cropped output width in pixels.
        height: Cropped output height in pixels.
        codec: Codec name, e.g. `"hevc"`.
        idr_indices: Decoded indices of IDR (keyframe) pictures (`outputs/inspection/*.json`
            `gop.idr_indices`).
        weather: Run-level qualitative weather tag, from a one-time visual read of each run.
        lighting: Run-level qualitative lighting tag.
        lens_occlusion: Camera-level qualitative occlusion tag.
    """

    run: str
    cam: str
    decoded_count: int
    src_file: str
    src_sha256: str
    fps_declared: float
    fps_measured: float
    width: int
    height: int
    codec: str
    idr_indices: tuple[int, ...]
    weather: str
    lighting: str
    lens_occlusion: str


def _frame_skeleton(facts: RunCameraFacts, n_lines: int) -> pd.DataFrame:
    """Build the identity + decode-status columns, one row per timestamp line.

    Args:
        facts: Static facts for this `(run, cam)`.
        n_lines: Number of timestamp lines for this run (`len(timestamps)`).

    Returns:
        `n_lines`-row frame with `run, cam, frame, decoded_index, src_file, decode_ok`.
    """
    frame = np.arange(n_lines, dtype=np.int64)
    decode_ok = frame < facts.decoded_count
    return pd.DataFrame(
        {
            "run": facts.run,
            "cam": facts.cam,
            "frame": frame,
            "decoded_index": np.where(decode_ok, frame, -1),
            "src_file": facts.src_file,
            "decode_ok": decode_ok,
        }
    )


def _attach_timing(
    df: pd.DataFrame,
    timestamps_ns: NDArray[np.int64],
    tail_zone_lines: int,
    irregular_lo_ms: float,
    irregular_hi_ms: float,
) -> pd.DataFrame:
    """Attach per-line timing columns from the raw timestamp array.

    Args:
        df: Output of `_frame_skeleton`, same row order as `timestamps_ns`.
        timestamps_ns: int64[n_lines] capture timestamps for this run (shared by both cameras).
        tail_zone_lines: Size of the "don't fully trust timing here" window at the file's tail,
            e.g. config.yaml `policy.tail_zone_lines`.
        irregular_lo_ms: Lower bound of the "regular" interval band, e.g.
            config.yaml `policy.interval_irregular_lo_ms`.
        irregular_hi_ms: Upper bound of the "regular" interval band.

    Returns:
        `df` with `t_capture_utc_ns, t_capture_iso, t_rel_s, ts_source, ts_confidence,
        dt_prev_ms, dt_next_ms, interval_irregular, tail_zone` added.
    """
    ts = timestamps_ns[df["frame"].to_numpy()]
    dt_ms = np.diff(timestamps_ns).astype(np.float64) / 1e6
    dt_prev = np.concatenate([[np.nan], dt_ms])[df["frame"].to_numpy()]
    dt_next = np.concatenate([dt_ms, [np.nan]])[df["frame"].to_numpy()]
    irregular = ((dt_prev < irregular_lo_ms) | (dt_prev > irregular_hi_ms)) | (
        (dt_next < irregular_lo_ms) | (dt_next > irregular_hi_ms)
    )
    tail_zone = df["frame"].to_numpy() >= (timestamps_ns.size - tail_zone_lines)

    out = df.copy()
    out["t_capture_utc_ns"] = ts
    out["t_capture_iso"] = [ns_to_iso(int(v)) for v in ts]
    out["t_rel_s"] = (ts - timestamps_ns[0]) / 1e9
    out["ts_source"] = "capture_timestamp_file"
    out["ts_confidence"] = np.where(tail_zone, "assumed", "measured")
    out["dt_prev_ms"] = dt_prev
    out["dt_next_ms"] = dt_next
    out["interval_irregular"] = np.nan_to_num(irregular, nan=False).astype(bool)
    out["tail_zone"] = tail_zone
    return out


def _attach_static(df: pd.DataFrame, facts: RunCameraFacts) -> pd.DataFrame:
    """Attach the per-file constants and the derived `is_keyframe` flag.

    Args:
        df: Output of `_attach_timing`.
        facts: Static facts for this `(run, cam)`.

    Returns:
        `df` with `src_sha256, fps_nominal_declared, fps_measured, width, height, codec,
        colour_range, is_keyframe, weather, lighting, lens_occlusion` added.
    """
    idr = set(facts.idr_indices)
    out = df.copy()
    out["src_sha256"] = facts.src_sha256
    out["fps_nominal_declared"] = facts.fps_declared
    out["fps_measured"] = facts.fps_measured
    out["width"] = facts.width
    out["height"] = facts.height
    out["codec"] = facts.codec
    # why: T1 found video_signal_type_present_flag=0 -- range is never signalled (CLAUDE.md §5)
    out["colour_range"] = "unsignalled"
    out["is_keyframe"] = df["frame"].isin(idr) & df["decode_ok"]
    out["weather"] = facts.weather
    out["lighting"] = facts.lighting
    out["lens_occlusion"] = facts.lens_occlusion
    return out


def _dedup_keep_mask(
    n: int, segments: list[tuple[int, int]], fps: float, keep_hz: float
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Compute `is_stationary`/`stationary_run_id`/`dedup_keep` for one camera's frame range.

    Args:
        n: Number of decoded frames.
        segments: `(start, end)` pairs, exclusive end, e.g. from `frames.stationary_segments`.
        fps: Measured frame rate, to convert `keep_hz` into a frame stride.
        keep_hz: Target kept-frame rate inside a stationary run, e.g.
            config.yaml `policy.dedup_keep_hz`.

    Returns:
        `(is_stationary[n], dedup_keep[n])` -- `dedup_keep` is `True` everywhere outside a
        segment (nothing to thin) and `True` only every `round(fps / keep_hz)`-th frame inside one.
    """
    is_stationary = np.zeros(n, dtype=bool)
    dedup_keep = np.ones(n, dtype=bool)
    stride = max(1, round(fps / keep_hz))
    for start, end in segments:
        is_stationary[start:end] = True
        offsets = np.arange(end - start)
        dedup_keep[start:end] = offsets % stride == 0
    return is_stationary, dedup_keep


def _attach_motion(
    df: pd.DataFrame,
    facts: RunCameraFacts,
    motion_energy: NDArray[np.float64],
    segments: list[tuple[int, int]],
    keep_hz: float,
) -> pd.DataFrame:
    """Attach `motion_energy, is_stationary, stationary_run_id, dedup_keep`.

    Args:
        df: Output of `_attach_static`.
        facts: Static facts for this `(run, cam)` (used for `fps_measured`).
        motion_energy: float64[decoded_count - 1] per-transition motion energy
            (`frames.motion_energy`) -- one value per `(k, k+1)` pair, so the last decoded frame
            has none.
        segments: This run's stationary segments, shared by both cameras
            (`frames.stationary_segments`).
        keep_hz: See `_dedup_keep_mask`.

    Returns:
        `df` with the four columns added; all four are null/`False` for `frame >= decoded_count`,
        and `motion_energy` is additionally null for the very last decoded frame (no transition).
    """
    n = facts.decoded_count
    is_stationary, dedup_keep = _dedup_keep_mask(n, segments, facts.fps_measured, keep_hz)
    run_id = np.full(n, -1, dtype=np.int64)
    for seg_id, (start, end) in enumerate(segments):
        run_id[start:end] = seg_id

    frame = df["frame"].to_numpy()
    decode_ok = df["decode_ok"].to_numpy()
    has_motion = decode_ok & (frame < motion_energy.size)
    out = df.copy()
    out["motion_energy"] = np.where(
        has_motion, motion_energy[np.clip(frame, 0, motion_energy.size - 1)], np.nan
    )
    out["is_stationary"] = np.where(decode_ok, is_stationary[np.clip(frame, 0, n - 1)], False)
    out["stationary_run_id"] = np.where(
        decode_ok & (run_id[np.clip(frame, 0, n - 1)] >= 0), run_id[np.clip(frame, 0, n - 1)], -1
    )
    out["dedup_keep"] = np.where(decode_ok, dedup_keep[np.clip(frame, 0, n - 1)], False)
    return out


def _attach_correspondence(df: pd.DataFrame, run: str, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach Task 2's correspondence (`outputs/mapping.csv`) to every row of one run.

    Only runA rows have a real correspondence -- Task 2's deliverable is directional, "for a
    frame in runA, which frame in runB shows the same place"; a reverse mapping was never
    computed or evaluated against ground truth, so runB rows get
    `corr_status="not_applicable"` rather than an invented, unvalidated reverse lookup.

    Args:
        df: One run's frame rows (both cameras will get an identical copy of these columns,
            since correspondence is a per-place, not per-camera, fact -- H0 means both cameras
            of a run share one timestamp line).
        run: `"runA"` or `"runB"`.
        mapping: `outputs/mapping.csv` contents (runA-indexed).

    Returns:
        `df` with `corr_run, corr_frame, corr_lo, corr_hi, corr_conf, corr_status, corr_method`
        added.
    """
    out = df.copy()
    if run == "runA":
        m = mapping.set_index("runA_frame")
        frame = df["frame"].to_numpy()
        out["corr_run"] = "runB"
        out["corr_frame"] = m["runB_frame"].reindex(frame).to_numpy()
        out["corr_lo"] = m["runB_frame_lo"].reindex(frame).to_numpy()
        out["corr_hi"] = m["runB_frame_hi"].reindex(frame).to_numpy()
        out["corr_conf"] = m["confidence"].reindex(frame).to_numpy()
        out["corr_status"] = m["status"].reindex(frame).to_numpy()
        out["corr_method"] = m["method"].reindex(frame).to_numpy()
    else:
        out["corr_run"] = "runA"
        out["corr_frame"] = np.nan
        out["corr_lo"] = np.nan
        out["corr_hi"] = np.nan
        out["corr_conf"] = np.nan
        out["corr_status"] = "not_applicable"
        out["corr_method"] = ""
    return out


def _place_id_for_runb_frame(
    runb_frame: NDArray[np.float64], bucket: int, depot_margin: int, runb_max: int
) -> NDArray[np.int64]:
    """Map a runB frame index to a `place_id`, merging the start/end depot into one bucket.

    Args:
        runb_frame: float64[n] runB frame indices (own, for runB rows; corresponded, for runA
            rows) -- `NaN` maps to `-1` (unplaced, distinct from `_DEPOT_PLACE_ID`, handled by
            the caller).
        bucket: Frames per place, e.g. config.yaml `policy.place_id_bucket_frames`.
        depot_margin: Frames from either end of the route counted as "the depot", e.g.
            config.yaml `policy.depot_runb_margin`.
        runb_max: Highest valid runB frame index (route's own last decoded frame).

    Returns:
        int64[n] place ids; `_DEPOT_PLACE_ID` for the merged depot, `-1` where `runb_frame` is NaN.
    """
    valid = ~np.isnan(runb_frame)
    is_depot = valid & ((runb_frame <= depot_margin) | (runb_frame >= runb_max - depot_margin))
    place = np.where(valid, np.floor_divide(np.nan_to_num(runb_frame), bucket), -1).astype(np.int64)
    place[is_depot] = _DEPOT_PLACE_ID
    place[~valid] = -1
    return place


def _interpolate_missing_corr_frame(
    runa_frame: NDArray[np.int64], corr_frame: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Fill a runA row's missing `corr_frame` from its nearest matched neighbours.

    A `no_match`/`no_frame` runA row still needs a `place_id` for the split rule, so its
    would-be runB position is linearly interpolated from the nearest runA rows that do have
    one -- not left null, and not silently treated as "place unknown".

    Args:
        runa_frame: int64[n] runA frame indices, sorted ascending.
        corr_frame: float64[n] `corr_frame` values, `NaN` where unmatched.

    Returns:
        float64[n] `corr_frame` with every `NaN` filled by interpolation (clamped to the nearest
        known value at either end, `numpy.interp`'s standard behaviour).
    """
    known = ~np.isnan(corr_frame)
    if known.sum() < 2:
        return corr_frame
    return np.interp(runa_frame, runa_frame[known], corr_frame[known])


def _assign_split(
    place_id: pd.Series, val_frac: float, test_frac: float, buffer_places: int
) -> pd.Series:
    """Assign train/val/test as contiguous blocks of `place_id`, never by frame index.

    Both runs' frames of one place always share a split, so blocking must
    happen on the sorted, deduplicated `place_id` sequence -- shuffling individual frames could
    put runA and runB frames of the same place on opposite sides of a split. The merged depot
    (`_DEPOT_PLACE_ID`) and any unplaced row (`place_id=-1`) are `"unassigned"`: the depot
    deliberately conflates two different points on the route, so it cannot honestly represent a
    single held-out place.

    Args:
        place_id: One row per frame, from `_place_id_for_runb_frame`.
        val_frac: Fraction of *places* (not frames) assigned to val, e.g.
            config.yaml `policy.split_val_frac`.
        test_frac: Fraction of places assigned to test.
        buffer_places: Unassigned places left between blocks, e.g.
            config.yaml `policy.split_buffer_places`.

    Returns:
        One `{"train", "val", "test", "unassigned"}` string per row of `place_id`.
    """
    places = sorted(p for p in place_id.unique() if p >= 0)
    n = len(places)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    test_start = n - n_test
    val_end = test_start - buffer_places
    val_start = val_end - n_val
    train_end = val_start - buffer_places

    label_by_place: dict[int, str] = {}
    for i, place in enumerate(places):
        if i >= test_start:
            label_by_place[place] = "test"
        elif val_start <= i < val_end:
            label_by_place[place] = "val"
        elif i < train_end:
            label_by_place[place] = "train"
        else:
            label_by_place[place] = "unassigned"
    return place_id.map(lambda p: label_by_place.get(p, "unassigned"))


def _apply_keep_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the one hard discard rule: no decoded frame, no use.

    Every other keep/discard-adjacent condition this manifest computes is a flag a downstream
    consumer filters on explicitly (`interval_irregular`, `dedup_keep`, `tail_zone`,
    `lens_occlusion`, `corr_status`), not a reason to remove the row. This rule fires on
    `decode_ok` alone; the two other candidate hard-discard conditions (corrupt decode, exact
    duplicates) were checked for real and found zero instances in this dataset -- not
    implemented as an additional condition here because there is nothing for one to catch, not
    because it was skipped.

    Args:
        df: Frame rows with `decode_ok` already attached.

    Returns:
        `df` with `keep` (bool) and `reject_reason` (string, empty iff `keep`) added -- empty
        string, not `None`/NaN, matching `confidence.fuse_confidence`'s `reason` column
        convention, and sidestepping pandas' `None` -> NaN coercion on object-dtype assignment.
    """
    out = df.copy()
    out["keep"] = df["decode_ok"]
    out["reject_reason"] = np.where(df["decode_ok"], "", "no decoded frame for this row")
    return out


_MANIFEST_COLUMNS = [
    "run", "cam", "frame", "decoded_index", "src_file", "src_sha256",
    "t_capture_utc_ns", "t_capture_iso", "t_rel_s", "ts_source", "ts_confidence",
    "dt_prev_ms", "dt_next_ms", "interval_irregular",
    "fps_nominal_declared", "fps_measured", "width", "height", "codec", "colour_range",
    "is_keyframe", "is_stationary", "stationary_run_id", "dedup_keep", "motion_energy",
    "weather", "lighting", "lens_occlusion", "tail_zone", "decode_ok",
    "corr_run", "corr_frame", "corr_lo", "corr_hi", "corr_conf", "corr_status", "corr_method",
    "place_id", "split", "keep", "reject_reason",
]  # fmt: skip


def _build_one_run_camera(
    facts: RunCameraFacts,
    timestamps_ns: NDArray[np.int64],
    motion_energy: NDArray[np.float64],
    segments: list[tuple[int, int]],
    mapping: pd.DataFrame,
    interval_irregular_lo_ms: float,
    interval_irregular_hi_ms: float,
    tail_zone_lines: int,
    dedup_keep_hz: float,
) -> pd.DataFrame:
    """Build one `(run, cam)`'s rows, identity through correspondence -- everything except the
    cross-run `place_id`/`split`/`keep` steps `build_manifest` does once, over every row.
    """
    df = _frame_skeleton(facts, timestamps_ns.size)
    df = _attach_timing(
        df, timestamps_ns, tail_zone_lines, interval_irregular_lo_ms, interval_irregular_hi_ms
    )
    df = _attach_static(df, facts)
    df = _attach_motion(df, facts, motion_energy, segments, dedup_keep_hz)
    return _attach_correspondence(df, facts.run, mapping)


def build_manifest(
    run_cameras: list[RunCameraFacts],
    timestamps_by_run: dict[str, NDArray[np.int64]],
    motion_by_run_cam: dict[tuple[str, str], NDArray[np.float64]],
    stationary_by_run: dict[str, list[tuple[int, int]]],
    mapping: pd.DataFrame,
    interval_irregular_lo_ms: float = 80.0,  # why: config.yaml policy.interval_irregular_lo_ms
    interval_irregular_hi_ms: float = 120.0,  # why: config.yaml policy.interval_irregular_hi_ms
    tail_zone_lines: int = 50,  # why: config.yaml policy.tail_zone_lines
    dedup_keep_hz: float = 1.0,  # why: config.yaml policy.dedup_keep_hz
    place_id_bucket_frames: int = 100,  # why: config.yaml policy.place_id_bucket_frames
    depot_runb_margin: int = 50,  # why: config.yaml policy.depot_runb_margin
    split_val_frac: float = 0.15,  # why: config.yaml policy.split_val_frac
    split_test_frac: float = 0.15,  # why: config.yaml policy.split_test_frac
    split_buffer_places: int = 1,  # why: config.yaml policy.split_buffer_places
) -> pd.DataFrame:
    """Build the per-frame keep/discard manifest with attached metadata: one row per
    `(run, cam, frame)`, including frames that never decoded, so nothing is silently dropped.

    Args:
        run_cameras: Static facts for every `(run, cam)` to include -- normally all 4.
        timestamps_by_run: Raw capture timestamps, keyed by run (`timestamps.load_timestamps`).
        motion_by_run_cam: Per-frame motion energy, keyed by `(run, cam)` (`frames.motion_energy`).
        stationary_by_run: Stationary segments, keyed by run -- shared by both of that run's
            cameras (`frames.stationary_segments`).
        mapping: `outputs/mapping.csv` contents.
        interval_irregular_lo_ms (float, optional): See `_attach_timing`. Defaults to 80.0.
        interval_irregular_hi_ms (float, optional): See `_attach_timing`. Defaults to 120.0.
        tail_zone_lines (int, optional): See `_attach_timing`. Defaults to 50.
        dedup_keep_hz (float, optional): See `_dedup_keep_mask`. Defaults to 1.0.
        place_id_bucket_frames (int, optional): See `_place_id_for_runb_frame`. Defaults to 100.
        depot_runb_margin (int, optional): See `_place_id_for_runb_frame`. Defaults to 50.
        split_val_frac (float, optional): See `_assign_split`. Defaults to 0.15.
        split_test_frac (float, optional): See `_assign_split`. Defaults to 0.15.
        split_buffer_places (int, optional): See `_assign_split`. Defaults to 1.

    Returns:
        One row per frame, per the manifest schema (`_MANIFEST_COLUMNS`); also written
        to `outputs/keep_manifest.csv`.
    """
    parts = [
        _build_one_run_camera(
            facts,
            timestamps_by_run[facts.run],
            motion_by_run_cam[(facts.run, facts.cam)],
            stationary_by_run[facts.run],
            mapping,
            interval_irregular_lo_ms,
            interval_irregular_hi_ms,
            tail_zone_lines,
            dedup_keep_hz,
        )
        for facts in run_cameras
    ]
    df = pd.concat(parts, ignore_index=True)

    is_a = (df["run"] == "runA").to_numpy()
    corr_frame = df["corr_frame"].to_numpy(dtype=np.float64, copy=True)
    corr_frame[is_a] = _interpolate_missing_corr_frame(
        df.loc[is_a, "frame"].to_numpy(), corr_frame[is_a]
    )
    own_or_corr = np.where(is_a, corr_frame, df["frame"].to_numpy(dtype=np.float64))

    runb_max = int(df.loc[df["run"] == "runB", "frame"].max())
    df["place_id"] = _place_id_for_runb_frame(
        own_or_corr, place_id_bucket_frames, depot_runb_margin, runb_max
    )
    df["split"] = _assign_split(
        df["place_id"], split_val_frac, split_test_frac, split_buffer_places
    )
    df = _apply_keep_rule(df)
    return df[_MANIFEST_COLUMNS]
