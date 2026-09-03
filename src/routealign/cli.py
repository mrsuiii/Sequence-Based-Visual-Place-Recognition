"""argparse CLI. The only module allowed `print`/`sys.exit` (CLAUDE.md §4). Each subcommand
equals one Makefile target.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import (
    align,
    confidence,
    descriptors,
    frames,
    groundtruth,
    io_video,
    policy,
    similarity,
    timestamps,
    verify,
    viz,
)
from .config import Config, load_config
from .frames import FrameStore

log = logging.getLogger(__name__)


def _cmd_verify_data(cfg: Config) -> int:
    """Check every dataset file's SHA-256 against dataset/SHA256SUMS.txt.

    Writes the full result to outputs/inspection/integrity.json.

    Args:
        cfg: Loaded configuration.

    Returns:
        0 if every file matched, 1 otherwise.
    """
    dataset_dir = cfg.path("dataset_dir")
    manifest_path = dataset_dir / "SHA256SUMS.txt"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found", file=sys.stderr)
        return 1

    expected: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if not name:
            raise ValueError(f"unparseable SHA256SUMS.txt line: {line!r}")
        expected[name.strip()] = digest.strip()

    results = []
    ok = True
    for run in cfg.runs:
        for cam in cfg.cameras:
            for suffix in ("_20_yuv420p_output.hevc", "_20_yuv420p_output.hevc.timestamps.txt"):
                rel = f"{run}/{cam}{suffix}"
                fpath = dataset_dir / rel
                exp = expected.get(rel)
                actual = io_video.sha256(fpath) if fpath.exists() else None
                match = actual is not None and exp is not None and actual == exp
                ok = ok and match
                results.append({"file": rel, "expected": exp, "actual": actual, "match": match})
                print(f"{'OK' if match else 'MISMATCH/MISSING':17s} {rel}")

    out_dir = cfg.path("outputs_dir") / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "integrity.json"
    out_path.write_text(json.dumps({"all_ok": ok, "files": results}, indent=2))
    print(f"\n{'all files verified' if ok else 'INTEGRITY CHECK FAILED'} -> {out_path}")
    return 0 if ok else 1


def _cmd_decode(cfg: Config) -> int:
    """Decode every run/camera to a JPEG cache.

    Checks the written JPEG count against a fresh, independent decode count for each file.

    Args:
        cfg: Loaded configuration.

    Returns:
        0 if every file's written count matched, 1 otherwise.
    """
    dataset_dir = cfg.path("dataset_dir")
    cache_dir = cfg.path("cache_dir") / "frames"
    ok = True
    for run in cfg.runs:
        for cam in cfg.cameras:
            src = dataset_dir / run / f"{cam}_20_yuv420p_output.hevc"
            out_dir = cache_dir / run / cam
            print(f"decoding {src} -> {out_dir} ...")
            expected = io_video.count_decoded(src)
            written = io_video.decode_to_jpegs(
                src,
                out_dir,
                width=cfg.decode.width,
                height=cfg.decode.height,
                jpeg_quality=cfg.decode.jpeg_quality,
                fps_mode=cfg.decode.fps_mode,
            )
            match = written == expected
            ok = ok and match
            status = "OK" if match else "MISMATCH"
            print(f"  {status}: wrote {written} JPEGs, count_decoded={expected}")
    return 0 if ok else 1


def _fmt(x: float | int | None, spec: str = ".2f") -> str:
    """Format a number for a markdown table cell.

    Args:
        x: Value to format, or `None`.
        spec (optional): `format()` spec string. Defaults to `".2f"`.

    Returns:
        The formatted string, or `"N/A"` if `x` is `None`.
    """
    return "N/A" if x is None else format(x, spec)


def _write_task1_table(rows: list[dict], out_path: Path) -> None:
    """Render the Task 1 per-camera records into a single markdown summary.

    Args:
        rows: One record per run/camera, as built by `_cmd_characterise`.
        out_path: File to write the markdown table to (outputs/inspection/task1_table.md).
    """
    lines = [
        "# Task 1 — characterisation summary",
        "",
        "`declared fps` below is the one label that traces to a real field about *this file's* "
        "content: the SPS/VUI `time_scale/num_units_in_tick`, independently confirmed against "
        "ffprobe's `r_frame_rate` (the two agree exactly). Two other candidate labels were found, "
        'investigated and rejected rather than silently omitted — see "Other frame-rate labels" '
        "below. `declared fps` is still not the actual rate; see "
        "`measured fps`, computed from `timestamps.txt` interval arithmetic. Full provenance "
        "for every figure is in the per-camera JSON files next to this table.",
        "",
        "| run/cam | frames (decoded / lines) | resolution | file size | declared fps (nominal) | "
        "measured fps | route time (s) | recorded (UTC) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fc, fr, res = r["frame_count"], r["frame_rate"], r["resolution"]["declared_cropped_wh"]
        lines.append(
            f"| {r['run']}/{r['camera']} | {fc['measured_decoded']} / {fc['timestamp_lines']} | "
            f"{res[0]}x{res[1]} | {r['file_size_bytes']:,} B | "
            f"{_fmt(fr['declared_nominal_fps'])} | "
            f"{_fmt(fr['measured_actual_fps'], '.3f')} | "
            f"{_fmt(r['route_time_s']['video_covered_span_H0'])} | {r['recorded_at_utc']} |"
        )

    lines.append("")
    lines.append("## Other frame-rate labels found — investigated and rejected, not used above")
    lines.append("")
    lines.append("| label | value | why it is not reported as the declared fps |")
    lines.append("|---|---|---|")
    rejected = rows[0]["frame_rate"]["other_labels_investigated_and_rejected"]
    lines.append(
        f"| filename token | {rejected['filename_token']['value']} | "
        f"{rejected['filename_token']['reason_rejected']} |"
    )
    lines.append(
        f"| ffprobe avg_frame_rate | {rejected['ffprobe_avg_frame_rate']['value']} | "
        f"{rejected['ffprobe_avg_frame_rate']['reason_rejected']} |"
    )

    lines.append("")
    lines.append("## Frame-count discrepancy (timestamp lines minus decoded frames)")
    lines.append("")
    lines.append("| run/cam | lines | decoded | nal picture count | discrepancy |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        fc = r["frame_count"]
        lines.append(
            f"| {r['run']}/{r['camera']} | {fc['timestamp_lines']} | {fc['measured_decoded']} | "
            f"{fc['measured_nal_pictures']} | {fc['discrepancy_lines_minus_decoded']} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def _cmd_characterise(cfg: Config) -> int:
    """Run Task 1: characterise every run/camera.

    Measures frame count, resolution, file size, frame rate (declared and measured), route time,
    recording time and the Δt distribution. Writes one JSON per run/camera, a combined markdown
    table and Δt figures, all under outputs/inspection/.

    Args:
        cfg: Loaded configuration.

    Returns:
        0 (always succeeds if the dataset files are readable).
    """
    dataset_dir = cfg.path("dataset_dir")
    out_dir = cfg.path("outputs_dir") / "inspection"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in cfg.runs:
        for cam in cfg.cameras:
            video_path = dataset_dir / run / f"{cam}_20_yuv420p_output.hevc"
            ts_path = dataset_dir / run / f"{cam}_20_yuv420p_output.hevc.timestamps.txt"
            print(f"characterising {run}/{cam} ...")

            file_size = video_path.stat().st_size
            sp = io_video.probe(video_path)
            vui = io_video.sps_vui(video_path)
            nal = io_video.scan_nals(video_path)  # n_pictures is an independent decoded-frame count
            decoded_count = nal.n_pictures

            ts = timestamps.load_timestamps(ts_path)
            stats = timestamps.interval_stats(ts)
            ft = timestamps.frame_times(ts, decoded_count)
            dt_ms = np.diff(ts).astype(np.float64) / 1e6

            span_all_s = (int(ts[-1]) - int(ts[0])) / 1e9
            span_video_s = (int(ft[-1]) - int(ft[0])) / 1e9
            recorded_utc = datetime.fromtimestamp(int(ts[0]) / 1e9, tz=UTC)
            bit_rate_bps = (file_size * 8 / span_video_s) if span_video_s > 0 else None
            actual_fps = ((decoded_count - 1) / span_video_s) if span_video_s > 0 else None

            record = {
                "run": run,
                "camera": cam,
                "frame_count": {
                    # ffprobe nb_frames is N/A for a raw elementary stream
                    "declared_nb_frames": None,
                    "timestamp_lines": int(ts.size),
                    "measured_decoded": int(decoded_count),
                    "measured_nal_pictures": int(nal.n_pictures),
                    "discrepancy_lines_minus_decoded": int(ts.size - decoded_count),
                },
                "resolution": {
                    "declared_ffprobe_wh": [sp.width, sp.height],
                    "declared_sps_coded_wh": [vui.pic_width_luma, vui.pic_height_luma],
                    "declared_cropped_wh": [vui.cropped_width, vui.cropped_height],
                    "pix_fmt": sp.pix_fmt,
                },
                "file_size_bytes": int(file_size),
                "frame_rate": {
                    "declared_nominal_fps": vui.vui_declared_fps,
                    "declared_nominal_source": (
                        "SPS/VUI time_scale/num_units_in_tick (ffmpeg -bsf:v trace_headers); "
                        "independently matches ffprobe's r_frame_rate exactly for this file"
                    ),
                    "measured_actual_fps": actual_fps,
                    "measured_median_interval_ms": stats["median_ms"],
                    "measured_source": (
                        "timestamps.txt interval arithmetic: "
                        "(decoded_count - 1) / video_covered_span_s"
                    ),
                    "other_labels_investigated_and_rejected": {
                        "filename_token": {
                            "value": 20,
                            "reason_rejected": (
                                "unexplained -- no field anywhere in the file confirms this "
                                "represents fps; listed only because of its position in the "
                                "filename, not because its meaning is known"
                            ),
                        },
                        "ffprobe_avg_frame_rate": {
                            "value": sp.avg_frame_rate,
                            "reason_rejected": (
                                "proven generic fallback for headerless elementary streams: a "
                                "synthetic test file encoded at a known true 17 fps still reports "
                                "avg_frame_rate=25/1, identical to this file"
                            ),
                        },
                    },
                },
                "route_time_s": {
                    "timestamp_span_all_lines": span_all_s,
                    "video_covered_span_H0": span_video_s,
                },
                "recorded_at_utc": recorded_utc.isoformat(),
                "interval_stats_ms": stats,
                "bit_rate_bps_measured": bit_rate_bps,
                "codec": {
                    "codec_name": sp.codec_name,
                    "level_idc": vui.level_idc,
                    "level": vui.level_idc / 30,
                    "max_dec_pic_buffering": vui.max_dec_pic_buffering,
                    "max_num_reorder_pics": vui.max_num_reorder_pics,
                    "video_signal_type_present": vui.video_signal_type_present,
                },
                "gop": {
                    "n_idr": nal.n_idr,
                    "idr_indices": nal.idr_indices,
                    "gop_sizes": nal.gop_sizes,
                    "pict_type_counts": nal.pict_type_counts,
                },
                "file_size_mod_262144": nal.file_size_mod_262144,
            }
            (out_dir / f"{run}_{cam}.json").write_text(json.dumps(record, indent=2))
            rows.append(record)

            viz.plot_interval_histogram(
                dt_ms,
                stats,
                title=f"{run}/{cam} — Δt distribution ({dt_ms.size} intervals)",
                out_path=fig_dir / f"{run}_{cam}_dt_hist.png",
            )
            viz.plot_interval_vs_index(
                dt_ms,
                title=f"{run}/{cam} — Δt vs. interval index",
                out_path=fig_dir / f"{run}_{cam}_dt_vs_index.png",
            )

    _write_task1_table(rows, out_dir / "task1_table.md")
    print(f"\nwrote {len(rows)} per-camera reports + task1_table.md -> {out_dir}")
    return 0


def _open_store(cfg: Config, run: str, cam: str) -> FrameStore:
    """Open a `FrameStore` for one run/camera using config.yaml's crop setting.

    Args:
        cfg: Loaded configuration.
        run: Run id.
        cam: Camera id.

    Returns:
        The opened frame store.
    """
    return FrameStore(
        cfg.path("cache_dir") / "frames", run, cam, crop_row_frac=cfg.raw["frames"]["crop_row_frac"]
    )


def _seqslam_desc(
    cfg: Config, run: str, cam: str, limit: int | None, force: bool
) -> tuple[NDArray[np.float32], int]:
    """Compute-or-load one run/camera's SeqSLAM descriptors, capped at `limit` frames.

    Shared by `describe` and `align` so the cache key logic lives in exactly one place.

    Args:
        cfg: Loaded configuration.
        run: Run id.
        cam: Camera id.
        limit: Cap on frame count (smoke runs), or `None` for the full run.
        force: Recompute even if a cache entry exists.

    Returns:
        `(descriptors, n)`: the descriptor array and the frame count actually used.
    """
    store = _open_store(cfg, run, cam)
    n = min(limit, store.n_frames) if limit else store.n_frames
    d = cfg.raw["descriptors"]["seqslam"]
    key = f"desc_seqslam_{run}_{cam}_n{n}_w{d['thumb_w']}_h{d['thumb_h']}_p{d['patch']}"
    arr = descriptors.cached(
        lambda: descriptors.seqslam_descriptors(
            store, np.arange(n, dtype=np.int64), d["thumb_w"], d["thumb_h"], d["patch"]
        ),
        key,
        cfg.path("cache_dir"),
        extra={"params": d, "run": run, "cam": cam, "n": n},
        force=force,
    )
    return arr, n


def _dinov2_desc(
    cfg: Config, run: str, cam: str, limit: int | None, force: bool
) -> tuple[NDArray[np.float32], int]:
    """Compute-or-load one run/camera's DINOv2 descriptors, capped at `limit` frames.

    Shared by `describe` and `align`, mirroring `_seqslam_desc`.

    Args:
        cfg: Loaded configuration.
        run: Run id.
        cam: Camera id.
        limit: Cap on frame count (smoke runs), or `None` for the full run.
        force: Recompute even if a cache entry exists.

    Returns:
        `(descriptors, n)`: the descriptor array and the frame count actually used.
    """
    store = _open_store(cfg, run, cam)
    n = min(limit, store.n_frames) if limit else store.n_frames
    d = cfg.raw["descriptors"]["dinov2"]
    key = f"desc_dinov2_{run}_{cam}_n{n}_{d['model']}"
    arr = descriptors.cached(
        lambda: descriptors.dino_descriptors(
            store, np.arange(n, dtype=np.int64), model_name=d["model"]
        ),
        key,
        cfg.path("cache_dir"),
        extra={"params": d, "run": run, "cam": cam, "n": n},
        force=force,
    )
    return arr, n


def _cmd_describe(cfg: Config, feat: str | None, limit: int | None, force: bool) -> int:
    """Compute and cache per-frame descriptors for every run/camera.

    DINOv2 (`--feat dinov2`) needs `torch.hub` network access to fetch weights on first use;
    if that is unavailable in this environment, it fails loudly here rather than silently
    falling back (CLAUDE.md §4) -- re-run with network access, or use `--feat seqslam` only.

    Args:
        cfg: Loaded configuration.
        feat: `"seqslam"`, `"dinov2"`, or `None` for both.
        limit: Cap on frame count per run/camera (smoke runs), or `None` for the full run.
        force: Recompute even if a cache entry exists.

    Returns:
        0 on success.
    """
    feats = [feat] if feat else ["seqslam", "dinov2"]
    for f in feats:
        desc_fn = _seqslam_desc if f == "seqslam" else _dinov2_desc
        for run in cfg.runs:
            for cam in cfg.cameras:
                arr, n = desc_fn(cfg, run, cam, limit, force)
                print(f"{f} {run}/{cam}: {arr.shape} {arr.dtype} (n={n})")
    return 0


def _cmd_align(cfg: Config, limit: int | None, force: bool) -> int:
    """Compute per-camera, per-descriptor and fused similarity + DTW paths; write a draft mapping.

    Two-stage fusion, symmetric in camera and descriptor: per (camera, descriptor) similarity is
    fused across descriptors within each camera (-> `S_cam0`/`S_cam5`, feeds `cam_disagree`) and
    across cameras within each descriptor (-> `S_seqslam`/`S_dinov2`, feeds `desc_disagree`); the
    full joint fusion (`S_joint`) combines all four and is what the draft mapping is built from.
    Also writes the argmax-vs-DTW diagnostic figures (`outputs/inspection/figures/step3_*`,
    `step4_*`) from this same `S_joint`/`path_joint`, so the report's diagnostic images and its
    quoted ablation numbers are never computed from two different things.

    Args:
        cfg: Loaded configuration.
        limit: Cap on frame count per run (smoke runs), or `None` for the full run.
        force: Recompute even if a cache entry exists.

    Returns:
        0 on success.
    """
    # window=50 (tuned for SeqSLAM's patch-level noise) collapses DINOv2's DTW path onto a
    # single attractor column -- each descriptor gets its own window instead.
    window_cfg = {
        "seqslam": cfg.raw["similarity"]["contrast_window_seqslam"],
        "dinov2": cfg.raw["similarity"]["contrast_window_dinov2"],
    }
    cam_weights_cfg = cfg.raw["similarity"]["camera_weights"]
    desc_weights_cfg = cfg.raw["similarity"]["descriptor_weights"]
    lam = cfg.raw["align"]["dtw"]["lam_default"]
    cache_dir = cfg.path("cache_dir")
    feats = ("seqslam", "dinov2")
    desc_fn = {"seqslam": _seqslam_desc, "dinov2": _dinov2_desc}

    raw_s: dict[tuple[str, str], NDArray[np.float64]] = {}
    n_b_max = 0
    for cam in cfg.cameras:
        for feat in feats:
            desc_a, n_a = desc_fn[feat](cfg, "runA", cam, limit, force)
            desc_b, n_b = desc_fn[feat](cfg, "runB", cam, limit, force)
            print(f"{feat} {cam}: runA n={n_a}, runB n={n_b}")
            raw_s[(cam, feat)] = similarity.local_contrast_norm(
                similarity.cosine(desc_a, desc_b), window=window_cfg[feat]
            )
            n_b_max = max(n_b_max, n_b)

    # cam0 and cam5 share one timestamp file (H0: same column index = same nominal instant,
    # CLAUDE.md §5), but can have different decoded counts (measured: runB cam0=2622,
    # cam5=2615) -- pad the shorter camera's columns with the worst possible
    # z-score (-3) rather than truncating both to the shorter length, so a camera that never
    # decoded those frames simply casts no vote there instead of silently losing real cam0
    # evidence for columns cam5 never had.
    for key, s in raw_s.items():
        if s.shape[1] < n_b_max:
            pad = np.full((s.shape[0], n_b_max - s.shape[1]), -3.0, dtype=np.float64)
            raw_s[key] = np.concatenate([s, pad], axis=1)

    cam_weights = np.array([cam_weights_cfg[c] for c in cfg.cameras], dtype=np.float64)
    desc_weights = np.array([desc_weights_cfg[f] for f in feats], dtype=np.float64)

    per_cam = {
        cam: similarity.fuse([raw_s[(cam, f)] for f in feats], desc_weights) for cam in cfg.cameras
    }
    per_feat = {
        f: similarity.fuse([raw_s[(c, f)] for c in cfg.cameras], cam_weights) for f in feats
    }
    s_joint = similarity.fuse([per_cam[c] for c in cfg.cameras], cam_weights)

    cost_joint = 3.0 - np.clip(s_joint, -3.0, 3.0)
    path_joint = align.dtw_open_ends(cost_joint, lam=lam)
    print(f"joint DTW: {s_joint.shape} -> path length {len(path_joint)}")

    np.save(cache_dir / "S_joint.npy", s_joint)
    np.save(cache_dir / "path_joint.npy", path_joint)

    # Step 3/4 diagnostic figures, on the exact S_joint/path_joint the shipped mapping and the
    # ablation table's "argmax-only" row both use -- so the figure and the quoted numbers are
    # never talking about two subtly different computations.
    fig_dir = cfg.path("outputs_dir") / "inspection" / "figures"
    argmax_per_row = np.argmax(s_joint, axis=1).astype(np.int32)
    argmax_path = np.column_stack([np.arange(s_joint.shape[0], dtype=np.int32), argmax_per_row])
    viz.plot_similarity_matrix(
        s_joint,
        argmax_path,
        "runA vs runB, joint (SeqSLAM+DINOv2, cam0+cam5) -- unconstrained argmax",
        fig_dir / "step3_similarity_argmax_diagnostic.png",
    )
    viz.plot_similarity_matrix(
        s_joint,
        path_joint,
        f"runA vs runB, joint (SeqSLAM+DINOv2, cam0+cam5) -- DTW open-ends path (lam={lam})",
        fig_dir / "step4_dtw_path_diagnostic.png",
    )
    for cam, s in per_cam.items():
        cost = 3.0 - np.clip(s, -3.0, 3.0)
        path = align.dtw_open_ends(cost, lam=lam)
        np.save(cache_dir / f"S_{cam}.npy", s)
        np.save(cache_dir / f"path_{cam}.npy", path)
    for feat, s in per_feat.items():
        cost = 3.0 - np.clip(s, -3.0, 3.0)
        path = align.dtw_open_ends(cost, lam=lam)
        np.save(cache_dir / f"S_{feat}.npy", s)
        np.save(cache_dir / f"path_{feat}.npy", path)

    # runB stationary segments, to widen path_to_mapping's [lo, hi]. frames.stationary_segments
    # returns Python-slice convention (end EXCLUSIVE); path_to_mapping's stationary_b expects
    # inclusive (lo, hi) pairs (its docstring/tests) -- convert once, here, at the boundary
    # between the two conventions, rather than silently being off-by-one.
    store_b0, store_b5 = _open_store(cfg, "runB", "cam0"), _open_store(cfg, "runB", "cam5")
    motion_b0 = frames.motion_energy(
        store_b0, cfg.raw["frames"]["motion_thumb_w"], cfg.raw["frames"]["motion_thumb_h"]
    )
    motion_b5 = frames.motion_energy(
        store_b5, cfg.raw["frames"]["motion_thumb_w"], cfg.raw["frames"]["motion_thumb_h"]
    )
    n_common = min(motion_b0.size, motion_b5.size)
    stationary_b_exclusive = frames.stationary_segments(
        motion_b0[:n_common],
        motion_b5[:n_common],
        cfg.raw["frames"]["stationary_threshold_frac"],
        cfg.raw["frames"]["stationary_min_length"],
    )
    stationary_b = [(s, e - 1) for s, e in stationary_b_exclusive]

    mapping_draft = align.path_to_mapping(path_joint, s_joint, stationary_b)
    out_dir = cfg.path("outputs_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    draft_path = out_dir / "mapping_draft.csv"
    mapping_draft.to_csv(draft_path, index=False)
    print(f"wrote draft mapping ({len(mapping_draft)} rows) -> {draft_path}")
    return 0


def _iso_utc(ns: int) -> str:
    """Format an int64-nanosecond epoch timestamp as an ISO-8601 UTC string.

    Args:
        ns: Nanoseconds since the Unix epoch.

    Returns:
        ISO-8601 string, e.g. `"2025-02-28T06:24:09.580000+00:00"`.
    """
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).isoformat()


def _cmd_verify(cfg: Config) -> int:
    """Add agreement cues and confidence to the draft mapping; write outputs/mapping.csv.

    Reads the draft mapping and cached paths/similarity `align` wrote.

    Args:
        cfg: Loaded configuration.

    Raises:
        FileNotFoundError: If `align` has not been run yet (no cached draft/paths to read).

    Returns:
        0 on success.
    """
    cache_dir = cfg.path("cache_dir")
    out_dir = cfg.path("outputs_dir")
    draft_path = out_dir / "mapping_draft.csv"
    if not draft_path.exists():
        raise FileNotFoundError(f"{draft_path} not found -- run `align` first")

    mapping = pd.read_csv(draft_path)
    s_joint = np.load(cache_dir / "S_joint.npy")
    path_joint = np.load(cache_dir / "path_joint.npy")
    path_cam0 = np.load(cache_dir / f"path_{cfg.cameras[0]}.npy")
    path_cam5 = np.load(cache_dir / f"path_{cfg.cameras[1]}.npy")
    path_seqslam = np.load(cache_dir / "path_seqslam.npy")
    path_dinov2 = np.load(cache_dir / "path_dinov2.npy")
    n_a = s_joint.shape[0]

    mapping["ridge_z"] = verify.ridge_strength(
        s_joint, path_joint, window=cfg.raw["verify"]["ridge_window"]
    )
    mapping["margin_z"] = verify.margin(s_joint, path_joint, band=cfg.raw["verify"]["margin_band"])
    mapping["cam_disagree"] = verify.path_disagreement(path_cam0, path_cam5)
    mapping["desc_disagree"] = verify.path_disagreement(path_seqslam, path_dinov2)

    # local_slope is one value per *path point* (align.py docstring); reduce to one per runA
    # row via the same last-visited-point convention path_to_mapping/_path_to_row_j already use
    slope_per_point = align.local_slope(path_joint, window=cfg.raw["align"]["local_slope_window"])
    row_edges = np.searchsorted(path_joint[:, 0], np.arange(n_a + 1))
    mapping["slope"] = slope_per_point[row_edges[1:] - 1]

    argmax_per_row = np.argmax(s_joint, axis=1).astype(np.int32)
    argmax_path = np.column_stack([np.arange(n_a, dtype=np.int32), argmax_per_row])
    mapping["argmax_disagree"] = verify.path_disagreement(path_joint, argmax_path)

    mapping["sift_inliers"] = np.nan  # SIFT verification not built (config.yaml verify.sift)
    mapping["ambiguity"] = mapping["hi"] - mapping["lo"]
    mapping["ambiguous_range"] = mapping["ambiguity"] > 0
    mapping["no_frame"] = False  # every row here came from a real decoded frame (align's n_a)

    fused = confidence.fuse_confidence(mapping, tau=cfg.raw["confidence"]["no_match_tau_default"])

    store_a0, store_a5 = _open_store(cfg, "runA", "cam0"), _open_store(cfg, "runA", "cam5")
    store_b0, store_b5 = _open_store(cfg, "runB", "cam0"), _open_store(cfg, "runB", "cam5")
    fused["runA_has_cam0"] = fused["runA_frame"] < store_a0.n_frames
    fused["runA_has_cam5"] = fused["runA_frame"] < store_a5.n_frames
    fused["runB_has_cam0"] = fused["runB_frame"] < store_b0.n_frames
    fused["runB_has_cam5"] = fused["runB_frame"] < store_b5.n_frames
    fused["method"] = "seqslam+dinov2"

    ts_a = timestamps.load_timestamps(
        cfg.path("dataset_dir") / "runA" / "cam0_20_yuv420p_output.hevc.timestamps.txt"
    )
    ts_b = timestamps.load_timestamps(
        cfg.path("dataset_dir") / "runB" / "cam0_20_yuv420p_output.hevc.timestamps.txt"
    )
    fused["runA_time_utc"] = [_iso_utc(int(ts_a[k])) for k in fused["runA_frame"]]
    fused["runB_time_utc"] = [
        _iso_utc(int(ts_b[j])) if j < ts_b.size else None for j in fused["runB_frame"]
    ]

    # H0 tail: timestamp lines beyond this run's decoded count have no frame at all (CLAUDE.md
    # §5) -- append them so outputs/mapping.csv covers every timestamp line, not just decoded
    # frames, matching the "2695 runA rows" definition of done (CLAUDE.md §7). Only meaningful
    # for a full (unlimited) run: under --limit, n_a is an artificial cap, not the true decoded
    # count, so the "tail" would misclassify a real frame as no_frame -- skipped in that case.
    full_decoded_a = store_a0.n_frames
    if n_a == full_decoded_a:
        tail_k = np.arange(full_decoded_a, ts_a.size)
        if tail_k.size:
            tail = pd.DataFrame(
                {
                    "runA_frame": tail_k,
                    "runB_frame": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                    "boundary_clamped": False,
                    "confidence": np.nan,
                    "status": "no_frame",
                    "reason": "no decoded frame for this row",
                    "runA_time_utc": [_iso_utc(int(ts_a[k])) for k in tail_k],
                    "runB_time_utc": None,
                    "ridge_z": np.nan,
                    "margin_z": np.nan,
                    "cam_disagree": np.nan,
                    "desc_disagree": np.nan,
                    "sift_inliers": np.nan,
                    "slope": np.nan,
                    "runA_has_cam0": False,
                    "runA_has_cam5": False,
                    "runB_has_cam0": False,
                    "runB_has_cam5": False,
                    "method": "seqslam+dinov2",
                }
            )
            fused = pd.concat([fused, tail], ignore_index=True)
            print(f"appended {tail_k.size} no_frame tail rows ({full_decoded_a}..{ts_a.size - 1})")

    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "mapping_full.csv"
    fused.to_csv(full_path, index=False)

    final_cols = [
        "runA_frame",
        "runB_frame",
        "confidence",
        "lo",
        "hi",
        "status",
        "reason",
        "runA_time_utc",
        "runB_time_utc",
        "ridge_z",
        "margin_z",
        "cam_disagree",
        "desc_disagree",
        "sift_inliers",
        "slope",
        "runA_has_cam0",
        "runA_has_cam5",
        "runB_has_cam0",
        "runB_has_cam5",
        "method",
    ]
    final = fused[final_cols].rename(columns={"lo": "runB_frame_lo", "hi": "runB_frame_hi"})
    # "empty runB_frame" is the no-match rule's actual output contract --
    # mapping_full.csv keeps the raw computed value (useful for debugging why a row was
    # abstained), but the public mapping.csv must not report a runB_frame we've explicitly said
    # we don't trust. ambiguous_range rows keep theirs -- "somewhere in this range" is still real
    # information; only no_match/no_frame rows are blanked.
    unmatched = final["status"].isin(["no_match", "no_frame"])
    final.loc[unmatched, ["runB_frame", "runB_frame_lo", "runB_frame_hi", "runB_time_utc"]] = np.nan
    final_path = out_dir / "mapping.csv"
    final.to_csv(final_path, index=False)
    print(f"wrote {full_path} and {final_path} ({len(fused)} rows)")

    n_matched = int((fused["status"] == "matched").sum())
    n_ambiguous = int((fused["status"] == "ambiguous_range").sum())
    n_no_match = int((fused["status"] == "no_match").sum())
    n_no_frame = int((fused["status"] == "no_frame").sum())
    print(
        f"status: matched={n_matched} ambiguous_range={n_ambiguous} "
        f"no_match={n_no_match} no_frame={n_no_frame}"
    )
    return 0


def _cmd_gt_sheets(cfg: Config) -> int:
    """Build stratum S's model-blind labelling sheets (anchor references + coarse runB overview).

    Only stratum S (22 anchors, a seeded deterministic sweep + fixed points of interest) is
    built here -- stratum H (8 anchors at the largest SeqSLAM/DINOv2 path disagreement) needs a
    function that ranks runA rows by `desc_disagree` and isn't implemented yet. Fine sheets are
    not built by this command either -- they need a per-anchor coarse pick from the labeller's
    first pass over the coarse overview, which does not exist until that pass happens.

    Args:
        cfg: Loaded configuration.

    Returns:
        0 on success.
    """
    gt_cfg = cfg.raw["groundtruth"]
    anchors = groundtruth.stratum_s_indices(seed=gt_cfg["stratum_s_seed"])

    store_a0, store_a5 = _open_store(cfg, "runA", "cam0"), _open_store(cfg, "runA", "cam5")
    store_b0, store_b5 = _open_store(cfg, "runB", "cam0"), _open_store(cfg, "runB", "cam5")

    out_dir = cfg.path("gt_dir") / "sheets"
    groundtruth.make_sheets(
        store_a0,
        store_a5,
        store_b0,
        store_b5,
        anchors,
        out_dir,
        coarse_step=gt_cfg["sheets"]["coarse_step"],
        fine_radius=gt_cfg["sheets"]["fine_radius"],
    )
    print(f"stratum S: {len(anchors)} anchors: {anchors.tolist()}")
    print(f"wrote anchor references + coarse overview -> {out_dir}")
    print(
        "stratum H not built yet -- needs a desc_disagree-ranking function; "
        "fine sheets need coarse picks from a first labelling pass"
    )
    return 0


def _cmd_evaluate(cfg: Config) -> int:
    """Score `outputs/mapping.csv` against the ground-truth anchors; write a real, inspectable
    per-anchor comparison table plus a summary, rather than leaving `evaluate`'s numbers as
    something only ever seen in a terminal.

    Args:
        cfg: Loaded configuration.

    Raises:
        FileNotFoundError: If `outputs/mapping.csv` doesn't exist yet (`align`/`verify` not run),
            or no ground-truth label file exists yet (`gt-sheets` + manual labelling not done).

    Returns:
        0 on success.
    """
    out_dir = cfg.path("outputs_dir")
    mapping_path = out_dir / "mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_path} not found -- run `align` then `verify` first")

    mapping = pd.read_csv(mapping_path)
    labels = groundtruth.load_labels(cfg.path("gt_dir"))
    anchors = labels[labels["labeller"] == "user"].copy()

    result = groundtruth.evaluate(mapping, anchors)

    per_anchor_path = out_dir / "evaluation_anchors.csv"
    result["per_anchor"].to_csv(per_anchor_path, index=False)

    summary = {k: v for k, v in result.items() if k != "per_anchor"}
    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    lo_ci, hi_ci = result["hit_at_5_wilson_ci"]
    print(f"n_scored={result['n_scored']}/{result['n_total_anchors']}")
    print(f"median_error={result['median_error']:.1f}  p90_error={result['p90_error']:.1f}")
    print(
        f"hit@2={result['hit_at_2']:.2f}  hit@5={result['hit_at_5']:.2f}  "
        f"hit@10={result['hit_at_10']:.2f}  (hit@5 95% CI: [{lo_ci:.2f}, {hi_ci:.2f}])"
    )
    print(
        f"abstain: true_positive={result['abstain_true_positive']} "
        f"false_negative={result['abstain_false_negative']} "
        f"false_positive={result['abstain_false_positive']}"
    )
    print("by_confidence_tier:")
    for tier, stats in sorted(result["by_confidence_tier"].items()):
        n, correct, acc = stats["n"], stats["n_correct"], stats["accuracy"]
        print(f"  {tier}: n={n}  correct={correct}  accuracy={acc:.2f}")
    print(f"\nwrote {per_anchor_path} and {summary_path}")
    return 0


# why: a one-time qualitative visual read per run/camera, not a per-frame measurement -- a
# defensible simplification given the time available, not a precise per-frame detector
_RUN_WEATHER = {"runA": "sunny", "runB": "overcast_light_rain"}
_RUN_LIGHTING = {"runA": "daylight_hard_shadow", "runB": "daylight_overcast"}
_CAM_LENS_OCCLUSION = {"cam0": "none", "cam5": "intermittent_traffic"}


def _run_camera_facts(
    cfg: Config, run: str, cam: str, sha256_by_file: dict[str, str]
) -> policy.RunCameraFacts:
    """Assemble one `(run, cam)`'s `RunCameraFacts` from Task 1's inspection JSON + integrity check.

    Args:
        cfg: Loaded configuration.
        run: Run id.
        cam: Camera id.
        sha256_by_file: `{"runA/cam0_..._output.hevc": "<sha256>", ...}`, from
            `outputs/inspection/integrity.json`.

    Returns:
        The assembled facts.
    """
    insp = json.loads((cfg.path("outputs_dir") / "inspection" / f"{run}_{cam}.json").read_text())
    src_file = f"{run}/{cam}_20_yuv420p_output.hevc"
    return policy.RunCameraFacts(
        run=run,
        cam=cam,
        decoded_count=insp["frame_count"]["measured_decoded"],
        src_file=src_file,
        src_sha256=sha256_by_file[src_file],
        fps_declared=insp["frame_rate"]["declared_nominal_fps"],
        fps_measured=insp["frame_rate"]["measured_actual_fps"],
        width=insp["resolution"]["declared_cropped_wh"][0],
        height=insp["resolution"]["declared_cropped_wh"][1],
        codec=insp["codec"]["codec_name"],
        idr_indices=tuple(insp["gop"]["idr_indices"]),
        weather=_RUN_WEATHER[run],
        lighting=_RUN_LIGHTING[run],
        lens_occlusion=_CAM_LENS_OCCLUSION[cam],
    )


def _cmd_manifest(cfg: Config) -> int:
    """Task 3: build the per-frame keep/discard manifest.

    Args:
        cfg: Loaded configuration.

    Raises:
        FileNotFoundError: If `outputs/mapping.csv` doesn't exist yet (`align`/`verify` not run).

    Returns:
        0 on success.
    """
    out_dir = cfg.path("outputs_dir")
    mapping_path = out_dir / "mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_path} not found -- run `align` then `verify` first")
    mapping = pd.read_csv(mapping_path)

    integrity = json.loads((out_dir / "inspection" / "integrity.json").read_text())
    sha256_by_file = {f["file"]: f["actual"] for f in integrity["files"]}

    run_cameras: list[policy.RunCameraFacts] = []
    timestamps_by_run: dict[str, NDArray[np.int64]] = {}
    motion_by_run_cam: dict[tuple[str, str], NDArray[np.float64]] = {}
    stationary_by_run: dict[str, list[tuple[int, int]]] = {}

    for run in cfg.runs:
        timestamps_by_run[run] = timestamps.load_timestamps(
            cfg.path("dataset_dir") / run / "cam0_20_yuv420p_output.hevc.timestamps.txt"
        )
        motions = []
        for cam in cfg.cameras:
            run_cameras.append(_run_camera_facts(cfg, run, cam, sha256_by_file))
            store = _open_store(cfg, run, cam)
            m = frames.motion_energy(
                store, cfg.raw["frames"]["motion_thumb_w"], cfg.raw["frames"]["motion_thumb_h"]
            )
            motion_by_run_cam[(run, cam)] = m
            motions.append(m)
        n_common = min(m.size for m in motions)
        stationary_by_run[run] = frames.stationary_segments(
            motions[0][:n_common],
            motions[1][:n_common],
            cfg.raw["frames"]["stationary_threshold_frac"],
            cfg.raw["frames"]["stationary_min_length"],
        )

    p = cfg.raw["policy"]
    manifest = policy.build_manifest(
        run_cameras,
        timestamps_by_run,
        motion_by_run_cam,
        stationary_by_run,
        mapping,
        interval_irregular_lo_ms=p["interval_irregular_lo_ms"],
        interval_irregular_hi_ms=p["interval_irregular_hi_ms"],
        tail_zone_lines=p["tail_zone_lines"],
        dedup_keep_hz=p["dedup_keep_hz"],
        place_id_bucket_frames=p["place_id_bucket_frames"],
        depot_runb_margin=p["depot_runb_margin"],
        split_val_frac=p["split_val_frac"],
        split_test_frac=p["split_test_frac"],
        split_buffer_places=p["split_buffer_places"],
    )

    out_path = out_dir / "keep_manifest.csv"
    manifest.to_csv(out_path, index=False)
    n_keep = int(manifest["keep"].sum())
    print(f"wrote {out_path} ({len(manifest)} rows)")
    print(f"keep={n_keep} discard={len(manifest) - n_keep}")
    print("split:")
    for label, n in manifest["split"].value_counts().items():
        print(f"  {label}: {n}")
    return 0


def _row(mapping: pd.DataFrame, runa_frame: int) -> pd.Series:
    """Look up one `outputs/mapping_full.csv` row by `runA_frame`, for figure captions."""
    return mapping.set_index("runA_frame").loc[runa_frame]


def _fig_start_plateau(stores: dict[str, FrameStore], mapping: pd.DataFrame, out_dir: Path) -> None:
    """Anchors 60/150: visually-similar depot frames collapse onto the same wrong runB match."""
    r60 = _row(mapping, 60)
    viz.plot_frame_comparison(
        [
            (stores["runA_cam0"].get_rgb(60), "runA #60 (query)"),
            (stores["runA_cam0"].get_rgb(150), "runA #150 (query, same depot)"),
            (
                stores["runB_cam0"].get_rgb(81),
                f"runB #81 -- pipeline's answer for both\nconf={r60['confidence']:.1f}, "
                f"ridge_z={r60['ridge_z']:.2f} (WRONG)",
            ),
            (stores["runB_cam0"].get_rgb(100), "runB #100 -- ground truth [98,102]"),
        ],
        "Start-plateau ambiguity: two visually-similar depot queries, one wrong shared answer",
        out_dir / "failure_start_plateau.png",
    )


def _fig_cam5_traffic(stores: dict[str, FrameStore], out_dir: Path) -> None:
    """runB cam5 ~90-110: a passing car occludes cam5 only, not the kerb-facing cam0."""
    viz.plot_frame_comparison(
        [
            (stores["runB_cam5"].get_rgb(85), "runB cam5 #85 (before)"),
            (stores["runB_cam5"].get_rgb(93), "runB cam5 #93 (car passing)"),
            (stores["runB_cam0"].get_rgb(93), "runB cam0 #93 (unaffected, kerb-facing)"),
        ],
        "cam5 passing-traffic occlusion: a transient absent from cam0's view of the same moment",
        out_dir / "failure_cam5_traffic.png",
    )


def _fig_rain_attractor(
    stores: dict[str, FrameStore], mapping: pd.DataFrame, out_dir: Path
) -> None:
    """runB #183: the low-contrast, rain-degraded scene SeqSLAM-alone late-fusion collapses onto
    for anchors 60, 150 and 183. Shown against runA #183 itself so the mismatch is visible: the
    query is a dry depot/carport frame, the attractor is an unrelated rainy gate scene -- the
    shared frame number is a coincidence (runA_frame=183 vs runB column 183), not a causal link."""
    r = _row(mapping, 183)
    viz.plot_frame_comparison(
        [
            (stores["runA_cam0"].get_rgb(183), "runA #183 (query, ground truth runB [102,104])"),
            (
                stores["runB_cam0"].get_rgb(183),
                f"runB cam0 #183 -- SeqSLAM-alone false attractor\n"
                f"(shipped pipeline lands near #100 instead, ridge_z={r['ridge_z']:.2f})",
            ),
            (stores["runB_cam5"].get_rgb(183), "runB cam5 #183"),
        ],
        "Rain-degraded false attractor: SeqSLAM-alone late-fusion pulls anchors 60/150/183 here",
        out_dir / "failure_rain_attractor.png",
    )


def _fig_loop_closure(stores: dict[str, FrameStore], mapping: pd.DataFrame, out_dir: Path) -> None:
    """Anchor 2600: the depot revisited at loop closure -- genuinely ambiguous, correctly
    abstained."""
    r = _row(mapping, 2600)
    caption_b = (
        f"runB #2597 (near route end -- pipeline's\n"
        f"raw candidate, ridge_z={r['ridge_z']:.2f} < tau, correctly abstained)"
    )
    viz.plot_frame_comparison(
        [
            (stores["runA_cam0"].get_rgb(2600), "runA #2600 (query, back at the depot)"),
            (stores["runB_cam0"].get_rgb(10), "runB #10 (near route start -- plausible)"),
            (stores["runB_cam0"].get_rgb(2597), caption_b),
        ],
        "Loop-closure ambiguity: the depot looks the same at both ends of the route",
        out_dir / "failure_loop_closure.png",
    )


def _fig_low_texture(stores: dict[str, FrameStore], mapping: pd.DataFrame, out_dir: Path) -> None:
    """runA 1659-1678: correct tracking (slope~1.0) but chronically weak margin_z -- a
    low-discriminativeness stretch, not a single attractor spike."""
    r = _row(mapping, 1660)
    viz.plot_frame_comparison(
        [
            (stores["runA_cam0"].get_rgb(1660), "runA #1660 (query)"),
            (
                stores["runB_cam0"].get_rgb(1588),
                f"runB #1588 -- matched, but\nmargin_z={r['margin_z']:.2f} (weak)",
            ),
            (stores["runB_cam0"].get_rgb(1598), "runB #1598 (+10 -- looks almost as good)"),
        ],
        "Low-texture stretch (runA 1659-1678): right track, chronically weak margin",
        out_dir / "failure_low_texture.png",
    )


def _cmd_figures(cfg: Config) -> int:
    """T2 failure-taxonomy figures: side-by-side runA/runB comparisons for the report's
    failure-cases section, each grounded in a real instance, not a constructed example.

    Args:
        cfg: Loaded configuration.

    Raises:
        FileNotFoundError: If `outputs/mapping_full.csv` doesn't exist yet.

    Returns:
        0 on success.
    """
    out_dir = cfg.path("outputs_dir")
    mapping_path = out_dir / "mapping_full.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_path} not found -- run `align` then `verify` first")
    mapping = pd.read_csv(mapping_path)

    stores = {f"{run}_{cam}": _open_store(cfg, run, cam) for run in cfg.runs for cam in cfg.cameras}
    fig_dir = out_dir / "inspection" / "figures"

    _fig_start_plateau(stores, mapping, fig_dir)
    _fig_cam5_traffic(stores, fig_dir)
    _fig_rain_attractor(stores, mapping, fig_dir)
    _fig_loop_closure(stores, mapping, fig_dir)
    _fig_low_texture(stores, mapping, fig_dir)

    print(f"wrote 5 failure-taxonomy figures to {fig_dir}")
    return 0


def _cmd_report(cfg: Config) -> int:
    """Build `report/report.pdf` from `report/report.tex` via `latexmk`.

    Args:
        cfg: Loaded configuration.

    Raises:
        FileNotFoundError: If `report/report.tex` doesn't exist.

    Returns:
        0 on success, 1 if `latexmk` fails (its stdout/stderr are passed through either way).
    """
    report_dir = cfg.path("report_dir")
    tex_path = report_dir / "report.tex"
    if not tex_path.exists():
        raise FileNotFoundError(f"{tex_path} not found")
    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
        cwd=report_dir,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(f"error: latexmk failed (exit {result.returncode})", file=sys.stderr)
        return 1
    print(f"wrote {report_dir / 'report.pdf'}")
    return 0


def _not_implemented(name: str) -> int:
    """Print a "not implemented" error for a subcommand that has no handler yet.

    Args:
        name: Subcommand name.

    Returns:
        1 (always an error).
    """
    print(f"error: '{name}' is not implemented yet", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the `routealign` argument parser, with one subcommand per Makefile target.

    Returns:
        The configured parser.
    """
    p = argparse.ArgumentParser(prog="routealign")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-data")
    sub.add_parser("characterise")
    sub.add_parser("decode")
    describe_p = sub.add_parser("describe")
    describe_p.add_argument("--feat", choices=["seqslam", "dinov2"], required=False)
    for name in ("align", "verify", "gt-sheets", "evaluate", "manifest", "figures", "report"):
        sub.add_parser(name)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m routealign`.

    Args:
        argv (optional): Argument list to parse. Defaults to `None`, which makes argparse read
            `sys.argv`.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "verify-data":
        return _cmd_verify_data(cfg)
    if args.command == "decode":
        return _cmd_decode(cfg)
    if args.command == "characterise":
        return _cmd_characterise(cfg)
    if args.command == "describe":
        return _cmd_describe(cfg, args.feat, args.limit, args.force)
    if args.command == "align":
        return _cmd_align(cfg, args.limit, args.force)
    if args.command == "verify":
        return _cmd_verify(cfg)
    if args.command == "gt-sheets":
        return _cmd_gt_sheets(cfg)
    if args.command == "evaluate":
        return _cmd_evaluate(cfg)
    if args.command == "manifest":
        return _cmd_manifest(cfg)
    if args.command == "figures":
        return _cmd_figures(cfg)
    if args.command == "report":
        return _cmd_report(cfg)
    return _not_implemented(args.command)


if __name__ == "__main__":
    sys.exit(main())
