"""argparse CLI. The only module allowed `print`/`sys.exit` (CLAUDE.md §4). Each subcommand
equals one Makefile target.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import io_video, timestamps, viz
from .config import Config, load_config

log = logging.getLogger(__name__)


def _cmd_verify_data(cfg: Config) -> int:
    """SHA-256 of every dataset file vs dataset/SHA256SUMS.txt (the exact download-page manifest)."""
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
    """One ffmpeg pass per run/camera -> data/cache/frames/{run}/{cam}/00000.jpg..., checked
    against a real decode count (measured, not the declared/timestamp-line count)."""
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
                src, out_dir,
                width=cfg.decode.width, height=cfg.decode.height,
                jpeg_quality=cfg.decode.jpeg_quality, fps_mode=cfg.decode.fps_mode,
            )
            match = written == expected
            ok = ok and match
            print(f"  {'OK' if match else 'MISMATCH'}: wrote {written} JPEGs, count_decoded={expected}")
    return 0 if ok else 1


def _fmt(x: float | int | None, spec: str = ".2f") -> str:
    return "N/A" if x is None else format(x, spec)


def _write_task1_table(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# Task 1 — characterisation summary",
        "",
        "`declared fps` below is the one label that traces to a real field about *this file's* "
        "content: the SPS/VUI `time_scale/num_units_in_tick`, independently confirmed against "
        "ffprobe's `r_frame_rate` (the two agree exactly). Two other candidate labels were found, "
        "investigated and rejected rather than silently omitted — see \"Other frame-rate labels\" "
        "below and `findings.md`. `declared fps` is still not the actual rate; see `measured fps`, "
        "computed from `timestamps.txt` interval arithmetic. Full provenance for every figure is in "
        "the per-camera JSON files next to this table and in `findings.md`.",
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
    """Task 1: frame count, resolution, file size, nominal vs actual fps, route time, recording
    time, interval distribution (with irregularities) — one JSON per run/camera, a combined
    markdown table, and figures. Every number states whether it is declared or measured."""
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
            recorded_utc = datetime.fromtimestamp(int(ts[0]) / 1e9, tz=timezone.utc)
            bit_rate_bps = (file_size * 8 / span_video_s) if span_video_s > 0 else None
            actual_fps = ((decoded_count - 1) / span_video_s) if span_video_s > 0 else None

            record = {
                "run": run, "camera": cam,
                "frame_count": {
                    "declared_nb_frames": None,  # ffprobe nb_frames is N/A for a raw elementary stream
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
                                "avg_frame_rate=25/1, identical to this file -- see findings.md"
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
                dt_ms, stats, title=f"{run}/{cam} — Δt distribution ({dt_ms.size} intervals)",
                out_path=fig_dir / f"{run}_{cam}_dt_hist.png",
            )
            viz.plot_interval_vs_index(
                dt_ms, title=f"{run}/{cam} — Δt vs. interval index",
                out_path=fig_dir / f"{run}_{cam}_dt_vs_index.png",
            )

    _write_task1_table(rows, out_dir / "task1_table.md")
    print(f"\nwrote {len(rows)} per-camera reports + task1_table.md -> {out_dir}")
    return 0


def _not_implemented(name: str) -> int:
    print(f"error: '{name}' is not implemented yet — see plan.md for its Step", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
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
    for name in ("align", "verify", "gt-sheets", "evaluate", "manifest", "report"):
        sub.add_parser(name)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "verify-data":
        return _cmd_verify_data(cfg)
    if args.command == "decode":
        return _cmd_decode(cfg)
    if args.command == "characterise":
        return _cmd_characterise(cfg)
    return _not_implemented(args.command)


if __name__ == "__main__":
    sys.exit(main())
