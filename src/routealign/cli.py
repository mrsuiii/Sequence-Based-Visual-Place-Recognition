"""argparse CLI. The only module allowed `print`/`sys.exit` (CLAUDE.md §4). Each subcommand
equals one Makefile target.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import io_video
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
    return _not_implemented(args.command)


if __name__ == "__main__":
    sys.exit(main())
