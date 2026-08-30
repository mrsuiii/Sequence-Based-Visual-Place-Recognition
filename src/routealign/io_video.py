"""ffprobe/ffmpeg subprocess wrappers. The only place video files are touched.

Video I/O goes through the ffmpeg/ffprobe CLI via subprocess only (CLAUDE.md §5) — never PyAV
(conflicts with OpenCV's bundled ffmpeg on macOS, confirmed when setting up this env — see
findings.md) and never `-hwaccel videotoolbox` (fails on these raw HEVC elementary streams).
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB


def sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes, streamed — never loads the whole file into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class StreamProbe:
    """ffprobe's declared stream fields. r_frame_rate/avg_frame_rate are declared, not measured —
    see findings.md for why they disagree with each other and with the measured rate on these files."""

    width: int
    height: int
    codec_name: str
    pix_fmt: str
    r_frame_rate: str
    avg_frame_rate: str


def probe(path: str | Path) -> StreamProbe:
    """ffprobe's declared metadata for the first video stream."""
    fields = "width,height,codec_name,pix_fmt,r_frame_rate,avg_frame_rate"
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", f"stream={fields}",
            "-of", "default=noprint_wrappers=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    values = dict(line.split("=", 1) for line in out.strip().splitlines())
    return StreamProbe(
        width=int(values["width"]),
        height=int(values["height"]),
        codec_name=values["codec_name"],
        pix_fmt=values["pix_fmt"],
        r_frame_rate=values["r_frame_rate"],
        avg_frame_rate=values["avg_frame_rate"],
    )


def count_decoded(path: str | Path) -> int:
    """Actually decode the file and count frames (measured, not declared). A full decode pass —
    slow; cache the result rather than calling this repeatedly."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.strip())


def decode_to_jpegs(
    path: str | Path,
    out_dir: str | Path,
    width: int,
    height: int,
    jpeg_quality: int,
    fps_mode: str = "passthrough",
) -> int:
    """One sequential ffmpeg pass: decode every frame of `path` to `out_dir/00000.jpg`, `00001.jpg`,
    ... scaled to width x height. Returns the number of JPEGs written. Does not itself validate that
    count against count_decoded(path) — that comparison is the caller's job (cli.py's `decode`
    subcommand), so a single file is never decoded twice just to check itself."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%05d.jpg")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(path),
            "-vf", f"scale={width}:{height}",
            "-fps_mode", fps_mode,
            "-start_number", "0",
            "-q:v", str(jpeg_quality),
            pattern,
        ],
        check=True,
    )
    return len(list(out_dir.glob("*.jpg")))


def scan_nals(path: str | Path) -> None:
    """NAL-unit scan: VPS/SPS/PPS/SEI/IDR positions, GOP structure, coded-picture count via
    `first_slice_segment_in_pic_flag`. Not yet implemented — Step 1 (plan.md §3)."""
    raise NotImplementedError("scan_nals: implement in Step 1 (plan.md §3)")


def sps_vui(path: str | Path) -> None:
    """Parse the HEVC SPS/VUI: coded size, conformance window, time_scale/num_units_in_tick.
    Not yet implemented — Step 1 (plan.md §3)."""
    raise NotImplementedError("sps_vui: implement in Step 1 (plan.md §3)")
