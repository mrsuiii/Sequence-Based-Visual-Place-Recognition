"""ffprobe/ffmpeg subprocess wrappers. The only place video files are touched.

Video I/O goes through the ffmpeg/ffprobe CLI via subprocess only, never PyAV (see decisions.md
D0) and never `-hwaccel videotoolbox` (fails on these raw HEVC streams; CLAUDE.md §5).
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB


def sha256(path: str | Path) -> str:
    """Compute a file's SHA-256, streamed so the whole file is never loaded into RAM.

    Args:
        path: File to hash.

    Returns:
        The hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class StreamProbe:
    """ffprobe's declared fields for the first video stream (CLAUDE.md §5: declared, not measured).

    Attributes:
        width: Declared width in pixels.
        height: Declared height in pixels.
        codec_name: Codec, e.g. `"hevc"`.
        pix_fmt: Pixel format, e.g. `"yuv420p"`.
        r_frame_rate: ffprobe's `r_frame_rate` field, as `"num/den"`.
        avg_frame_rate: ffprobe's `avg_frame_rate` field, as `"num/den"`. For a headerless
            elementary stream this is a content-independent fallback, not a real average
            (see findings.md) -- never use it as the declared frame rate.
    """

    width: int
    height: int
    codec_name: str
    pix_fmt: str
    r_frame_rate: str
    avg_frame_rate: str


def probe(path: str | Path) -> StreamProbe:
    """Read ffprobe's declared metadata for a video's first stream.

    Args:
        path: Video file to probe.

    Returns:
        The declared stream fields.
    """
    fields = "width,height,codec_name,pix_fmt,r_frame_rate,avg_frame_rate"
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            f"stream={fields}",
            "-of",
            "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
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
    """Fully decode a video and count its frames (measured, not declared).

    Slow -- a full decode pass. Cache the result rather than calling this repeatedly;
    `scan_nals` gives a second, independent count for cross-checking.

    Args:
        path: Video file to decode.

    Returns:
        The number of decoded frames.
    """
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
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
    """Decode every frame of a video to numbered JPEGs in one sequential ffmpeg pass.

    Writes `out_dir/00000.jpg`, `out_dir/00001.jpg`, ..., scaled to `width` x `height`. Does not
    itself check the count against `count_decoded(path)`; that comparison is the caller's job
    (see `cli.py`'s `decode` subcommand), so a file is never decoded twice just to check itself.

    Args:
        path: Source video file.
        out_dir: Directory to write JPEGs into (created if missing).
        width: Output width in pixels.
        height: Output height in pixels.
        jpeg_quality: ffmpeg `-q:v` value (2 = best, 31 = worst).
        fps_mode (optional): ffmpeg `-fps_mode` value. Defaults to `"passthrough"`, which writes
            exactly one JPEG per decoded frame with no resampling.

    Returns:
        The number of JPEGs written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%05d.jpg")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"scale={width}:{height}",
            "-fps_mode",
            fps_mode,
            "-start_number",
            "0",
            "-q:v",
            str(jpeg_quality),
            pattern,
        ],
        check=True,
    )
    return len(list(out_dir.glob("*.jpg")))


@dataclass(frozen=True)
class NalStats:
    """Picture/GOP structure from `ffprobe -show_frames` -- a picture-level scan (key_frame,
    pict_type per frame), not a byte-level Annex-B NAL parse (decisions.md D0).

    Attributes:
        n_pictures: Total decoded picture count -- an independent cross-check of `count_decoded`.
        n_idr: Number of IDR (keyframe) pictures.
        idr_indices: Frame index of each IDR picture.
        gop_sizes: Frame count between consecutive IDR pictures.
        pict_type_counts: Count of each picture type, e.g. `{"I": 11, "P": 674, "B": 1989}`.
        file_size_bytes: Size of the video file in bytes.
        file_size_mod_262144: `file_size_bytes % 262144` -- see findings.md for why this is
            checked (weak, circumstantial evidence about a truncated final frame).
    """

    n_pictures: int
    n_idr: int
    idr_indices: list[int]
    gop_sizes: list[int]
    pict_type_counts: dict[str, int]
    file_size_bytes: int
    file_size_mod_262144: int


def scan_nals(path: str | Path) -> NalStats:
    """Scan a video's picture/GOP structure via `ffprobe -show_frames`.

    Args:
        path: Video file to scan.

    Returns:
        The picture/GOP structure and file-size check.
    """
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=key_frame,pict_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = [line.split(",") for line in out.strip().splitlines() if line.strip()]
    key_frame_flags = [row[0] == "1" for row in rows]
    pict_types = [row[1] for row in rows]

    idr_indices = [i for i, is_kf in enumerate(key_frame_flags) if is_kf]
    gop_sizes = [b - a for a, b in zip(idr_indices, idr_indices[1:], strict=False)]
    counts: dict[str, int] = {}
    for pt in pict_types:
        counts[pt] = counts.get(pt, 0) + 1

    size = Path(path).stat().st_size
    return NalStats(
        n_pictures=len(pict_types),
        n_idr=len(idr_indices),
        idr_indices=idr_indices,
        gop_sizes=gop_sizes,
        pict_type_counts=counts,
        file_size_bytes=size,
        file_size_mod_262144=size % 262144,
    )


@dataclass(frozen=True)
class SpsVui:
    """Declared HEVC SPS/VUI fields (CLAUDE.md §5: declared, not measured -- though decode would
    fail outright if the dimensions were wrong, which makes this an unusually trustworthy
    declared value in practice).

    Attributes:
        pic_width_luma: SPS coded width in luma samples, before conformance cropping.
        pic_height_luma: SPS coded height in luma samples, before conformance cropping.
        conf_win_left: Conformance-window left offset, in chroma samples.
        conf_win_right: Conformance-window right offset, in chroma samples.
        conf_win_top: Conformance-window top offset, in chroma samples.
        conf_win_bottom: Conformance-window bottom offset, in chroma samples.
        cropped_width: Displayed width after conformance cropping.
        cropped_height: Displayed height after conformance cropping.
        level_idc: HEVC `general_level_idc`; divide by 30 for the level number (e.g. 120 -> 4.0).
        max_dec_pic_buffering: Decoded picture buffer size in frames.
        max_num_reorder_pics: Maximum picture reorder depth.
        vui_timing_info_present: Whether the VUI declares a frame rate at all.
        vui_num_units_in_tick: VUI `num_units_in_tick`, or `None` if not present.
        vui_time_scale: VUI `time_scale`, or `None` if not present.
        vui_declared_fps: `vui_time_scale / vui_num_units_in_tick`, or `None` if not present.
        video_signal_type_present: Whether colour range/primaries are signalled at all.
    """

    pic_width_luma: int
    pic_height_luma: int
    conf_win_left: int
    conf_win_right: int
    conf_win_top: int
    conf_win_bottom: int
    cropped_width: int
    cropped_height: int
    level_idc: int
    max_dec_pic_buffering: int
    max_num_reorder_pics: int
    vui_timing_info_present: bool
    vui_num_units_in_tick: int | None
    vui_time_scale: int | None
    vui_declared_fps: float | None
    video_signal_type_present: bool


_TRACE_FIELD_RE = re.compile(r"^\[trace_headers[^\]]*\]\s+\d+\s+(\S+)\s+[01]+\s+=\s+(-?\d+)\s*$")


def sps_vui(path: str | Path) -> SpsVui:
    """Parse a video's first SPS/VUI via `ffmpeg -bsf:v trace_headers`.

    Uses ffmpeg's own HEVC bitstream parser rather than a hand-rolled Exp-Golomb/RBSP parser
    (decisions.md D0). Requires `-c:v copy`, since `trace_headers` reads the coded bitstream and
    errors if the stream is decoded first.

    Args:
        path: Video file to parse.

    Returns:
        The declared SPS/VUI fields.

    Raises:
        ValueError: If an expected field is missing from the trace output (e.g. the ffmpeg
            version changed the `trace_headers` format).
    """
    out = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-c:v",
            "copy",
            "-bsf:v",
            "trace_headers",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    ).stderr

    fields: dict[str, int] = {}
    in_sps = False
    for line in out.splitlines():
        if "Sequence Parameter Set" in line:
            in_sps = True
            continue
        if "Picture Parameter Set" in line:
            in_sps = False
            continue
        if not in_sps:
            continue
        m = _TRACE_FIELD_RE.match(line)
        if m and m.group(1) not in fields:  # first SPS occurrence only
            fields[m.group(1)] = int(m.group(2))

    def get(name: str) -> int:
        if name not in fields:
            raise ValueError(
                f"{path}: sps_vui could not find field {name!r} in trace_headers output "
                f"(ffmpeg version / trace_headers format may have changed)"
            )
        return fields[name]

    width = get("pic_width_in_luma_samples")
    height = get("pic_height_in_luma_samples")
    left, right = get("conf_win_left_offset"), get("conf_win_right_offset")
    top, bottom = get("conf_win_top_offset"), get("conf_win_bottom_offset")
    # 4:2:0 chroma: SubWidthC = SubHeightC = 2 (HEVC spec eq. 7-25/7-26) -- offsets are in chroma
    # sample units, cropped luma size subtracts 2x the offset.
    cropped_w = width - 2 * (left + right)
    cropped_h = height - 2 * (top + bottom)

    timing_present = bool(fields.get("vui_timing_info_present_flag", 0))
    num_units = fields.get("vui_num_units_in_tick") if timing_present else None
    time_scale = fields.get("vui_time_scale") if timing_present else None
    declared_fps = (time_scale / num_units) if (timing_present and num_units) else None

    return SpsVui(
        pic_width_luma=width,
        pic_height_luma=height,
        conf_win_left=left,
        conf_win_right=right,
        conf_win_top=top,
        conf_win_bottom=bottom,
        cropped_width=cropped_w,
        cropped_height=cropped_h,
        level_idc=get("general_level_idc"),
        max_dec_pic_buffering=get("sps_max_dec_pic_buffering_minus1[0]") + 1,
        max_num_reorder_pics=get("sps_max_num_reorder_pics[0]"),
        vui_timing_info_present=timing_present,
        vui_num_units_in_tick=num_units,
        vui_time_scale=time_scale,
        vui_declared_fps=declared_fps,
        video_signal_type_present=bool(fields.get("video_signal_type_present_flag", 0)),
    )
