"""Config: the single source of truth for tunable parameters (CLAUDE.md §4).

Only the sections with a real consumer today (paths, runs, cameras, decode) are strongly typed;
`raw` is an escape hatch for the rest and should shrink as later steps add their own typed sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DecodeConfig:
    """Parameters for `io_video.decode_to_jpegs`.

    Attributes:
        width: Output JPEG width in pixels.
        height: Output JPEG height in pixels.
        jpeg_quality: ffmpeg `-q:v` value (2 = best, 31 = worst).
        fps_mode: ffmpeg `-fps_mode` value. Must be `"passthrough"` (CLAUDE.md §5) so every
            decoded frame produces exactly one JPEG.
    """

    width: int
    height: int
    jpeg_quality: int
    fps_mode: str


@dataclass(frozen=True)
class Config:
    """Frozen, typed view of config.yaml. Nothing else in the package should read the YAML directly.

    Attributes:
        paths: Named directories (e.g. `"dataset_dir"`, `"outputs_dir"`), keyed by name from
            config.yaml's `paths` section. Resolve with `path()`.
        runs: Run ids in the order they should be processed, e.g. `["runA", "runB"]`.
        cameras: Camera ids in the order they should be processed, e.g. `["cam0", "cam5"]`.
        decode: Parameters for the JPEG-cache decode step.
        raw: The full parsed config.yaml, for sections not yet given their own typed field.
    """

    paths: dict[str, str]
    runs: list[str]
    cameras: list[str]
    decode: DecodeConfig
    raw: dict[str, Any] = field(repr=False)

    def path(self, key: str) -> Path:
        """Resolve a named directory from the `paths` section.

        Args:
            key: Name of the directory, e.g. `"dataset_dir"`.

        Returns:
            The directory as a `Path`.
        """
        return Path(self.paths[key])


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into `base`, without mutating either input.

    Args:
        base: The starting dictionary.
        overrides: Values to layer on top; nested dicts are merged key-by-key instead of
            replacing the whole nested dict.

    Returns:
        A new, merged dictionary.
    """
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    path: str | Path = "config.yaml", overrides: dict[str, Any] | None = None
) -> Config:
    """Load config.yaml into a frozen `Config`, optionally layering CLI overrides on top.

    Args:
        path (optional): Path to the YAML file. Defaults to `"config.yaml"`.
        overrides (optional): Values to deep-merge on top of the file, e.g. from `--limit`.
            Defaults to `None` (no overrides).

    Returns:
        The loaded configuration.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(
        paths=raw["paths"],
        runs=raw["runs"],
        cameras=raw["cameras"],
        decode=DecodeConfig(**raw["decode"]),
        raw=raw,
    )
