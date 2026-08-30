"""Config: the single source of truth for tunable parameters (CLAUDE.md §4).

Only the sections a real consumer exists for (paths, runs, cameras, decode) are strongly typed so
far; `raw` is an escape hatch for the rest and should shrink as each step's modules are built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DecodeConfig:
    width: int
    height: int
    jpeg_quality: int
    fps_mode: str


@dataclass(frozen=True)
class Config:
    """Frozen, typed view of config.yaml. Nothing else in the package reads config.yaml directly."""

    paths: dict[str, str]
    runs: list[str]
    cameras: list[str]
    decode: DecodeConfig
    raw: dict[str, Any] = field(repr=False)

    def path(self, key: str) -> Path:
        return Path(self.paths[key])


def _deep_merge(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path = "config.yaml", overrides: dict[str, Any] | None = None) -> Config:
    """Load config.yaml (plus optional overrides, deep-merged) into a frozen Config."""
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
