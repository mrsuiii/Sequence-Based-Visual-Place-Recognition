"""Analysis figure: motion energy + stationary segments, 2x2 grid (run x camera).

Each panel shows one (run, cam)'s motion-energy curve, that camera's own threshold line
(stationary_threshold_frac x its median), a light band where that camera alone is below its
threshold, and darker spans for the final run-level stationary segments -- which require BOTH
cameras below threshold for >= stationary_min_length frames, so the darker spans are identical
between the two panels of a run while the light bands differ. That difference is the point of
the both-cameras rule: one camera alone dipping low (a blank wall, a low-texture stretch) is not
a stop.

Exploration only: writes notebooks/results/stationary_segments.png, nothing in outputs/.
Run from the repo root: conda run -n phase2-vpr python notebooks/stationary_segments_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from routealign import frames
from routealign.config import Config, load_config
from routealign.frames import FrameStore

OUT_PATH = Path(__file__).parent / "results" / "stationary_segments.png"


def _store(cfg: Config, run: str, cam: str) -> FrameStore:
    return FrameStore(
        cfg.path("cache_dir") / "frames", run, cam, crop_row_frac=cfg.raw["frames"]["crop_row_frac"]
    )


def main() -> None:
    cfg = load_config("config.yaml")
    fcfg = cfg.raw["frames"]
    w, h = fcfg["motion_thumb_w"], fcfg["motion_thumb_h"]
    frac, min_len = fcfg["stationary_threshold_frac"], fcfg["stationary_min_length"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="row")
    for row, run in enumerate(cfg.runs):
        print(f"computing motion energy for {run} ...")
        motion = {cam: frames.motion_energy(_store(cfg, run, cam), w, h) for cam in cfg.cameras}
        n_common = min(m.size for m in motion.values())
        truncated = {cam: m[:n_common] for cam, m in motion.items()}
        segments = frames.stationary_segments(
            truncated[cfg.cameras[0]], truncated[cfg.cameras[1]], frac, min_len
        )
        seg_note = ", ".join(f"{s}-{e - 1} ({e - s}f)" for s, e in segments)
        print(f"  {run}: {len(segments)} joint stationary segment(s): {seg_note}")

        for col, cam in enumerate(cfg.cameras):
            ax = axes[row, col]
            m = truncated[cam]
            threshold = frac * float(np.median(m))
            ax.plot(m, lw=0.4, color="#1f4e79", label="motion energy")
            ax.axhline(
                threshold,
                color="#c0392b",
                lw=1.0,
                ls="--",
                label=f"{frac} x median = {threshold:.2f}",
            )
            below = m < threshold
            ax.fill_between(
                np.arange(m.size),
                0,
                m.max(),
                where=below,
                color="#f5b041",
                alpha=0.35,
                linewidth=0,
                label="this cam below threshold",
            )
            for s, e in segments:
                ax.axvspan(s, e - 1, color="#27ae60", alpha=0.45)
            ax.set_title(f"{run} {cam} -- {len(segments)} joint segment(s)", fontsize=10)
            ax.set_ylabel("mean |diff| (gray 64x48)", fontsize=8)
            ax.set_xlabel("transition index (frame k -> k+1)", fontsize=8)
            ax.tick_params(labelsize=8)
            if row == 0 and col == 0:
                # one legend entry for the joint spans, drawn via a proxy patch
                from matplotlib.patches import Patch

                handles, labels = ax.get_legend_handles_labels()
                handles.append(Patch(color="#27ae60", alpha=0.45))
                labels.append("joint stationary segment (both cams, >= 10f)")
                ax.legend(handles, labels, fontsize=7, loc="upper right")

    fig.suptitle(
        "Motion energy and stationary segments per run/camera "
        "(green spans = final run-level segments, identical within a run by construction)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=130)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
