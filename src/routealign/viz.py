"""Every figure in outputs/figures/. No computation lives here -- arrays and DataFrames in,
PNGs out.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never opens a window, safe to call from the CLI

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_interval_histogram(
    dt_ms: NDArray[np.float64], stats: dict, title: str, out_path: str | Path
) -> None:
    """Save a Δt histogram (log-scale y-axis) with the median marked.

    Args:
        dt_ms: float64[N] interval values in milliseconds, e.g. from `np.diff(ts)`.
        stats: Summary dict from `timestamps.interval_stats`; only `"median_ms"` is used.
        title: Plot title.
        out_path: File to save the PNG to (parent directories are created if missing).
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dt_ms, bins=100, color="#4C72B0", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("interval between consecutive frames (ms)")
    ax.set_ylabel("count (log scale)")
    ax.set_title(title)
    ax.axvline(
        stats["median_ms"],
        color="#C44E52",
        linestyle="--",
        label=f"median = {stats['median_ms']:.2f} ms",
    )
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_interval_vs_index(dt_ms: NDArray[np.float64], title: str, out_path: str | Path) -> None:
    """Save a plot of Δt against interval index, to show *where* irregularities happen.

    Args:
        dt_ms: float64[N] interval values in milliseconds, e.g. from `np.diff(ts)`.
        title: Plot title.
        out_path: File to save the PNG to (parent directories are created if missing).
    """
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(np.arange(dt_ms.size), dt_ms, linewidth=0.6, color="#4C72B0")
    ax.set_xlabel("interval index")
    ax.set_ylabel("Δt (ms)")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_similarity_matrix(
    similarity: NDArray[np.float64],
    path: NDArray[np.int64],
    title: str,
    out_path: str | Path,
) -> None:
    """Save a similarity matrix as a heatmap with a path overlaid.

    Generic over what `path` represents: the unconstrained per-row argmax produces the "corner
    aliasing" diagnostic (showing where naive nearest-neighbour matching would jump around);
    the same function is reused for the actual DTW path once one exists.

    Args:
        similarity: float64[NA, NB] similarity matrix (runA rows, runB columns).
        path: int64[L, 2] path as (i, j) pairs to overlay.
        title: Plot title.
        out_path: File to save the PNG to (parent directories are created if missing).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(similarity, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.plot(path[:, 1], path[:, 0], color="#E8433D", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("runB frame")
    ax.set_ylabel("runA frame")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="similarity")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _brighten(img: NDArray[np.uint8], gamma: float) -> NDArray[np.uint8]:
    """Gamma-correct an image for display only -- these dashcam frames run dark (overcast,
    tree shade, rain), and a report figure a reader cannot actually see is useless. Never used
    on the path the matching pipeline itself reads (`descriptors.py`/`similarity.py` read the
    JPEG cache directly, not through this function).

    Args:
        img: uint8[H, W, 3] RGB image.
        gamma: Exponent applied to normalised pixel values; `<1` brightens, `1` is a no-op.

    Returns:
        uint8[H, W, 3] gamma-corrected image.
    """
    normalised = img.astype(np.float64) / 255.0
    return np.clip(np.power(normalised, gamma) * 255.0, 0, 255).astype(np.uint8)


def plot_frame_comparison(
    panels: list[tuple[NDArray[np.uint8], str]],
    suptitle: str,
    out_path: str | Path,
    gamma: float = 0.6,  # why: these frames run dark; 0.6 keeps colour but lifts shadows enough
) -> None:  #      to read on a printed page (one failure-case panel was unreadable at 1.0)
    """Lay `panels` out side by side with per-panel captions, for the T2 failure-taxonomy
    figures: a runA query frame next to its predicted and/or ground-truth runB match(es), so a
    reader sees the same evidence the reported numbers are based on.

    Args:
        panels: `(image, caption)` pairs, left to right. Images are `uint8[H, W, 3]` RGB
            (`frames.FrameStore.get_rgb`), any height/width -- not required to match between
            panels.
        suptitle: Figure title.
        out_path: File to save the PNG to (parent directories are created if missing).
        gamma (float, optional): Display-only brightening, see `_brighten`. Defaults to 0.6.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6))
    for ax, (img, caption) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(_brighten(img, gamma))
        ax.set_title(caption, fontsize=9)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
