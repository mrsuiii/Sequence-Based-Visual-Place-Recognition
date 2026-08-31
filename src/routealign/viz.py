"""Every figure in outputs/figures/. No computation lives here — arrays and DataFrames in,
PNGs out.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never opens a window, safe to call from the CLI

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_interval_histogram(dt_ms: NDArray[np.float64], stats: dict, title: str, out_path: str | Path) -> None:
    """Δt histogram (log-y) with the median marked, for the Task 1 report."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dt_ms, bins=100, color="#4C72B0", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("interval between consecutive frames (ms)")
    ax.set_ylabel("count (log scale)")
    ax.set_title(title)
    ax.axvline(
        stats["median_ms"], color="#C44E52", linestyle="--",
        label=f"median = {stats['median_ms']:.2f} ms",
    )
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_interval_vs_index(dt_ms: NDArray[np.float64], title: str, out_path: str | Path) -> None:
    """Δt against frame index — shows *where* irregularities happen (start/end/scattered), not
    just their overall distribution."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(np.arange(dt_ms.size), dt_ms, linewidth=0.6, color="#4C72B0")
    ax.set_xlabel("interval index")
    ax.set_ylabel("Δt (ms)")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_similarity_matrix(*args, **kwargs):
    """Similarity matrix with the chosen DTW path, rejected spans, and corner aliasing visible.
    Not yet implemented — Step 4/5."""
    raise NotImplementedError("plot_similarity_matrix: implement in Step 4/5 (plan.md §6-7)")
