"""FrameStore: read-through access to the JPEG cache, crops, LRU. Motion energy and stationarity
detection. Step 2 (plan.md §4).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class FrameStore:
    """Read-through, LRU-cached access to one run/camera's JPEG cache.

    Every frame is cropped to rows `[0, crop_row_frac * H)` before being cached or returned --
    this drops the wing-mirror/vignette band along the bottom of the frame.

    Attributes:
        n_frames: Number of cached JPEGs found for this run/camera.
    """

    def __init__(
        self, cache_dir: str | Path, run: str, cam: str, crop_row_frac: float, max_cached: int = 512
    ) -> None:
        """Open a JPEG cache directory for one run/camera.

        Args:
            cache_dir: Root of the JPEG cache, e.g. `"data/cache/frames"`.
            run: Run id, `"runA"` or `"runB"`.
            cam: Camera id, `"cam0"` or `"cam5"`.
            crop_row_frac: Keep rows `[0, crop_row_frac * H)`; see config.yaml
                `frames.crop_row_frac`.
            max_cached (optional): Maximum number of frames kept in the LRU cache at once.
                Defaults to 512.

        Raises:
            ValueError: If no cached JPEGs are found for this run/camera.
        """
        self._dir = Path(cache_dir) / run / cam
        self._crop_row_frac = crop_row_frac
        self._max_cached = max_cached
        self._cache: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self.n_frames = len(list(self._dir.glob("*.jpg")))
        if self.n_frames == 0:
            raise ValueError(f"{self._dir}: no cached JPEGs found -- run `make decode` first")

    def _path(self, k: int) -> Path:
        """Path to frame `k`'s JPEG file.

        Args:
            k: Frame index.

        Returns:
            The JPEG path (may not exist).
        """
        return self._dir / f"{k:05d}.jpg"

    def get_rgb(self, k: int) -> NDArray[np.uint8]:
        """Read (or fetch from cache) one frame as cropped RGB.

        Args:
            k: Frame index.

        Returns:
            uint8[crop_h, W, 3] RGB image.

        Raises:
            ValueError: If the JPEG for frame `k` is missing or fails to decode.
        """
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        bgr = cv2.imread(str(self._path(k)))
        if bgr is None:
            raise ValueError(f"{self._path(k)}: failed to read (missing or corrupt JPEG)")
        crop_h = int(bgr.shape[0] * self._crop_row_frac)
        rgb = cv2.cvtColor(bgr[:crop_h], cv2.COLOR_BGR2RGB)
        self._cache[k] = rgb
        self._cache.move_to_end(k)
        if len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return rgb

    def get_gray(self, k: int) -> NDArray[np.uint8]:
        """Read (or fetch from cache) one frame as cropped grayscale.

        Args:
            k: Frame index.

        Returns:
            uint8[crop_h, W] grayscale image.
        """
        return cv2.cvtColor(self.get_rgb(k), cv2.COLOR_RGB2GRAY)


def motion_energy(store: FrameStore, thumb_w: int, thumb_h: int) -> NDArray[np.float64]:
    """Measure how much each frame changes from the one before it.

    Downsamples consecutive frames to `thumb_w` x `thumb_h` gray before differencing, so JPEG
    noise and minor camera shake average out and only genuine large-scale motion remains. Same
    array convention as `timestamps.interval_stats`' Δt array: one value per (k-1, k) transition.

    Args:
        store: Frame source for one run/camera.
        thumb_w: Downsample width, e.g. config.yaml `frames.motion_thumb_w`.
        thumb_h: Downsample height, e.g. config.yaml `frames.motion_thumb_h`.

    Returns:
        float64[store.n_frames - 1] mean absolute pixel difference per transition.
    """
    diffs = np.empty(store.n_frames - 1, dtype=np.float64)
    for k in range(store.n_frames - 1):
        gray0 = cv2.resize(store.get_gray(k), (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        gray1 = cv2.resize(store.get_gray(k + 1), (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(gray1, gray0)
        diff_mean = np.mean(diff)
        diffs[k] = diff_mean
    return diffs


def stationary_segments(
    motion_cam0: NDArray[np.float64],
    motion_cam5: NDArray[np.float64],
    threshold_frac: float,
    min_length: int,
) -> list[tuple[int, int]]:
    """Find frame ranges where the vehicle is stationary.

    A frame is "low motion" when its value is below `threshold_frac` of that camera's own
    median. A run counts as stationary only where *both* cameras agree (plan.md §4) -- a single
    camera showing low motion can just mean a low-texture scene, not a stop -- and only if the
    run is at least `min_length` frames long.

    Args:
        motion_cam0: float64[N] motion energy for cam0, e.g. from `motion_energy`.
        motion_cam5: float64[N] motion energy for cam5, same length as `motion_cam0`.
        threshold_frac: Fraction of each camera's median motion energy used as the low-motion
            cutoff, e.g. config.yaml `frames.stationary_threshold_frac` (0.15).
        min_length: Minimum run length in frames to count as stationary, e.g. config.yaml
            `frames.stationary_min_length` (10).

    Returns:
        List of `(start, end)` index pairs (inclusive start, exclusive end), one per stationary run.
    """
    cam0_threshold = np.median(motion_cam0) * threshold_frac
    cam5_threshold = np.median(motion_cam5) * threshold_frac
    is_low = (motion_cam0 < cam0_threshold) & (motion_cam5 < cam5_threshold)

    # pad with False on both sides so a run touching index 0 or the last index still gets a
    # proper open/close edge; diff on the int-cast array then marks +1 at each run start and
    # -1 at each run end (np.diff refuses to subtract bool arrays directly).
    padded = np.concatenate(([False], is_low, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True) if e - s >= min_length]
