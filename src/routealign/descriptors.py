"""Per-frame descriptors: SeqSLAM-style thumbnails (R1) and DINOv2 embeddings (R2). R2 is a
committed part of the default pipeline, fused with R1, not an optional extra.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from numpy.typing import NDArray

from .frames import FrameStore

log = logging.getLogger(__name__)


def seqslam_descriptors(
    store: FrameStore,
    indices: NDArray[np.int64],
    thumb_w: int,
    thumb_h: int,
    patch: int,
) -> NDArray[np.float32]:
    """Compute SeqSLAM-style descriptors: downsample, patch-normalise, flatten, L2-normalise.

    Divides each downsampled frame into non-overlapping `patch` x `patch` blocks and normalises
    each block by its own local (mean, std) -- this is what makes the descriptor robust to global
    illumination change between runs (a block that is uniformly brighter still normalises to a
    similar distribution), unlike normalising by one whole-image mean/std.

    Args:
        store: Frame source to read from.
        indices: int64[N] frame indices to describe.
        thumb_w: Downsample width, e.g. config.yaml `descriptors.seqslam.thumb_w` (64). Must be
            evenly divisible by `patch`.
        thumb_h: Downsample height, e.g. config.yaml `descriptors.seqslam.thumb_h` (40). Must be
            evenly divisible by `patch`.
        patch: Non-overlapping patch side in pixels, e.g. config.yaml
            `descriptors.seqslam.patch` (8).

    Returns:
        float32[N, thumb_h * thumb_w] L2-normalised descriptors, one row per index.
    """
    out = np.empty((indices.size, thumb_h * thumb_w), dtype=np.float32)
    for row, k in enumerate(indices):
        img = cv2.resize(
            store.get_gray(int(k)), (thumb_w, thumb_h), interpolation=cv2.INTER_AREA
        ).astype(np.float32)

        # Non-overlapping patches via reshape: after reshape(h//patch, patch, w//patch, patch),
        # axes 0 and 2 index the patch grid position, axes 1 and 3 are the pixels within one
        # patch -- reducing over axes (1, 3) gives one (mean, std) per patch, vectorised.
        blocks = img.reshape(thumb_h // patch, patch, thumb_w // patch, patch)
        means = blocks.mean(axis=(1, 3), keepdims=True)
        stds = blocks.std(axis=(1, 3), keepdims=True)
        normalised = ((blocks - means) / (stds + 1)).reshape(thumb_h, thumb_w)
        normalised = np.clip(normalised, -3, 3)

        flat = normalised.flatten()
        norm = np.linalg.norm(flat)
        out[row] = flat / norm if norm > 0 else flat
    return out


def _gem_pool(patch_tokens: torch.Tensor, p: float, eps: float = 1e-6) -> torch.Tensor:
    """Generalised-mean pool over the patch dimension.

    Args:
        patch_tokens: float32[B, num_patches, D] patch tokens.
        p: Power parameter (1 = average pooling; higher values move it closer to max pooling).
        eps (optional): Floor applied before raising to the power `p`. Patch tokens (post
            LayerNorm) can be negative, and a negative base raised to a fractional exponent
            (`1/p`) is not defined in real arithmetic -- without this clamp, a negative mean
            silently produces `nan` instead of raising. Defaults to `1e-6`.

    Returns:
        float32[B, D] pooled summary.
    """
    return patch_tokens.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


def dino_descriptors(
    store: FrameStore,
    indices: NDArray[np.int64],
    model_name: str = "dinov2_vits14",
    device: str = "mps",
    batch_size: int = 32,
    gem_p: float = 3.0,
) -> NDArray[np.float32]:
    """Compute DINOv2 ViT-S/14 embeddings (CLS token concatenated with a GeM-pooled patch summary).

    Args:
        store: Frame source to read from.
        indices: int64[N] frame indices to describe.
        model_name (optional): Name of the DINOv2 model to use. Defaults to `"dinov2_vits14"`.
        device (optional): Torch device to use if available, e.g. `"mps"`; falls back to `"cpu"`
            automatically when it is not. Defaults to `"mps"`.
        batch_size (optional): Frames per forward pass, e.g. config.yaml
            `descriptors.dinov2.batch_size`. Defaults to `32`.
        gem_p (optional): GeM pooling power over the patch tokens. Defaults to `3.0`.

    Returns:
        float32[N, 768] L2-normalised descriptors: `L2(concat(L2(CLS), L2(GeM(patch tokens))))`.
    """
    dinov2_model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    device = torch.device(device) if torch.backends.mps.is_available() else torch.device("cpu")
    dinov2_model.to(device)
    dinov2_model.eval()
    transform = T.Compose(
        [
            T.ToTensor(),  # must run before Resize -- Resize errors on a raw numpy array
            T.Resize((280, 448)),  # (height, width); matches the ~1.6:1 crop aspect, not a square
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    out = np.empty((indices.size, 768), dtype=np.float32)
    for batch_start in range(0, indices.size, batch_size):
        batch_idx = indices[batch_start : batch_start + batch_size]
        tensors = [transform(store.get_rgb(int(k))) for k in batch_idx]
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            features = dinov2_model.forward_features(batch)
            cls = F.normalize(features["x_norm_clstoken"], dim=-1)
            gem = F.normalize(_gem_pool(features["x_norm_patchtokens"], p=gem_p), dim=-1)
            descriptor = F.normalize(torch.cat([cls, gem], dim=-1), dim=-1)

        out[batch_start : batch_start + len(batch_idx)] = descriptor.cpu().numpy()
    return out


def cached(
    fn: Callable[[], NDArray],
    key: str,
    cache_dir: str | Path,
    extra: dict[str, object] | None = None,
    force: bool = False,
) -> NDArray:
    """Compute-or-load an array, writing a `.npy` plus a provenance `.json` sidecar.

    `cached` cannot see what `fn` (a zero-argument closure) depends on, so `key` must already
    encode every parameter that changes the result, e.g. `"seqslam_runA_cam0_w64_h40_p8"` --
    a stale key with unchanged parameters is not detected. `extra` is where the caller records
    that same provenance (source file SHA-256 prefixes, the parameters `fn` was built with,
    package versions) in machine-readable form for the sidecar (CLAUDE.md §4).

    Args:
        fn: Zero-argument callable that computes the array on a cache miss.
        key: Cache key; determines the output filename. Must already encode every parameter
            that affects the result.
        cache_dir: Directory the `.npy`/`.json` pair is stored in.
        extra (optional): Extra provenance fields to record in the sidecar. Defaults to `None`
            (no extra fields).
        force (optional): Recompute even if a cache entry exists. Defaults to `False`.

    Returns:
        The array, either loaded from cache or freshly computed.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path = cache_dir / f"{key}.npy"
    json_path = cache_dir / f"{key}.json"

    if not force and npy_path.exists():
        log.info("cache hit: %s", npy_path)
        return np.load(npy_path)

    log.info("cache miss: %s (computing)", npy_path)
    arr = fn()
    np.save(npy_path, arr)
    sidecar = {
        "key": key,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numpy_version": np.__version__,
        **(extra or {}),
    }
    json_path.write_text(json.dumps(sidecar, indent=2, default=str))
    return arr
