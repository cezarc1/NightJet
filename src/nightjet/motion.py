from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

DEFAULT_MOTION_BUDGET = 0.045

type F32Array = NDArray[np.float32]
type U8Array = NDArray[np.uint8]


def motion_window_size(motions: Sequence[float], budget: float) -> int:
    """Number of trailing window frames whose cumulative inter-frame motion fits."""
    keep = 1
    cumulative = 0.0
    for motion in reversed(motions):
        cumulative += motion
        if cumulative > budget:
            break
        keep += 1
    return keep


def block_mean_luma(luma: np.ndarray, block: int = 8) -> F32Array:
    if luma.ndim != 2:
        raise ValueError(f"expected 2D luma frame, got shape {luma.shape}")
    array = luma.astype(np.float32, copy=False)
    blocks_h, blocks_w = array.shape[0] // block, array.shape[1] // block
    if blocks_h == 0 or blocks_w == 0:
        return array
    trimmed = array[: blocks_h * block, : blocks_w * block]
    return trimmed.reshape(blocks_h, block, blocks_w, block).mean(axis=(1, 3), dtype=np.float32)


def block_mean_luma_u8(luma: U8Array, block: int = 8) -> F32Array:
    if luma.ndim != 2:
        raise ValueError(f"expected 2D luma frame, got shape {luma.shape}")
    if luma.dtype != np.dtype(np.uint8):
        raise ValueError(f"expected luma dtype uint8, got {luma.dtype}")
    blocks_h, blocks_w = luma.shape[0] // block, luma.shape[1] // block
    scale = np.float32(1.0 / 255.0)
    if blocks_h == 0 or blocks_w == 0:
        return luma.astype(np.float32) * scale
    trimmed = luma[: blocks_h * block, : blocks_w * block].astype(np.float32)
    means = trimmed.reshape(blocks_h, block, blocks_w, block).mean(axis=(1, 3), dtype=np.float32)
    return means * scale


def mean_abs_luma_delta(current: np.ndarray, previous: np.ndarray) -> float:
    return float(np.mean(np.abs(current - previous), dtype=np.float32))
