from __future__ import annotations

import math

import numpy as np


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def edge_mean(frames: np.ndarray) -> float:
    dx = np.abs(frames[..., :, 1:] - frames[..., :, :-1])
    dy = np.abs(frames[..., 1:, :] - frames[..., :-1, :])
    return float(dx.mean() + dy.mean())


def temporal_diff_mean(frames: np.ndarray) -> float:
    if frames.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(frames[1:] - frames[:-1])))


def flat_region_noise(frames: np.ndarray, raw: np.ndarray) -> float:
    threshold = np.quantile(edge_map(raw), 0.25)
    mask = edge_map(raw) <= threshold
    if not np.any(mask):
        return 0.0
    per_frame = []
    for frame in frames:
        values = frame[mask]
        per_frame.append(float(values.std()) if values.size else 0.0)
    return float(np.mean(per_frame))


def edge_map(frames: np.ndarray) -> np.ndarray:
    mean_frame = frames.mean(axis=0) if frames.ndim == 3 else frames
    dy = np.zeros_like(mean_frame, dtype=np.float32)
    dx = np.zeros_like(mean_frame, dtype=np.float32)
    dx[:, 1:] = np.abs(mean_frame[:, 1:] - mean_frame[:, :-1])
    dy[1:, :] = np.abs(mean_frame[1:, :] - mean_frame[:-1, :])
    return dx + dy


def clipping_rate(frames: np.ndarray) -> float:
    return float(np.mean((frames <= 0.001) | (frames >= 0.999)))
