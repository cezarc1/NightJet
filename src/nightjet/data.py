from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BundleManifest:
    bundle_dir: Path
    input_luma: Path
    target_luma: Path
    frame_count: int
    height: int
    width: int
    input_sha256: str
    target_sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "bundle_dir": str(self.bundle_dir),
            "input_luma": str(self.input_luma),
            "target_luma": str(self.target_luma),
            "frame_count": self.frame_count,
            "height": self.height,
            "width": self.width,
            "input_sha256": self.input_sha256,
            "target_sha256": self.target_sha256,
        }


class TemporalLumaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        input_frames: int,
        crop_size: int | None,
        target_indices: list[int] | None = None,
    ) -> None:
        if inputs.ndim != 3 or targets.ndim != 3:
            raise ValueError("inputs and targets must be TxHxW luma arrays")
        if inputs.shape != targets.shape:
            raise ValueError("inputs and targets must have matching TxHxW shapes")
        if input_frames not in {1, 3, 5}:
            raise ValueError("input_frames must be one of 1, 3, or 5")
        if inputs.shape[0] < input_frames:
            raise ValueError(f"need at least {input_frames} frames, got {inputs.shape[0]}")
        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size must be positive when set")
        self.inputs = np.ascontiguousarray(inputs.astype(np.float32, copy=False))
        self.targets = np.ascontiguousarray(targets.astype(np.float32, copy=False))
        self.input_frames = input_frames
        self.crop_size = crop_size
        self.target_indices = (
            list(target_indices)
            if target_indices is not None
            else list(range(input_frames - 1, inputs.shape[0]))
        )
        if not self.target_indices:
            raise ValueError("target_indices must not be empty")
        if min(self.target_indices) < 0 or max(self.target_indices) >= inputs.shape[0]:
            raise ValueError("target_indices must be inside frame range")

    @classmethod
    def from_arrays(
        cls,
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        input_frames: int,
        crop_size: int | None,
        target_indices: list[int] | None = None,
    ) -> TemporalLumaDataset:
        return cls(
            inputs,
            targets,
            input_frames=input_frames,
            crop_size=crop_size,
            target_indices=target_indices,
        )

    @classmethod
    def from_bundle(
        cls,
        bundle_uri: str,
        *,
        input_frames: int,
        crop_size: int | None,
        target_indices: list[int] | None = None,
    ) -> TemporalLumaDataset:
        bundle_dir = Path(bundle_uri)
        inputs = np.load(bundle_dir / "input_luma.npy")
        targets = np.load(bundle_dir / "target_luma.npy")
        return cls.from_arrays(
            inputs,
            targets,
            input_frames=input_frames,
            crop_size=crop_size,
            target_indices=target_indices,
        )

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        target_index = self.target_indices[index]
        start = target_index - self.input_frames + 1
        indices = [
            min(max(start + offset, 0), self.inputs.shape[0] - 1)
            for offset in range(self.input_frames)
        ]
        window = np.stack([self.inputs[frame_index] for frame_index in indices], axis=0)
        target = self.targets[target_index]
        window, target = self._crop(window, target, index=index)
        return torch.from_numpy(window.copy()), torch.from_numpy(target[None, :, :].copy())

    def _crop(
        self,
        window: np.ndarray,
        target: np.ndarray,
        *,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.crop_size is None:
            return window, target
        height, width = target.shape
        crop = min(self.crop_size, height, width)
        max_y = height - crop
        max_x = width - crop
        # Stable pseudo-random crop per item keeps tests reproducible while still sampling space.
        y0 = 0 if max_y == 0 else (index * 37) % (max_y + 1)
        x0 = 0 if max_x == 0 else (index * 53) % (max_x + 1)
        return window[:, y0 : y0 + crop, x0 : x0 + crop], target[y0 : y0 + crop, x0 : x0 + crop]


def package_arrays(
    input_luma: np.ndarray, target_luma: np.ndarray, output_dir: Path
) -> BundleManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input_luma.npy"
    target_path = output_dir / "target_luma.npy"
    np.save(input_path, input_luma.astype(np.float32, copy=False))
    np.save(target_path, target_luma.astype(np.float32, copy=False))
    manifest = BundleManifest(
        bundle_dir=output_dir,
        input_luma=input_path,
        target_luma=target_path,
        frame_count=int(input_luma.shape[0]),
        height=int(input_luma.shape[1]),
        width=int(input_luma.shape[2]),
        input_sha256=sha256_file(input_path),
        target_sha256=sha256_file(target_path),
    )
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
