from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from nightjet.data import BundleManifest, package_arrays, sha256_file
from nightjet.teacher_manifest import read_luma_video


def bundle_from_video_frames(
    *,
    input_video: Path,
    target_frame_dir: Path,
    output_dir: Path,
    target_glob: str = "*.png",
    max_frames: int | None = None,
) -> BundleManifest:
    target_paths = sorted_frame_paths(target_frame_dir, glob=target_glob)
    if not target_paths:
        raise ValueError(f"no target frames match {target_glob} in {target_frame_dir}")
    frame_count = (
        min(len(target_paths), max_frames) if max_frames is not None else len(target_paths)
    )
    input_luma = read_luma_video(input_video, max_frames=frame_count)
    target_luma = read_luma_frames(target_paths[:frame_count])
    frame_count = min(input_luma.shape[0], target_luma.shape[0])
    if frame_count == 0:
        raise ValueError("input video and target frame directory produced no frames")
    bundle = package_arrays(input_luma[:frame_count], target_luma[:frame_count], output_dir)
    metadata_path = bundle.bundle_dir / "bundle_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "input_video": str(input_video),
            "input_video_sha256": sha256_file(input_video),
            "target_frame_dir": str(target_frame_dir),
            "target_glob": target_glob,
            "target_frame_count": frame_count,
            "target_frames": [str(path) for path in target_paths[:frame_count]],
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle


def sorted_frame_paths(frame_dir: Path, *, glob: str) -> list[Path]:
    return sorted(frame_dir.glob(glob), key=_natural_path_key)


def read_luma_frames(paths: list[Path]) -> np.ndarray:
    frames: list[np.ndarray] = []
    for path in paths:
        array = np.asarray(iio.imread(path))
        if array.ndim == 2:
            luma = array.astype(np.float32)
        elif array.ndim == 3:
            rgb = array[..., :3].astype(np.float32)
            luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        else:
            raise ValueError(f"unsupported frame shape in {path}: {array.shape}")
        frames.append(np.ascontiguousarray(luma / 255.0))
    if not frames:
        raise ValueError("no target frames provided")
    return np.stack(frames, axis=0).astype(np.float32)


def _natural_path_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]
