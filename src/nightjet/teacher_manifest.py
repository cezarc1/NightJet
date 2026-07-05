from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from nightjet.data import BundleManifest, package_arrays, sha256_file


@dataclass(frozen=True)
class TeacherManifest:
    path: Path
    teacher_model: str
    source_clip: Path
    input_luma_clip: Path
    target_luma_clip: Path
    frame_height: int
    frame_width: int
    frames: int
    payload: dict[str, Any]


def load_teacher_manifest(path: Path) -> TeacherManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("teacher manifest must be a JSON object")
    required = ["teacher_model", "input_luma_clip", "target_luma_clip", "frames"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"teacher manifest missing required fields: {', '.join(missing)}")
    return TeacherManifest(
        path=path,
        teacher_model=str(payload["teacher_model"]),
        source_clip=_resolve_manifest_path(path, str(payload.get("source_clip", ""))),
        input_luma_clip=_resolve_manifest_path(path, str(payload["input_luma_clip"])),
        target_luma_clip=_resolve_manifest_path(path, str(payload["target_luma_clip"])),
        frame_height=int(payload.get("frame_height", 0)),
        frame_width=int(payload.get("frame_width", 0)),
        frames=int(payload["frames"]),
        payload=payload,
    )


def bundle_from_teacher_manifest(manifest_path: Path, *, output_dir: Path) -> BundleManifest:
    teacher = load_teacher_manifest(manifest_path)
    input_luma = read_luma_video(teacher.input_luma_clip, max_frames=teacher.frames)
    target_luma = read_luma_video(teacher.target_luma_clip, max_frames=teacher.frames)
    if input_luma.shape != target_luma.shape:
        raise ValueError(
            "teacher input and target luma shapes do not match: "
            f"{input_luma.shape} vs {target_luma.shape}"
        )
    bundle = package_arrays(input_luma, target_luma, output_dir)
    metadata_path = bundle.bundle_dir / "bundle_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "source_manifest": str(manifest_path),
            "teacher_model": teacher.teacher_model,
            "source_clip": str(teacher.source_clip),
            "input_luma_clip": str(teacher.input_luma_clip),
            "target_luma_clip": str(teacher.target_luma_clip),
            "source_manifest_sha256": sha256_file(manifest_path),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle


def read_luma_video(path: Path, *, max_frames: int | None = None) -> np.ndarray:
    frames: list[np.ndarray] = []
    for index, frame in enumerate(iio.imiter(path)):
        if max_frames is not None and index >= max_frames:
            break
        array = np.asarray(frame)
        if array.ndim == 2:
            luma = array
        elif array.ndim == 3:
            rgb = array[..., :3].astype(np.float32)
            luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        else:
            raise ValueError(f"unsupported frame shape in {path}: {array.shape}")
        frames.append(np.ascontiguousarray(luma.astype(np.float32) / 255.0))
    if not frames:
        raise ValueError(f"no frames read from {path}")
    return np.stack(frames, axis=0).astype(np.float32)


def _resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    if not raw_path:
        return Path()
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [manifest_path.parent / path]
    candidates.extend(parent / path for parent in [manifest_path.parent, *manifest_path.parents])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return manifest_path.parent / path
