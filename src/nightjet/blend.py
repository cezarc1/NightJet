from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nightjet.data import BundleManifest, package_arrays


@dataclass(frozen=True)
class BlendComponent:
    name: str
    weight: float
    bundle_dir: Path


def blend_bundles(components: list[BlendComponent], *, output_dir: Path) -> BundleManifest:
    if len(components) < 2:
        raise ValueError("at least two blend components are required")
    total_weight = sum(component.weight for component in components)
    if total_weight <= 0:
        raise ValueError("blend weights must sum to a positive value")
    normalized = [
        BlendComponent(
            name=component.name,
            weight=component.weight / total_weight,
            bundle_dir=component.bundle_dir,
        )
        for component in components
    ]
    raw_arrays = [np.load(component.bundle_dir / "input_luma.npy") for component in normalized]
    target_arrays = [np.load(component.bundle_dir / "target_luma.npy") for component in normalized]
    frame_count = min(array.shape[0] for array in raw_arrays + target_arrays)
    shape = raw_arrays[0].shape[1:]
    if any(array.shape[1:] != shape for array in raw_arrays + target_arrays):
        raise ValueError("all blend components must have matching frame height/width")
    raw = raw_arrays[0][:frame_count]
    target = np.zeros_like(target_arrays[0][:frame_count], dtype=np.float32)
    for component, target_array in zip(normalized, target_arrays, strict=True):
        target += component.weight * target_array[:frame_count].astype(np.float32)
    target = np.clip(target, 0.0, 1.0)
    bundle = package_arrays(raw.astype(np.float32), target.astype(np.float32), output_dir)
    metadata_path = bundle.bundle_dir / "bundle_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "teacher_model": "+".join(
                f"{component.name}-{component.weight:.3f}" for component in normalized
            ),
            "components": [
                {
                    "name": component.name,
                    "weight": component.weight,
                    "bundle_dir": str(component.bundle_dir),
                }
                for component in normalized
            ],
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle


def parse_blend_component(value: str) -> BlendComponent:
    try:
        name, weight, bundle_dir = value.split("=", maxsplit=2)
    except ValueError as exc:
        raise ValueError("component must use NAME=WEIGHT=BUNDLE_DIR") from exc
    return BlendComponent(name=name, weight=float(weight), bundle_dir=Path(bundle_dir))
