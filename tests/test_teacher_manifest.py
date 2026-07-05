import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nightjet.teacher_manifest import bundle_from_teacher_manifest, load_teacher_manifest


def _write_luma_video(path: Path, frames: np.ndarray) -> None:
    rgb = np.repeat(np.clip(frames * 255.0, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)
    iio.imwrite(path, rgb, fps=30, macro_block_size=1)


def test_bundle_from_teacher_manifest_writes_luma_arrays_and_manifest(tmp_path: Path) -> None:
    input_frames = np.linspace(0.0, 0.4, num=4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    target_frames = np.clip(input_frames * 1.8, 0.0, 1.0)
    _write_luma_video(tmp_path / "input.mp4", input_frames)
    _write_luma_video(tmp_path / "target.mp4", target_frames)
    manifest_path = tmp_path / "teacher_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "teacher_model": "synthetic-detail-teacher",
                "source_clip": "source.mp4",
                "input_luma_clip": "input.mp4",
                "target_luma_clip": "target.mp4",
                "frame_height": 8,
                "frame_width": 8,
                "frames": 4,
            }
        ),
        encoding="utf-8",
    )

    manifest = load_teacher_manifest(manifest_path)
    bundle = bundle_from_teacher_manifest(manifest_path, output_dir=tmp_path / "bundle")

    assert manifest.teacher_model == "synthetic-detail-teacher"
    assert bundle.frame_count == 4
    assert bundle.height == 8
    assert bundle.width == 8
    assert np.load(bundle.input_luma).shape == (4, 8, 8)
    assert np.load(bundle.target_luma).shape == (4, 8, 8)
    metadata = json.loads((bundle.bundle_dir / "bundle_manifest.json").read_text())
    assert metadata["teacher_model"] == "synthetic-detail-teacher"
    assert metadata["source_manifest"] == str(manifest_path)
