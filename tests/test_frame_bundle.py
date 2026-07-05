import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nightjet.frame_bundle import bundle_from_video_frames, sorted_frame_paths


def test_sorted_frame_paths_uses_natural_numeric_order(tmp_path: Path) -> None:
    for name in ["10_normal.png", "2_normal.png", "1_normal.png"]:
        (tmp_path / name).write_text("x", encoding="utf-8")

    paths = sorted_frame_paths(tmp_path, glob="*_normal.png")

    assert [path.name for path in paths] == ["1_normal.png", "2_normal.png", "10_normal.png"]


def test_bundle_from_video_frames_packages_matching_luma_arrays(tmp_path: Path) -> None:
    video_path = tmp_path / "input.mp4"
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    input_frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    input_frames[0, :, :, :] = 10
    input_frames[1, :, :, :] = 20
    input_frames[2, :, :, :] = 30
    iio.imwrite(video_path, input_frames, fps=3, macro_block_size=1)
    for index, value in [(1, 40), (2, 80), (10, 120)]:
        iio.imwrite(frame_dir / f"{index}_normal.png", np.full((8, 8, 3), value, dtype=np.uint8))

    bundle = bundle_from_video_frames(
        input_video=video_path,
        target_frame_dir=frame_dir,
        output_dir=tmp_path / "bundle",
        target_glob="*_normal.png",
    )

    manifest = json.loads((bundle.bundle_dir / "bundle_manifest.json").read_text())
    inputs = np.load(bundle.input_luma)
    targets = np.load(bundle.target_luma)
    assert bundle.frame_count == 3
    assert inputs.shape == targets.shape == (3, 8, 8)
    assert targets[0].mean() < targets[1].mean() < targets[2].mean()
    assert manifest["input_video"] == str(video_path)
    assert manifest["target_frame_count"] == 3
