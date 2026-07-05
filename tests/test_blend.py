import json
from pathlib import Path

import numpy as np

from nightjet.blend import BlendComponent, blend_bundles
from nightjet.data import package_arrays


def test_blend_bundles_uses_weighted_targets_and_min_frame_count(tmp_path: Path) -> None:
    raw = np.zeros((4, 4, 4), dtype=np.float32) + 0.1
    first_target = np.zeros((4, 4, 4), dtype=np.float32) + 0.8
    second_target = np.zeros((3, 4, 4), dtype=np.float32) + 0.2
    first = package_arrays(raw, first_target, tmp_path / "first")
    second = package_arrays(raw[:3], second_target, tmp_path / "second")

    bundle = blend_bundles(
        [
            BlendComponent(name="first", weight=0.75, bundle_dir=first.bundle_dir),
            BlendComponent(name="second", weight=0.25, bundle_dir=second.bundle_dir),
        ],
        output_dir=tmp_path / "blend",
    )

    target = np.load(bundle.target_luma)
    assert target.shape == (3, 4, 4)
    np.testing.assert_allclose(target, np.zeros((3, 4, 4), dtype=np.float32) + 0.65)
    metadata = json.loads((bundle.bundle_dir / "bundle_manifest.json").read_text())
    assert metadata["teacher_model"] == "first-0.750+second-0.250"
