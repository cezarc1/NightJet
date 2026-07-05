import numpy as np

from nightjet.data import TemporalLumaDataset, package_arrays


def test_temporal_dataset_uses_causal_three_frame_windows() -> None:
    inputs = np.arange(5 * 4 * 6, dtype=np.float32).reshape(5, 4, 6) / 255.0
    targets = inputs + 0.25

    dataset = TemporalLumaDataset.from_arrays(inputs, targets, input_frames=3, crop_size=None)
    x, y = dataset[0]

    assert len(dataset) == 3
    assert x.shape == (3, 4, 6)
    assert y.shape == (1, 4, 6)
    np.testing.assert_allclose(x.numpy(), inputs[0:3])
    np.testing.assert_allclose(y.numpy()[0], targets[2])


def test_temporal_dataset_can_subset_by_target_indices() -> None:
    inputs = np.arange(6 * 4 * 6, dtype=np.float32).reshape(6, 4, 6) / 255.0
    targets = inputs + 0.25

    dataset = TemporalLumaDataset.from_arrays(
        inputs,
        targets,
        input_frames=3,
        crop_size=None,
        target_indices=[3, 5],
    )
    x, y = dataset[0]

    assert len(dataset) == 2
    np.testing.assert_allclose(x.numpy(), inputs[1:4])
    np.testing.assert_allclose(y.numpy()[0], targets[3])


def test_package_arrays_writes_bundle_manifest(tmp_path) -> None:
    inputs = np.zeros((3, 4, 6), dtype=np.float32)
    targets = np.ones((3, 4, 6), dtype=np.float32)

    manifest = package_arrays(inputs, targets, tmp_path / "bundle")

    assert manifest.input_luma.exists()
    assert manifest.target_luma.exists()
    assert (manifest.bundle_dir / "bundle_manifest.json").exists()
    assert manifest.frame_count == 3
    assert len(manifest.input_sha256) == 64
