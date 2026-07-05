import json
from pathlib import Path

import numpy as np
import torch

from nightjet.config import load_config
from nightjet.training import train


def test_cpu_smoke_training_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    inputs = np.random.default_rng(0).random((6, 16, 16), dtype=np.float32)
    targets = np.clip(inputs * 1.15, 0.0, 1.0)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.save(bundle / "input_luma.npy", inputs)
    np.save(bundle / "target_luma.npy", targets)

    cfg = load_config(Path("configs/nightjet_edge_v1_reco_s2_c16_f3.yaml"))
    cfg.data.bundle_uri = str(bundle)
    cfg.data.crop_size = None
    cfg.training.batch_size = 2
    cfg.training.max_steps = 2
    cfg.training.amp = False

    result = train(cfg, output_dir=tmp_path / "run", device="cpu")

    assert result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["final_step"] == 2
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert {"model_state_dict", "optimizer_state_dict", "config", "step"} <= set(checkpoint)


def test_cpu_smoke_training_can_write_ema_checkpoint_state(tmp_path: Path) -> None:
    inputs = np.random.default_rng(1).random((6, 16, 16), dtype=np.float32)
    targets = np.clip(inputs * 1.2, 0.0, 1.0)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.save(bundle / "input_luma.npy", inputs)
    np.save(bundle / "target_luma.npy", targets)

    cfg = load_config(Path("configs/candidates/edge_v1_reco_s2_c16_f3.yaml"))
    cfg.data.bundle_uri = str(bundle)
    cfg.data.crop_size = None
    cfg.data.split_uri = None
    cfg.training.batch_size = 2
    cfg.training.max_steps = 2
    cfg.training.warmup_steps = 1
    cfg.training.amp = False

    result = train(cfg, output_dir=tmp_path / "run", device="cpu")

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["ema_model_state_dict"] is not None


def test_cpu_training_uses_configured_split_indices(tmp_path: Path) -> None:
    inputs = np.random.default_rng(2).random((8, 16, 16), dtype=np.float32)
    targets = np.clip(inputs * 1.2, 0.0, 1.0)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.save(bundle / "input_luma.npy", inputs)
    np.save(bundle / "target_luma.npy", targets)
    split_path = bundle / "splits.json"
    split_path.write_text(
        json.dumps({"train": [2, 3, 4], "val": [5], "test": [6, 7]}), encoding="utf-8"
    )

    cfg = load_config(Path("configs/nightjet_edge_v1_reco_s2_c16_f3.yaml"))
    cfg.data.bundle_uri = str(bundle)
    cfg.data.split_uri = str(split_path)
    cfg.data.split_name = "train"
    cfg.data.crop_size = None
    cfg.training.batch_size = 2
    cfg.training.max_steps = 2
    cfg.training.amp = False

    result = train(cfg, output_dir=tmp_path / "run", device="cpu")

    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["training_frames"] == 3
