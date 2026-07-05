from pathlib import Path

from nightjet.config import load_config


def test_loads_training_config() -> None:
    cfg = load_config(Path("configs/nightjet_edge_v1_reco_s2_c16_f3.yaml"))

    assert cfg.run.name == "nightjet-edge-v1-reco-s2-c16-f3"
    assert cfg.model.input_frames == 3
    assert cfg.model.base_channels == 16
    assert cfg.data.resolution == (720, 1280)
    assert cfg.training.max_steps == 1500


def test_candidate_training_fields_are_supported() -> None:
    cfg = load_config(Path("configs/candidates/edge_v1_reco_s2_c16_f3.yaml"))

    assert cfg.training.scheduler == "cosine"
    assert cfg.training.warmup_steps == 500
    assert cfg.training.ema_decay == 0.999
    assert cfg.loss.smoothness_weight == 0.05
