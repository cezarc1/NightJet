from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from PIL import Image

from nightjet.config import ModelConfig
from nightjet.models import NightJetEdgeV1


def test_space_prediction_defaults_to_grayscale_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _load_space_app()
    checkpoint = _write_identity_checkpoint(tmp_path)
    monkeypatch.setattr(app, "_download_weights", lambda: checkpoint)
    app.clear_model_cache()

    enhanced, comparison = app.enhance_demo_image(_color_image(), preserve_color=False)

    pixels = np.asarray(enhanced)
    assert enhanced.mode == "RGB"
    assert np.array_equal(pixels[..., 0], pixels[..., 1])
    assert np.array_equal(pixels[..., 1], pixels[..., 2])
    assert comparison.size == (enhanced.size[0] * 2, enhanced.size[1])


def test_space_prediction_can_preserve_original_chroma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _load_space_app()
    checkpoint = _write_identity_checkpoint(tmp_path)
    monkeypatch.setattr(app, "_download_weights", lambda: checkpoint)
    app.clear_model_cache()

    enhanced, _comparison = app.enhance_demo_image(_color_image(), preserve_color=True)

    pixels = np.asarray(enhanced)
    assert not np.array_equal(pixels[..., 0], pixels[..., 1])


def test_space_prediction_resizes_large_inputs_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _load_space_app()
    checkpoint = _write_identity_checkpoint(tmp_path)
    monkeypatch.setattr(app, "_download_weights", lambda: checkpoint)
    app.clear_model_cache()
    image = Image.new("RGB", (2000, 1000), color=(32, 64, 96))

    enhanced, comparison = app.enhance_demo_image(image, preserve_color=False, max_long_edge=1000)

    assert enhanced.size == (1000, 500)
    assert comparison.size == (2000, 500)


def _load_space_app() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "huggingface" / "space" / "app.py"
    spec = importlib.util.spec_from_file_location("nightjet_hf_space_app", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_identity_checkpoint(tmp_path: Path) -> Path:
    model_config = ModelConfig(
        name="hf-space-test",
        input_frames=5,
        base_channels=8,
        detail_channels=4,
        trunk_blocks=1,
        trunk_scale=2,
        residual_scale=0.45,
    )
    model = NightJetEdgeV1(model_config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = tmp_path / "nightjet-edge-v1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.model_dump(mode="json"),
        },
        checkpoint,
    )
    return checkpoint


def _color_image() -> Image.Image:
    pixels = np.zeros((8, 16, 3), dtype=np.uint8)
    pixels[:, :8] = np.array([180, 30, 45], dtype=np.uint8)
    pixels[:, 8:] = np.array([20, 160, 90], dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")
