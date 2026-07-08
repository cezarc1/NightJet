from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from imageio.typing import ArrayLike
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

    enhanced = app.enhance_demo_image(_color_image(), preserve_color=False)

    pixels = np.asarray(enhanced)
    assert enhanced.mode == "RGB"
    assert np.array_equal(pixels[..., 0], pixels[..., 1])
    assert np.array_equal(pixels[..., 1], pixels[..., 2])


def test_space_prediction_can_preserve_original_chroma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _load_space_app()
    checkpoint = _write_identity_checkpoint(tmp_path)
    monkeypatch.setattr(app, "_download_weights", lambda: checkpoint)
    app.clear_model_cache()

    enhanced = app.enhance_demo_image(_color_image(), preserve_color=True)

    pixels = np.asarray(enhanced)
    assert not np.array_equal(pixels[..., 0], pixels[..., 1])


def test_space_prediction_preserves_large_input_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _load_space_app()
    checkpoint = _write_identity_checkpoint(tmp_path)
    monkeypatch.setattr(app, "_download_weights", lambda: checkpoint)
    app.clear_model_cache()
    image = Image.new("RGB", (2000, 1000), color=(32, 64, 96))

    enhanced = app.enhance_demo_image(image, preserve_color=False, max_long_edge=1000)

    assert enhanced.size == (2000, 1000)


def test_space_default_example_image_exists() -> None:
    app = _load_space_app()

    assert app.DEFAULT_EXAMPLE_IMAGE.exists()
    with Image.open(app.DEFAULT_EXAMPLE_IMAGE) as image:
        assert image.size == (1280, 720)


def test_space_video_conversion_writes_enhanced_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _load_space_app()
    checkpoint = _write_identity_checkpoint(tmp_path)
    monkeypatch.setattr(app, "_download_weights", lambda: checkpoint)
    app.clear_model_cache()
    input_video = tmp_path / "input.mp4"
    frames: list[ArrayLike] = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (20, 80, 140)]
    imageio.mimsave(input_video, frames, fps=3, macro_block_size=1)

    output_path = Path(
        app.enhance_demo_video(
            str(input_video),
            preserve_color=False,
            max_seconds=1.0,
            max_long_edge=8,
        )
    )

    assert output_path.exists()
    output_frames = imageio.mimread(output_path)
    assert len(output_frames) == 3
    assert np.asarray(output_frames[0]).shape[:2] == (8, 8)


def test_space_video_conversion_preserves_frame_size_after_bounded_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _load_space_app()
    checkpoint = _write_identity_checkpoint(tmp_path)
    monkeypatch.setattr(app, "_download_weights", lambda: checkpoint)
    app.clear_model_cache()
    input_video = tmp_path / "input.mp4"
    frames: list[ArrayLike] = [
        np.full((90, 160, 3), value, dtype=np.uint8) for value in (20, 80, 140)
    ]
    imageio.mimsave(input_video, frames, fps=3, macro_block_size=1)

    output_path = Path(
        app.enhance_demo_video(
            str(input_video),
            preserve_color=False,
            max_seconds=1.0,
            max_long_edge=80,
        )
    )

    output_frames = imageio.mimread(output_path)
    height, width = np.asarray(output_frames[0]).shape[:2]
    assert (height, width) == (90, 160)
    assert height % 2 == 0
    assert width % 2 == 0


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
