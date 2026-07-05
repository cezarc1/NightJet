from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from imageio.typing import ArrayLike
from PIL import Image

from nightjet.config import ModelConfig
from nightjet.inference import NightJetEnhancer
from nightjet.models import NightJetEdgeV1


def test_enhance_window_pads_short_static_window(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=5)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")
    window = np.stack(
        [
            np.full((6, 8), 0.10, dtype=np.float32),
            np.full((6, 8), 0.40, dtype=np.float32),
        ],
        axis=0,
    )

    enhanced = enhancer.enhance_window(window)

    assert enhanced.shape == (6, 8)
    assert np.allclose(enhanced, 0.40, atol=1e-6)


def test_enhance_image_defaults_to_grayscale_rgb(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    input_path = _write_color_image(tmp_path)
    output_path = tmp_path / "enhanced.png"
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    result = enhancer.enhance_image(input_path, output_path=output_path)

    assert output_path.exists()
    assert result.mode == "RGB"
    pixels = np.asarray(result)
    assert np.array_equal(pixels[..., 0], pixels[..., 1])
    assert np.array_equal(pixels[..., 1], pixels[..., 2])


def test_enhance_image_can_preserve_original_chroma_and_write_comparison(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    input_path = _write_color_image(tmp_path)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    preserved = enhancer.enhance_image(input_path, preserve_color=True)
    comparison = enhancer.enhance_image(input_path, side_by_side=True)

    preserved_pixels = np.asarray(preserved)
    assert not np.array_equal(preserved_pixels[..., 0], preserved_pixels[..., 1])
    assert comparison.size == (32, 8)


def test_enhance_video_writes_rgb_frames(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    frames: list[ArrayLike] = [
        np.full((16, 16, 3), value, dtype=np.uint8) for value in (20, 80, 140)
    ]
    imageio.mimsave(input_path, frames, fps=5, macro_block_size=1)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    returned = enhancer.enhance_video(input_path, output_path)

    assert returned == output_path
    assert output_path.exists()
    output_frames = imageio.mimread(output_path)
    assert len(output_frames) == 3
    assert np.asarray(output_frames[0]).shape[-1] == 3


def _write_identity_checkpoint(tmp_path: Path, *, input_frames: int) -> Path:
    model_config = ModelConfig(
        name=f"identity-f{input_frames}",
        input_frames=input_frames,
        base_channels=8,
        detail_channels=4,
        trunk_blocks=1,
        trunk_scale=2,
        residual_scale=0.45,
    )
    model = NightJetEdgeV1(model_config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint_path = tmp_path / f"identity-f{input_frames}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.model_dump(mode="json"),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_color_image(tmp_path: Path) -> Path:
    pixels = np.zeros((8, 16, 3), dtype=np.uint8)
    pixels[:, :8] = np.array([180, 30, 45], dtype=np.uint8)
    pixels[:, 8:] = np.array([20, 160, 90], dtype=np.uint8)
    input_path = tmp_path / "input.png"
    Image.fromarray(pixels, mode="RGB").save(input_path)
    return input_path
