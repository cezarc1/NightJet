import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest
import torch
from imageio.typing import ArrayLike
from PIL import Image

import nightjet.inference as nightjet_inference
from nightjet.config import ModelConfig
from nightjet.inference import (
    MODEL_INPUT_DTYPE,
    NightJetEnhancer,
    _estimate_frame_count,
    _iter_video_frames,
    _open_video_reader,
    _rgb_to_luma,
    _rgb_to_luma_array,
    _to_uint8,
)
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


def test_enhance_window_rejects_dtype_override(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")
    window = np.zeros((3, 6, 8), dtype=np.float32)

    with pytest.raises(TypeError):
        cast(Any, enhancer.enhance_window)(window, dtype=torch.float16)


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
    _write_gray_video(input_path, fps=5, macro_block_size=1)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    returned = enhancer.enhance_video(input_path, output_path)

    assert returned == output_path
    assert output_path.exists()
    output_frames = imageio.mimread(output_path)
    assert len(output_frames) == 3
    assert np.asarray(output_frames[0]).shape[-1] == 3


def test_enhance_video_accepts_show_progress_flag(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_gray_video(input_path, fps=5, macro_block_size=1)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    enhancer.enhance_video(input_path, output_path, show_progress=False)

    assert output_path.exists()
    assert len(imageio.mimread(output_path)) == 3


def test_enhance_video_decodes_with_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    _write_gray_video(input_path, fps=5, macro_block_size=1)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    real_get_reader = imageio.get_reader
    reader_kwargs: list[dict[str, Any]] = []

    def spy_get_reader(uri: Any, *args: Any, **kwargs: Any) -> Any:
        reader_kwargs.append(kwargs)
        return real_get_reader(uri, *args, **kwargs)

    monkeypatch.setattr("nightjet.inference.imageio.get_reader", spy_get_reader)
    enhancer.enhance_video(input_path, output_path)

    assert len(imageio.mimread(output_path)) == 3
    (kwargs,) = reader_kwargs
    assert kwargs["format"] == "FFMPEG"
    assert kwargs["output_params"] == ["-vsync", "0"]


def test_open_video_reader_preserves_vfr_frame_sequence(tmp_path: Path) -> None:
    input_path = _write_vfr_video(tmp_path)

    with closing(_open_video_reader(input_path)) as reader:
        frame_means = [
            float(np.asarray(frame, dtype=np.float32).mean())
            for frame in _iter_video_frames(reader)
        ]

    assert len(frame_means) == 4
    assert np.allclose(frame_means, [30.0, 120.0, 220.0, 220.0], atol=3.0)


def test_enhance_video_closes_reader_when_writer_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    input_path = tmp_path / "input.mp4"
    _write_gray_video(input_path, fps=5, macro_block_size=1)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    opened: list[Any] = []

    def spy_open_reader(path: Path) -> Any:
        reader = _open_video_reader(path)
        opened.append(reader)
        return reader

    def failing_open_writer(path: Path, fps: float) -> Any:
        raise RuntimeError("writer failed")

    monkeypatch.setattr("nightjet.inference._open_video_reader", spy_open_reader)
    monkeypatch.setattr("nightjet.inference._open_video_writer", failing_open_writer)
    with pytest.raises(RuntimeError, match="writer failed"):
        enhancer.enhance_video(input_path, tmp_path / "output.mp4")

    (reader,) = opened
    assert reader.closed


def test_enhance_video_reads_gif_input(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    input_path = tmp_path / "input.gif"
    output_path = tmp_path / "output.mp4"
    _write_gray_video(input_path)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    enhancer.enhance_video(input_path, output_path)

    assert len(imageio.mimread(output_path)) == 3


def test_enhance_video_preserve_color_and_side_by_side(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    input_path = tmp_path / "input.mp4"
    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    pixels[:, :8] = np.array([180, 30, 45], dtype=np.uint8)
    pixels[:, 8:] = np.array([20, 160, 90], dtype=np.uint8)
    frames: list[ArrayLike] = [pixels] * 3
    imageio.mimsave(input_path, frames, fps=5, macro_block_size=1)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")

    color_path = tmp_path / "color.mp4"
    enhancer.enhance_video(input_path, color_path, preserve_color=True)
    color_frame = np.asarray(imageio.mimread(color_path)[0])
    assert not np.array_equal(color_frame[..., 0], color_frame[..., 1])

    comparison_path = tmp_path / "comparison.mp4"
    enhancer.enhance_video(input_path, comparison_path, side_by_side=True)
    comparison_frame = np.asarray(imageio.mimread(comparison_path)[0])
    assert comparison_frame.shape[1] == 32


def test_streaming_window_matches_enhance_window(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")
    rng = np.random.default_rng(7)
    base = rng.integers(0, 200, size=(6, 8, 3), dtype=np.uint8)
    history: list[np.ndarray] = []
    for step in range(4):
        # Low-motion frames keep the full history inside the motion budget.
        rgb = np.clip(base.astype(np.int16) + 2 * step, 0, 255).astype(np.uint8)
        history.append(_rgb_to_luma_array(rgb))

        streamed = enhancer._enhance_video_frame(rgb, side_by_side=False, preserve_color=False)

        reference_luma = enhancer.enhance_window(np.stack(history, axis=0))
        expected = np.stack([_to_uint8(reference_luma)] * 3, axis=-1)
        delta = np.abs(streamed.astype(np.int16) - expected.astype(np.int16))
        assert int(delta.max()) <= 1


def test_streaming_video_frame_uses_model_input_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")
    rgb = np.full((6, 8, 3), 40, dtype=np.uint8)

    def fake_luma_array(_rgb: np.ndarray) -> np.ndarray:
        return np.zeros((6, 8), dtype=np.float64)

    monkeypatch.setattr(nightjet_inference, "_rgb_to_luma_array", fake_luma_array)
    enhancer._enhance_video_frame(rgb, side_by_side=False, preserve_color=False)

    assert enhancer._luma_history[-1].dtype == MODEL_INPUT_DTYPE


def test_effective_history_shrinks_under_motion_budget(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=5)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")
    tensors = [torch.full((4, 4), float(i)) for i in range(5)]
    enhancer._luma_history.extend(tensors)
    enhancer._motion_history.extend([0.001, 0.001, 0.05, 0.001])

    effective = enhancer._effective_history()

    assert len(effective) == 2
    assert effective[0] is tensors[3]
    assert effective[1] is tensors[4]


def test_video_history_collapses_on_fast_pan(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")
    calm = np.full((16, 16, 3), 40, dtype=np.uint8)
    for _ in range(3):
        enhancer._enhance_video_frame(calm, side_by_side=False, preserve_color=False)
    assert len(enhancer._effective_history()) == 3

    jump = np.full((16, 16, 3), 200, dtype=np.uint8)
    enhancer._enhance_video_frame(jump, side_by_side=False, preserve_color=False)
    assert len(enhancer._effective_history()) == 1

    enhancer._enhance_video_frame(jump, side_by_side=False, preserve_color=False)
    assert len(enhancer._effective_history()) == 2


def test_motion_budget_can_be_disabled_for_full_causal_history(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    enhancer = NightJetEnhancer.from_checkpoint(
        checkpoint, device="cpu", motion_budget=None
    )
    calm = np.full((16, 16, 3), 40, dtype=np.uint8)
    jump = np.full((16, 16, 3), 200, dtype=np.uint8)

    enhancer._enhance_video_frame(calm, side_by_side=False, preserve_color=False)
    enhancer._enhance_video_frame(calm, side_by_side=False, preserve_color=False)
    enhancer._enhance_video_frame(jump, side_by_side=False, preserve_color=False)

    assert len(enhancer._effective_history()) == 3


def test_rgb_to_luma_array_matches_pil() -> None:
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

    pil_luma = _rgb_to_luma(Image.fromarray(rgb))
    array_luma = _rgb_to_luma_array(rgb)

    assert array_luma.dtype == np.float32
    assert float(np.max(np.abs(pil_luma - array_luma))) <= 1.0 / 255.0 + 1e-6


def test_estimate_frame_count() -> None:
    assert _estimate_frame_count({"fps": 5.0, "duration": 2.0}) == 10
    assert _estimate_frame_count({"nframes": 42, "fps": 5.0, "duration": 2.0}) == 42
    assert _estimate_frame_count({"nframes": float("inf"), "fps": 5.0, "duration": 2.0}) == 10
    assert _estimate_frame_count({"fps": 5.0}) is None
    assert _estimate_frame_count({"fps": 5.0, "duration": float("inf")}) is None
    assert _estimate_frame_count({}) is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_enhance_window_mps_matches_cpu(tmp_path: Path) -> None:
    checkpoint = _write_identity_checkpoint(tmp_path, input_frames=3)
    cpu_enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="cpu")
    mps_enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device="mps")
    window = np.random.default_rng(5).random((3, 6, 8), dtype=np.float32)

    assert np.allclose(
        cpu_enhancer.enhance_window(window),
        mps_enhancer.enhance_window(window),
        atol=1e-4,
    )


def _write_gray_video(path: Path, **save_kwargs: Any) -> None:
    frames: list[ArrayLike] = [
        np.full((16, 16, 3), value, dtype=np.uint8) for value in (20, 80, 140)
    ]
    imageio.mimsave(path, frames, **save_kwargs)


def _write_vfr_video(tmp_path: Path) -> Path:
    frame_dir = tmp_path / "vfr"
    frame_dir.mkdir()
    for index, value in enumerate((30, 120, 220)):
        frame = np.full((32, 32, 3), value, dtype=np.uint8)
        Image.fromarray(frame, mode="RGB").save(frame_dir / f"frame{index}.png")
    concat_path = frame_dir / "frames.txt"
    concat_path.write_text(
        "\n".join(
            [
                "ffconcat version 1.0",
                "file 'frame0.png'",
                "duration 0.10",
                "file 'frame1.png'",
                "duration 0.45",
                "file 'frame2.png'",
                "duration 0.10",
                "file 'frame2.png'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vsync",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
        cwd=frame_dir,
    )
    return output_path


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
