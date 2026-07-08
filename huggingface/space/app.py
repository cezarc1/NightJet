from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Callable
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from nightjet.inference import (
    NightJetEnhancer,
    _coerce_rgb_array,
    _estimate_frame_count,
    _iter_video_frames,
    _open_video_reader,
    _open_video_writer,
)

MODEL_REPO_ID = os.environ.get("NIGHTJET_MODEL_REPO_ID", "ggamecrazy/nightjet-edge-v1")
WEIGHTS_FILENAME = os.environ.get("NIGHTJET_WEIGHTS_FILENAME", "nightjet-edge-v1.pt")
MAX_LONG_EDGE = int(os.environ.get("NIGHTJET_MAX_LONG_EDGE", "1280"))
MAX_VIDEO_LONG_EDGE = int(os.environ.get("NIGHTJET_MAX_VIDEO_LONG_EDGE", "720"))
MAX_VIDEO_SECONDS = float(os.environ.get("NIGHTJET_MAX_VIDEO_SECONDS", "8"))
MAX_VIDEO_FPS = float(os.environ.get("NIGHTJET_MAX_VIDEO_FPS", "12"))
SPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_EXAMPLE_IMAGE = SPACE_ROOT / "examples" / "frame-01.jpg"


try:
    spaces = importlib.import_module("spaces")
except ImportError:
    spaces = None


def _gpu(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if spaces is None:
        return lambda function: function
    return spaces.GPU(**kwargs)


def clear_model_cache() -> None:
    _load_enhancer.cache_clear()


@_gpu(duration=30)
def enhance_demo_image(
    input_image: Image.Image | np.ndarray | None,
    preserve_color: bool = False,
    max_long_edge: int = MAX_LONG_EDGE,
) -> tuple[Image.Image, Image.Image]:
    if input_image is None:
        raise ValueError("Upload an image before running NightJet.")
    original = _prepare_image(input_image, max_long_edge=max_long_edge)
    enhancer = _load_enhancer(_device())
    enhanced = enhancer.enhance_image(original, preserve_color=preserve_color)
    return enhanced, _side_by_side(original, enhanced)


@_gpu(duration=120)
def enhance_demo_video(
    input_video: str | Path | None,
    preserve_color: bool = False,
    max_seconds: float = MAX_VIDEO_SECONDS,
    max_long_edge: int = MAX_VIDEO_LONG_EDGE,
) -> str:
    if input_video is None:
        raise ValueError("Upload a short video clip before running NightJet.")
    input_path = Path(input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"video not found: {input_path}")
    output_path = Path(tempfile.mkdtemp(prefix="nightjet-space-")) / "nightjet-side-by-side.mp4"
    enhancer = _load_enhancer(_device())
    enhancer.reset()

    with (
        closing(_open_video_reader(input_path)) as reader,
        closing(
            _open_video_writer(output_path, _output_video_fps(reader.get_meta_data()))
        ) as writer,
    ):
        metadata = reader.get_meta_data()
        max_frames = _max_video_frames(metadata, max_seconds=max_seconds)
        for index, frame in enumerate(_iter_video_frames(reader)):
            if index >= max_frames:
                break
            rgb = _resize_for_demo(
                _array_to_image(_coerce_rgb_array(frame)),
                max_long_edge=max_long_edge,
            )
            enhanced = enhancer._enhance_video_frame(
                np.asarray(rgb, dtype=np.uint8),
                side_by_side=True,
                preserve_color=preserve_color,
            )
            writer.append_data(enhanced)
    return str(output_path)


def build_demo() -> Any:
    gr = importlib.import_module("gradio")

    with gr.Blocks(title="NightJet") as demo:
        gr.Markdown(
            "# NightJet\nTiny luma-first low-light enhancement for passive night-vision images."
        )
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    input_image = gr.Image(
                        value=str(DEFAULT_EXAMPLE_IMAGE),
                        type="pil",
                        label="Input image",
                    )
                    with gr.Column():
                        preserve_color = gr.Checkbox(
                            label="Preserve original color",
                            value=False,
                        )
                        run_button = gr.Button("Enhance image", variant="primary")
                with gr.Row():
                    enhanced = gr.Image(type="pil", label="NightJet output")
                    comparison = gr.Image(type="pil", label="Before / after")
                run_button.click(
                    fn=enhance_demo_image,
                    inputs=[input_image, preserve_color],
                    outputs=[enhanced, comparison],
                    api_name="enhance",
                )
            with gr.Tab("Video"):
                gr.Markdown(
                    f"Upload a short clip. The demo converts up to {MAX_VIDEO_SECONDS:g}s "
                    f"at up to {MAX_VIDEO_FPS:g} fps and {MAX_VIDEO_LONG_EDGE}px long edge."
                )
                with gr.Row():
                    input_video = gr.Video(label="Input clip")
                    with gr.Column():
                        video_preserve_color = gr.Checkbox(
                            label="Preserve original color",
                            value=False,
                        )
                        run_video_button = gr.Button("Enhance clip", variant="primary")
                output_video = gr.Video(label="Before / after clip")
                run_video_button.click(
                    fn=enhance_demo_video,
                    inputs=[input_video, video_preserve_color],
                    outputs=[output_video],
                    api_name="enhance_video",
                )
        gr.Markdown(
            "Model: "
            "[ggamecrazy/nightjet-edge-v1](https://huggingface.co/ggamecrazy/nightjet-edge-v1). "
            "Source: [cezarc1/nightjet](https://github.com/cezarc1/nightjet)."
        )
    return demo


@lru_cache(maxsize=2)
def _load_enhancer(device: str) -> NightJetEnhancer:
    return NightJetEnhancer.from_checkpoint(_download_weights(), device=device)


def _download_weights() -> Path:
    hub = importlib.import_module("huggingface_hub")

    return Path(hub.hf_hub_download(repo_id=MODEL_REPO_ID, filename=WEIGHTS_FILENAME))


def _prepare_image(input_image: Image.Image | np.ndarray, *, max_long_edge: int) -> Image.Image:
    if isinstance(input_image, Image.Image):
        image = input_image.convert("RGB")
    else:
        image = _array_to_image(np.asarray(input_image))
    return _resize_for_demo(image, max_long_edge=max_long_edge)


def _array_to_image(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(_to_uint8(array)).convert("RGB")
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        return Image.fromarray(_to_uint8(array[..., :3])).convert("RGB")
    raise ValueError(f"expected image array with shape HxW, HxWx3, or HxWx4; got {array.shape}")


def _resize_for_demo(image: Image.Image, *, max_long_edge: int) -> Image.Image:
    if max_long_edge < 1:
        raise ValueError("max_long_edge must be positive")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_long_edge:
        return image
    scale = max_long_edge / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _max_video_frames(metadata: dict[str, Any], *, max_seconds: float) -> int:
    fps = _output_video_fps(metadata)
    estimated_frames = _estimate_frame_count(metadata)
    max_frames = max(1, round(fps * max_seconds))
    if estimated_frames is None:
        return max_frames
    return max(1, min(estimated_frames, max_frames))


def _output_video_fps(metadata: dict[str, Any]) -> float:
    fps = metadata.get("fps")
    if not isinstance(fps, int | float) or not np.isfinite(fps) or fps <= 0:
        return min(MAX_VIDEO_FPS, 12.0)
    return float(min(float(fps), MAX_VIDEO_FPS))


def _side_by_side(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    original = original.convert("RGB")
    enhanced = enhanced.convert("RGB")
    size = (original.width + enhanced.width, max(original.height, enhanced.height))
    canvas = Image.new("RGB", size)
    canvas.paste(original, (0, 0))
    canvas.paste(enhanced, (original.width, 0))
    return canvas


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    build_demo().launch()
