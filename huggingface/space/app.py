from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from nightjet.inference import NightJetEnhancer

MODEL_REPO_ID = os.environ.get("NIGHTJET_MODEL_REPO_ID", "ggamecrazy/nightjet-edge-v1")
WEIGHTS_FILENAME = os.environ.get("NIGHTJET_WEIGHTS_FILENAME", "nightjet-edge-v1.pt")
MAX_LONG_EDGE = int(os.environ.get("NIGHTJET_MAX_LONG_EDGE", "1280"))


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


def build_demo() -> Any:
    gr = importlib.import_module("gradio")

    with gr.Blocks(title="NightJet") as demo:
        gr.Markdown(
            "# NightJet\nTiny luma-first low-light enhancement for passive night-vision images."
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input image")
            with gr.Column():
                preserve_color = gr.Checkbox(
                    label="Preserve original color",
                    value=False,
                )
                run_button = gr.Button("Enhance", variant="primary")
        with gr.Row():
            enhanced = gr.Image(type="pil", label="NightJet output")
            comparison = gr.Image(type="pil", label="Before / after")
        run_button.click(
            fn=enhance_demo_image,
            inputs=[input_image, preserve_color],
            outputs=[enhanced, comparison],
            api_name="enhance",
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
