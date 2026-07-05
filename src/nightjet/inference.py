from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from nightjet.config import ModelConfig
from nightjet.models import NightJetEdgeV1

DEFAULT_WEIGHTS_PATH = Path("weights/nightjet-edge-v1.pt")
VIDEO_SUFFIXES = {".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


class NightJetEnhancer:
    def __init__(
        self,
        model: NightJetEdgeV1,
        *,
        checkpoint_path: Path,
        device: torch.device,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.metadata = metadata or {}
        self.model_config = model.config
        self._luma_history: deque[np.ndarray] = deque(maxlen=self.model_config.input_frames)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path, *, device: str | None = None
    ) -> NightJetEnhancer:
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = ModelConfig.model_validate(checkpoint["model_config"])
        model = NightJetEdgeV1(model_config)
        state_dict = _checkpoint_state_dict(checkpoint)
        model.load_state_dict(state_dict)
        model.to(torch_device)
        model.eval()
        return cls(
            model,
            checkpoint_path=checkpoint_path,
            device=torch_device,
            metadata=checkpoint.get("metadata"),
        )

    def reset(self) -> None:
        self._luma_history.clear()

    def enhance_window(self, window: np.ndarray) -> np.ndarray:
        luma_window = _normalize_luma_window(window)
        luma_window = _pad_or_trim_window(luma_window, self.model_config.input_frames)
        tensor = torch.from_numpy(luma_window).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            enhanced = self.model(tensor).squeeze(0).squeeze(0).detach().cpu().numpy()
        return np.clip(enhanced.astype(np.float32, copy=False), 0.0, 1.0)

    def enhance_image(
        self,
        input_image: Path | Image.Image | np.ndarray,
        *,
        output_path: Path | None = None,
        side_by_side: bool = False,
        preserve_color: bool = False,
    ) -> Image.Image:
        rgb_image = _load_rgb_image(input_image)
        luma = _rgb_to_luma(rgb_image)
        enhanced_luma = self.enhance_window(luma)
        enhanced_rgb = _compose_rgb(rgb_image, enhanced_luma, preserve_color=preserve_color)
        if side_by_side:
            enhanced_rgb = np.concatenate(
                [np.asarray(rgb_image, dtype=np.uint8), enhanced_rgb], axis=1
            )
        result = Image.fromarray(enhanced_rgb)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
        return result

    def enhance_video(
        self,
        input_path: Path,
        output_path: Path,
        *,
        side_by_side: bool = False,
        preserve_color: bool = False,
        fps: float | None = None,
    ) -> Path:
        self.reset()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reader = imageio.get_reader(input_path)
        metadata = reader.get_meta_data()
        output_fps = fps or float(metadata.get("fps") or 30.0)
        writer = imageio.get_writer(output_path, fps=output_fps, macro_block_size=1)
        try:
            index = 0
            while True:
                try:
                    frame = reader.get_data(index)
                except IndexError:
                    break
                rgb_image = _load_rgb_image(frame)
                enhanced_rgb = self._enhance_video_frame(
                    rgb_image,
                    side_by_side=side_by_side,
                    preserve_color=preserve_color,
                )
                writer.append_data(enhanced_rgb)
                index += 1
        finally:
            writer.close()
            reader.close()
        return output_path

    def _enhance_video_frame(
        self,
        rgb_image: Image.Image,
        *,
        side_by_side: bool,
        preserve_color: bool,
    ) -> np.ndarray:
        luma = _rgb_to_luma(rgb_image)
        self._luma_history.append(luma)
        enhanced_luma = self.enhance_window(np.stack(tuple(self._luma_history), axis=0))
        enhanced_rgb = _compose_rgb(rgb_image, enhanced_luma, preserve_color=preserve_color)
        if side_by_side:
            return np.concatenate([np.asarray(rgb_image, dtype=np.uint8), enhanced_rgb], axis=1)
        return enhanced_rgb


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def _checkpoint_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise KeyError("checkpoint must contain model_state_dict")


def _load_rgb_image(input_image: Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(input_image, Path):
        with Image.open(input_image) as image:
            return image.convert("RGB")
    if isinstance(input_image, Image.Image):
        return input_image.convert("RGB")
    array = np.asarray(input_image)
    if array.ndim == 2:
        return Image.fromarray(_to_uint8(array)).convert("RGB")
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        return Image.fromarray(_to_uint8(array[..., :3])).convert("RGB")
    raise ValueError(f"expected image array with shape HxW, HxWx3, or HxWx4; got {array.shape}")


def _normalize_luma_window(window: np.ndarray) -> np.ndarray:
    array = np.asarray(window)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"expected luma window with shape HxW or FxHxW; got {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        normalized = array.astype(np.float32) / 255.0
    else:
        normalized = array.astype(np.float32, copy=False)
    return np.clip(normalized, 0.0, 1.0)


def _pad_or_trim_window(window: np.ndarray, input_frames: int) -> np.ndarray:
    if window.shape[0] == input_frames:
        return window
    if window.shape[0] > input_frames:
        return window[-input_frames:]
    pad_count = input_frames - window.shape[0]
    padding = np.repeat(window[:1], pad_count, axis=0)
    return np.concatenate([padding, window], axis=0)


def _rgb_to_luma(image: Image.Image) -> np.ndarray:
    y_channel = image.convert("YCbCr").split()[0]
    return np.asarray(y_channel, dtype=np.float32) / 255.0


def _compose_rgb(
    original_image: Image.Image,
    enhanced_luma: np.ndarray,
    *,
    preserve_color: bool,
) -> np.ndarray:
    y_channel = Image.fromarray(_to_uint8(enhanced_luma))
    if preserve_color:
        _, cb_channel, cr_channel = original_image.convert("YCbCr").split()
        return np.asarray(Image.merge("YCbCr", (y_channel, cb_channel, cr_channel)).convert("RGB"))
    y = np.asarray(y_channel, dtype=np.uint8)
    return np.stack([y, y, y], axis=-1)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(array, 0, 255).astype(np.uint8, copy=False)
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
