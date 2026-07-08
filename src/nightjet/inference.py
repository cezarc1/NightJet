from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from nightjet.config import ModelConfig
from nightjet.devices import resolve_device
from nightjet.metrics import mae
from nightjet.models import NightJetEdgeV1

DEFAULT_WEIGHTS_PATH = Path("weights/nightjet-edge-v1.pt")
VIDEO_SUFFIXES = {".avi", ".gif", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
# Cumulative inter-frame motion (mean abs block-luma delta) tolerated inside the window.
DEFAULT_MOTION_BUDGET = 0.045


class NightJetEnhancer:
    def __init__(
        self,
        model: NightJetEdgeV1,
        *,
        checkpoint_path: Path,
        device: torch.device,
        metadata: dict[str, Any] | None = None,
        motion_budget: float | None = DEFAULT_MOTION_BUDGET,
    ) -> None:
        if motion_budget is not None and motion_budget < 0:
            raise ValueError("motion_budget must be non-negative or None")
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.metadata = metadata or {}
        self.model_config = model.config
        self.motion_budget = motion_budget
        self._luma_history: deque[torch.Tensor] = deque(maxlen=self.model_config.input_frames)
        self._motion_history: deque[float] = deque(maxlen=self.model_config.input_frames - 1)
        self._last_block_luma: np.ndarray | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device: str | None = None,
        motion_budget: float | None = DEFAULT_MOTION_BUDGET,
    ) -> NightJetEnhancer:
        torch_device = resolve_device(device)
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
            motion_budget=motion_budget,
        )

    def reset(self) -> None:
        self._luma_history.clear()
        self._motion_history.clear()
        self._last_block_luma = None

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
        show_progress: bool = True,
    ) -> Path:
        self.reset()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_open_video_reader(input_path)) as reader:
            metadata = reader.get_meta_data()
            output_fps = fps or float(metadata.get("fps") or 30.0)
            with (
                closing(_open_video_writer(output_path, output_fps)) as writer,
                tqdm(
                    total=_estimate_frame_count(metadata),
                    desc=input_path.name,
                    unit="frame",
                    disable=None if show_progress else True,
                ) as progress,
            ):
                for frame in _iter_video_frames(reader):
                    enhanced_rgb = self._enhance_video_frame(
                        _coerce_rgb_array(frame),
                        side_by_side=side_by_side,
                        preserve_color=preserve_color,
                    )
                    writer.append_data(enhanced_rgb)
                    progress.update(1)
        return output_path

    def _enhance_video_frame(
        self,
        rgb: np.ndarray,
        *,
        side_by_side: bool,
        preserve_color: bool,
    ) -> np.ndarray:
        luma = _rgb_to_luma_array(rgb)
        if self.model_config.input_frames > 1:
            block_luma = _block_mean_luma(luma)
            if self._last_block_luma is not None:
                self._motion_history.append(mae(block_luma, self._last_block_luma))
            self._last_block_luma = block_luma
        with torch.inference_mode():
            self._luma_history.append(torch.from_numpy(luma).to(self.device))
        enhanced_luma = self._enhance_history_window()
        if preserve_color:
            enhanced_rgb = _compose_rgb(Image.fromarray(rgb), enhanced_luma, preserve_color=True)
        else:
            y = _to_uint8(enhanced_luma)
            enhanced_rgb = np.stack([y, y, y], axis=-1)
        if side_by_side:
            return np.concatenate([rgb, enhanced_rgb], axis=1)
        return enhanced_rgb

    def _effective_history(self) -> list[torch.Tensor]:
        if self.motion_budget is None:
            return list(self._luma_history)
        keep = _motion_window_size(tuple(self._motion_history), self.motion_budget)
        return list(self._luma_history)[-keep:]

    def _enhance_history_window(self) -> np.ndarray:
        history = self._effective_history()
        pad_count = self.model_config.input_frames - len(history)
        if pad_count > 0:
            history = [history[0]] * pad_count + history
        with torch.inference_mode():
            enhanced = self.model(torch.stack(history, dim=0).unsqueeze(0)).squeeze(0).squeeze(0)
        return np.clip(enhanced.cpu().numpy().astype(np.float32, copy=False), 0.0, 1.0)


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def _open_video_reader(path: Path) -> Any:
    if path.suffix.lower() == ".gif":
        return imageio.get_reader(path)
    # -vsync 0: decode VFR sources without CFR frame duplication.
    return imageio.get_reader(
        path,
        format="FFMPEG",  # ty: ignore[invalid-argument-type]
        output_params=["-vsync", "0"],
    )


def _open_video_writer(path: Path, fps: float) -> Any:
    return imageio.get_writer(path, fps=fps, macro_block_size=1)


def _iter_video_frames(reader: Any) -> Iterator[Any]:
    index = 0
    while True:
        try:
            yield reader.get_data(index)
        except (EOFError, IndexError):
            return
        index += 1


def _motion_window_size(motions: Sequence[float], budget: float) -> int:
    """Number of trailing window frames whose cumulative inter-frame motion fits the budget."""
    keep = 1
    cumulative = 0.0
    for motion in reversed(motions):
        cumulative += motion
        if cumulative > budget:
            break
        keep += 1
    return keep


def _estimate_frame_count(metadata: dict[str, Any]) -> int | None:
    nframes = metadata.get("nframes")
    if isinstance(nframes, int | float) and math.isfinite(nframes) and nframes > 0:
        return int(nframes)
    fps = metadata.get("fps")
    duration = metadata.get("duration")
    if not isinstance(fps, int | float) or not isinstance(duration, int | float):
        return None
    if not (math.isfinite(fps) and math.isfinite(duration)):
        return None
    estimate = round(float(fps) * float(duration))
    return estimate if estimate > 0 else None


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


# ITU-R BT.601 luma weights; results round to the uint8 grid to match PIL's Y channel.
_BT601_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _coerce_rgb_array(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = _to_uint8(array)
        return np.stack([gray, gray, gray], axis=-1)
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        return _to_uint8(array[..., :3])
    raise ValueError(f"expected image array with shape HxW, HxWx3, or HxWx4; got {array.shape}")


def _rgb_to_luma_array(rgb: np.ndarray) -> np.ndarray:
    return np.rint(rgb.astype(np.float32) @ _BT601_LUMA_WEIGHTS) / 255.0


def _block_mean_luma(luma: np.ndarray, block: int = 8) -> np.ndarray:
    blocks_h, blocks_w = luma.shape[0] // block, luma.shape[1] // block
    if blocks_h == 0 or blocks_w == 0:
        return luma
    trimmed = luma[: blocks_h * block, : blocks_w * block]
    return trimmed.reshape(blocks_h, block, blocks_w, block).mean(axis=(1, 3))


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
