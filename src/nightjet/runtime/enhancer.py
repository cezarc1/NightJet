from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from nightjet.inference import (
    _compose_rgb,
    _estimate_frame_count,
    _iter_video_frames,
    _load_rgb_image,
    _open_video_reader,
    _open_video_writer,
)
from nightjet.motion import DEFAULT_MOTION_BUDGET
from nightjet.runtime.tensorrt import TensorRTLumaEnhancer, TensorRTLumaWindowEnhancer
from nightjet.runtime.tensors import U8_FRAME_DTYPE, U8Frame


class TensorRTNightJetEnhancer:
    """High-level image/video enhancer backed by a NightJet TensorRT engine."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.engine_path = engine.engine_path
        self.last_metrics: dict[str, float] = {}

    @classmethod
    def from_engine(
        cls,
        engine_path: Path,
        *,
        motion_budget: float | None = DEFAULT_MOTION_BUDGET,
    ) -> TensorRTNightJetEnhancer:
        try:
            return cls(TensorRTLumaWindowEnhancer(engine_path, motion_budget=motion_budget))
        except ValueError as exc:
            if "1xNxHxW" not in str(exc):
                raise
        return cls(TensorRTLumaEnhancer(engine_path))

    def reset(self) -> None:
        self.engine.reset()
        self.last_metrics = {}

    def process_luma_u8(self, luma: U8Frame) -> U8Frame:
        process_next = getattr(self.engine, "process_next", None)
        if callable(process_next):
            output, metrics = process_next(luma)  # type: ignore[operator]
        else:
            output, metrics = self.engine.process(luma)
        self.last_metrics = metrics
        return output

    def enhance_image(
        self,
        input_image: Path | Image.Image | np.ndarray,
        *,
        output_path: Path | None = None,
        side_by_side: bool = False,
        preserve_color: bool = False,
    ) -> Image.Image:
        self.reset()
        rgb_image = _load_rgb_image(input_image)
        enhanced_luma = self.process_luma_u8(_rgb_to_luma_u8(rgb_image))
        enhanced_rgb = _compose_rgb(rgb_image, enhanced_luma, preserve_color=preserve_color)
        if side_by_side:
            enhanced_rgb = np.concatenate(
                [np.asarray(rgb_image, dtype=U8_FRAME_DTYPE), enhanced_rgb], axis=1
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
        # closing() instead of bare with: imageio's __exit__ skips close() on exceptions.
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
                    rgb_image = _load_rgb_image(frame)
                    enhanced_luma = self.process_luma_u8(_rgb_to_luma_u8(rgb_image))
                    enhanced_rgb = _compose_rgb(
                        rgb_image, enhanced_luma, preserve_color=preserve_color
                    )
                    if side_by_side:
                        enhanced_rgb = np.concatenate(
                            [np.asarray(rgb_image, dtype=U8_FRAME_DTYPE), enhanced_rgb], axis=1
                        )
                    writer.append_data(enhanced_rgb)
                    progress.update(1)
        return output_path


def _rgb_to_luma_u8(image: Image.Image) -> U8Frame:
    return np.ascontiguousarray(np.asarray(image.convert("YCbCr").split()[0], dtype=np.uint8))
