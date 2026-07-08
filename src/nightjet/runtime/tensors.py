from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from nightjet.motion import (
    DEFAULT_MOTION_BUDGET,
    block_mean_luma_u8,
    mean_abs_luma_delta,
    motion_window_size,
)

U8_FRAME_DTYPE = np.uint8
FLOAT_TENSOR_DTYPE = np.float32

type U8Frame = NDArray[np.uint8]
type F32Tensor = NDArray[np.float32]


def u8_luma_to_nchw_float(luma: U8Frame) -> F32Tensor:
    if luma.ndim != 2:
        raise ValueError(f"expected 2D luma frame, got shape {luma.shape}")
    output = np.empty((1, 1, luma.shape[0], luma.shape[1]), dtype=FLOAT_TENSOR_DTYPE)
    return write_u8_luma_to_nchw_float(luma, output)


def write_u8_luma_to_nchw_float(luma: U8Frame, output: F32Tensor) -> F32Tensor:
    if luma.ndim != 2:
        raise ValueError(f"expected 2D luma frame, got shape {luma.shape}")
    _validate_u8_tensor_dtype(luma, name="luma")
    _validate_float_tensor_dtype(output, name="output")
    expected = (1, 1, luma.shape[0], luma.shape[1])
    if output.shape != expected:
        raise ValueError(f"expected output shape {expected}, got {output.shape}")
    np.multiply(luma, 1.0 / 255.0, out=output[0, 0], casting="unsafe")
    return output


def nchw_float_to_luma_u8(tensor: F32Tensor) -> U8Frame:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError(f"expected NCHW tensor with shape 1x1xHxW, got {tensor.shape}")
    clipped = np.clip(tensor[0, 0], 0.0, 1.0)
    return np.ascontiguousarray(np.rint(clipped * 255.0).astype(U8_FRAME_DTYPE))


def bgr_u8_to_nchw_rgb_float(frame: U8Frame) -> F32Tensor:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected BGR frame with shape HxWx3, got {frame.shape}")
    output = np.empty((1, 3, frame.shape[0], frame.shape[1]), dtype=FLOAT_TENSOR_DTYPE)
    return write_bgr_u8_to_nchw_rgb_float(frame, output)


def write_bgr_u8_to_nchw_rgb_float(frame: U8Frame, output: F32Tensor) -> F32Tensor:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected BGR frame with shape HxWx3, got {frame.shape}")
    _validate_u8_tensor_dtype(frame, name="frame")
    _validate_float_tensor_dtype(output, name="output")
    expected = (1, 3, frame.shape[0], frame.shape[1])
    if output.shape != expected:
        raise ValueError(f"expected output shape {expected}, got {output.shape}")
    np.multiply(frame[:, :, 2], 1.0 / 255.0, out=output[0, 0], casting="unsafe")
    np.multiply(frame[:, :, 1], 1.0 / 255.0, out=output[0, 1], casting="unsafe")
    np.multiply(frame[:, :, 0], 1.0 / 255.0, out=output[0, 2], casting="unsafe")
    return output


def nchw_rgb_float_to_bgr_u8(tensor: F32Tensor) -> U8Frame:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError(f"expected NCHW tensor with shape 1x3xHxW, got {tensor.shape}")
    clipped = np.clip(tensor[0], 0.0, 1.0)
    bgr = np.empty((tensor.shape[2], tensor.shape[3], 3), dtype=U8_FRAME_DTYPE)
    bgr[:, :, 2] = np.rint(clipped[0] * 255.0).astype(U8_FRAME_DTYPE)
    bgr[:, :, 1] = np.rint(clipped[1] * 255.0).astype(U8_FRAME_DTYPE)
    bgr[:, :, 0] = np.rint(clipped[2] * 255.0).astype(U8_FRAME_DTYPE)
    return np.ascontiguousarray(bgr)


class CausalLumaWindowPacker:
    """Maintain a padded, causal 1xNxHxW luma window in an existing float buffer."""

    def __init__(
        self,
        host_input: F32Tensor,
        *,
        motion_budget: float | None = DEFAULT_MOTION_BUDGET,
    ) -> None:
        if host_input.ndim != 4 or host_input.shape[0] != 1 or host_input.shape[1] < 2:
            raise ValueError(f"expected host input shape 1xNxHxW, got {host_input.shape}")
        if motion_budget is not None and motion_budget < 0:
            raise ValueError("motion_budget must be non-negative or None")
        _validate_float_tensor_dtype(host_input, name="host input")
        self._host_input = host_input
        self.motion_budget = motion_budget
        self._fill = 0
        self._effective_fill = 0
        self._motion_history: deque[float] = deque(maxlen=int(host_input.shape[1]) - 1)
        self._last_block_luma: F32Tensor | None = None

    @property
    def fill(self) -> int:
        return self._fill

    @property
    def effective_fill(self) -> int:
        return self._effective_fill

    @property
    def expected_hw(self) -> tuple[int, int]:
        return int(self._host_input.shape[2]), int(self._host_input.shape[3])

    def reset(self) -> None:
        self._fill = 0
        self._effective_fill = 0
        self._motion_history.clear()
        self._last_block_luma = None

    def write_next(self, luma: U8Frame) -> None:
        if tuple(luma.shape) != self.expected_hw:
            raise ValueError(f"expected luma shape {self.expected_hw}, got {luma.shape}")
        block_luma = block_mean_luma_u8(luma)
        if self._last_block_luma is not None:
            self._motion_history.append(mean_abs_luma_delta(block_luma, self._last_block_luma))
        self._last_block_luma = block_luma

        if self._fill == 0:
            np.multiply(
                luma,
                1.0 / 255.0,
                out=self._host_input[0, -1],
                casting="unsafe",
            )
            self._host_input[0, :-1] = self._host_input[0, -1]
        else:
            self._host_input[0, :-1] = self._host_input[0, 1:].copy()
            np.multiply(
                luma,
                1.0 / 255.0,
                out=self._host_input[0, -1],
                casting="unsafe",
            )
        self._fill = min(self._fill + 1, int(self._host_input.shape[1]))
        self._effective_fill = self._compute_effective_fill()
        pad_count = int(self._host_input.shape[1]) - self._effective_fill
        if pad_count > 0:
            # Older frames excluded by the monotone motion budget cannot re-enter later.
            self._host_input[0, :pad_count] = self._host_input[0, pad_count]

    def _compute_effective_fill(self) -> int:
        if self._fill == 0:
            return 0
        if self.motion_budget is None:
            return self._fill
        keep = motion_window_size(tuple(self._motion_history), self.motion_budget)
        return min(keep, self._fill)


def _validate_float_tensor_dtype(array: np.ndarray, *, name: str) -> None:
    expected = np.dtype(FLOAT_TENSOR_DTYPE)
    if array.dtype != expected:
        raise ValueError(f"expected {name} dtype {expected.name}, got {array.dtype}")


def _validate_u8_tensor_dtype(array: np.ndarray, *, name: str) -> None:
    expected = np.dtype(U8_FRAME_DTYPE)
    if array.dtype != expected:
        raise ValueError(f"expected {name} dtype {expected.name}, got {array.dtype}")
