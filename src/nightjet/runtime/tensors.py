from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

U8Frame = NDArray[np.uint8]
F32Tensor = NDArray[np.float32]


def u8_luma_to_nchw_float(luma: U8Frame) -> F32Tensor:
    if luma.ndim != 2:
        raise ValueError(f"expected 2D luma frame, got shape {luma.shape}")
    output = np.empty((1, 1, luma.shape[0], luma.shape[1]), dtype=np.float32)
    return write_u8_luma_to_nchw_float(luma, output)


def write_u8_luma_to_nchw_float(luma: U8Frame, output: F32Tensor) -> F32Tensor:
    if luma.ndim != 2:
        raise ValueError(f"expected 2D luma frame, got shape {luma.shape}")
    expected = (1, 1, luma.shape[0], luma.shape[1])
    if output.shape != expected:
        raise ValueError(f"expected output shape {expected}, got {output.shape}")
    np.multiply(luma, 1.0 / 255.0, out=output[0, 0], casting="unsafe")
    return output


def nchw_float_to_luma_u8(tensor: F32Tensor) -> U8Frame:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError(f"expected NCHW tensor with shape 1x1xHxW, got {tensor.shape}")
    clipped = np.clip(tensor[0, 0], 0.0, 1.0)
    return np.ascontiguousarray(np.rint(clipped * 255.0).astype(np.uint8))


def bgr_u8_to_nchw_rgb_float(frame: U8Frame) -> F32Tensor:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected BGR frame with shape HxWx3, got {frame.shape}")
    output = np.empty((1, 3, frame.shape[0], frame.shape[1]), dtype=np.float32)
    return write_bgr_u8_to_nchw_rgb_float(frame, output)


def write_bgr_u8_to_nchw_rgb_float(frame: U8Frame, output: F32Tensor) -> F32Tensor:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected BGR frame with shape HxWx3, got {frame.shape}")
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
    bgr = np.empty((tensor.shape[2], tensor.shape[3], 3), dtype=np.uint8)
    bgr[:, :, 2] = np.rint(clipped[0] * 255.0).astype(np.uint8)
    bgr[:, :, 1] = np.rint(clipped[1] * 255.0).astype(np.uint8)
    bgr[:, :, 0] = np.rint(clipped[2] * 255.0).astype(np.uint8)
    return np.ascontiguousarray(bgr)


class CausalLumaWindowPacker:
    """Maintain a padded, causal 1xNxHxW luma window in an existing float buffer."""

    def __init__(self, host_input: F32Tensor) -> None:
        if host_input.ndim != 4 or host_input.shape[0] != 1 or host_input.shape[1] < 2:
            raise ValueError(f"expected host input shape 1xNxHxW, got {host_input.shape}")
        self._host_input = host_input
        self._fill = 0

    @property
    def fill(self) -> int:
        return self._fill

    @property
    def expected_hw(self) -> tuple[int, int]:
        return int(self._host_input.shape[2]), int(self._host_input.shape[3])

    def reset(self) -> None:
        self._fill = 0

    def write_next(self, luma: U8Frame) -> None:
        if tuple(luma.shape) != self.expected_hw:
            raise ValueError(f"expected luma shape {self.expected_hw}, got {luma.shape}")
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
