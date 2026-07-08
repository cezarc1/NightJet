from typing import Any, cast

import numpy as np
import pytest

from nightjet.runtime.tensors import (
    FLOAT_TENSOR_DTYPE,
    U8_FRAME_DTYPE,
    CausalLumaWindowPacker,
    bgr_u8_to_nchw_rgb_float,
    nchw_float_to_luma_u8,
    nchw_rgb_float_to_bgr_u8,
    u8_luma_to_nchw_float,
    write_bgr_u8_to_nchw_rgb_float,
    write_u8_luma_to_nchw_float,
)


def test_luma_tensor_round_trip_clips_to_u8() -> None:
    luma = np.array([[0, 127], [255, 32]], dtype=np.uint8)

    tensor = u8_luma_to_nchw_float(luma)
    restored = nchw_float_to_luma_u8(tensor)

    assert tensor.shape == (1, 1, 2, 2)
    assert tensor.dtype == FLOAT_TENSOR_DTYPE
    assert restored.dtype == U8_FRAME_DTYPE
    assert np.array_equal(restored, luma)


def test_luma_tensor_rejects_non_luma_input() -> None:
    with pytest.raises(ValueError, match="expected 2D luma frame"):
        u8_luma_to_nchw_float(np.zeros((2, 2, 3), dtype=np.uint8))


def test_tensor_helpers_reject_dtype_overrides() -> None:
    luma = np.zeros((2, 2), dtype=np.uint8)
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_tensor = np.zeros((1, 3, 2, 2), dtype=np.float32)

    with pytest.raises(TypeError):
        cast(Any, u8_luma_to_nchw_float)(luma, dtype=np.uint8)
    with pytest.raises(TypeError):
        cast(Any, bgr_u8_to_nchw_rgb_float)(bgr, dtype=np.float16)
    with pytest.raises(TypeError):
        cast(Any, nchw_rgb_float_to_bgr_u8)(rgb_tensor, dtype=np.float32)


def test_runtime_tensor_writers_reject_non_float32_buffers() -> None:
    luma = np.zeros((2, 2), dtype=np.uint8)
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="expected output dtype float32"):
        write_u8_luma_to_nchw_float(luma, np.empty((1, 1, 2, 2), dtype=np.float16))
    with pytest.raises(ValueError, match="expected output dtype float32"):
        write_bgr_u8_to_nchw_rgb_float(bgr, np.empty((1, 3, 2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="expected host input dtype float32"):
        CausalLumaWindowPacker(np.empty((1, 3, 2, 2), dtype=np.float16))


def test_luma_tensor_helpers_reject_non_uint8_luma_input() -> None:
    luma = np.zeros((2, 2), dtype=np.float32)
    output = np.empty((1, 1, 2, 2), dtype=np.float32)
    host_input = np.empty((1, 3, 2, 2), dtype=np.float32)
    packer = CausalLumaWindowPacker(host_input)

    with pytest.raises(ValueError, match="expected luma dtype uint8"):
        u8_luma_to_nchw_float(luma)
    with pytest.raises(ValueError, match="expected luma dtype uint8"):
        write_u8_luma_to_nchw_float(luma, output)
    with pytest.raises(ValueError, match="expected luma dtype uint8"):
        packer.write_next(luma)


def test_causal_window_packer_pads_first_frame_then_rolls() -> None:
    host_input = np.empty((1, 3, 2, 2), dtype=np.float32)
    packer = CausalLumaWindowPacker(host_input, motion_budget=None)

    first = np.full((2, 2), 25, dtype=np.uint8)
    second = np.full((2, 2), 100, dtype=np.uint8)
    packer.write_next(first)
    assert packer.fill == 1
    assert packer.effective_fill == 1
    assert np.allclose(host_input[0], 25 / 255.0)

    packer.write_next(second)
    assert packer.fill == 2
    assert packer.effective_fill == 2
    assert np.allclose(host_input[0, 0], 25 / 255.0)
    assert np.allclose(host_input[0, 1], 25 / 255.0)
    assert np.allclose(host_input[0, 2], 100 / 255.0)


def test_causal_window_packer_keeps_low_motion_window() -> None:
    host_input = np.empty((1, 3, 16, 16), dtype=np.float32)
    packer = CausalLumaWindowPacker(host_input)

    for value in (40, 41, 42):
        packer.write_next(np.full((16, 16), value, dtype=np.uint8))

    assert packer.fill == 3
    assert packer.effective_fill == 3
    assert np.allclose(host_input[0, 0], 40 / 255.0)
    assert np.allclose(host_input[0, 1], 41 / 255.0)
    assert np.allclose(host_input[0, 2], 42 / 255.0)


def test_causal_window_packer_effective_window_shrinks_under_motion_budget() -> None:
    host_input = np.empty((1, 5, 16, 16), dtype=np.float32)
    packer = CausalLumaWindowPacker(host_input)

    for value in (10, 11, 12, 32, 33):
        packer.write_next(np.full((16, 16), value, dtype=np.uint8))

    assert packer.fill == 5
    assert packer.effective_fill == 2
    assert np.allclose(host_input[0, 0], 32 / 255.0)
    assert np.allclose(host_input[0, 1], 32 / 255.0)
    assert np.allclose(host_input[0, 2], 32 / 255.0)
    assert np.allclose(host_input[0, 3], 32 / 255.0)
    assert np.allclose(host_input[0, 4], 33 / 255.0)


def test_causal_window_packer_reset_clears_motion_state() -> None:
    host_input = np.empty((1, 3, 16, 16), dtype=np.float32)
    packer = CausalLumaWindowPacker(host_input)

    packer.write_next(np.full((16, 16), 40, dtype=np.uint8))
    packer.write_next(np.full((16, 16), 200, dtype=np.uint8))
    assert packer.effective_fill == 1

    packer.reset()
    assert packer.fill == 0
    assert packer.effective_fill == 0

    packer.write_next(np.full((16, 16), 60, dtype=np.uint8))
    assert packer.fill == 1
    assert packer.effective_fill == 1
    assert np.allclose(host_input[0], 60 / 255.0)
