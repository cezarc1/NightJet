import numpy as np
import pytest

from nightjet.runtime.tensors import (
    CausalLumaWindowPacker,
    nchw_float_to_luma_u8,
    u8_luma_to_nchw_float,
)


def test_luma_tensor_round_trip_clips_to_u8() -> None:
    luma = np.array([[0, 127], [255, 32]], dtype=np.uint8)

    tensor = u8_luma_to_nchw_float(luma)
    restored = nchw_float_to_luma_u8(tensor)

    assert tensor.shape == (1, 1, 2, 2)
    assert tensor.dtype == np.float32
    assert np.array_equal(restored, luma)


def test_luma_tensor_rejects_non_luma_input() -> None:
    with pytest.raises(ValueError, match="expected 2D luma frame"):
        u8_luma_to_nchw_float(np.zeros((2, 2, 3), dtype=np.uint8))


def test_causal_window_packer_pads_first_frame_then_rolls() -> None:
    host_input = np.empty((1, 3, 2, 2), dtype=np.float32)
    packer = CausalLumaWindowPacker(host_input)

    first = np.full((2, 2), 25, dtype=np.uint8)
    second = np.full((2, 2), 100, dtype=np.uint8)
    packer.write_next(first)
    assert packer.fill == 1
    assert np.allclose(host_input[0], 25 / 255.0)

    packer.write_next(second)
    assert packer.fill == 2
    assert np.allclose(host_input[0, 0], 25 / 255.0)
    assert np.allclose(host_input[0, 1], 25 / 255.0)
    assert np.allclose(host_input[0, 2], 100 / 255.0)
