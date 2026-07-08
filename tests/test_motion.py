import numpy as np
import pytest

from nightjet.motion import block_mean_luma, block_mean_luma_u8, motion_window_size


def test_motion_window_size_keeps_trailing_budget_suffix() -> None:
    assert motion_window_size([0.001, 0.001, 0.05, 0.001], 0.045) == 2
    assert motion_window_size([], 0.045) == 1


def test_block_mean_luma_u8_matches_float_domain_normalization() -> None:
    luma_u8 = np.zeros((16, 16), dtype=np.uint8)
    luma_u8[:8, :8] = 32
    luma_u8[:8, 8:] = 96
    luma_u8[8:, :8] = 160
    luma_u8[8:, 8:] = 224

    normalized = luma_u8.astype(np.float32) / 255.0
    from_u8 = block_mean_luma_u8(luma_u8)
    from_float = block_mean_luma(normalized)

    assert from_u8.dtype == np.float32
    assert from_float.dtype == np.float32
    assert np.allclose(from_u8, from_float)


def test_block_mean_luma_u8_rejects_non_uint8_input() -> None:
    with pytest.raises(ValueError, match="expected luma dtype uint8"):
        block_mean_luma_u8(np.zeros((16, 16), dtype=np.float32))
