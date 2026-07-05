import torch

from nightjet.config import ModelConfig
from nightjet.models import NightJetEdgeV1


def test_nightjet_edge_v1_preserves_luma_shape() -> None:
    model = NightJetEdgeV1(
        ModelConfig(
            name="test",
            input_frames=3,
            base_channels=16,
            detail_channels=8,
            trunk_blocks=2,
            trunk_scale=2,
            residual_scale=0.45,
        )
    )
    x = torch.rand(2, 3, 32, 48)

    y = model(x)

    assert y.shape == (2, 1, 32, 48)
    assert torch.isfinite(y).all()
    assert 1_000 < model.parameter_count() < 50_000
