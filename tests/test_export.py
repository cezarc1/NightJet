from pathlib import Path

import torch

from nightjet.config import ModelConfig
from nightjet.export import export_onnx
from nightjet.models import NightJetEdgeV1


def test_export_onnx_writes_model_file(tmp_path: Path) -> None:
    model_config = ModelConfig(
        name="test",
        input_frames=3,
        base_channels=8,
        detail_channels=4,
        trunk_blocks=1,
        trunk_scale=2,
        residual_scale=0.45,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": NightJetEdgeV1(model_config).state_dict(),
            "model_config": model_config.model_dump(mode="json"),
        },
        checkpoint_path,
    )

    output_path = export_onnx(
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "nightjet.onnx",
        input_shape=(1, 3, 16, 16),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
