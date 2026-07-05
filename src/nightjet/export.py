from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nightjet.config import ModelConfig
from nightjet.models import NightJetEdgeV1


def export_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
    input_shape: tuple[int, int, int, int],
    opset_version: int = 18,
) -> Path:
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig.model_validate(checkpoint["model_config"])
    model = NightJetEdgeV1(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    example = torch.zeros(input_shape, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        output_path,
        input_names=["luma_window"],
        output_names=["enhanced_luma"],
        opset_version=opset_version,
        dynamic_axes=None,
        dynamo=False,
    )
    return output_path
