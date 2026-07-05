from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nightjet.config import ModelConfig


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class NightJetEdgeV1(nn.Module):
    """Small residual luma enhancer designed for ONNX/TensorRT export."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        trunk_layers: list[nn.Module] = [
            nn.Conv2d(config.input_frames, config.base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        trunk_layers.extend(
            DepthwiseSeparableBlock(config.base_channels) for _ in range(config.trunk_blocks)
        )
        self.trunk = nn.Sequential(*trunk_layers)
        self.detail = nn.Sequential(
            nn.Conv2d(config.input_frames, config.detail_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.detail_channels, config.detail_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(config.base_channels + config.detail_channels + 1, 1, 3, padding=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != self.config.input_frames:
            raise ValueError(
                f"expected Bx{self.config.input_frames}xHxW luma frames, got {tuple(frames.shape)}"
            )
        center = frames[:, -1:, :, :]
        trunk_input = frames
        if self.config.trunk_scale > 1:
            trunk_input = F.avg_pool2d(
                frames,
                kernel_size=self.config.trunk_scale,
                stride=self.config.trunk_scale,
            )
        trunk_features = self.trunk(trunk_input)
        if self.config.trunk_scale > 1:
            trunk_features = F.interpolate(
                trunk_features,
                size=(frames.shape[2], frames.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        detail_features = self.detail(frames)
        fused = torch.cat([center, trunk_features, detail_features], dim=1)
        residual = self.config.residual_scale * torch.tanh(self.fuse(fused))
        return torch.clamp(center + residual, 0.0, 1.0)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
