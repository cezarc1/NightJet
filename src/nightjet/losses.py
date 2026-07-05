from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NightJetLossWeights:
    reconstruction: float = 1.0
    edge: float = 0.2
    brightness: float = 0.35
    smoothness: float = 0.0


class NightJetLoss(nn.Module):
    def __init__(self, weights: NightJetLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or NightJetLossWeights()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        reconstruction = F.l1_loss(prediction, target)
        edge = gradient_l1_loss(prediction, target)
        brightness = F.l1_loss(
            prediction.mean(dim=(-2, -1)),
            target.mean(dim=(-2, -1)),
        )
        smoothness = flat_region_smoothness(prediction, target)
        total = (
            self.weights.reconstruction * reconstruction
            + self.weights.edge * edge
            + self.weights.brightness * brightness
            + self.weights.smoothness * smoothness
        )
        components = {
            "reconstruction": float(reconstruction.detach().cpu()),
            "edge": float(edge.detach().cpu()),
            "brightness": float(brightness.detach().cpu()),
            "smoothness": float(smoothness.detach().cpu()),
        }
        return total, components


def gradient_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def flat_region_smoothness(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    threshold_x = torch.quantile(target_dx.detach().flatten(), 0.35)
    threshold_y = torch.quantile(target_dy.detach().flatten(), 0.35)
    pred_dx = torch.abs(prediction[:, :, :, 1:] - prediction[:, :, :, :-1])
    pred_dy = torch.abs(prediction[:, :, 1:, :] - prediction[:, :, :-1, :])
    smooth_x = pred_dx[target_dx <= threshold_x]
    smooth_y = pred_dy[target_dy <= threshold_y]
    if smooth_x.numel() == 0 or smooth_y.numel() == 0:
        return prediction.new_tensor(0.0)
    return smooth_x.mean() + smooth_y.mean()
