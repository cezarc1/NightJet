import torch

from nightjet.losses import NightJetLoss, NightJetLossWeights


def test_loss_returns_finite_components() -> None:
    criterion = NightJetLoss(NightJetLossWeights(edge=0.2, brightness=0.35))
    prediction = torch.full((2, 1, 8, 8), 0.4)
    target = torch.full((2, 1, 8, 8), 0.5)

    total, components = criterion(prediction, target)

    assert torch.isfinite(total)
    assert components["reconstruction"] > 0
    assert set(components) == {"reconstruction", "edge", "brightness", "smoothness"}
