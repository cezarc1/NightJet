"""Device resolution for NightJet inference."""

from __future__ import annotations

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve the torch device used for inference.

    An explicitly requested device always wins. Otherwise pick the best
    available backend in order: cuda > mps > cpu.
    """
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
