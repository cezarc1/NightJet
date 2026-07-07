import torch

from nightjet.devices import resolve_device


def test_resolve_device_honors_explicit_request() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_prefers_best_available_backend() -> None:
    resolved = resolve_device(None)
    if torch.cuda.is_available():
        assert resolved.type == "cuda"
    elif torch.backends.mps.is_available():
        assert resolved.type == "mps"
    else:
        assert resolved.type == "cpu"
