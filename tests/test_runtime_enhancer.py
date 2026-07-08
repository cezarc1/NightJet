from pathlib import Path
from typing import Any

import numpy as np

from nightjet.runtime.enhancer import TensorRTNightJetEnhancer
from nightjet.runtime.tensorrt import TensorRTLumaWindowEnhancer
from nightjet.runtime.tensors import CausalLumaWindowPacker


def test_from_engine_passes_motion_budget_to_window_enhancer(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    engine_path = tmp_path / "nightjet.plan"
    engine_path.write_bytes(b"engine")
    seen: dict[str, Any] = {}

    class FakeWindowEnhancer:
        def __init__(
            self,
            engine_path: Path,
            *,
            motion_budget: float | None,
        ) -> None:
            self.engine_path = engine_path
            seen["engine_path"] = engine_path
            seen["motion_budget"] = motion_budget

    monkeypatch.setattr(
        "nightjet.runtime.enhancer.TensorRTLumaWindowEnhancer",
        FakeWindowEnhancer,
    )

    enhancer = TensorRTNightJetEnhancer.from_engine(engine_path, motion_budget=0.12)

    assert enhancer.engine_path == engine_path
    assert seen == {"engine_path": engine_path, "motion_budget": 0.12}


def test_from_engine_ignores_motion_budget_for_static_engine(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    engine_path = tmp_path / "static.plan"
    engine_path.write_bytes(b"engine")
    seen: dict[str, Any] = {}

    class FakeWindowEnhancer:
        def __init__(
            self,
            engine_path: Path,
            *,
            motion_budget: float | None,
        ) -> None:
            seen["window_motion_budget"] = motion_budget
            raise ValueError("expected TensorRT input shape 1xNxHxW, got (1, 1, 8, 8)")

    class FakeStaticEnhancer:
        def __init__(self, engine_path: Path) -> None:
            self.engine_path = engine_path
            seen["static_engine_path"] = engine_path

    monkeypatch.setattr(
        "nightjet.runtime.enhancer.TensorRTLumaWindowEnhancer",
        FakeWindowEnhancer,
    )
    monkeypatch.setattr(
        "nightjet.runtime.enhancer.TensorRTLumaEnhancer",
        FakeStaticEnhancer,
    )

    enhancer = TensorRTNightJetEnhancer.from_engine(engine_path, motion_budget=None)

    assert enhancer.engine_path == engine_path
    assert seen == {"window_motion_budget": None, "static_engine_path": engine_path}


def test_window_enhancer_process_next_reports_physical_and_effective_fill() -> None:
    host_input = np.empty((1, 3, 16, 16), dtype=np.float32)
    enhancer = object.__new__(TensorRTLumaWindowEnhancer)
    enhancer._input_shape = (1, 3, 16, 16)
    enhancer._causal_packer = CausalLumaWindowPacker(host_input)

    def fake_execute_current_input(
        _start: float, _pack_ms: float
    ) -> tuple[np.ndarray, dict[str, float]]:
        return np.zeros((16, 16), dtype=np.uint8), {}

    enhancer._execute_current_input = fake_execute_current_input

    enhancer.process_next(np.full((16, 16), 40, dtype=np.uint8))
    _, metrics = enhancer.process_next(np.full((16, 16), 200, dtype=np.uint8))

    assert metrics["causal_window_fill"] == 2.0
    assert metrics["causal_window_effective_fill"] == 1.0
