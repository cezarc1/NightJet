import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np

import nightjet.runtime.tensorrt as tensorrt_runtime
from nightjet.runtime.tensorrt import (
    TensorRTLumaEnhancer,
    TensorRTLumaWindowEnhancer,
    _nvtx_range,
)


def test_nvtx_range_uses_torch_cuda_range_when_available() -> None:
    observed: list[str] = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            nvtx=SimpleNamespace(range=lambda name: observed.append(name) or nullcontext())
        )
    )

    with _nvtx_range(torch, "nightjet.tensorrt.execute"):
        pass

    assert observed == ["nightjet.tensorrt.execute"]


def test_nvtx_range_falls_back_when_nvtx_is_unavailable() -> None:
    with _nvtx_range(SimpleNamespace(cuda=SimpleNamespace()), "nightjet.tensorrt.execute"):
        pass


def test_static_tensorrt_execution_is_wrapped_in_nvtx_range(monkeypatch: Any) -> None:
    active_ranges: list[str] = []
    observed_during_execute: list[list[str]] = []

    @contextmanager
    def record_range(_torch: Any, name: str) -> Any:
        active_ranges.append(name)
        try:
            yield
        finally:
            active_ranges.pop()

    class FakeTensor:
        def __init__(self, array: np.ndarray | None = None) -> None:
            self.array = array

        def copy_(self, _other: Any, *, non_blocking: bool) -> None:
            return None

        def numpy(self) -> np.ndarray:
            assert self.array is not None
            return self.array

    class FakeContext:
        def execute_async_v3(self, _stream: Any) -> bool:
            observed_during_execute.append(list(active_ranges))
            return True

    enhancer = object.__new__(TensorRTLumaEnhancer)
    enhancer._torch = SimpleNamespace(
        cuda=SimpleNamespace(stream=lambda _stream: nullcontext())
    )
    enhancer._stream = SimpleNamespace(cuda_stream=object(), synchronize=lambda: None)
    enhancer._context = FakeContext()
    enhancer._input_shape = (1, 1, 2, 2)
    enhancer._host_input = np.empty((1, 1, 2, 2), dtype=np.float32)
    enhancer._input_tensor = FakeTensor()
    enhancer._input_cpu_tensor = FakeTensor()
    enhancer._output_tensor = FakeTensor()
    enhancer._output_cpu_tensor = FakeTensor(np.zeros((1, 1, 2, 2), dtype=np.float32))

    monkeypatch.setattr(tensorrt_runtime, "_nvtx_range", record_range)
    enhancer.process(np.zeros((2, 2), dtype=np.uint8))

    assert observed_during_execute == [["nightjet.tensorrt.execute"]]


def test_temporal_tensorrt_execution_is_wrapped_in_nvtx_range(monkeypatch: Any) -> None:
    active_ranges: list[str] = []
    observed_during_execute: list[list[str]] = []

    @contextmanager
    def record_range(_torch: Any, name: str) -> Any:
        active_ranges.append(name)
        try:
            yield
        finally:
            active_ranges.pop()

    class FakeEvent:
        def record(self, _stream: Any) -> None:
            return None

        def elapsed_time(self, _other: Any) -> float:
            return 1.0

    class FakeTensor:
        def __init__(self, array: np.ndarray | None = None) -> None:
            self.array = array

        def copy_(self, _other: Any, *, non_blocking: bool) -> None:
            return None

        def numpy(self) -> np.ndarray:
            assert self.array is not None
            return self.array

    class FakeContext:
        def execute_async_v3(self, _stream: Any) -> bool:
            observed_during_execute.append(list(active_ranges))
            return True

    stream = SimpleNamespace(cuda_stream=object(), synchronize=lambda: None)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            stream=lambda _stream: nullcontext(),
            Event=lambda **_kwargs: FakeEvent(),
        )
    )
    enhancer = object.__new__(TensorRTLumaWindowEnhancer)
    enhancer._torch = fake_torch
    enhancer._stream = stream
    enhancer._context = FakeContext()
    enhancer._input_tensor = FakeTensor()
    enhancer._input_cpu_tensor = FakeTensor()
    enhancer._output_tensor = FakeTensor()
    enhancer._output_cpu_tensor = FakeTensor(np.zeros((1, 1, 2, 2), dtype=np.float32))
    enhancer._input_cpu_pinned = False
    enhancer._output_cpu_pinned = False

    monkeypatch.setattr(tensorrt_runtime, "_nvtx_range", record_range)
    enhancer._execute_current_input(time.perf_counter(), 0.0)

    assert observed_during_execute == [["nightjet.tensorrt.execute"]]
