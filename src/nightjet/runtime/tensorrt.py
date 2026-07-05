from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nightjet.runtime.tensors import (
    CausalLumaWindowPacker,
    U8Frame,
    nchw_float_to_luma_u8,
    write_u8_luma_to_nchw_float,
)


class TensorRTLumaEnhancer:
    """Run a static 1x1xHxW luma TensorRT engine using PyTorch CUDA buffers."""

    def __init__(self, engine_path: Path | str) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self._torch = importlib.import_module("torch")
        trt = importlib.import_module("tensorrt")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()
        self._input_name, self._output_name = _tensor_names(self._engine, trt)
        input_shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(self._input_name))
        output_shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(self._output_name))
        if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 1:
            raise ValueError(f"expected TensorRT input shape 1x1xHxW, got {input_shape}")
        if output_shape != input_shape:
            raise ValueError(f"expected TensorRT output shape {input_shape}, got {output_shape}")
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._input_cpu_tensor, _ = _empty_cpu_tensor(
            self._torch, self._input_shape, pin_memory=False
        )
        self._host_input = self._input_cpu_tensor.numpy()
        self._output_cpu_tensor, _ = _empty_cpu_tensor(
            self._torch, self._output_shape, pin_memory=False
        )
        self._input_tensor = self._torch.empty(
            self._input_shape, dtype=self._torch.float32, device="cuda"
        )
        self._output_tensor = self._torch.empty(
            self._output_shape, dtype=self._torch.float32, device="cuda"
        )
        self._stream = self._torch.cuda.Stream()
        self._context.set_tensor_address(self._input_name, int(self._input_tensor.data_ptr()))
        self._context.set_tensor_address(self._output_name, int(self._output_tensor.data_ptr()))

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return self._input_shape

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_shape[3], self._input_shape[2]

    def reset(self) -> None:
        return None

    def process(self, luma: U8Frame) -> tuple[U8Frame, dict[str, float]]:
        expected_hw = self._input_shape[2:]
        if tuple(luma.shape) != expected_hw:
            raise ValueError(f"expected luma shape {expected_hw}, got {luma.shape}")

        start = time.perf_counter()
        write_u8_luma_to_nchw_float(luma, self._host_input)
        with self._torch.cuda.stream(self._stream):
            self._input_tensor.copy_(self._input_cpu_tensor, non_blocking=True)
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            self._output_cpu_tensor.copy_(self._output_tensor, non_blocking=False)
        self._stream.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return nchw_float_to_luma_u8(self._output_cpu_tensor.numpy()), {
            "tensorrt_ms": elapsed_ms
        }


class TensorRTLumaWindowEnhancer:
    """Run a static 1xNxHxW NightJet luma-window TensorRT engine."""

    def __init__(self, engine_path: Path | str) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self._torch = importlib.import_module("torch")
        trt = importlib.import_module("tensorrt")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()
        self._input_name, self._output_name = _tensor_names(self._engine, trt)
        input_shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(self._input_name))
        output_shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(self._output_name))
        if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] < 2:
            raise ValueError(f"expected TensorRT input shape 1xNxHxW, got {input_shape}")
        expected_output = (1, 1, input_shape[2], input_shape[3])
        if output_shape != expected_output:
            raise ValueError(
                f"expected TensorRT output shape {expected_output}, got {output_shape}"
            )
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._input_cpu_tensor, self._input_cpu_pinned = _empty_cpu_tensor(
            self._torch, self._input_shape, pin_memory=True
        )
        self._host_input = self._input_cpu_tensor.numpy()
        self._causal_packer = CausalLumaWindowPacker(self._host_input)
        self._output_cpu_tensor, self._output_cpu_pinned = _empty_cpu_tensor(
            self._torch, self._output_shape, pin_memory=True
        )
        self._input_tensor = self._torch.empty(
            self._input_shape, dtype=self._torch.float32, device="cuda"
        )
        self._output_tensor = self._torch.empty(
            self._output_shape, dtype=self._torch.float32, device="cuda"
        )
        self._stream = self._torch.cuda.Stream()
        self._context.set_tensor_address(self._input_name, int(self._input_tensor.data_ptr()))
        self._context.set_tensor_address(self._output_name, int(self._output_tensor.data_ptr()))

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return self._input_shape

    @property
    def input_frames(self) -> int:
        return self._input_shape[1]

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_shape[3], self._input_shape[2]

    def reset(self) -> None:
        self._causal_packer.reset()

    def process(self, window: Sequence[U8Frame]) -> tuple[U8Frame, dict[str, float]]:
        if len(window) != self.input_frames:
            raise ValueError(f"expected {self.input_frames} luma frames, got {len(window)}")
        expected_hw = self._input_shape[2:]
        for index, luma in enumerate(window):
            if tuple(luma.shape) != expected_hw:
                raise ValueError(
                    f"expected luma window frame {index} shape {expected_hw}, got {luma.shape}"
                )

        start = time.perf_counter()
        pack_start = time.perf_counter()
        for index, luma in enumerate(window):
            np.multiply(luma, 1.0 / 255.0, out=self._host_input[0, index], casting="unsafe")
        pack_ms = (time.perf_counter() - pack_start) * 1000.0
        return self._execute_current_input(start, pack_ms)

    def process_next(self, luma: U8Frame) -> tuple[U8Frame, dict[str, float]]:
        expected_hw = self._input_shape[2:]
        if tuple(luma.shape) != expected_hw:
            raise ValueError(f"expected luma shape {expected_hw}, got {luma.shape}")
        start = time.perf_counter()
        pack_start = time.perf_counter()
        self._causal_packer.write_next(luma)
        pack_ms = (time.perf_counter() - pack_start) * 1000.0
        output, metrics = self._execute_current_input(start, pack_ms)
        metrics["causal_window_fill"] = float(self._causal_packer.fill)
        return output, metrics

    def _execute_current_input(
        self,
        start: float,
        pack_ms: float,
    ) -> tuple[U8Frame, dict[str, float]]:
        with self._torch.cuda.stream(self._stream):
            input_copy_start = time.perf_counter()
            self._input_tensor.copy_(
                self._input_cpu_tensor,
                non_blocking=_copy_non_blocking(pinned_cpu_tensor=self._input_cpu_pinned),
            )
            input_copy_submit_ms = (time.perf_counter() - input_copy_start) * 1000.0
            trt_start_event = self._torch.cuda.Event(enable_timing=True)
            trt_end_event = self._torch.cuda.Event(enable_timing=True)
            copy_start_event = self._torch.cuda.Event(enable_timing=True)
            copy_end_event = self._torch.cuda.Event(enable_timing=True)
            trt_start_event.record(self._stream)
            execute_start = time.perf_counter()
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            execute_submit_ms = (time.perf_counter() - execute_start) * 1000.0
            trt_end_event.record(self._stream)
            copy_start_event.record(self._stream)
            output_copy_start = time.perf_counter()
            self._output_cpu_tensor.copy_(
                self._output_tensor,
                non_blocking=_copy_non_blocking(pinned_cpu_tensor=self._output_cpu_pinned),
            )
            output_copy_ms = (time.perf_counter() - output_copy_start) * 1000.0
            copy_end_event.record(self._stream)
        sync_start = time.perf_counter()
        self._stream.synchronize()
        stream_sync_ms = (time.perf_counter() - sync_start) * 1000.0
        trt_gpu_ms = float(trt_start_event.elapsed_time(trt_end_event))
        output_copy_gpu_ms = float(copy_start_event.elapsed_time(copy_end_event))
        output = nchw_float_to_luma_u8(self._output_cpu_tensor.numpy())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return output, {
            "tensorrt_ms": elapsed_ms,
            "pack_ms": pack_ms,
            "input_copy_submit_ms": input_copy_submit_ms,
            "execute_submit_ms": execute_submit_ms,
            "output_copy_ms": output_copy_ms,
            "stream_sync_ms": stream_sync_ms,
            "trt_gpu_ms": trt_gpu_ms,
            "output_copy_gpu_ms": output_copy_gpu_ms,
            "input_cpu_pinned": float(self._input_cpu_pinned),
            "output_cpu_pinned": float(self._output_cpu_pinned),
        }


def _tensor_names(engine: Any, trt: Any) -> tuple[str, str]:
    input_names: list[str] = []
    output_names: list[str] = []
    for index in range(int(engine.num_io_tensors)):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError(
            f"expected one TensorRT input and one output, got {input_names=} {output_names=}"
        )
    return input_names[0], output_names[0]


def _empty_cpu_tensor(
    torch: Any,
    shape: tuple[int, ...],
    *,
    pin_memory: bool,
) -> tuple[Any, bool]:
    if pin_memory:
        try:
            return torch.empty(shape, dtype=torch.float32, pin_memory=True), True
        except (RuntimeError, TypeError):
            pass
    return torch.empty(shape, dtype=torch.float32), False


def _copy_non_blocking(*, pinned_cpu_tensor: bool) -> bool:
    return pinned_cpu_tensor
