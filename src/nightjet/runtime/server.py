from __future__ import annotations

import importlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from nightjet.motion import DEFAULT_MOTION_BUDGET
from nightjet.runtime.enhancer import TensorRTNightJetEnhancer


@dataclass
class RuntimeServerConfig:
    engine_path: Path
    source: str | None = None
    camera: Path | None = None
    resolution: str = "1280x720"
    pixel_format: str | None = None
    fps: float | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    max_frames: int | None = None
    exit_after_max_frames: bool = False
    motion_budget: float | None = DEFAULT_MOTION_BUDGET
    model_id: str = "nightjet-edge-v1"


@dataclass
class RuntimeMetrics:
    model_id: str = "nightjet-edge-v1"
    ready: bool = False
    frames_total: int = 0
    errors_total: int = 0
    started_at: float = field(default_factory=time.time)
    last_frame_at: float | None = None
    last_error: str | None = None
    last_inference_ms: float = 0.0
    last_metrics: dict[str, float] = field(default_factory=dict)
    capture_fps: float = 0.0
    inference_duration_sum: float = 0.0
    inference_duration_count: int = 0
    inference_duration_buckets: list[int] = field(
        default_factory=lambda: [0] * len(INFERENCE_DURATION_BUCKETS)
    )
    _frame_timestamps: deque[float] = field(default_factory=deque, repr=False)

    def record_frame(
        self,
        *,
        inference_seconds: float,
        timestamp: float,
        runtime_metrics: dict[str, float] | None = None,
    ) -> None:
        self.ready = True
        self.frames_total += 1
        self.last_frame_at = timestamp
        self.last_inference_ms = inference_seconds * 1000.0
        self.last_metrics = runtime_metrics or {}
        self.last_error = None
        self.inference_duration_sum += inference_seconds
        self.inference_duration_count += 1
        for index, upper_bound in enumerate(INFERENCE_DURATION_BUCKETS):
            if inference_seconds <= upper_bound:
                self.inference_duration_buckets[index] += 1

        self._frame_timestamps.append(timestamp)
        cutoff = timestamp - 5.0
        while len(self._frame_timestamps) > 1 and self._frame_timestamps[0] < cutoff:
            self._frame_timestamps.popleft()
        if len(self._frame_timestamps) > 1:
            elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
            self.capture_fps = (len(self._frame_timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "frames_total": self.frames_total,
            "errors_total": self.errors_total,
            "uptime_seconds": max(0.0, time.time() - self.started_at),
            "last_frame_at": self.last_frame_at,
            "last_error": self.last_error,
            "last_inference_ms": self.last_inference_ms,
            "last_metrics": self.last_metrics,
        }


def run_runtime_server(config: RuntimeServerConfig) -> RuntimeMetrics:
    metrics = RuntimeMetrics(model_id=config.model_id)
    enhancer = TensorRTNightJetEnhancer.from_engine(
        config.engine_path,
        motion_budget=config.motion_budget,
    )

    if config.exit_after_max_frames:
        _capture_loop(config, enhancer, metrics)
        return metrics

    stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=_capture_loop,
        args=(config, enhancer, metrics, stop_event),
        name="nightjet-capture",
        daemon=True,
    )
    capture_thread.start()

    handler = _handler_for(metrics)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    try:
        server.serve_forever()
    finally:
        stop_event.set()
        server.server_close()
        capture_thread.join(timeout=5.0)
    return metrics


def render_prometheus_metrics(metrics: RuntimeMetrics) -> str:
    lines = [
        "# HELP nightjet_runtime_info Static NightJet runtime labels.",
        "# TYPE nightjet_runtime_info gauge",
        (
            'nightjet_runtime_info{task="vision",runtime="tensorrt-fp16",'
            f'model_id="{_escape_label(metrics.model_id)}"}} 1'
        ),
        "# HELP nightjet_ready Whether the NightJet runtime has processed a frame.",
        "# TYPE nightjet_ready gauge",
        f"nightjet_ready {1 if metrics.ready else 0}",
        "# HELP nightjet_frames_total Frames processed by the NightJet runtime.",
        "# TYPE nightjet_frames_total counter",
        f"nightjet_frames_total {metrics.frames_total}",
        "# HELP nightjet_errors_total Capture or inference errors.",
        "# TYPE nightjet_errors_total counter",
        f"nightjet_errors_total {metrics.errors_total}",
        "# HELP nightjet_last_inference_ms Last measured inference latency.",
        "# TYPE nightjet_last_inference_ms gauge",
        f"nightjet_last_inference_ms {metrics.last_inference_ms:.6f}",
        "# HELP nightjet_capture_fps Rolling capture and inference frame rate.",
        "# TYPE nightjet_capture_fps gauge",
        f"nightjet_capture_fps {metrics.capture_fps:.6f}",
        "# HELP nightjet_last_frame_timestamp_seconds Unix timestamp of the last processed frame.",
        "# TYPE nightjet_last_frame_timestamp_seconds gauge",
        f"nightjet_last_frame_timestamp_seconds {(metrics.last_frame_at or 0.0):.6f}",
        "# HELP nightjet_inference_duration_seconds TensorRT inference duration.",
        "# TYPE nightjet_inference_duration_seconds histogram",
    ]
    for upper_bound, count in zip(
        INFERENCE_DURATION_BUCKETS,
        metrics.inference_duration_buckets,
        strict=True,
    ):
        lines.append(f'nightjet_inference_duration_seconds_bucket{{le="{upper_bound:g}"}} {count}')
    lines.extend(
        [
            "nightjet_inference_duration_seconds_bucket"
            f'{{le="+Inf"}} {metrics.inference_duration_count}',
            f"nightjet_inference_duration_seconds_sum {metrics.inference_duration_sum:.6f}",
            f"nightjet_inference_duration_seconds_count {metrics.inference_duration_count}",
        ]
    )
    for name in sorted(RUNTIME_METRIC_NAMES):
        if name in metrics.last_metrics:
            lines.append(f"nightjet_runtime_{name} {float(metrics.last_metrics[name]):.6f}")
    return "\n".join(lines) + "\n"


INFERENCE_DURATION_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
RUNTIME_METRIC_NAMES = {
    "causal_window_effective_fill",
    "causal_window_fill",
    "execute_submit_ms",
    "input_copy_submit_ms",
    "input_cpu_pinned",
    "output_copy_gpu_ms",
    "output_copy_ms",
    "output_cpu_pinned",
    "pack_ms",
    "stream_sync_ms",
    "tensorrt_ms",
    "trt_gpu_ms",
}


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _capture_loop(
    config: RuntimeServerConfig,
    enhancer: TensorRTNightJetEnhancer,
    metrics: RuntimeMetrics,
    stop_event: threading.Event | None = None,
) -> None:
    cv2 = _import_cv2()
    source = _capture_source(config)
    capture = cv2.VideoCapture(source)
    width, height = _parse_resolution(config.resolution)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if config.fps is not None:
        capture.set(cv2.CAP_PROP_FPS, config.fps)
    if config.pixel_format:
        fourcc = cv2.VideoWriter_fourcc(*config.pixel_format[:4])
        capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                metrics.errors_total += 1
                metrics.last_error = f"could not read frame from {source}"
                time.sleep(0.1)
                continue
            start = time.perf_counter()
            enhancer.process_luma_u8(_frame_to_luma_u8(cv2, frame))
            inference_seconds = time.perf_counter() - start
            metrics.record_frame(
                inference_seconds=inference_seconds,
                timestamp=time.time(),
                runtime_metrics=enhancer.last_metrics,
            )
            if config.max_frames is not None and metrics.frames_total >= config.max_frames:
                return
    except Exception as exc:
        metrics.errors_total += 1
        metrics.last_error = str(exc)
        raise
    finally:
        capture.release()


def _handler_for(metrics: RuntimeMetrics) -> type[BaseHTTPRequestHandler]:
    class NightJetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in {"/healthz", "/readyz"}:
                _write_response(self, "application/json", json.dumps(metrics.to_json_dict()))
            elif self.path == "/metrics":
                _write_response(
                    self,
                    "text/plain; version=0.0.4",
                    render_prometheus_metrics(metrics),
                )
            else:
                _write_response(self, "text/plain", "NightJet runtime\n")

        def log_message(self, format: str, *args: Any) -> None:
            return None

    return NightJetHandler


def _write_response(handler: BaseHTTPRequestHandler, content_type: str, body: str) -> None:
    payload = body.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _import_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError("nightjet serve requires the optional 'orin' dependencies") from exc


def _capture_source(config: RuntimeServerConfig) -> str | int:
    source = str(config.camera) if config.camera is not None else config.source
    if source is None:
        raise ValueError("nightjet serve requires --camera or --source")
    if source.isdigit():
        return int(source)
    return source


def _parse_resolution(value: str) -> tuple[int, int]:
    width, sep, height = value.lower().partition("x")
    if sep != "x":
        raise ValueError(f"expected resolution WIDTHxHEIGHT, got {value!r}")
    return int(width), int(height)


def _frame_to_luma_u8(cv2: Any, frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.ascontiguousarray(frame.astype(np.uint8, copy=False))
    if frame.ndim == 3 and frame.shape[2] == 3:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        return np.ascontiguousarray(ycrcb[:, :, 0])
    raise ValueError(f"expected grayscale or BGR frame, got shape {frame.shape}")
