from __future__ import annotations

import importlib
import json
import threading
import time
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


@dataclass
class RuntimeMetrics:
    ready: bool = False
    frames_total: int = 0
    errors_total: int = 0
    started_at: float = field(default_factory=time.time)
    last_frame_at: float | None = None
    last_error: str | None = None
    last_inference_ms: float = 0.0
    last_metrics: dict[str, float] = field(default_factory=dict)

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
    metrics = RuntimeMetrics()
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
    ]
    for name, value in sorted(metrics.last_metrics.items()):
        safe_name = name.replace("-", "_")
        lines.append(f"nightjet_runtime_{safe_name} {float(value):.6f}")
    return "\n".join(lines) + "\n"


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
            metrics.last_inference_ms = (time.perf_counter() - start) * 1000.0
            metrics.last_metrics = enhancer.last_metrics
            metrics.frames_total += 1
            metrics.last_frame_at = time.time()
            metrics.ready = True
            metrics.last_error = None
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
