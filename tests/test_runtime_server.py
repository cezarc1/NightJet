from pathlib import Path
from typing import Any

from nightjet.runtime.server import (
    RuntimeMetrics,
    RuntimeServerConfig,
    render_prometheus_metrics,
    run_runtime_server,
)


def test_runtime_server_config_preserves_source_positional_argument(tmp_path: Path) -> None:
    engine_path = tmp_path / "nightjet.plan"

    config = RuntimeServerConfig(engine_path, "0")

    assert config.engine_path == engine_path
    assert config.source == "0"
    assert config.motion_budget == 0.045


def test_run_runtime_server_passes_motion_budget_to_engine(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    engine_path = tmp_path / "nightjet.plan"
    engine_path.write_bytes(b"engine")
    seen: dict[str, Any] = {}

    class FakeEnhancer:
        @classmethod
        def from_engine(
            cls,
            engine_path: Path,
            *,
            motion_budget: float | None,
        ) -> "FakeEnhancer":
            seen["engine_path"] = engine_path
            seen["motion_budget"] = motion_budget
            return cls()

    def fake_capture_loop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr("nightjet.runtime.server.TensorRTNightJetEnhancer", FakeEnhancer)
    monkeypatch.setattr("nightjet.runtime.server._capture_loop", fake_capture_loop)

    run_runtime_server(
        RuntimeServerConfig(
            engine_path=engine_path,
            motion_budget=0.12,
            exit_after_max_frames=True,
        )
    )

    assert seen == {"engine_path": engine_path, "motion_budget": 0.12}


def test_runtime_metrics_report_model_fps_freshness_and_inference_histogram() -> None:
    metrics = RuntimeMetrics(model_id="nightjet-edge-v1-detail")

    metrics.record_frame(inference_seconds=0.012, timestamp=100.0)
    metrics.record_frame(inference_seconds=0.020, timestamp=101.0)

    rendered = render_prometheus_metrics(metrics)

    assert (
        'nightjet_runtime_info{task="vision",runtime="tensorrt-fp16",'
        'model_id="nightjet-edge-v1-detail"} 1'
    ) in rendered
    assert "nightjet_capture_fps 1.000000" in rendered
    assert "nightjet_last_frame_timestamp_seconds 101.000000" in rendered
    assert 'nightjet_inference_duration_seconds_bucket{le="0.025"} 2' in rendered
    assert "nightjet_inference_duration_seconds_sum 0.032000" in rendered
    assert "nightjet_inference_duration_seconds_count 2" in rendered


def test_runtime_metrics_escape_model_label_values() -> None:
    metrics = RuntimeMetrics(model_id='nightjet"detail\\candidate')

    rendered = render_prometheus_metrics(metrics)

    assert 'model_id="nightjet\\"detail\\\\candidate"' in rendered


def test_runtime_metrics_only_export_known_engine_measurements() -> None:
    metrics = RuntimeMetrics()
    metrics.record_frame(
        inference_seconds=0.01,
        timestamp=100.0,
        runtime_metrics={
            "causal_window_effective_fill": 2.0,
            "trt_gpu_ms": 8.5,
            'bad metric{label="x"}': 99.0,
        },
    )

    rendered = render_prometheus_metrics(metrics)

    assert "nightjet_runtime_causal_window_effective_fill 2.000000" in rendered
    assert "nightjet_runtime_trt_gpu_ms 8.500000" in rendered
    assert "bad metric" not in rendered
