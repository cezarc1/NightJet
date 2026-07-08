from pathlib import Path
from typing import Any

from nightjet.runtime.server import RuntimeServerConfig, run_runtime_server


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
