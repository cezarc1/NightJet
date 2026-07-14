import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from typer.testing import CliRunner

from nightjet.cli import app
from nightjet.config import ModelConfig
from nightjet.data import package_arrays
from nightjet.models import NightJetEdgeV1


def test_cli_writes_split_json(tmp_path: Path) -> None:
    runner = CliRunner()
    bundle = package_arrays(
        np.zeros((10, 4, 4), dtype=np.float32),
        np.ones((10, 4, 4), dtype=np.float32),
        tmp_path / "bundle",
    )

    result = runner.invoke(
        app,
        [
            "data",
            "split",
            "--bundle-dir",
            str(bundle.bundle_dir),
            "--output",
            str(tmp_path / "splits.json"),
            "--seed",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "splits.json").read_text())
    assert sorted(payload) == ["test", "train", "val"]


def test_cli_eval_checkpoint_writes_report(tmp_path: Path) -> None:
    runner = CliRunner()
    bundle = package_arrays(
        np.random.default_rng(3).random((5, 10, 10), dtype=np.float32),
        np.random.default_rng(4).random((5, 10, 10), dtype=np.float32),
        tmp_path / "bundle",
    )
    model_config = ModelConfig(
        name="test",
        input_frames=3,
        base_channels=8,
        detail_channels=4,
        trunk_blocks=1,
        trunk_scale=2,
        residual_scale=0.45,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": NightJetEdgeV1(model_config).state_dict(),
            "model_config": model_config.model_dump(mode="json"),
        },
        checkpoint,
    )
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps({"train": [0, 1, 2], "val": [2, 3], "test": [3, 4]}), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--bundle-dir",
            str(bundle.bundle_dir),
            "--output-dir",
            str(tmp_path / "eval"),
            "--device",
            "cpu",
            "--split-json",
            str(split_path),
            "--split",
            "val",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "eval" / "eval_report.json").read_text())
    assert report["frames_evaluated"] == 2
    assert (tmp_path / "eval" / "preview.mp4").exists()


def test_cli_train_accepts_bundle_override(tmp_path: Path) -> None:
    runner = CliRunner()
    bundle = package_arrays(
        np.random.default_rng(5).random((5, 10, 10), dtype=np.float32),
        np.random.default_rng(6).random((5, 10, 10), dtype=np.float32),
        tmp_path / "bundle",
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            "configs/nightjet_edge_v1_reco_s2_c16_f3.yaml",
            "--output-dir",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--bundle-uri",
            str(bundle.bundle_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "checkpoint.pt").exists()


def test_cli_stages_kubetorch_source(tmp_path: Path) -> None:
    runner = CliRunner()
    source = tmp_path / "repo"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'nightjet'\n", encoding="utf-8")
    (source / "src" / "nightjet").mkdir(parents=True)
    (source / "src" / "nightjet" / "__init__.py").write_text("", encoding="utf-8")
    bundle = source / "data" / "bundles" / "detail"
    bundle.mkdir(parents=True)
    (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "input_luma.npy").write_text("input", encoding="utf-8")
    (bundle / "target_luma.npy").write_text("target", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "kt-stage-source",
            "--source-dir",
            str(source),
            "--output-dir",
            str(tmp_path / "staged" / "nightjet"),
            "--bundle-dir",
            str(bundle),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output_dir"] == str(tmp_path / "staged" / "nightjet")
    assert payload["bundle_dirs"] == [
        str(tmp_path / "staged" / "nightjet" / "data" / "bundles" / "detail")
    ]


def test_cli_get_output_requires_run_id(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "kt-get-output",
            "--output-dir",
            str(tmp_path / "retrieved"),
        ],
    )

    assert result.exit_code != 0
    assert "Missing option" in result.output


def test_cli_enhance_writes_image_output(tmp_path: Path) -> None:
    runner = CliRunner()
    model_config = ModelConfig(
        name="identity",
        input_frames=3,
        base_channels=8,
        detail_channels=4,
        trunk_blocks=1,
        trunk_scale=2,
        residual_scale=0.45,
    )
    model = NightJetEdgeV1(model_config)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config.model_dump(mode="json"),
        },
        checkpoint,
    )
    input_image = tmp_path / "input.png"
    Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8), mode="RGB").save(input_image)
    output_image = tmp_path / "output.png"

    result = runner.invoke(
        app,
        [
            "enhance",
            "--input",
            str(input_image),
            "--output",
            str(output_image),
            "--weights",
            str(checkpoint),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["output"] == str(output_image)
    assert output_image.exists()


def test_cli_enhance_can_route_to_tensorrt_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    input_image = tmp_path / "input.png"
    output_image = tmp_path / "output.png"
    engine = tmp_path / "nightjet.plan"
    Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8), mode="RGB").save(input_image)
    engine.write_bytes(b"engine")

    class FakeEngineEnhancer:
        def __init__(self, engine_path: Path) -> None:
            self.engine_path = engine_path

        @classmethod
        def from_engine(
            cls,
            engine_path: Path,
            *,
            motion_budget: float | None = None,
        ) -> "FakeEngineEnhancer":
            return cls(engine_path)

        def enhance_image(
            self,
            input_image: Path,
            *,
            output_path: Path,
            side_by_side: bool,
            preserve_color: bool,
        ) -> Image.Image:
            assert input_image == input_image
            assert side_by_side is False
            assert preserve_color is False
            image = Image.new("RGB", (2, 2), color=(10, 10, 10))
            image.save(output_path)
            return image

    import nightjet.cli as nightjet_cli

    monkeypatch.setattr(nightjet_cli, "TensorRTNightJetEnhancer", FakeEngineEnhancer, raising=False)

    result = runner.invoke(
        app,
        [
            "enhance",
            "--input",
            str(input_image),
            "--output",
            str(output_image),
            "--engine",
            str(engine),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output"] == str(output_image)
    assert payload["engine"] == str(engine)
    assert output_image.exists()


def test_cli_enhance_passes_motion_budget_to_tensorrt_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _run_fake_tensorrt_video_enhance(
        tmp_path,
        monkeypatch,
        extra_args=["--motion-budget", "0.12"],
    )

    assert seen["motion_budget"] == 0.12


def test_cli_enhance_can_disable_motion_budget_for_tensorrt_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _run_fake_tensorrt_video_enhance(
        tmp_path,
        monkeypatch,
        extra_args=["--disable-motion-budget"],
    )

    assert seen["motion_budget"] is None


def test_cli_enhance_passes_motion_budget_to_pytorch_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _run_fake_pytorch_video_enhance(
        tmp_path,
        monkeypatch,
        extra_args=["--motion-budget", "0.12"],
    )

    assert seen["motion_budget"] == 0.12


def test_cli_enhance_can_disable_motion_budget_for_pytorch_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _run_fake_pytorch_video_enhance(
        tmp_path,
        monkeypatch,
        extra_args=["--disable-motion-budget"],
    )

    assert seen["motion_budget"] is None


def test_cli_serve_passes_motion_budget_to_runtime_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    engine = tmp_path / "nightjet.plan"
    engine.write_bytes(b"engine")
    seen: dict[str, Any] = {}

    class FakeMetrics:
        def to_json_dict(self) -> dict[str, bool]:
            return {"ready": True}

    def fake_run_runtime_server(config: Any) -> FakeMetrics:
        seen["engine_path"] = config.engine_path
        seen["motion_budget"] = config.motion_budget
        return FakeMetrics()

    import nightjet.cli as nightjet_cli

    monkeypatch.setattr(nightjet_cli, "run_runtime_server", fake_run_runtime_server)

    result = runner.invoke(
        app,
        [
            "serve",
            "--engine",
            str(engine),
            "--exit-after-max-frames",
            "--motion-budget",
            "0.12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == {"engine_path": engine, "motion_budget": 0.12}


def test_cli_serve_can_disable_motion_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    engine = tmp_path / "nightjet.plan"
    engine.write_bytes(b"engine")
    seen: dict[str, Any] = {}

    class FakeMetrics:
        def to_json_dict(self) -> dict[str, bool]:
            return {"ready": True}

    def fake_run_runtime_server(config: Any) -> FakeMetrics:
        seen["motion_budget"] = config.motion_budget
        return FakeMetrics()

    import nightjet.cli as nightjet_cli

    monkeypatch.setattr(nightjet_cli, "run_runtime_server", fake_run_runtime_server)

    result = runner.invoke(
        app,
        [
            "serve",
            "--engine",
            str(engine),
            "--exit-after-max-frames",
            "--disable-motion-budget",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["motion_budget"] is None


def test_cli_serve_passes_explicit_model_id_to_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nightjet.cli as nightjet_cli
    from nightjet.runtime.server import RuntimeMetrics

    engine = tmp_path / "nightjet.plan"
    engine.write_bytes(b"engine")
    seen: dict[str, Any] = {}

    def fake_run(config: Any) -> RuntimeMetrics:
        seen["model_id"] = config.model_id
        return RuntimeMetrics(model_id=config.model_id)

    monkeypatch.setattr(nightjet_cli, "run_runtime_server", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--engine",
            str(engine),
            "--source",
            "0",
            "--model-id",
            "nightjet-edge-v1-detail",
            "--exit-after-max-frames",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["model_id"] == "nightjet-edge-v1-detail"


def test_cli_serve_reads_model_id_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nightjet.cli as nightjet_cli
    from nightjet.runtime.server import RuntimeMetrics

    engine = tmp_path / "nightjet.plan"
    engine.write_bytes(b"engine")
    seen: dict[str, Any] = {}

    def fake_run(config: Any) -> RuntimeMetrics:
        seen["model_id"] = config.model_id
        return RuntimeMetrics(model_id=config.model_id)

    monkeypatch.setenv("NIGHTJET_MODEL_ID", "nightjet-edge-v1-detail")
    monkeypatch.setattr(nightjet_cli, "run_runtime_server", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--engine",
            str(engine),
            "--source",
            "0",
            "--exit-after-max-frames",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["model_id"] == "nightjet-edge-v1-detail"


def test_cli_build_engine_dry_run_uses_exported_input_name(tmp_path: Path) -> None:
    runner = CliRunner()
    onnx = tmp_path / "nightjet.onnx"
    engine = tmp_path / "nightjet.plan"
    onnx.write_bytes(b"onnx")

    result = runner.invoke(
        app,
        [
            "build-engine",
            "--onnx",
            str(onnx),
            "--output",
            str(engine),
            "--height",
            "32",
            "--width",
            "64",
            "--input-frames",
            "5",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    command = json.loads(result.output)["command"]
    assert "--minShapes=luma_window:1x5x32x64" not in command
    assert "--fp16" in command
    assert "--explicitBatch" not in command


def _run_fake_pytorch_video_enhance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_args: list[str],
) -> dict[str, Any]:
    runner = CliRunner()
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    checkpoint = tmp_path / "checkpoint.pt"
    input_video.write_bytes(b"video")
    checkpoint.write_bytes(b"checkpoint")
    seen: dict[str, Any] = {}

    class FakeEnhancer:
        @classmethod
        def from_checkpoint(
            cls,
            checkpoint_path: Path,
            *,
            device: str | None = None,
            motion_budget: float | None = None,
        ) -> "FakeEnhancer":
            seen["checkpoint"] = checkpoint_path
            seen["device"] = device
            seen["motion_budget"] = motion_budget
            return cls()

        def enhance_video(
            self,
            input_path: Path,
            output_path: Path,
            *,
            side_by_side: bool,
            preserve_color: bool,
            fps: float | None,
            show_progress: bool,
        ) -> Path:
            seen["input"] = input_path
            seen["side_by_side"] = side_by_side
            seen["preserve_color"] = preserve_color
            seen["fps"] = fps
            seen["show_progress"] = show_progress
            output_path.write_bytes(b"output")
            return output_path

    import nightjet.cli as nightjet_cli

    monkeypatch.setattr(nightjet_cli, "NightJetEnhancer", FakeEnhancer)
    result = runner.invoke(
        app,
        [
            "enhance",
            "--input",
            str(input_video),
            "--output",
            str(output_video),
            "--weights",
            str(checkpoint),
            *extra_args,
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output"] == str(output_video)
    assert payload["weights"] == str(checkpoint)
    assert seen["checkpoint"] == checkpoint
    assert seen["input"] == input_video
    return seen


def _run_fake_tensorrt_video_enhance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_args: list[str],
) -> dict[str, Any]:
    runner = CliRunner()
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    engine = tmp_path / "nightjet.plan"
    input_video.write_bytes(b"video")
    engine.write_bytes(b"engine")
    seen: dict[str, Any] = {}

    class FakeEngineEnhancer:
        @classmethod
        def from_engine(
            cls,
            engine_path: Path,
            **kwargs: Any,
        ) -> "FakeEngineEnhancer":
            seen["engine"] = engine_path
            seen["motion_budget"] = kwargs.get("motion_budget", "missing")
            return cls()

        def enhance_video(
            self,
            input_path: Path,
            output_path: Path,
            *,
            side_by_side: bool,
            preserve_color: bool,
            fps: float | None,
            show_progress: bool,
        ) -> Path:
            seen["input"] = input_path
            seen["side_by_side"] = side_by_side
            seen["preserve_color"] = preserve_color
            seen["fps"] = fps
            seen["show_progress"] = show_progress
            output_path.write_bytes(b"output")
            return output_path

    import nightjet.cli as nightjet_cli

    monkeypatch.setattr(nightjet_cli, "TensorRTNightJetEnhancer", FakeEngineEnhancer)
    result = runner.invoke(
        app,
        [
            "enhance",
            "--input",
            str(input_video),
            "--output",
            str(output_video),
            "--engine",
            str(engine),
            *extra_args,
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output"] == str(output_video)
    assert payload["engine"] == str(engine)
    assert seen["engine"] == engine
    assert seen["input"] == input_video
    return seen


def test_cli_build_engine_dynamic_shapes_dry_run_uses_exported_input_name(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    onnx = tmp_path / "nightjet.onnx"
    engine = tmp_path / "nightjet.plan"
    onnx.write_bytes(b"onnx")

    result = runner.invoke(
        app,
        [
            "build-engine",
            "--onnx",
            str(onnx),
            "--output",
            str(engine),
            "--height",
            "32",
            "--width",
            "64",
            "--input-frames",
            "5",
            "--dynamic-shapes",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    command = json.loads(result.output)["command"]
    assert "--minShapes=luma_window:1x5x32x64" in command
    assert "--fp16" in command
    assert "--explicitBatch" not in command
