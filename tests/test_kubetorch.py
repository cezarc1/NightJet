import os
from pathlib import Path

from nightjet.kubetorch import (
    build_submit_payload,
    publish_run_output,
    retrieve_run_output,
    submit_training_run,
)


def test_kubetorch_dry_run_payload_contains_gpu_and_pull_secret() -> None:
    payload = build_submit_payload(
        name="nightjet-smoke",
        namespace="kubetorch",
        image="ghcr.io/cezarc1/nightjet-train:test",
        config_path=Path("configs/nightjet_edge_v1_reco_s2_c16_f3.yaml"),
        source_dir=Path("/repo/nightjet"),
        output_dir="outputs/nightjet-smoke",
        image_pull_secret="ghcr-pull-secret",
        gpu_count=1,
        max_steps=2,
        bundle_uri="data/bundles/detail",
        split_uri="data/bundles/detail/splits.json",
        split_name="train",
    )

    assert payload["namespace"] == "kubetorch"
    assert payload["imagePullSecrets"] == ["ghcr-pull-secret"]
    assert payload["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert payload["env"] == {"PYTHONPATH": "src"}
    assert "--max-steps 2" in payload["command"]
    assert "--bundle-uri data/bundles/detail" in payload["command"]
    assert "--split-uri data/bundles/detail/splits.json" in payload["command"]
    assert "--split-name train" in payload["command"]


def test_kubetorch_submit_uses_batch_run_contract() -> None:
    payload = build_submit_payload(
        name="nightjet-smoke",
        namespace="kubetorch",
        image="ghcr.io/cezarc1/nightjet-train:test",
        config_path=Path("configs/nightjet_edge_v1_reco_s2_c16_f3.yaml"),
        source_dir=Path("/repo/nightjet"),
        output_dir="outputs/nightjet-smoke",
        image_pull_secret="ghcr-pull-secret",
        gpu_count=1,
        max_steps=2,
    )
    calls = {}

    def fake_submitter(**kwargs):
        calls.update(kwargs)
        return {"run_id": "nightjet-smoke-abc123"}

    result = submit_training_run(payload, submitter=fake_submitter)

    assert result["run_id"] == "nightjet-smoke-abc123"
    assert calls["namespace"] == "kubetorch"
    assert calls["command"][0:3] == ["python", "-m", "nightjet.cli"]
    assert calls["env"] == {"PYTHONPATH": "src"}
    assert calls["image_pull_secrets"] == ["ghcr-pull-secret"]


def test_kubetorch_submit_uses_staged_source_rsync_filters(monkeypatch) -> None:
    monkeypatch.delenv("KT_RSYNC_FILTERS", raising=False)
    payload = build_submit_payload(
        name="nightjet-smoke",
        namespace="kubetorch",
        image="ghcr.io/cezarc1/nightjet-train:test",
        config_path=Path("configs/nightjet_edge_v1_reco_s2_c16_f3.yaml"),
        source_dir=Path("/repo/nightjet"),
        output_dir="outputs/nightjet-smoke",
        image_pull_secret="ghcr-pull-secret",
        gpu_count=1,
        max_steps=2,
    )
    captured_filters = {}

    def fake_submitter(**kwargs):
        captured_filters["value"] = os.environ.get("KT_RSYNC_FILTERS")
        return {"run_id": "nightjet-smoke-abc123"}

    submit_training_run(payload, submitter=fake_submitter)

    assert captured_filters["value"] == (
        "--exclude='*.pyc' --exclude='__pycache__' --exclude='.venv' --exclude='.git'"
    )
    assert "KT_RSYNC_FILTERS" not in os.environ


def test_publish_run_output_noops_outside_kubetorch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("KT_RUN_ID", raising=False)

    assert publish_run_output(tmp_path) is None


def test_publish_run_output_uploads_and_registers_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KT_RUN_ID", "run-123")
    monkeypatch.setenv("KT_NAMESPACE", "kubetorch")
    output_dir = tmp_path / "outputs" / "run"
    output_dir.mkdir(parents=True)
    (output_dir / "run.json").write_text("{}", encoding="utf-8")
    calls = {}

    def fake_putter(**kwargs):
        calls["put"] = kwargs
        calls["put_env"] = os.environ.get("KT_RSYNC_FILTERS")

    def fake_artifact_recorder(**kwargs):
        calls["artifact"] = kwargs
        return {"name": kwargs["name"], "uri": kwargs["uri"]}

    result = publish_run_output(
        output_dir,
        putter=fake_putter,
        artifact_recorder=fake_artifact_recorder,
    )

    assert result == {
        "name": "nightjet-output",
        "uri": "kt://kubetorch/runs/run-123/artifacts/nightjet-output",
    }
    assert calls["put"] == {
        "key": "runs/run-123/artifacts/nightjet-output",
        "src": output_dir,
        "contents": True,
        "namespace": "kubetorch",
        "force": True,
        "filter_options": (
            "--include='*/' --include='*.json' --include='*.pt' --include='*.pth' "
            "--include='*.ckpt' --include='*.csv' --include='*.png' --include='*.mp4' "
            "--include='*.onnx' --exclude='*'"
        ),
    }
    assert calls["put_env"] == (
        "--exclude='*.pyc' --exclude='__pycache__' --exclude='.venv' --exclude='.git'"
    )
    assert calls["artifact"]["name"] == "nightjet-output"
    assert calls["artifact"]["kind"] == "kt-data-store"
    assert "KT_RSYNC_FILTERS" not in os.environ


def test_retrieve_run_output_creates_destination_and_uses_filters(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("KT_RSYNC_FILTERS", raising=False)
    calls = {}
    output_dir = tmp_path / "retrieved" / "run"

    def fake_getter(**kwargs):
        calls["get"] = kwargs
        calls["get_env"] = os.environ.get("KT_RSYNC_FILTERS")

    result = retrieve_run_output(
        run_id="run-123",
        output_dir=output_dir,
        namespace="kubetorch",
        getter=fake_getter,
    )

    assert output_dir.exists()
    assert result == {
        "artifact_uri": "kt://kubetorch/runs/run-123/artifacts/nightjet-output",
        "output_dir": str(output_dir),
    }
    assert calls["get"] == {
        "key": "runs/run-123/artifacts/nightjet-output",
        "dest": output_dir,
        "contents": True,
        "namespace": "kubetorch",
        "force": True,
    }
    assert calls["get_env"] == (
        "--exclude='*.pyc' --exclude='__pycache__' --exclude='.venv' --exclude='.git'"
    )
    assert "KT_RSYNC_FILTERS" not in os.environ
