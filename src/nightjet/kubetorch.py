from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any, cast

STAGED_SOURCE_RSYNC_FILTERS = (
    "--exclude='*.pyc' --exclude='__pycache__' --exclude='.venv' --exclude='.git'"
)
ARTIFACT_RSYNC_FILTERS = (
    "--include='*/' "
    "--include='*.json' "
    "--include='*.pt' "
    "--include='*.pth' "
    "--include='*.ckpt' "
    "--include='*.csv' "
    "--include='*.png' "
    "--include='*.mp4' "
    "--include='*.onnx' "
    "--exclude='*'"
)


def build_submit_payload(
    *,
    name: str,
    namespace: str,
    image: str,
    config_path: Path,
    source_dir: Path,
    output_dir: str,
    image_pull_secret: str,
    gpu_count: int,
    max_steps: int | None = None,
    bundle_uri: str | None = None,
    split_uri: str | None = None,
    split_name: str | None = None,
) -> dict[str, Any]:
    command = [
        "python",
        "-m",
        "nightjet.cli",
        "train",
        "--config",
        str(config_path),
        "--output-dir",
        output_dir,
        "--device",
        "cuda",
    ]
    if max_steps is not None:
        command.extend(["--max-steps", str(max_steps)])
    if bundle_uri is not None:
        command.extend(["--bundle-uri", bundle_uri])
    if split_uri is not None:
        command.extend(["--split-uri", split_uri])
    if split_name is not None:
        command.extend(["--split-name", split_name])
    return {
        "name": name,
        "namespace": namespace,
        "image": image,
        "sourceDir": str(source_dir),
        "command": " ".join(command),
        "env": {"PYTHONPATH": "src"},
        "imagePullSecrets": [image_pull_secret],
        "resources": {
            "limits": {"nvidia.com/gpu": str(gpu_count)},
            "requests": {"cpu": "2", "memory": "8Gi"},
        },
    }


def payload_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def submit_training_run(
    payload: dict[str, Any],
    *,
    submitter: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if submitter is None:
        try:
            runs_module = import_module("kubetorch.runs")
            submitter = cast(Callable[..., dict[str, Any]], runs_module.submit_batch_run)
        except ImportError as exc:
            raise RuntimeError(
                "KubeTorch client is not installed. Install the local KubeTorch fork in the "
                "training image or use --dry-run."
            ) from exc

    with _temporary_default_env("KT_RSYNC_FILTERS", STAGED_SOURCE_RSYNC_FILTERS):
        return submitter(
            command=shlex.split(str(payload["command"])),
            namespace=str(payload["namespace"]),
            source_dir=Path(str(payload["sourceDir"])),
            image=str(payload["image"]),
            intent=f"Train {payload['name']} with NightJet",
            resources=payload["resources"],
            env=dict(payload["env"]),
            labels={"app.kubernetes.io/name": "nightjet", "nightjet.dev/run": str(payload["name"])},
            image_pull_secrets=list(payload["imagePullSecrets"]),
            name=str(payload["name"]),
        )


def publish_run_output(
    output_dir: Path,
    *,
    artifact_name: str = "nightjet-output",
    run_id: str | None = None,
    namespace: str | None = None,
    putter: Callable[..., Any] | None = None,
    artifact_recorder: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    resolved_run_id = run_id or os.environ.get("KT_RUN_ID")
    if not resolved_run_id:
        return None
    resolved_namespace = namespace or os.environ.get("KT_NAMESPACE", "kubetorch")
    key = f"runs/{resolved_run_id}/artifacts/{artifact_name}"
    uri = f"kt://{resolved_namespace}/{key}"

    if putter is None:
        try:
            data_store_module = import_module("kubetorch.data_store")
            putter = cast(Callable[..., Any], data_store_module.put)
        except ImportError as exc:
            raise RuntimeError(
                "KubeTorch data store client is required to publish run output."
            ) from exc
    if artifact_recorder is None:
        try:
            runs_module = import_module("kubetorch.runs")
            artifact_recorder = cast(Callable[..., dict[str, Any]], runs_module.artifact)
        except ImportError as exc:
            raise RuntimeError("KubeTorch runs client is required to publish run output.") from exc

    with _temporary_default_env("KT_RSYNC_FILTERS", STAGED_SOURCE_RSYNC_FILTERS):
        putter(
            key=key,
            src=output_dir,
            contents=True,
            namespace=resolved_namespace,
            force=True,
            filter_options=ARTIFACT_RSYNC_FILTERS,
        )
    return artifact_recorder(
        name=artifact_name,
        uri=uri,
        kind="kt-data-store",
        metadata={"path": str(output_dir)},
        run_id=resolved_run_id,
    )


def retrieve_run_output(
    *,
    run_id: str,
    output_dir: Path,
    artifact_name: str = "nightjet-output",
    namespace: str = "kubetorch",
    getter: Callable[..., Any] | None = None,
) -> dict[str, str]:
    key = f"runs/{run_id}/artifacts/{artifact_name}"
    uri = f"kt://{namespace}/{key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if getter is None:
        try:
            data_store_module = import_module("kubetorch.data_store")
            getter = cast(Callable[..., Any], data_store_module.get)
        except ImportError as exc:
            raise RuntimeError(
                "KubeTorch data store client is required to retrieve run output."
            ) from exc

    with _temporary_default_env("KT_RSYNC_FILTERS", STAGED_SOURCE_RSYNC_FILTERS):
        getter(
            key=key,
            dest=output_dir,
            contents=True,
            namespace=namespace,
            force=True,
        )
    return {"artifact_uri": uri, "output_dir": str(output_dir)}


@contextmanager
def _temporary_default_env(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    if previous is None:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
