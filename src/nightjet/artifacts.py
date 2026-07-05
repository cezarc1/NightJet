from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any


def kt_uri(namespace: str, key: str) -> str:
    return f"kt://{namespace}/{key.strip('/')}"


def publish_output_dir(
    output_dir: Path,
    *,
    name: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    namespace = os.getenv("KT_NAMESPACE")
    run_id = os.getenv("KT_RUN_ID")
    if not namespace or not run_id:
        return False
    key = f"runs/{run_id}/nightjet/{output_dir.name}"
    try:
        runs = import_module("kubetorch.runs")
        data_store = import_module("kubetorch.data_store")
        data_store_client = data_store.DataStoreClient(namespace=namespace)

        data_store_client.put(
            key=key,
            src=output_dir,
            contents=True,
            force=True,
        )
        runs.artifact(
            name=name,
            uri=kt_uri(namespace, key),
            kind="nightjet-output-dir",
            metadata=metadata,
        )
        return True
    except Exception:
        return False
