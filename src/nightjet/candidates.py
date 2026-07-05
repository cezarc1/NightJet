from __future__ import annotations

from pathlib import Path

from nightjet.config import TrainingConfig, load_config


def load_candidate_configs(path: Path) -> list[TrainingConfig]:
    configs = [load_config(config_path) for config_path in sorted(path.glob("*.yaml"))]
    if not configs:
        raise ValueError(f"no candidate configs found in {path}")
    return configs
