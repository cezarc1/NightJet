from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 1337


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    input_frames: int = 3
    base_channels: int = 16
    detail_channels: int = 8
    trunk_blocks: int = 2
    trunk_scale: int = 2
    residual_scale: float = 0.45

    @field_validator("input_frames")
    @classmethod
    def validate_input_frames(cls, value: int) -> int:
        if value not in {1, 3, 5}:
            raise ValueError("input_frames must be one of 1, 3, or 5")
        return value

    @field_validator("base_channels", "detail_channels", "trunk_blocks", "trunk_scale")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("model integer fields must be positive")
        return value

    @field_validator("trunk_scale")
    @classmethod
    def validate_trunk_scale(cls, value: int) -> int:
        if value not in {1, 2, 4}:
            raise ValueError("trunk_scale must be one of 1, 2, or 4")
        return value

    @field_validator("residual_scale")
    @classmethod
    def validate_residual_scale(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("residual_scale must be positive")
        return value


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle_uri: str
    resolution: tuple[int, int] = (720, 1280)
    crop_size: int | None = 192
    input_frames: int = 3
    split_uri: str | None = None
    split_name: str = "train"

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or value[0] <= 0 or value[1] <= 0:
            raise ValueError("resolution must be (height, width)")
        return value

    @field_validator("input_frames")
    @classmethod
    def validate_input_frames(cls, value: int) -> int:
        if value not in {1, 3, 5}:
            raise ValueError("input_frames must be one of 1, 3, or 5")
        return value


class TrainingHyperparams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 8
    max_steps: int = 1500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    amp: bool = True
    warmup_steps: int = 0
    scheduler: str = "none"
    ema_decay: float | None = None

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, value: str) -> str:
        if value not in {"none", "cosine"}:
            raise ValueError("scheduler must be 'none' or 'cosine'")
        return value


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reconstruction_weight: float = 1.0
    edge_weight: float = 0.2
    brightness_weight: float = 0.35
    smoothness_weight: float = 0.0


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingHyperparams
    loss: LossConfig = Field(default_factory=LossConfig)


def load_config(path: Path) -> TrainingConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return TrainingConfig.model_validate(payload)


def config_to_json_dict(config: TrainingConfig) -> dict[str, Any]:
    return config.model_dump(mode="json")
