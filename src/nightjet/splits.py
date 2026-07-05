from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitSpec:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 1337

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train + val + test must equal 1.0")
        if min(self.train, self.val, self.test) <= 0:
            raise ValueError("split fractions must be positive")


@dataclass(frozen=True)
class FrameSplits:
    train: list[int]
    val: list[int]
    test: list[int]

    def to_json_dict(self) -> dict[str, list[int]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def build_frame_splits(frame_count: int, spec: SplitSpec) -> FrameSplits:
    if frame_count < 3:
        raise ValueError("frame_count must be at least 3")
    rng = np.random.default_rng(spec.seed)
    indices = np.arange(frame_count)
    rng.shuffle(indices)
    train_count = round(frame_count * spec.train)
    val_count = round(frame_count * spec.val)
    if train_count <= 0 or val_count <= 0 or train_count + val_count >= frame_count:
        raise ValueError("split fractions produce an empty split")
    train = sorted(indices[:train_count].astype(int).tolist())
    val = sorted(indices[train_count : train_count + val_count].astype(int).tolist())
    test = sorted(indices[train_count + val_count :].astype(int).tolist())
    return FrameSplits(train=train, val=val, test=test)


def load_split_indices(path: Path, split_name: str) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or split_name not in payload:
        raise ValueError(f"split JSON must contain split {split_name!r}")
    values = payload[split_name]
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"split {split_name!r} must be a list of integers")
    if not values:
        raise ValueError(f"split {split_name!r} must not be empty")
    return values
