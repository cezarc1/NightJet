from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

SKIPPED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ty",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "checkpoints",
    "data",
    "dist",
    "outputs",
    "wandb",
}


@dataclass(frozen=True)
class StageResult:
    output_dir: Path
    bundle_dirs: list[Path]
    files_copied: int
    bytes_copied: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "bundle_dirs": [str(path) for path in self.bundle_dirs],
            "files_copied": self.files_copied,
            "bytes_copied": self.bytes_copied,
        }


def stage_kubetorch_source(
    *,
    source_dir: Path,
    output_dir: Path,
    bundle_dirs: list[Path] | None = None,
) -> StageResult:
    source_root = source_dir.resolve()
    destination_root = output_dir.resolve()
    if source_root == destination_root:
        raise ValueError("output_dir must be different from source_dir")
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"source_dir does not exist or is not a directory: {source_dir}")

    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True)

    stats = _CopyStats()
    _copy_source_tree(source_root, destination_root, stats)

    staged_bundles: list[Path] = []
    for bundle_dir in bundle_dirs or []:
        staged_bundles.append(_copy_bundle(bundle_dir.resolve(), destination_root, stats))

    return StageResult(
        output_dir=output_dir,
        bundle_dirs=staged_bundles,
        files_copied=stats.files_copied,
        bytes_copied=stats.bytes_copied,
    )


@dataclass
class _CopyStats:
    files_copied: int = 0
    bytes_copied: int = 0


def _copy_source_tree(source_root: Path, destination_root: Path, stats: _CopyStats) -> None:
    for source_path in sorted(source_root.rglob("*")):
        if _should_skip(source_path, source_root, destination_root):
            continue
        relative_path = source_path.relative_to(source_root)
        destination_path = destination_root / relative_path
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            _copy_file(source_path, destination_path, stats)


def _copy_bundle(bundle_dir: Path, destination_root: Path, stats: _CopyStats) -> Path:
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        raise FileNotFoundError(f"bundle_dir does not exist or is not a directory: {bundle_dir}")
    required_files = ["bundle_manifest.json", "input_luma.npy", "target_luma.npy"]
    missing = [name for name in required_files if not (bundle_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"bundle_dir is missing required files: {', '.join(missing)}")

    destination_bundle = destination_root / "data" / "bundles" / bundle_dir.name
    destination_bundle.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(bundle_dir.rglob("*")):
        if _has_skipped_part(source_path.relative_to(bundle_dir)):
            continue
        relative_path = source_path.relative_to(bundle_dir)
        destination_path = destination_bundle / relative_path
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            _copy_file(source_path, destination_path, stats)
    return destination_bundle


def _copy_file(source_path: Path, destination_path: Path, stats: _CopyStats) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    stats.files_copied += 1
    stats.bytes_copied += source_path.stat().st_size


def _should_skip(source_path: Path, source_root: Path, destination_root: Path) -> bool:
    if _is_relative_to(source_path, destination_root):
        return True
    relative_path = source_path.relative_to(source_root)
    return _has_skipped_part(relative_path)


def _has_skipped_part(relative_path: Path) -> bool:
    return any(part in SKIPPED_DIR_NAMES for part in relative_path.parts)


def _is_relative_to(path: Path, maybe_parent: Path) -> bool:
    try:
        path.resolve().relative_to(maybe_parent)
    except ValueError:
        return False
    return True
