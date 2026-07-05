from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_NAMES = {"raw", "classical-luma", "teacher"}


@dataclass(frozen=True)
class ReportEntrySpec:
    name: str
    split: str
    report_path: Path
    run_path: Path | None = None
    checkpoint_path: Path | None = None
    onnx_path: Path | None = None


@dataclass(frozen=True)
class ReportEntry:
    spec: ReportEntrySpec
    frames_evaluated: int
    metrics: dict[str, float]
    scores: dict[str, float]
    run: dict[str, Any] | None = None


def parse_report_entry(value: str) -> ReportEntrySpec:
    parts = value.split("|")
    if len(parts) not in {3, 4, 5, 6}:
        raise ValueError(
            "report entries must be NAME|SPLIT|REPORT_JSON[|RUN_JSON][|CHECKPOINT][|ONNX]"
        )
    name, split, report_path, *optional_paths = parts
    if not name or not split or not report_path:
        raise ValueError("report entry name, split, and report path are required")
    padded = [*optional_paths, None, None, None]
    return ReportEntrySpec(
        name=name,
        split=split,
        report_path=Path(report_path),
        run_path=Path(padded[0]) if padded[0] else None,
        checkpoint_path=Path(padded[1]) if padded[1] else None,
        onnx_path=Path(padded[2]) if padded[2] else None,
    )


def build_leaderboard_report(
    specs: list[ReportEntrySpec],
    *,
    title: str = "NightJet Candidate Leaderboard",
) -> str:
    entries = [_load_entry(spec) for spec in specs]
    lines = [
        f"# {title}",
        "",
        "This report is generated from explicit eval and run artifacts. It should be",
        "regenerated when any source eval JSON changes.",
        "",
        "## Training Runs",
        "",
    ]
    lines.extend(_training_table(entries))
    lines.append("")
    for split in _ordered_splits(entries):
        split_entries = [entry for entry in entries if entry.spec.split == split]
        lines.extend(_split_section(split, split_entries))
        lines.append("")
    lines.extend(_promotion_section(entries))
    return "\n".join(lines).rstrip() + "\n"


def write_leaderboard_report(
    specs: list[ReportEntrySpec],
    *,
    output_path: Path,
    title: str = "NightJet Candidate Leaderboard",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_leaderboard_report(specs, title=title), encoding="utf-8")
    return output_path


def _load_entry(spec: ReportEntrySpec) -> ReportEntry:
    report = json.loads(spec.report_path.read_text(encoding="utf-8"))
    run = json.loads(spec.run_path.read_text(encoding="utf-8")) if spec.run_path else None
    return ReportEntry(
        spec=spec,
        frames_evaluated=int(report["frames_evaluated"]),
        metrics={key: float(value) for key, value in report["metrics"].items()},
        scores={key: float(value) for key, value in report["scores"].items()},
        run=run,
    )


def _training_table(entries: list[ReportEntry]) -> list[str]:
    run_entries: dict[str, ReportEntry] = {}
    for entry in entries:
        if entry.run is not None and entry.spec.name not in run_entries:
            run_entries[entry.spec.name] = entry
    if not run_entries:
        return ["No training run metadata was provided."]
    lines = [
        "| Candidate | Train time | Final step | Final loss | Checkpoint | ONNX |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, entry in run_entries.items():
        run = entry.run or {}
        checkpoint_size = _file_size(entry.spec.checkpoint_path)
        onnx = "yes" if entry.spec.onnx_path is not None and entry.spec.onnx_path.exists() else "no"
        lines.append(
            f"| `{name}` | {_format_seconds(run.get('elapsed_seconds'))} | "
            f"{_format_int(run.get('final_step'))} | {_format_float(run.get('final_loss'), 5)} | "
            f"{checkpoint_size} | {onnx} |"
        )
    return lines


def _split_section(split: str, entries: list[ReportEntry]) -> list[str]:
    frames = entries[0].frames_evaluated if entries else 0
    table_header = (
        "| Method | Teacher PSNR | Teacher MAE | Brightness gain | Detail gain | "
        "Flat noise | Clipping | Flicker ratio | Score | Teacher agreement | "
        "Cleanliness | Temporal |"
    )
    lines = [
        f"## {split}",
        "",
        f"Frames evaluated: {frames}",
        "",
        table_header,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        metrics = entry.metrics
        scores = entry.scores
        lines.append(
            f"| `{entry.spec.name}` | {_format_float(metrics['teacher_psnr'])} | "
            f"{_format_float(metrics['teacher_mae'], 5)} | "
            f"{_format_float(metrics['brightness_gain'])} | "
            f"{_format_float(metrics['detail_gain'])} | "
            f"{_format_float(metrics['flat_region_noise'], 5)} | "
            f"{_format_float(metrics['clipping_rate'], 6)} | "
            f"{_format_float(metrics['temporal_flicker_ratio'])} | "
            f"{_format_float(scores['detail_seeking_score'])} | "
            f"{_format_float(scores['teacher_agreement'])} | "
            f"{_format_float(scores['cleanliness'])} | "
            f"{_format_float(scores['temporal'])} |"
        )
    return lines


def _promotion_section(entries: list[ReportEntry]) -> list[str]:
    overall = max(entries, key=lambda entry: entry.scores["detail_seeking_score"])
    learned = [entry for entry in entries if entry.spec.name not in BASELINE_NAMES]
    if not learned:
        return [
            "## Promotion",
            "",
            f"Best overall score: `{overall.spec.name}`.",
            "",
            "No learned candidates were provided.",
        ]
    learned_scores: dict[str, list[float]] = {}
    for entry in learned:
        learned_scores.setdefault(entry.spec.name, []).append(entry.scores["detail_seeking_score"])
    promoted_name, promoted_scores = max(
        learned_scores.items(), key=lambda item: sum(item[1]) / len(item[1])
    )
    promoted_score = sum(promoted_scores) / len(promoted_scores)
    return [
        "## Promotion",
        "",
        f"Best overall score: `{overall.spec.name}`.",
        f"Current learned-model promotion: `{promoted_name}`.",
        f"Mean learned score across provided splits: {_format_float(promoted_score)}.",
    ]


def _ordered_splits(entries: list[ReportEntry]) -> list[str]:
    splits: list[str] = []
    for entry in entries:
        if entry.spec.split not in splits:
            splits.append(entry.spec.split)
    return splits


def _format_float(value: float | int | str | None, precision: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{precision}f}"


def _format_int(value: int | str | None) -> str:
    if value is None:
        return ""
    return str(int(value))


def _format_seconds(value: float | int | str | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f} s"


def _file_size(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return f"{path.stat().st_size:,} B"
