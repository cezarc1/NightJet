from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw

from nightjet.config import ModelConfig
from nightjet.metrics import (
    clipping_rate,
    edge_mean,
    flat_region_noise,
    mae,
    psnr,
    temporal_diff_mean,
)
from nightjet.models import NightJetEdgeV1


@dataclass(frozen=True)
class EvalReport:
    teacher_name: str
    frames_evaluated: int
    metrics: dict[str, float]
    scores: dict[str, float]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "teacher_name": self.teacher_name,
            "frames_evaluated": self.frames_evaluated,
            "metrics": self.metrics,
            "scores": self.scores,
        }


@dataclass(frozen=True)
class EvalResult:
    report_path: Path
    csv_path: Path
    contact_sheet_path: Path
    preview_path: Path


def evaluate_arrays(
    *,
    raw: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    teacher_name: str,
) -> EvalReport:
    _validate_eval_shapes(raw, target, prediction)
    raw_edge = max(edge_mean(raw), 1e-6)
    target_edge = max(edge_mean(target), 1e-6)
    raw_temporal = max(temporal_diff_mean(raw), 1e-6)
    metrics = {
        "teacher_mae": mae(prediction, target),
        "teacher_psnr": psnr(prediction, target),
        "raw_teacher_mae": mae(raw, target),
        "raw_teacher_psnr": psnr(raw, target),
        "brightness_gain": float(prediction.mean() / max(float(raw.mean()), 1e-6)),
        "contrast_gain": float(prediction.std() / max(float(raw.std()), 1e-6)),
        "detail_gain": float(edge_mean(prediction) / raw_edge),
        "target_detail_ratio": float(edge_mean(prediction) / target_edge),
        "temporal_flicker_ratio": float(temporal_diff_mean(prediction) / raw_temporal),
        "flat_region_noise": flat_region_noise(prediction, raw),
        "clipping_rate": clipping_rate(prediction),
    }
    scores = _score_metrics(metrics)
    return EvalReport(
        teacher_name=teacher_name,
        frames_evaluated=int(raw.shape[0]),
        metrics=metrics,
        scores=scores,
    )


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    bundle_dir: Path,
    output_dir: Path,
    device: str | None,
    split_indices: list[int] | None = None,
    teacher_name: str = "teacher",
) -> EvalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(bundle_dir / "input_luma.npy")
    target = np.load(bundle_dir / "target_luma.npy")
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig.model_validate(checkpoint["model_config"])
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = NightJetEdgeV1(model_config).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    indices = split_indices if split_indices is not None else list(range(raw.shape[0]))
    prediction = _predict_luma(
        model, raw, indices=indices, input_frames=model_config.input_frames, device=torch_device
    )
    eval_raw = raw[indices]
    eval_target = target[indices]
    report = evaluate_arrays(
        raw=eval_raw,
        target=eval_target,
        prediction=prediction,
        teacher_name=teacher_name,
    )
    report_path = output_dir / "eval_report.json"
    csv_path = output_dir / "eval_metrics.csv"
    contact_sheet_path = output_dir / "contact_sheet.png"
    preview_path = output_dir / "preview.mp4"
    report_path.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_metrics_csv(csv_path, report)
    _write_contact_sheet(contact_sheet_path, eval_raw, eval_target, prediction)
    _write_preview(preview_path, eval_raw, eval_target, prediction)
    return EvalResult(
        report_path=report_path,
        csv_path=csv_path,
        contact_sheet_path=contact_sheet_path,
        preview_path=preview_path,
    )


def evaluate_baseline(
    *,
    method: str,
    bundle_dir: Path,
    output_dir: Path,
    split_indices: list[int] | None = None,
    teacher_name: str = "teacher",
) -> EvalResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(bundle_dir / "input_luma.npy")
    target = np.load(bundle_dir / "target_luma.npy")
    indices = split_indices if split_indices is not None else list(range(raw.shape[0]))
    eval_raw = raw[indices]
    eval_target = target[indices]
    if method == "teacher":
        prediction = np.ascontiguousarray(eval_target.astype(np.float32))
    else:
        prediction = _predict_baseline(raw, indices=indices, method=method)
    report = evaluate_arrays(
        raw=eval_raw,
        target=eval_target,
        prediction=prediction,
        teacher_name=teacher_name,
    )
    report_path = output_dir / "eval_report.json"
    csv_path = output_dir / "eval_metrics.csv"
    contact_sheet_path = output_dir / "contact_sheet.png"
    preview_path = output_dir / "preview.mp4"
    report_path.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_metrics_csv(csv_path, report)
    _write_contact_sheet(contact_sheet_path, eval_raw, eval_target, prediction)
    _write_preview(preview_path, eval_raw, eval_target, prediction)
    return EvalResult(
        report_path=report_path,
        csv_path=csv_path,
        contact_sheet_path=contact_sheet_path,
        preview_path=preview_path,
    )


def _predict_luma(
    model: NightJetEdgeV1,
    raw: np.ndarray,
    *,
    indices: list[int],
    input_frames: int,
    device: torch.device,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for index in indices:
            start = index - input_frames + 1
            window_indices = [
                min(max(start + offset, 0), raw.shape[0] - 1) for offset in range(input_frames)
            ]
            window = np.stack([raw[window_index] for window_index in window_indices], axis=0)
            tensor = torch.from_numpy(window[None, ...]).to(device=device, dtype=torch.float32)
            output = model(tensor).detach().cpu().numpy()[0, 0]
            predictions.append(np.ascontiguousarray(output.astype(np.float32)))
    return np.stack(predictions, axis=0)


def _predict_baseline(raw: np.ndarray, *, indices: list[int], method: str) -> np.ndarray:
    if method == "raw":
        return np.ascontiguousarray(raw[indices].astype(np.float32))
    if method != "classical-luma":
        raise ValueError(f"unsupported baseline method: {method}")

    predictions: list[np.ndarray] = []
    for index in indices:
        window_indices = [min(max(index + offset, 0), raw.shape[0] - 1) for offset in (-1, 0, 1)]
        denoised = np.mean(raw[window_indices], axis=0)
        stretched = np.arcsinh(8.0 * denoised) / np.arcsinh(8.0)
        brightened = np.power(np.clip(stretched, 0.0, 1.0), 0.72)
        predictions.append(np.ascontiguousarray(np.clip(brightened, 0.0, 1.0).astype(np.float32)))
    return np.stack(predictions, axis=0)


def _score_metrics(metrics: dict[str, float]) -> dict[str, float]:
    teacher_agreement = max(0.0, min(100.0, 100.0 - metrics["teacher_mae"] * 200.0))
    detail = max(0.0, min(100.0, metrics["detail_gain"] * 25.0))
    cleanliness = max(
        0.0,
        min(100.0, 100.0 - metrics["flat_region_noise"] * 250.0 - metrics["clipping_rate"] * 100.0),
    )
    temporal = max(
        0.0, min(100.0, 100.0 - max(0.0, metrics["temporal_flicker_ratio"] - 1.0) * 25.0)
    )
    detail_seeking = 0.35 * teacher_agreement + 0.35 * detail + 0.15 * cleanliness + 0.15 * temporal
    return {
        "teacher_agreement": round(teacher_agreement, 6),
        "detail": round(detail, 6),
        "cleanliness": round(cleanliness, 6),
        "temporal": round(temporal, 6),
        "detail_seeking_score": round(detail_seeking, 6),
    }


def _write_metrics_csv(path: Path, report: EvalReport) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "value"])
        for key, value in sorted(report.metrics.items()):
            writer.writerow([key, value])
        for key, value in sorted(report.scores.items()):
            writer.writerow([key, value])


def _write_contact_sheet(
    path: Path, raw: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> None:
    count = min(3, raw.shape[0])
    rows = []
    for index in range(count):
        rows.append(
            _labeled_row(raw[index], target[index], prediction[index], label=f"frame {index}")
        )
    sheet = np.concatenate(rows, axis=0)
    iio.imwrite(path, sheet)


def _write_preview(path: Path, raw: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    frames = [
        _labeled_row(raw[index], target[index], prediction[index], label=f"frame {index}")
        for index in range(raw.shape[0])
    ]
    iio.imwrite(path, np.stack(frames, axis=0), fps=30, macro_block_size=1)


def _labeled_row(
    raw: np.ndarray, target: np.ndarray, prediction: np.ndarray, *, label: str
) -> np.ndarray:
    panels = [
        ("raw", raw),
        ("teacher", target),
        ("nightjet", prediction),
    ]
    panel_images = [
        _panel_with_label(name, frame, prefix=label if index == 0 else "")
        for index, (name, frame) in enumerate(panels)
    ]
    return np.concatenate(panel_images, axis=1)


def _panel_with_label(name: str, frame: np.ndarray, *, prefix: str = "") -> np.ndarray:
    image = Image.fromarray(np.clip(frame * 255.0, 0, 255).astype(np.uint8), mode="L").convert(
        "RGB"
    )
    draw = ImageDraw.Draw(image)
    text = f"{prefix} {name}".strip()
    draw.rectangle((0, 0, min(image.width, 180), 16), fill=(0, 0, 0))
    draw.text((3, 2), text, fill=(255, 255, 255))
    return np.asarray(image)


def _validate_eval_shapes(raw: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    if raw.shape != target.shape or raw.shape != prediction.shape:
        raise ValueError(
            "raw, target, and prediction must have matching shapes: "
            f"{raw.shape}, {target.shape}, {prediction.shape}"
        )
    if raw.ndim != 3:
        raise ValueError("eval arrays must be TxHxW")
