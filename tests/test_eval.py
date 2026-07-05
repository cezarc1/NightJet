import json
from pathlib import Path

import numpy as np
import torch

from nightjet.config import ModelConfig
from nightjet.eval import evaluate_arrays, evaluate_baseline, evaluate_checkpoint
from nightjet.models import NightJetEdgeV1


def test_evaluate_arrays_scores_detail_noise_and_temporal_terms() -> None:
    raw = np.zeros((4, 8, 8), dtype=np.float32) + 0.05
    target = raw.copy()
    target[:, 2:6, 2:6] = 0.65
    prediction = target * 0.9

    report = evaluate_arrays(
        raw=raw,
        target=target,
        prediction=prediction,
        teacher_name="synthetic-teacher",
    )

    assert report.metrics["teacher_mae"] > 0
    assert report.metrics["detail_gain"] > 1
    assert report.metrics["clipping_rate"] == 0
    assert 0 <= report.scores["detail_seeking_score"] <= 100


def test_evaluate_checkpoint_writes_json_csv_and_contact_sheet(tmp_path: Path) -> None:
    raw = np.random.default_rng(1).random((5, 12, 12), dtype=np.float32) * 0.2
    target = np.clip(raw * 2.0, 0.0, 1.0)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.save(bundle / "input_luma.npy", raw)
    np.save(bundle / "target_luma.npy", target)
    model_config = ModelConfig(
        name="test",
        input_frames=3,
        base_channels=8,
        detail_channels=4,
        trunk_blocks=1,
        trunk_scale=2,
        residual_scale=0.45,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": NightJetEdgeV1(model_config).state_dict(),
            "model_config": model_config.model_dump(mode="json"),
        },
        checkpoint_path,
    )

    output_dir = tmp_path / "eval"
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        bundle_dir=bundle,
        output_dir=output_dir,
        device="cpu",
        split_indices=[0, 1, 2],
        teacher_name="synthetic-teacher",
    )

    assert result.report_path.exists()
    assert result.csv_path.exists()
    assert result.contact_sheet_path.exists()
    assert result.preview_path.exists()
    report = json.loads(result.report_path.read_text())
    assert report["teacher_name"] == "synthetic-teacher"
    assert report["frames_evaluated"] == 3
    assert "raw_teacher_mae" in report["metrics"]


def test_evaluate_baseline_writes_classical_comparator(tmp_path: Path) -> None:
    raw = np.zeros((5, 12, 12), dtype=np.float32) + 0.08
    raw[:, 4:8, 4:8] = 0.16
    target = np.clip(raw * 3.0, 0.0, 1.0)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.save(bundle / "input_luma.npy", raw)
    np.save(bundle / "target_luma.npy", target)

    result = evaluate_baseline(
        method="classical-luma",
        bundle_dir=bundle,
        output_dir=tmp_path / "baseline",
        split_indices=[1, 2, 3],
        teacher_name="synthetic-teacher",
    )

    assert result.report_path.exists()
    assert result.csv_path.exists()
    assert result.contact_sheet_path.exists()
    assert result.preview_path.exists()
    report = json.loads(result.report_path.read_text())
    assert report["teacher_name"] == "synthetic-teacher"
    assert report["frames_evaluated"] == 3
    assert report["metrics"]["brightness_gain"] > 1.0


def test_evaluate_baseline_supports_teacher_oracle(tmp_path: Path) -> None:
    raw = np.zeros((3, 8, 8), dtype=np.float32) + 0.05
    target = raw.copy()
    target[:, 2:6, 2:6] = 0.7
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.save(bundle / "input_luma.npy", raw)
    np.save(bundle / "target_luma.npy", target)

    result = evaluate_baseline(
        method="teacher",
        bundle_dir=bundle,
        output_dir=tmp_path / "teacher",
        teacher_name="synthetic-teacher",
    )

    report = json.loads(result.report_path.read_text())
    assert report["metrics"]["teacher_mae"] == 0.0
    assert report["metrics"]["teacher_psnr"] == 99.0
    assert report["scores"]["teacher_agreement"] == 100.0
