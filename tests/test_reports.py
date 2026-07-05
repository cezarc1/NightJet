import json
from pathlib import Path

from nightjet.reports import build_leaderboard_report, parse_report_entry


def _write_report(path: Path, *, score: float, psnr: float = 25.0) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "frames_evaluated": 3,
                "teacher_name": "synthetic",
                "metrics": {
                    "brightness_gain": 2.5,
                    "clipping_rate": 0.001,
                    "teacher_psnr": psnr,
                    "teacher_mae": 0.03,
                    "detail_gain": 2.0,
                    "flat_region_noise": 0.02,
                    "temporal_flicker_ratio": 1.5,
                },
                "scores": {
                    "detail_seeking_score": score,
                    "teacher_agreement": 94.0,
                    "cleanliness": 95.0,
                    "temporal": 87.5,
                },
            }
        ),
        encoding="utf-8",
    )


def test_parse_report_entry_accepts_optional_paths(tmp_path: Path) -> None:
    spec = parse_report_entry(
        f"c16-f5|hvi|{tmp_path / 'eval.json'}|{tmp_path / 'run.json'}|"
        f"{tmp_path / 'checkpoint.pt'}|{tmp_path / 'model.onnx'}"
    )

    assert spec.name == "c16-f5"
    assert spec.split == "hvi"
    assert spec.report_path == tmp_path / "eval.json"
    assert spec.run_path == tmp_path / "run.json"
    assert spec.checkpoint_path == tmp_path / "checkpoint.pt"
    assert spec.onnx_path == tmp_path / "model.onnx"


def test_build_leaderboard_report_promotes_best_learned_candidate(tmp_path: Path) -> None:
    raw_report = tmp_path / "raw" / "eval_report.json"
    classical_report = tmp_path / "classical" / "eval_report.json"
    teacher_report = tmp_path / "teacher" / "eval_report.json"
    student_a_report = tmp_path / "student-a" / "eval_report.json"
    student_b_report = tmp_path / "student-b" / "eval_report.json"
    run_path = tmp_path / "run.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    onnx_path = tmp_path / "model.onnx"

    _write_report(raw_report, score=60.0, psnr=15.0)
    _write_report(classical_report, score=90.0, psnr=22.0)
    _write_report(teacher_report, score=100.0, psnr=99.0)
    _write_report(student_a_report, score=70.0, psnr=26.0)
    _write_report(student_b_report, score=80.0, psnr=25.5)
    run_path.write_text(
        json.dumps({"elapsed_seconds": 12.34, "final_step": 5000, "final_loss": 0.0123}),
        encoding="utf-8",
    )
    checkpoint_path.write_bytes(b"checkpoint")
    onnx_path.write_bytes(b"onnx")

    markdown = build_leaderboard_report(
        [
            parse_report_entry(f"raw|hvi|{raw_report}"),
            parse_report_entry(f"classical-luma|hvi|{classical_report}"),
            parse_report_entry(f"teacher|hvi|{teacher_report}"),
            parse_report_entry(f"c16-f3|hvi|{student_a_report}"),
            parse_report_entry(
                f"c16-f5|hvi|{student_b_report}|{run_path}|{checkpoint_path}|{onnx_path}"
            ),
        ],
        title="Synthetic Leaderboard",
    )

    assert "# Synthetic Leaderboard" in markdown
    assert "| `teacher` | 99.0000" in markdown
    assert "Brightness gain" in markdown
    assert "Clipping" in markdown
    assert "| `c16-f5` | 12.34 s | 5000 | 0.01230 | 10 B | yes |" in markdown
    assert "Current learned-model promotion: `c16-f5`" in markdown
    assert "Best overall score: `teacher`" in markdown
