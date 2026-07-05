#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run nightjet report leaderboard \
  --title "NightJet Generated Candidate Leaderboard" \
  --output docs/reports/generated-candidate-leaderboard.md \
  --entry 'raw|hvi-darkir-test|outputs/eval/baseline-raw/hvi-darkir-test/eval_report.json' \
  --entry 'classical-luma|hvi-darkir-test|outputs/eval/baseline-classical-luma/hvi-darkir-test/eval_report.json' \
  --entry 'teacher|hvi-darkir-test|outputs/eval/baseline-teacher/hvi-darkir-test/eval_report.json' \
  --entry 'c16-f3|hvi-darkir-test|outputs/eval/edge-v1-reco-s2-c16-f3-detail-v1-5000/hvi-darkir-test/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f3-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f3-detail-v1-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f3-detail-v1-5000.onnx' \
  --entry 'c24-f3|hvi-darkir-test|outputs/eval/edge-v1-reco-s2-c24-f3-detail-v1-5000/hvi-darkir-test/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c24-f3-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c24-f3-detail-v1-5000/checkpoint.pt' \
  --entry 'c16-f5|hvi-darkir-test|outputs/eval/edge-v1-reco-s2-c16-f5-detail-v1-5000/hvi-darkir-test/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-detail-v1-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f5-detail-v1-5000.onnx' \
  --entry 'c16-f5-reddit10|hvi-darkir-test|outputs/eval/edge-v1-reco-s2-c16-f5-reddit10-5000/hvi-darkir-test/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-reddit10-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-reddit10-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f5-reddit10-5000.onnx' \
  --entry 'raw|reddit-detail-probe|outputs/eval/baseline-raw/reddit-detail-probe/eval_report.json' \
  --entry 'classical-luma|reddit-detail-probe|outputs/eval/baseline-classical-luma/reddit-detail-probe/eval_report.json' \
  --entry 'teacher|reddit-detail-probe|outputs/eval/baseline-teacher/reddit-detail-probe/eval_report.json' \
  --entry 'c16-f3|reddit-detail-probe|outputs/eval/edge-v1-reco-s2-c16-f3-detail-v1-5000/reddit-detail-probe/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f3-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f3-detail-v1-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f3-detail-v1-5000.onnx' \
  --entry 'c24-f3|reddit-detail-probe|outputs/eval/edge-v1-reco-s2-c24-f3-detail-v1-5000/reddit-detail-probe/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c24-f3-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c24-f3-detail-v1-5000/checkpoint.pt' \
  --entry 'c16-f5|reddit-detail-probe|outputs/eval/edge-v1-reco-s2-c16-f5-detail-v1-5000/reddit-detail-probe/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-detail-v1-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f5-detail-v1-5000.onnx' \
  --entry 'c16-f5-reddit10|reddit-detail-probe|outputs/eval/edge-v1-reco-s2-c16-f5-reddit10-5000/reddit-detail-probe/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-reddit10-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-reddit10-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f5-reddit10-5000.onnx' \
  --entry 'raw|hvi-darkir-reddit10-test|outputs/eval/baseline-raw/hvi-darkir-reddit10-test/eval_report.json' \
  --entry 'classical-luma|hvi-darkir-reddit10-test|outputs/eval/baseline-classical-luma/hvi-darkir-reddit10-test/eval_report.json' \
  --entry 'teacher|hvi-darkir-reddit10-test|outputs/eval/baseline-teacher/hvi-darkir-reddit10-test/eval_report.json' \
  --entry 'c16-f3|hvi-darkir-reddit10-test|outputs/eval/edge-v1-reco-s2-c16-f3-detail-v1-5000/hvi-darkir-reddit10-test/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f3-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f3-detail-v1-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f3-detail-v1-5000.onnx' \
  --entry 'c24-f3|hvi-darkir-reddit10-test|outputs/eval/edge-v1-reco-s2-c24-f3-detail-v1-5000/hvi-darkir-reddit10-test/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c24-f3-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c24-f3-detail-v1-5000/checkpoint.pt' \
  --entry 'c16-f5|hvi-darkir-reddit10-test|outputs/eval/edge-v1-reco-s2-c16-f5-detail-v1-5000/hvi-darkir-reddit10-test/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-detail-v1-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-detail-v1-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f5-detail-v1-5000.onnx' \
  --entry 'c16-f5-reddit10|hvi-darkir-reddit10-test|outputs/eval/edge-v1-reco-s2-c16-f5-reddit10-5000/hvi-darkir-reddit10-test/eval_report.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-reddit10-5000/run.json|outputs/retrieved/edge-v1-reco-s2-c16-f5-reddit10-5000/checkpoint.pt|outputs/onnx/edge-v1-reco-s2-c16-f5-reddit10-5000.onnx'
