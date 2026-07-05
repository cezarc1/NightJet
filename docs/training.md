# Training

NightJet training turns low-light luma frame bundles plus teacher-target luma
bundles into tiny `NightJetEdgeV1` student checkpoints.

## Local Smoke Train

```bash
uv run nightjet train \
  --config configs/candidates/edge_v1_reco_s2_c16_f3.yaml \
  --output-dir outputs/smoke/edge-v1-reco-s2-c16-f3 \
  --device cpu \
  --max-steps 2
```

Outputs go under `outputs/`, which is intentionally ignored.

## Data Bundles

NightJet expects two NumPy arrays:

- `input_luma.npy`: raw low-light luma frames shaped `N x H x W`
- `target_luma.npy`: teacher target luma frames shaped `N x H x W`

Package arrays:

```bash
uv run nightjet data package \
  --input-luma data/local/input_luma.npy \
  --target-luma data/local/target_luma.npy \
  --output-dir data/bundles/example
```

Build train/val/test splits:

```bash
uv run nightjet data split \
  --bundle-dir data/bundles/example \
  --output data/bundles/example/splits.json \
  --train 0.70 \
  --val 0.15 \
  --test 0.15 \
  --seed 1337
```

Datasets and teacher bundles are not committed.

## KubeTorch Runs

[KubeTorch](https://github.com/cezarc1/kubetorch) is the homelab GPU batch
runner used for longer training jobs. It is not required for public inference.

Stage the source and selected bundles:

```bash
uv run nightjet kt-stage-source \
  --output-dir outputs/kubetorch-source/nightjet \
  --bundle-dir data/bundles/hvi-darkir-60f-low-light-720p
```

Dry-run a training payload:

```bash
uv run nightjet kt-submit-train \
  --name nightjet-edge-v1-smoke \
  --image ghcr.io/cezarc1/nightjet-train:<immutable-tag> \
  --config configs/candidates/edge_v1_reco_s2_c16_f5.yaml \
  --source-dir outputs/kubetorch-source/nightjet \
  --bundle-uri data/bundles/hvi-darkir-60f-low-light-720p \
  --split-uri data/bundles/hvi-darkir-60f-low-light-720p/splits.json \
  --split-name train \
  --max-steps 5000 \
  --dry-run
```

See [docs/kubetorch.md](kubetorch.md) for the longer
[KubeTorch](https://github.com/cezarc1/kubetorch) handoff and
[docs/reports/detail-seeking-campaign.md](reports/detail-seeking-campaign.md)
for the campaign runbook that produced the public v0.1 candidates.

## Evaluation

```bash
uv run nightjet eval checkpoint \
  --checkpoint outputs/runs/edge-v1-reco-s2-c16-f5/checkpoint.pt \
  --bundle-dir data/bundles/hvi-darkir-60f-low-light-720p \
  --output-dir outputs/eval/edge-v1-reco-s2-c16-f5/hvi-darkir-test \
  --split-json data/bundles/hvi-darkir-60f-low-light-720p/splits.json \
  --split test \
  --teacher-name hvi-darkir-60f
```

Evaluation writes JSON metrics, CSV metrics, contact sheets, and preview video
under `outputs/eval/`.
