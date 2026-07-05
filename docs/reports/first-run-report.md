# NightJet First Run Report

Date: 2026-06-18

## Scope

This report captures the first real NightJet Edge candidate run after the
[KubeTorch](https://github.com/cezarc1/kubetorch) staging, artifact, and
retrieval path was hardened.

Candidate:

- `edge-v1-reco-s2-c16-f3`
- 3 luma input frames
- base channels: 16
- detail channels: 8
- trunk blocks: 2
- training target: `hvi-darkir-60f-low-light-720p`

Run:

- [KubeTorch](https://github.com/cezarc1/kubetorch) run id:
  `edge-v1-reco-s2-c16-f3-detail-v1-5000-91cbe2ee`
- image: `ghcr.io/cezarc1/nightjet-train:0.2.0-201b3c3`
- source branch: `nightjet-campaign-eval`
- latest source commit after harness fixes: `63ab298`
- GPU node: `cezar-4090`
- steps: 5000
- elapsed training time: 51.556105 seconds
- final loss: 0.02969243749976158
- initial loss: 0.14884620904922485

Artifacts:

- retrieved checkpoint:
  `outputs/retrieved/edge-v1-reco-s2-c16-f3-detail-v1-5000/checkpoint.pt`
- retrieved metrics:
  `outputs/retrieved/edge-v1-reco-s2-c16-f3-detail-v1-5000/run.json`
- ONNX:
  `outputs/onnx/edge-v1-reco-s2-c16-f3-detail-v1-5000.onnx`
- ONNX check: passed, 39 nodes, input `luma_window`, output `enhanced_luma`

## Evaluation

### HVI/DarkIR Held-Out Test

Path:
`outputs/eval/edge-v1-reco-s2-c16-f3-detail-v1-5000/hvi-darkir-test`

Frames evaluated: 9

| Metric | Value |
| --- | ---: |
| teacher PSNR | 26.5800 |
| teacher MAE | 0.02861 |
| detail gain | 1.9954 |
| brightness gain | 4.8870 |
| contrast gain | 2.4322 |
| flat-region noise | 0.01728 |
| clipping rate | 0.000156 |
| temporal flicker ratio | 2.4381 |
| detail-seeking score | 74.4134 |
| teacher agreement | 94.2777 |
| cleanliness | 95.6642 |
| temporal score | 64.0474 |

### ReDDiT Detail Probe

Path:
`outputs/eval/edge-v1-reco-s2-c16-f3-detail-v1-5000/reddit-detail-probe`

Frames evaluated: 2

| Metric | Value |
| --- | ---: |
| teacher PSNR | 26.3566 |
| teacher MAE | 0.03230 |
| detail gain | 2.0112 |
| brightness gain | 4.9473 |
| contrast gain | 2.3956 |
| flat-region noise | 0.04376 |
| clipping rate | 0.000146 |
| temporal flicker ratio | 1.9724 |
| detail-seeking score | 75.0480 |
| teacher agreement | 93.5407 |
| cleanliness | 89.0464 |
| temporal score | 75.6894 |

## Findings

- The first trainable Edge candidate fits the intended tiny-model profile and
  trains quickly on the 4090.
- The model tracks the HVI/DarkIR teacher closely after 5000 steps.
- The ReDDiT probe is also reasonable despite not being the main 60-frame
  target, which suggests the distilled output is not purely memorizing one
  teacher style.
- Temporal behavior remains the main weakness. The HVI/DarkIR test reports a
  temporal flicker ratio of `2.4381`.
- The naive training image is too large for fast iteration. First pull of
  `ghcr.io/cezarc1/nightjet-train:0.2.0-201b3c3` took roughly eight minutes on
  the 4090 node because it was built from `python:3.12.12-slim` plus CUDA/Torch
  wheels.
- [KubeTorch](https://github.com/cezarc1/kubetorch) rsync behavior was the
  biggest harness gotcha. Both upload and download paths inherit repo ignore
  rules unless `KT_RSYNC_FILTERS` is overridden. This affected staged bundles,
  checkpoint artifact upload, and checkpoint retrieval.

## Next Steps

1. Open the HVI/DarkIR and ReDDiT contact sheets for subjective inspection.
2. Train `edge_v1_reco_s2_c24_f3` and `edge_v1_reco_s2_c16_f5`.
3. Add a leaderboard generator that compares smoke, c16-f3, c24-f3, c16-f5,
   raw/classical baselines, and teacher targets in one CSV/Markdown table.
4. Build a slimmer training image based on a CUDA/PyTorch image with the
   [KubeTorch](https://github.com/cezarc1/kubetorch) client added.
5. Move the ONNX file to the Jetson path and build TensorRT FP16 there; do not
   claim Jetson FPS until the Orin run measures it.
