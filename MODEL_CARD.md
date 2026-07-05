# NightJet Edge v1 Model Card

## Model Summary

NightJet Edge v1 is a tiny temporal luma enhancement model for passive
low-light imagery. It accepts a causal 5-frame luma window and predicts an
enhanced luma frame for the latest input.

Public artifacts:

- Default: `weights/nightjet-edge-v1.pt`
- Detail variant: `weights/nightjet-edge-v1-detail.pt`
- Matching ONNX files: `weights/nightjet-edge-v1.onnx`,
  `weights/nightjet-edge-v1-detail.onnx`
- Hugging Face default model repo:
  [`ggamecrazy/nightjet-edge-v1`](https://huggingface.co/ggamecrazy/nightjet-edge-v1)
- Interactive demo:
  [`ggamecrazy/nightjet`](https://huggingface.co/spaces/ggamecrazy/nightjet)

## Intended Use

Use NightJet for local experiments, demos, and edge-runtime prototyping where a
small luma-only low-light enhancer is more useful than a large RGB restoration
model. The default weight is the recommended public starting point.

NightJet is not intended for safety-critical perception, surveillance
identification, forensic enhancement, medical use, or any workflow where
hallucinated or unstable detail could cause harm.

## Architecture

The model is `NightJetEdgeV1`, a residual convolutional network with:

- 5 input luma frames
- 16 base channels
- 8 detail channels
- 2 depthwise-separable trunk blocks
- trunk scale 2
- bounded residual scale 0.45

The output is:

```text
clamp(latest_luma + residual_scale * tanh(residual), 0, 1)
```

See [docs/architecture.md](docs/architecture.md) for details.

## Training Data And Teachers

The public weights were trained on local low-light luma bundles derived from a
passive night-vision camera path. Targets were distilled from stronger teacher
enhancement outputs, including HVI-CIDNet, DarkIR, and ReDDiT-derived detail
targets.

Datasets, teacher outputs, and raw clips are not included in this public repo.
The committed checkpoints are inference-only payloads derived from local
[KubeTorch](https://github.com/cezarc1/kubetorch) run outputs.

## Evaluation Snapshot

Metrics below come from the committed `weights/manifest.json` and tiny held-out
eval splits. Treat them as model-selection evidence, not broad benchmark claims.

| Model | Split | Teacher MAE | Detail gain | Temporal flicker ratio | Detail score | Temporal score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `nightjet-edge-v1` | ReDDiT detail probe | `0.032482` | `2.079988` | `1.759828` | `51.999704` | `81.004304` |
| `nightjet-edge-v1-detail` | ReDDiT detail probe | `0.032337` | `3.055698` | `3.497916` | `76.392455` | `37.55211` |

The detail variant is sharper, but the measured flicker ratio is much worse.
Use the default model unless the extra detail is worth that temporal risk.

## Limitations

- Luma-only enhancement; color is either grayscale RGB or original chroma
  recombined with enhanced luma.
- Small eval splits and teacher-derived targets.
- Detail-seeking behavior can amplify noise and flicker.
- No guarantee of robustness across sensors, lenses, compression pipelines, or
  lighting regimes.
- TensorRT performance must be measured on the target Jetson runtime.

## License And Provenance Notes

The repository source code is MIT licensed. Model weights have separate
provenance considerations because they were distilled from teacher-model outputs
and local camera data. Review upstream teacher-model licenses and data rights
before redistributing, commercializing, or embedding the weights in a product.
The Hugging Face model repo uses a separate `WEIGHTS_LICENSE.md` file to keep
that distinction explicit.
