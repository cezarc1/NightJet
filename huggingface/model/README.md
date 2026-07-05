---
language:
  - en
license: other
license_name: nightjet-research-weights
license_link: https://huggingface.co/ggamecrazy/nightjet-edge-v1/blob/main/WEIGHTS_LICENSE.md
library_name: pytorch
pipeline_tag: image-to-image
tags:
  - low-light-enhancement
  - night-vision
  - jetson
  - edge-ai
  - pytorch
  - onnx
widget:
  - src: examples/input.jpg
    example_title: Low-light sample
---

# NightJet Edge v1

NightJet Edge v1 is a tiny temporal luma enhancement model for passive
low-light imagery. It accepts a causal 5-frame luma window and predicts an
enhanced luma frame for the latest input.

The easiest way to try it is the
[NightJet Space](https://huggingface.co/spaces/ggamecrazy/nightjet).

> The canonical developer repo is
> [github.com/cezarc1/nightjet](https://github.com/cezarc1/nightjet). Use
> Hugging Face for the hosted demo and model artifacts; use GitHub for the
> installable CLI/API, source code, training/export docs, and Jetson/KubeJet
> deployment path.

## Files

| File | Purpose |
| --- | --- |
| `nightjet-edge-v1.pt` | Inference-only PyTorch checkpoint |
| `nightjet-edge-v1.onnx` | Fixed-shape ONNX exchange artifact |
| `manifest.json` | Architecture, input/output contract, provenance, hashes, and eval notes |
| `examples/input.jpg` | Sample low-light input |
| `examples/output.jpg` | Output from the default model |
| `examples/comparison.jpg` | Before/after strip |

The sharper `nightjet-edge-v1-detail` variant is intentionally not included in
this first Hugging Face repo. It has more visible detail but a much worse
temporal flicker score, so it should be published as a separate model repo if
we decide to expose it later.

## Input And Output

- Input tensor: `luma_window`
- Input shape: `1 x 5 x H x W`
- Input dtype/range: `float32`, `[0, 1]`
- Temporal order: causal, latest frame last
- Output tensor: `enhanced_luma`
- Output shape: `1 x 1 x H x W`
- Output dtype/range: `float32`, `[0, 1]`

For normal image use, install NightJet and let the public API handle RGB/luma
conversion:

```python
from pathlib import Path

from nightjet.inference import NightJetEnhancer

enhancer = NightJetEnhancer.from_checkpoint(Path("nightjet-edge-v1.pt"))
enhancer.enhance_image(
    Path("examples/input.jpg"),
    output_path=Path("nightjet-output.png"),
    preserve_color=False,
)
```

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

## Training Data And Teachers

The public weights were trained on local low-light luma bundles derived from a
passive night-vision camera path. Targets were distilled from stronger teacher
enhancement outputs, including HVI-CIDNet, DarkIR, and ReDDiT-derived detail
targets.

Datasets, teacher outputs, and raw clips are not included. The checkpoint is an
inference-only payload derived from local KubeTorch run outputs.

## Evaluation Snapshot

Metrics below come from the committed `manifest.json` and tiny held-out eval
splits. Treat them as model-selection evidence, not broad benchmark claims.

| Split | Teacher MAE | Detail gain | Temporal flicker ratio | Detail score | Temporal score |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReDDiT detail probe | `0.032482` | `2.079988` | `1.759828` | `51.999704` | `81.004304` |

## Limitations

- Luma-only enhancement; color is either grayscale RGB or original chroma
  recombined with enhanced luma.
- Small eval splits and teacher-derived targets.
- Detail-seeking behavior can amplify noise and flicker.
- No guarantee of robustness across sensors, lenses, compression pipelines, or
  lighting regimes.
- TensorRT performance must be measured on the target Jetson runtime.

## License And Provenance

NightJet source code is MIT licensed. Model weights are covered by the separate
`WEIGHTS_LICENSE.md` file in this Hugging Face repo because they were distilled
from teacher-model outputs and local camera data. Review upstream teacher-model
licenses and data rights before redistributing, commercializing, or embedding
the weights in a product.

## Links

- GitHub: https://github.com/cezarc1/nightjet
- Space: https://huggingface.co/spaces/ggamecrazy/nightjet
- KubeJet: https://github.com/cezarc1/kubejet
