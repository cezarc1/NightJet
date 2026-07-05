# NightJet Weights

This directory contains the canonical public NightJet v0.1 inference artifacts.

| ID | PyTorch | ONNX | Role | Eval note |
| --- | --- | --- | --- | --- |
| `nightjet-edge-v1` | `nightjet-edge-v1.pt` | `nightjet-edge-v1.onnx` | Default | Conservative 5-frame model; temporal score `81.004304` on the ReDDiT detail probe. |
| `nightjet-edge-v1-detail` | `nightjet-edge-v1-detail.pt` | `nightjet-edge-v1-detail.onnx` | Detail variant | Higher detail score `76.392455`, but temporal score drops to `37.55211`. |

Full architecture config, source runs, eval reports, sizes, and SHA-256 hashes
are in [manifest.json](manifest.json).

## Input And Output Contract

- Input name: `luma_window`
- Input shape: `1 x 5 x H x W`
- Input dtype/range: `float32`, `[0, 1]`
- Temporal order: causal, with the latest frame last
- Output name: `enhanced_luma`
- Output shape: `1 x 1 x H x W`
- Output dtype/range: `float32`, `[0, 1]`

The CLI handles image/video conversion for you. Direct model users should feed
normalized luma windows and convert the output luma back to their desired image
container.

## Provenance

The public `.pt` files were re-saved from retrieved
[KubeTorch](https://github.com/cezarc1/kubetorch) checkpoints as inference-only
payloads. They keep `model_state_dict`, `model_config`, and metadata, but omit
optimizer state, AMP scaler state, loss history, and EMA tensors.

The default model comes from `edge-v1-reco-s2-c16-f5-detail-v1-5000`. The detail
variant comes from `edge-v1-reco-s2-c16-f5-reddit10-5000`.

## TensorRT

The ONNX files are canonical exchange artifacts. TensorRT `.plan` and `.engine`
files are not committed because they are specific to the Jetson model, TensorRT
version, CUDA stack, precision flags, and runtime image that built them.

## License Notes

Repository source code is MIT licensed. These model weights are research/demo
artifacts distilled from teacher-model outputs. Review the model card and the
licenses or terms for upstream teacher models/data before redistributing,
commercializing, or embedding the weights in a product.
