# Export

NightJet exports PyTorch checkpoints to fixed-shape ONNX files for downstream
runtime work.

## ONNX Export

```bash
uv run nightjet export-onnx \
  --checkpoint weights/nightjet-edge-v1.pt \
  --output outputs/nightjet-edge-v1.onnx \
  --height 720 \
  --width 1280 \
  --input-frames 5
```

The exported model uses:

- input: `luma_window`, shape `1 x 5 x H x W`, `float32`, range `[0, 1]`
- output: `enhanced_luma`, shape `1 x 1 x H x W`, `float32`, range `[0, 1]`
- opset: 18 by default

The public ONNX files under `weights/` are canonical exchange artifacts for the
public weights. Regenerated ONNX files in `outputs/` stay ignored unless they are
deliberately promoted.

## TensorRT

Build TensorRT engines on the target Jetson/runtime image, not on a development
machine. TensorRT `.plan` and `.engine` files are not portable across hardware,
TensorRT versions, CUDA stacks, or precision/build settings.

Use the NightJet CLI as a thin `trtexec` wrapper:

```bash
uv run nightjet build-engine \
  --onnx weights/nightjet-edge-v1.onnx \
  --output outputs/nightjet-edge-v1-fp16.plan \
  --fp16
```

The committed public ONNX files are fixed-shape models, so the default
`build-engine` command lets TensorRT read the input shape from the model. Use
`--dynamic-shapes` with `--height`, `--width`, and `--input-frames` only for
future dynamic-shape ONNX exports. The default TensorRT input name is
`luma_window`, matching NightJet's ONNX export. Use `--dry-run` to print the
exact command before running it on the Jetson.

Recommended promotion sequence:

1. Start from a committed ONNX artifact in `weights/`.
2. Build FP16 TensorRT on the target Orin runtime image.
3. Validate saved clips before live camera use.
4. Measure FPS and latency on the Jetson.
5. Keep plans/engines in release assets or runtime repos, not in git.

For Jetson handoff notes, see [docs/jetson.md](jetson.md).
