---
title: NightJet
colorFrom: gray
colorTo: blue
sdk: gradio
python_version: 3.12.12
app_file: app.py
pinned: false
models:
  - ggamecrazy/nightjet-edge-v1
---

# NightJet

Interactive demo for `ggamecrazy/nightjet-edge-v1`, a tiny luma-first low-light
enhancement model.

Start with the bundled Jetson/Arducam example image or upload your own image.
The image demo returns the enhanced image and a before/after comparison.

The video tab converts a short uploaded clip into a before/after comparison on
ZeroGPU. Longer local video inference, ONNX export, TensorRT engine builds, and
Jetson deployment are documented in the
[NightJet GitHub repo](https://github.com/cezarc1/nightjet).

> Stay on Hugging Face to test an image quickly or download the default model
> artifacts. Go to
> [github.com/cezarc1/nightjet](https://github.com/cezarc1/nightjet) when you
> want longer local image/video inference, the Python API, training/export
> details, or the Jetson/KubeJet deployment workflow.
