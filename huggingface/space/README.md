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

Upload an image, choose whether to preserve the original color, and run the
model. The demo returns the enhanced image and a before/after comparison.

This Space is intended for image demos only. Video inference, ONNX export,
TensorRT engine builds, and Jetson deployment are documented in the
[NightJet GitHub repo](https://github.com/cezarc1/nightjet).

> Stay on Hugging Face to test an image quickly or download the default model
> artifacts. Go to
> [github.com/cezarc1/nightjet](https://github.com/cezarc1/nightjet) when you
> want local image/video inference, the Python API, training/export details, or
> the Jetson/KubeJet deployment workflow.
