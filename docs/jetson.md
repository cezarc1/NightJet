# Jetson Handoff

You do not need a Jetson to run the public NightJet image/video examples. The
Jetson path starts after local inference works and you want a live camera or
managed edge deployment.

## Runtime Boundary

NightJet owns:

- public inference API and CLI;
- optional ONNX/TensorRT runtime primitives and a headless `nightjet serve`
  contract;
- model architecture experiments;
- dataset bundle packaging;
- training and evaluation;
- ONNX export;
- public checkpoint and ONNX promotion.

`night-vision-orin` remains the private Arducam/homelab reference runtime and
owns hardware-specific work:

- Arducam xISP camera bring-up;
- V4L2/OpenCV/GStreamer capture checks;
- polished MJPEG demo UI and local metrics services;
- saved-clip and live-camera validation.

[KubeJet](https://github.com/cezarc1/kubejet) owns managed k3s deployment
through `EdgeVisionPipeline`.
[KubeTorch](https://github.com/cezarc1/kubetorch) owns GPU batch training and
artifact retrieval. Neither is required for a first `nightjet enhance` run.

## Jetson Promotion Checklist

Before calling a NightJet model deployed on Jetson:

1. export or choose a committed ONNX file from `weights/`;
2. build TensorRT FP16 on the Jetson/runtime image;
3. validate the engine against saved clips;
4. validate the local camera stream;
5. measure FPS and latency on the Jetson Orin Nano target;
6. update any
   [KubeJet `EdgeVisionPipeline` manifest](https://github.com/cezarc1/kubejet/blob/main/examples/nightjet-tensorrt.yaml)
   only after local runtime validation passes.

Do not commit TensorRT plans or engines here. Keep them in runtime release
assets or target-local storage with checksums and build metadata.

## Companion Repos

- `night-vision-orin`: advanced Jetson runtime and camera path.
- [`kubejet`](https://github.com/cezarc1/kubejet): optional managed k3s
  deployment path for edge inference services. Start with the
  [Helm chart](https://github.com/cezarc1/kubejet/tree/main/charts/kubejet) or
  the
  [`EdgeVisionPipeline` example](https://github.com/cezarc1/kubejet/blob/main/examples/nightjet-tensorrt.yaml).
- [`kubetorch`](https://github.com/cezarc1/kubetorch): optional GPU training
  runner used for longer NightJet campaigns.
