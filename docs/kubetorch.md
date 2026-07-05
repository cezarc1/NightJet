# KubeTorch Training

NightJet uses [our KubeTorch fork](https://github.com/cezarc1/kubetorch) as the
homelab GPU training runner. The source repo is snapshotted into the run, while
datasets and teacher outputs remain outside Git.

## Build the Training Image

```bash
IMAGE_TAG="0.1.0-$(git rev-parse --short HEAD)" \
  scripts/build_train_image.sh
```

The image defaults to:

```text
ghcr.io/cezarc1/nightjet-train:${IMAGE_TAG}
```

It requires a local checkout of the
[KubeTorch fork](https://github.com/cezarc1/kubetorch) as a Docker build context
so the in-cluster run has the matching client and `rsync`. Set
`KUBETORCH_CONTEXT` to the checkout path before running the build script.

## Dry Run a Submission

First stage only the source files and selected data bundles that the GPU job
should see:

```bash
uv run nightjet kt-stage-source \
  --output-dir outputs/kubetorch-source/nightjet \
  --bundle-dir data/bundles/hvi-darkir-60f-low-light-720p \
  --bundle-dir data/bundles/hvi-darkir-reddit-detail-v1
```

The staged directory excludes `.venv`, `.git`, caches, `outputs/`, checkpoints,
and unselected datasets. The [KubeTorch](https://github.com/cezarc1/kubetorch)
payload also sets `PYTHONPATH=src` so the run imports this staged source tree
rather than relying on whatever package was installed when the image was built.
`kt-submit-train` also overrides
[KubeTorch](https://github.com/cezarc1/kubetorch)'s rsync filters during
submission so selected staged bundles are not dropped by this repo's
artifact-oriented `.gitignore`.

```bash
uv run nightjet kt-submit-train \
  --name nightjet-edge-v1-smoke \
  --image ghcr.io/cezarc1/nightjet-train:${IMAGE_TAG} \
  --config configs/candidates/edge_v1_reco_s2_c16_f3.yaml \
  --source-dir outputs/kubetorch-source/nightjet \
  --output-dir outputs/nightjet-edge-v1-smoke \
  --bundle-uri data/bundles/hvi-darkir-60f-low-light-720p \
  --split-uri data/bundles/hvi-darkir-60f-low-light-720p/splits.json \
  --split-name train \
  --max-steps 10 \
  --dry-run
```

## Submit to [KubeTorch](https://github.com/cezarc1/kubetorch)

```bash
uv run nightjet kt-submit-train \
  --name nightjet-edge-v1-train \
  --image ghcr.io/cezarc1/nightjet-train:${IMAGE_TAG} \
  --config configs/candidates/edge_v1_reco_s2_c16_f3.yaml \
  --source-dir outputs/kubetorch-source/nightjet \
  --output-dir outputs/nightjet-edge-v1-train \
  --bundle-uri data/bundles/hvi-darkir-60f-low-light-720p \
  --split-uri data/bundles/hvi-darkir-60f-low-light-720p/splits.json \
  --split-name train \
  --no-dry-run
```

The default namespace is `kubetorch`, and the default private image pull secret
is `ghcr-pull-secret`.

The local homelab currently runs a forked
[KubeTorch](https://github.com/cezarc1/kubetorch) controller build. If the
editable client reports a version mismatch during local submission, use:

```bash
KUBETORCH_IGNORE_VERSION_MISMATCH=1 uv run nightjet kt-submit-train ...
```

When a NightJet training command runs inside
[KubeTorch](https://github.com/cezarc1/kubetorch), it publishes the output
directory before exit as:

```text
kt://kubetorch/runs/<run-id>/artifacts/nightjet-output
```

That artifact is the durable source for `checkpoint.pt`, `run.json`, and later
eval exports. The artifact transfer explicitly includes NightJet output types
such as `.json`, `.pt`, `.csv`, `.png`, `.mp4`, and `.onnx`, because repo-level
ignore rules intentionally exclude generated model artifacts. Do not rely on
`kubectl cp` from completed Job pods.

Retrieve output artifacts with:

```bash
KUBETORCH_IGNORE_VERSION_MISMATCH=1 uv run nightjet kt-get-output \
  --run-id <run-id> \
  --output-dir outputs/retrieved/<run-id>
```

The command creates the destination and applies the download-side rsync override
needed for checkpoint files.
