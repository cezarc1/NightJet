# Hugging Face Release

NightJet uses two Hugging Face repos:

- model repo: `ggamecrazy/nightjet-edge-v1`;
- interactive Space: `ggamecrazy/nightjet`.

The model repo owns durable artifacts and download metrics. The Space owns the
interactive upload-and-enhance demo. A native model-page widget is not the
reliable first path for NightJet because the model uses custom PyTorch code
rather than a built-in Transformers/Diffusers pipeline.

## Prerequisites

- NightJet wheel built from the local checkout for the Space. This keeps the
  demo buildable while the GitHub repo is private. After the GitHub repo is
  public and tagged, `huggingface/space/requirements.txt` can switch back to a
  GitHub ref.
- Hugging Face token with write access to `ggamecrazy`;
- HF Pro enabled if hosting the Space on ZeroGPU from a personal account;
- `huggingface_hub` available locally:

```bash
uv run --with huggingface_hub python -c "import huggingface_hub"
```

## Publish The Model Repo

```bash
uv run --with huggingface_hub python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, upload_folder

api = HfApi()
repo_id = "ggamecrazy/nightjet-edge-v1"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "examples").mkdir()
    for src, dst in [
        ("huggingface/model/README.md", "README.md"),
        ("huggingface/model/WEIGHTS_LICENSE.md", "WEIGHTS_LICENSE.md"),
        ("weights/nightjet-edge-v1.pt", "nightjet-edge-v1.pt"),
        ("weights/nightjet-edge-v1.onnx", "nightjet-edge-v1.onnx"),
        ("weights/manifest.json", "manifest.json"),
        ("examples/assets/input.jpg", "examples/input.jpg"),
        ("examples/assets/output.jpg", "examples/output.jpg"),
        ("examples/assets/comparison.jpg", "examples/comparison.jpg"),
    ]:
        target = root / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(src).read_bytes())

    upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=root,
        commit_message="Publish NightJet Edge v1",
    )
PY
```

## Publish The Space

```bash
uv build --wheel --out-dir /tmp/nightjet-hf-wheel
uv run --with huggingface_hub python - <<'PY'
from pathlib import Path
from shutil import copy2, copytree
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, upload_folder

api = HfApi()
repo_id = "ggamecrazy/nightjet"
wheel = next(Path("/tmp/nightjet-hf-wheel").glob("nightjet-*.whl"))
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
    private=False,
)
with TemporaryDirectory() as tmp:
    root = Path(tmp) / "space"
    copytree(
        "huggingface/space",
        root,
        ignore=lambda _dir, names: {"__pycache__"} & set(names),
    )
    vendor = root / "vendor"
    vendor.mkdir()
    copy2(wheel, vendor / wheel.name)
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = [
        (
            f"nightjet @ https://huggingface.co/spaces/{repo_id}/resolve/main/vendor/{wheel.name}"
            if line.startswith("nightjet @ ")
            else line
        )
        for line in requirements
    ]
    (root / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=root,
        commit_message="Add NightJet image demo",
        delete_patterns=["__pycache__/*"],
    )
api.request_space_hardware(repo_id, hardware="zero-a10g")
PY
```

ZeroGPU is a Gradio-only runtime; the app uses `@spaces.GPU(duration=30)` around
inference so GPU is requested only for the model call. If ZeroGPU is
unavailable, leave the Space on CPU Basic and switch hardware later.

## Smoke Test

1. Open `https://huggingface.co/spaces/ggamecrazy/nightjet`.
2. Upload `examples/assets/input.jpg`.
3. Run with `Preserve original color` disabled.
4. Confirm the enhanced output is grayscale RGB while the input image remains
   visible in the upload pane.
5. Enable `Preserve original color` and confirm the output keeps visible chroma.
6. Upload a short video clip in the `Video` tab and confirm the output is a
   side-by-side before/after clip.

## Update Policy

- Keep `ggamecrazy/nightjet-edge-v1` focused on the default conservative weight.
- Publish the detail variant separately only if we want a second public model
  URL, e.g. `ggamecrazy/nightjet-edge-v1-detail`.
- Do not upload TensorRT `.plan` or `.engine` files to the model repo.
- Do not upload datasets, teacher bundles, raw clips, or retrieved training
  outputs.
