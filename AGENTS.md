# AGENTS.md

This repo trains NightJet, a compact passive low-light enhancement model for
Jetson Orin Nano deployment.

## Tooling

- Python is pinned with `.python-version` to `3.12.12`.
- Use `uv` for dependency management and commands.
- Use `ruff` for formatting and linting.
- Use `ty` for type checking and editor/LSP feedback.
- Do not add Black, isort, Flake8, Pyright, or Mypy unless explicitly asked.

## Canonical Commands

```bash
uv sync --locked
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
just check
```

## Boundaries

- This repo owns training, evaluation, export, KubeTorch submission, and generic
  ONNX/TensorRT deployment primitives for NightJet artifacts.
- Do not add camera bring-up, Arducam-specific tuning, or Jetson live demo UI
  code here.
- Do not commit datasets, teacher clips, ad hoc checkpoints, TensorRT engines,
  or large generated artifacts. The small canonical public artifacts under
  `weights/` are the only committed checkpoint/ONNX exception.
- Final FPS and latency claims must come from the Jetson Orin Nano, not the
  4090 k3s training node.
