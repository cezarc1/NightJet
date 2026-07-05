from __future__ import annotations

import subprocess
from pathlib import Path


def build_trtexec_command(
    *,
    onnx_path: Path,
    output_path: Path,
    input_shape: tuple[int, int, int, int] | None = None,
    input_name: str = "luma_window",
    fp16: bool = True,
    trtexec: str = "trtexec",
) -> list[str]:
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
    ]
    if fp16:
        command.append("--fp16")
    if input_shape is not None:
        shape = "x".join(str(part) for part in input_shape)
        command.extend(
            [
                f"--minShapes={input_name}:{shape}",
                f"--optShapes={input_name}:{shape}",
                f"--maxShapes={input_name}:{shape}",
            ]
        )
    return command


def build_tensorrt_engine(
    *,
    onnx_path: Path,
    output_path: Path,
    input_shape: tuple[int, int, int, int] | None = None,
    input_name: str = "luma_window",
    fp16: bool = True,
    trtexec: str = "trtexec",
) -> Path:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_trtexec_command(
        onnx_path=onnx_path,
        output_path=output_path,
        input_shape=input_shape,
        input_name=input_name,
        fp16=fp16,
        trtexec=trtexec,
    )
    subprocess.run(command, check=True)
    return output_path
