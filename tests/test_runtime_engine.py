from pathlib import Path

from nightjet.runtime.engine import build_trtexec_command


def test_build_trtexec_command_uses_static_onnx_shape_by_default(tmp_path: Path) -> None:
    onnx = tmp_path / "nightjet.onnx"
    engine = tmp_path / "nightjet.plan"
    onnx.write_bytes(b"onnx")

    command = build_trtexec_command(
        onnx_path=onnx,
        output_path=engine,
        fp16=True,
    )

    assert command == [
        "trtexec",
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        "--fp16",
    ]


def test_build_trtexec_command_can_emit_dynamic_shape_profile(tmp_path: Path) -> None:
    onnx = tmp_path / "nightjet.onnx"
    engine = tmp_path / "nightjet.plan"
    onnx.write_bytes(b"onnx")

    command = build_trtexec_command(
        onnx_path=onnx,
        output_path=engine,
        input_shape=(1, 5, 720, 1280),
        fp16=True,
    )

    assert command == [
        "trtexec",
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        "--fp16",
        "--minShapes=luma_window:1x5x720x1280",
        "--optShapes=luma_window:1x5x720x1280",
        "--maxShapes=luma_window:1x5x720x1280",
    ]
