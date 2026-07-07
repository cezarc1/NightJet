from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console

from nightjet.blend import blend_bundles, parse_blend_component
from nightjet.config import load_config
from nightjet.data import package_arrays
from nightjet.eval import evaluate_baseline, evaluate_checkpoint
from nightjet.export import export_onnx
from nightjet.frame_bundle import bundle_from_video_frames
from nightjet.inference import DEFAULT_WEIGHTS_PATH, NightJetEnhancer, is_video_path
from nightjet.kubetorch import (
    build_submit_payload,
    payload_to_json,
    publish_run_output,
    retrieve_run_output,
    submit_training_run,
)
from nightjet.reports import parse_report_entry, write_leaderboard_report
from nightjet.runtime.engine import build_tensorrt_engine, build_trtexec_command
from nightjet.runtime.enhancer import TensorRTNightJetEnhancer
from nightjet.runtime.server import RuntimeServerConfig, run_runtime_server
from nightjet.splits import SplitSpec, build_frame_splits, load_split_indices
from nightjet.staging import stage_kubetorch_source
from nightjet.teacher_manifest import bundle_from_teacher_manifest
from nightjet.training import train as train_model

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(report_app, name="report")
console = Console()


@app.command()
def train(
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
    device: Annotated[str | None, typer.Option()] = None,
    max_steps: Annotated[int | None, typer.Option()] = None,
    bundle_uri: Annotated[str | None, typer.Option()] = None,
    split_uri: Annotated[str | None, typer.Option()] = None,
    split_name: Annotated[str | None, typer.Option()] = None,
) -> None:
    cfg = load_config(config)
    if max_steps is not None:
        cfg.training.max_steps = max_steps
    if bundle_uri is not None:
        cfg.data.bundle_uri = bundle_uri
    if split_uri is not None:
        cfg.data.split_uri = split_uri
    if split_name is not None:
        cfg.data.split_name = split_name
    result = train_model(cfg, output_dir=output_dir, device=device)
    artifact = publish_run_output(output_dir)
    payload = {
        "checkpoint": str(result.checkpoint_path),
        "metrics": str(result.metrics_path),
    }
    if artifact is not None:
        payload["artifact"] = artifact
    console.print_json(data=payload)


@app.command("export-onnx")
def export_onnx_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    height: Annotated[int, typer.Option()] = 720,
    width: Annotated[int, typer.Option()] = 1280,
    input_frames: Annotated[int, typer.Option()] = 3,
) -> None:
    output_path = export_onnx(
        checkpoint_path=checkpoint,
        output_path=output,
        input_shape=(1, input_frames, height, width),
    )
    console.print_json(data={"onnx": str(output_path)})


@app.command()
def enhance(
    input: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    weights: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    engine: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    device: Annotated[str | None, typer.Option()] = None,
    side_by_side: Annotated[bool, typer.Option("--side-by-side")] = False,
    preserve_color: Annotated[bool, typer.Option("--preserve-color")] = False,
    fps: Annotated[float | None, typer.Option()] = None,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
) -> None:
    if weights is not None and engine is not None:
        raise typer.BadParameter("--weights and --engine are mutually exclusive")
    if engine is not None:
        enhancer = TensorRTNightJetEnhancer.from_engine(engine)
        if is_video_path(input):
            output_path = enhancer.enhance_video(
                input,
                output,
                side_by_side=side_by_side,
                preserve_color=preserve_color,
                fps=fps,
                show_progress=progress,
            )
        else:
            enhancer.enhance_image(
                input,
                output_path=output,
                side_by_side=side_by_side,
                preserve_color=preserve_color,
            )
            output_path = output
        console.print_json(data={"output": str(output_path), "engine": str(engine)})
        return

    checkpoint = weights or DEFAULT_WEIGHTS_PATH
    enhancer = NightJetEnhancer.from_checkpoint(checkpoint, device=device)
    if is_video_path(input):
        output_path = enhancer.enhance_video(
            input,
            output,
            side_by_side=side_by_side,
            preserve_color=preserve_color,
            fps=fps,
            show_progress=progress,
        )
    else:
        enhancer.enhance_image(
            input,
            output_path=output,
            side_by_side=side_by_side,
            preserve_color=preserve_color,
        )
        output_path = output
    console.print_json(data={"output": str(output_path), "weights": str(checkpoint)})


@app.command("build-engine")
def build_engine(
    onnx: Annotated[Path, typer.Option(exists=True, readable=True)],
    output: Annotated[Path, typer.Option()],
    height: Annotated[int, typer.Option()] = 720,
    width: Annotated[int, typer.Option()] = 1280,
    input_frames: Annotated[int, typer.Option()] = 5,
    input_name: Annotated[str, typer.Option()] = "luma_window",
    dynamic_shapes: Annotated[bool, typer.Option("--dynamic-shapes")] = False,
    fp16: Annotated[bool, typer.Option("--fp16/--fp32")] = True,
    trtexec: Annotated[str, typer.Option()] = "trtexec",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    input_shape = (1, input_frames, height, width) if dynamic_shapes else None
    if dry_run:
        command = build_trtexec_command(
            onnx_path=onnx,
            output_path=output,
            input_shape=input_shape,
            input_name=input_name,
            fp16=fp16,
            trtexec=trtexec,
        )
        console.print_json(data={"command": command, "engine": str(output)})
        return
    engine_path = build_tensorrt_engine(
        onnx_path=onnx,
        output_path=output,
        input_shape=input_shape,
        input_name=input_name,
        fp16=fp16,
        trtexec=trtexec,
    )
    console.print_json(data={"engine": str(engine_path)})


@app.command()
def serve(
    engine: Annotated[Path, typer.Option(exists=True, readable=True)],
    source: Annotated[str | None, typer.Option()] = None,
    camera: Annotated[Path | None, typer.Option()] = None,
    resolution: Annotated[str, typer.Option()] = "1280x720",
    pixel_format: Annotated[str | None, typer.Option()] = None,
    fps: Annotated[float | None, typer.Option()] = None,
    host: Annotated[str, typer.Option()] = "0.0.0.0",
    port: Annotated[int, typer.Option()] = 8000,
    max_frames: Annotated[int | None, typer.Option()] = None,
    exit_after_max_frames: Annotated[bool, typer.Option("--exit-after-max-frames")] = False,
) -> None:
    metrics = run_runtime_server(
        RuntimeServerConfig(
            engine_path=engine,
            source=source,
            camera=camera,
            resolution=resolution,
            pixel_format=pixel_format,
            fps=fps,
            host=host,
            port=port,
            max_frames=max_frames,
            exit_after_max_frames=exit_after_max_frames,
        )
    )
    if exit_after_max_frames:
        console.print_json(data=metrics.to_json_dict())


@app.command("kt-submit-train")
def kt_submit_train(
    name: Annotated[str, typer.Option()],
    image: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option(exists=True, readable=True)],
    source_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[str, typer.Option()] = "outputs/nightjet-train",
    namespace: Annotated[str, typer.Option()] = "kubetorch",
    image_pull_secret: Annotated[str, typer.Option()] = "ghcr-pull-secret",
    gpu_count: Annotated[int, typer.Option()] = 1,
    max_steps: Annotated[int | None, typer.Option()] = None,
    bundle_uri: Annotated[str | None, typer.Option()] = None,
    split_uri: Annotated[str | None, typer.Option()] = None,
    split_name: Annotated[str | None, typer.Option()] = None,
    dry_run: Annotated[bool, typer.Option()] = True,
) -> None:
    payload = build_submit_payload(
        name=name,
        namespace=namespace,
        image=image,
        config_path=config,
        source_dir=source_dir,
        output_dir=output_dir,
        image_pull_secret=image_pull_secret,
        gpu_count=gpu_count,
        max_steps=max_steps,
        bundle_uri=bundle_uri,
        split_uri=split_uri,
        split_name=split_name,
    )
    if dry_run:
        print(payload_to_json(payload), end="")
        return
    console.print_json(data=submit_training_run(payload))


@app.command("kt-stage-source")
def kt_stage_source(
    output_dir: Annotated[Path, typer.Option()],
    source_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("."),
    bundle_dir: Annotated[list[Path] | None, typer.Option(exists=True, file_okay=False)] = None,
) -> None:
    result = stage_kubetorch_source(
        source_dir=source_dir,
        output_dir=output_dir,
        bundle_dirs=bundle_dir or [],
    )
    console.print_json(data=result.to_json_dict())


@app.command("kt-get-output")
def kt_get_output(
    run_id: Annotated[str, typer.Option()],
    output_dir: Annotated[Path, typer.Option()],
    namespace: Annotated[str, typer.Option()] = "kubetorch",
    artifact_name: Annotated[str, typer.Option()] = "nightjet-output",
) -> None:
    console.print_json(
        data=retrieve_run_output(
            run_id=run_id,
            output_dir=output_dir,
            namespace=namespace,
            artifact_name=artifact_name,
        )
    )


@data_app.command("package")
def data_package(
    input_luma: Annotated[Path, typer.Option(exists=True, readable=True)],
    target_luma: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
) -> None:
    manifest = package_arrays(
        np.load(input_luma),
        np.load(target_luma),
        output_dir,
    )
    console.print_json(data=manifest.to_json_dict())


@data_app.command("from-teacher-manifest")
def data_from_teacher_manifest(
    manifest: Annotated[Path, typer.Option(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option()],
) -> None:
    bundle = bundle_from_teacher_manifest(manifest, output_dir=output_dir)
    console.print_json(data=bundle.to_json_dict())


@data_app.command("from-video-frames")
def data_from_video_frames(
    input_video: Annotated[Path, typer.Option(exists=True, readable=True)],
    target_frame_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    target_glob: Annotated[str, typer.Option()] = "*.png",
    max_frames: Annotated[int | None, typer.Option()] = None,
) -> None:
    bundle = bundle_from_video_frames(
        input_video=input_video,
        target_frame_dir=target_frame_dir,
        output_dir=output_dir,
        target_glob=target_glob,
        max_frames=max_frames,
    )
    console.print_json(data=bundle.to_json_dict())


@data_app.command("blend")
def data_blend(
    component: Annotated[list[str], typer.Option(help="Repeat NAME=WEIGHT=BUNDLE_DIR")],
    output_dir: Annotated[Path, typer.Option()],
) -> None:
    bundle = blend_bundles(
        [parse_blend_component(value) for value in component],
        output_dir=output_dir,
    )
    console.print_json(data=bundle.to_json_dict())


@data_app.command("split")
def data_split(
    bundle_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
    train_fraction: Annotated[float, typer.Option("--train")] = 0.7,
    val_fraction: Annotated[float, typer.Option("--val")] = 0.15,
    test_fraction: Annotated[float, typer.Option("--test")] = 0.15,
    seed: Annotated[int, typer.Option()] = 1337,
) -> None:
    input_luma = np.load(bundle_dir / "input_luma.npy", mmap_mode="r")
    splits = build_frame_splits(
        frame_count=int(input_luma.shape[0]),
        spec=SplitSpec(train=train_fraction, val=val_fraction, test=test_fraction, seed=seed),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_dumps(splits.to_json_dict()), encoding="utf-8")
    console.print_json(data=splits.to_json_dict())


@eval_app.command("checkpoint")
def eval_checkpoint_command(
    checkpoint: Annotated[Path, typer.Option(exists=True, readable=True)],
    bundle_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    device: Annotated[str | None, typer.Option()] = None,
    teacher_name: Annotated[str, typer.Option()] = "teacher",
    split_json: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    split: Annotated[str, typer.Option()] = "test",
) -> None:
    split_indices = load_split_indices(split_json, split) if split_json is not None else None
    result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        bundle_dir=bundle_dir,
        output_dir=output_dir,
        device=device,
        split_indices=split_indices,
        teacher_name=teacher_name,
    )
    console.print_json(
        data={
            "report": str(result.report_path),
            "csv": str(result.csv_path),
            "contact_sheet": str(result.contact_sheet_path),
        }
    )


@eval_app.command("baseline")
def eval_baseline_command(
    method: Annotated[str, typer.Option()],
    bundle_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    teacher_name: Annotated[str, typer.Option()] = "teacher",
    split_json: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    split: Annotated[str, typer.Option()] = "test",
) -> None:
    split_indices = load_split_indices(split_json, split) if split_json is not None else None
    result = evaluate_baseline(
        method=method,
        bundle_dir=bundle_dir,
        output_dir=output_dir,
        split_indices=split_indices,
        teacher_name=teacher_name,
    )
    console.print_json(
        data={
            "report": str(result.report_path),
            "csv": str(result.csv_path),
            "contact_sheet": str(result.contact_sheet_path),
        }
    )


@report_app.command("leaderboard")
def report_leaderboard_command(
    entry: Annotated[
        list[str],
        typer.Option(help="Repeat NAME|SPLIT|REPORT_JSON[|RUN_JSON][|CHECKPOINT][|ONNX]."),
    ],
    output: Annotated[Path, typer.Option()],
    title: Annotated[str, typer.Option()] = "NightJet Candidate Leaderboard",
) -> None:
    output_path = write_leaderboard_report(
        [parse_report_entry(value) for value in entry],
        output_path=output,
        title=title,
    )
    console.print_json(data={"report": str(output_path)})


def json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    app()
