from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nightjet.config import TrainingConfig, config_to_json_dict
from nightjet.data import TemporalLumaDataset
from nightjet.losses import NightJetLoss, NightJetLossWeights
from nightjet.models import NightJetEdgeV1
from nightjet.splits import load_split_indices


@dataclass(frozen=True)
class TrainResult:
    output_dir: Path
    checkpoint_path: Path
    metrics_path: Path


def train(config: TrainingConfig, *, output_dir: Path, device: str | None = None) -> TrainResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.run.seed)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target_indices = (
        load_split_indices(Path(config.data.split_uri), config.data.split_name)
        if config.data.split_uri
        else None
    )
    dataset = TemporalLumaDataset.from_bundle(
        config.data.bundle_uri,
        input_frames=config.data.input_frames,
        crop_size=config.data.crop_size,
        target_indices=target_indices,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    model = NightJetEdgeV1(config.model).to(torch_device)
    criterion = NightJetLoss(
        NightJetLossWeights(
            reconstruction=config.loss.reconstruction_weight,
            edge=config.loss.edge_weight,
            brightness=config.loss.brightness_weight,
            smoothness=config.loss.smoothness_weight,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = _build_scheduler(
        optimizer,
        max_steps=config.training.max_steps,
        warmup_steps=config.training.warmup_steps,
        scheduler=config.training.scheduler,
    )
    ema_state = _clone_state_dict(model) if config.training.ema_decay is not None else None
    amp_enabled = bool(config.training.amp and torch_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    losses: list[float] = []
    component_history: list[dict[str, float]] = []

    model.train()
    start = time.perf_counter()
    step = 0
    while step < config.training.max_steps:
        for batch_x, batch_y in loader:
            step += 1
            batch_x = batch_x.to(device=torch_device, dtype=torch.float32)
            batch_y = batch_y.to(device=torch_device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch_device.type, enabled=amp_enabled):
                prediction = model(batch_x)
                loss, components = criterion(prediction, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema_state is not None and config.training.ema_decay is not None:
                _update_ema_state(ema_state, model, decay=config.training.ema_decay)
            losses.append(float(loss.detach().cpu()))
            component_history.append(components)
            if step >= config.training.max_steps:
                break

    elapsed_seconds = time.perf_counter() - start
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config_to_json_dict(config),
            "model_config": config.model.model_dump(mode="json"),
            "ema_model_state_dict": ema_state,
            "step": step,
            "losses": losses,
            "seed": config.run.seed,
        },
        checkpoint_path,
    )
    metrics_path = output_dir / "run.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_name": config.run.name,
                "final_step": step,
                "device": str(torch_device),
                "elapsed_seconds": round(elapsed_seconds, 6),
                "initial_loss": losses[0] if losses else None,
                "final_loss": losses[-1] if losses else None,
                "components": component_history[-1] if component_history else {},
                "checkpoint": str(checkpoint_path),
                "ema_enabled": ema_state is not None,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "training_frames": len(dataset),
                "split_uri": config.data.split_uri,
                "split_name": config.data.split_name if config.data.split_uri else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainResult(
        output_dir=output_dir, checkpoint_path=checkpoint_path, metrics_path=metrics_path
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    warmup_steps: int,
    scheduler: str,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, float(step + 1) / float(warmup_steps))
        if scheduler == "cosine":
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress))))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _update_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    *,
    decay: float,
) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
