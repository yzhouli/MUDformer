from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from .model import ModelConfig, MUDformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str | Path) -> ModelConfig:
    return ModelConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def classification_metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    predictions = [int(value >= 0.5) for value in probabilities]
    report = classification_report(
        labels, predictions, labels=[0, 1], target_names=["normal", "spammer"],
        output_dict=True, zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(labels, predictions),
        "normal": report["normal"],
        "spammer": report["spammer"],
        "macro_avg": report["macro avg"],
    }
    return metrics


def save_checkpoint(
    path: str | Path,
    model: MUDformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    history: list[dict[str, Any]] | None = None,
    best_score: float | None = None,
    best_epoch: int | None = None,
    stale_epochs: int = 0,
    generator_state: torch.Tensor | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "split_identity": getattr(model, "split_identity", None),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "config": model.config.to_dict(),
        "history": history or [],
        "best_score": best_score,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "generator_state": generator_state,
        "rng_state": {
            "python": random.getstate(),
            "numpy": (lambda x: [x[0], x[1].tolist(), x[2], x[3], x[4]])(np.random.get_state()),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("rng_state")
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def load_model_checkpoint(path: str | Path, device: torch.device) -> MUDformer:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MUDformer(ModelConfig.from_dict(checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device)
