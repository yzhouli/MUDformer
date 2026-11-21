#!/usr/bin/env python3
"""Fit an audit-logit threshold using only fixed-training users."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_data import SyntheticRiskBlockDataset
from mudformer.data import (  # noqa: E402
    AuxiliaryFeatureStore,
    UserFeatureDataset,
    collate_users,
)
from mudformer.runtime import load_model_checkpoint, move_batch, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-space", choices=("probability", "margin"), default="probability")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--aux-features", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def auxiliary_kwargs(batch: dict[str, object]) -> dict[str, torch.Tensor]:
    return {
        name: batch[name]
        for name in ("profile", "graph", "profile_present", "graph_present", "behavior_present")
        if isinstance(batch.get(name), torch.Tensor)
    }


@torch.inference_mode()
def probabilities(model: torch.nn.Module, loader: DataLoader, device: torch.device, score_space: str = "probability") -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    values: list[float] = []
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(
            batch["text"], batch["image"], batch["sequence_mask"], batch["image_present"],
            sample_latent=False, **auxiliary_kwargs(batch),
        )
        logits = outputs.get("audit_logits")
        if logits is None:
            raise ValueError("checkpoint does not emit audit_logits")
        ids.extend(str(value) for value in batch["user_ids"])
        scores = logits[:, 1].double() - logits[:, 0].double() if score_space == "margin" else logits.softmax(dim=-1)[:, 1]
        values.extend(float(value) for value in scores.cpu().tolist())
    return ids, np.asarray(values, dtype=np.float64)


def candidate_thresholds(clean: np.ndarray, injected: np.ndarray, score_space: str = "probability") -> np.ndarray:
    values = np.unique(np.concatenate((clean, injected)))
    if score_space == "margin":
        if not np.isfinite(values).all(): raise ValueError("non-finite calibration margins")
        mids = values[:-1] + (values[1:] - values[:-1]) / 2.0
        return np.unique(np.concatenate(([np.nextafter(values[0], -np.inf)], mids, [np.nextafter(values[-1], np.inf)])))
    if values.size == 1:
        return np.asarray([min(max(float(values[0]), 1e-6), 1.0 - 1e-6)])
    mids = (values[:-1] + values[1:]) / 2.0
    return np.unique(np.clip(np.concatenate(([1e-6], mids, [1.0 - 1e-6])), 1e-6, 1.0 - 1e-6))


def select_threshold(clean: np.ndarray, injected: np.ndarray, score_space: str = "probability") -> tuple[float, dict[str, float]]:
    best: tuple[tuple[float, float, float, float], float, dict[str, float]] | None = None
    for threshold in candidate_thresholds(clean, injected, score_space):
        clean_negative = clean < threshold
        injected_positive = injected >= threshold
        paired = float(np.mean(clean_negative & injected_positive))
        true_negative_rate = float(np.mean(clean_negative))
        true_positive_rate = float(np.mean(injected_positive))
        false_positive_rate = 1.0 - true_negative_rate
        balanced_accuracy = 0.5 * (true_negative_rate + true_positive_rate)
        metrics = {
            "paired_detection_rate": paired,
            "clean_true_negative_rate": true_negative_rate,
            "injected_true_positive_rate": true_positive_rate,
            "clean_false_positive_rate": false_positive_rate,
            "balanced_accuracy": balanced_accuracy,
        }
        key = (paired, balanced_accuracy, -false_positive_rate, -abs(float(threshold) - (0.0 if score_space == "margin" else 0.5)))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise AssertionError("threshold search produced no candidate")
    return best[1], best[2]


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.max_length < 5 or args.batch_size <= 0:
        raise ValueError("invalid max length or batch size")
    seed_everything(args.seed)
    device = torch.device(args.device)
    auxiliary = AuxiliaryFeatureStore(args.aux_features) if args.aux_features else None
    clean = UserFeatureDataset(
        args.train_manifest, args.feature_root, args.max_length, training=False,
        auxiliary_features=auxiliary,
    )
    source_train_users = len(clean.records)
    clean.records = [record for record in clean.records if record.label == 0 and record.behavior_count >= 50]
    injected = SyntheticRiskBlockDataset(
        args.train_manifest, args.train_manifest, args.feature_root, args.max_length,
        auxiliary_features=auxiliary,
    )
    clean_ids = [record.user_id for record in clean.records]
    injected_ids = [record.user_id for record in injected.records]
    if clean_ids != injected_ids:
        raise ValueError("clean and injected train-only recipients are not aligned")
    loader_options = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True,
        "collate_fn": collate_users,
    }
    model = load_model_checkpoint(args.checkpoint, device).eval()
    clean_output_ids, clean_probabilities = probabilities(model, DataLoader(clean, **loader_options), device, args.score_space)
    injected_output_ids, injected_probabilities = probabilities(model, DataLoader(injected, **loader_options), device, args.score_space)
    if clean_output_ids != injected_output_ids or clean_output_ids != clean_ids:
        raise ValueError("calibration predictions are not recipient-aligned")
    threshold, metrics = select_threshold(clean_probabilities, injected_probabilities, args.score_space)
    plan_lines = [
        f"{user_id},{injected.plan[user_id].donor_id},{injected.plan[user_id].block_index}"
        for user_id in clean_ids
    ]
    receipt = {
        "schema_version": 2,
        "seed": args.seed,
        "decision_threshold": threshold,
        "decision_score_space": args.score_space,
        "selection_metrics": metrics,
        "calibration_population": {
            "fixed_training_users": source_train_users,
            "normal_recipients": len(clean_ids),
            "normal_recipient_ids_sha256": hashlib.sha256("\n".join(clean_ids).encode()).hexdigest(),
            "synthetic_injection_plan_sha256": hashlib.sha256("\n".join(plan_lines).encode()).hexdigest(),
        },
        "protocol": {
            "name": "train_only_temporal_audit_threshold_v2",
            "decision_score_space": args.score_space,
            "selection_objective": "maximize paired(clean<threshold and injected>=threshold) rate; then balanced accuracy; then lower clean FPR; then threshold closest to zero margin (or probability 0.5)",
            "calibration_inputs": "clean fixed-training normal histories and deterministic spammer-block injections whose donors are also fixed-training users",
            "test_manifest_accessed": False,
            "train_or_checkpoint_modification": False,
            "max_length": args.max_length,
            "audit_logit_key": "audit_logits",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "artifacts": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": sha256_file(args.train_manifest),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / "calibration.json.tmp"
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "calibration.json")
    with (args.output_dir / "calibration_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("user_id", "clean_score", "injected_score"))
        writer.writeheader()
        writer.writerows(
            {
                "user_id": user_id,
                "clean_score": float(clean_probability),
                "injected_score": float(injected_probability),
            }
            for user_id, clean_probability, injected_probability in zip(
                clean_ids, clean_probabilities, injected_probabilities, strict=True
            )
        )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
