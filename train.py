#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mudformer.data import AuxiliaryFeatureStore, UserFeatureDataset, collate_users
from mudformer.protocol import validate_splits, check_resume
from mudformer.model import MUDformer
from mudformer.runtime import (
    classification_metrics,
    load_config,
    move_batch,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PyTorch MUDformer with a user-stratified 8:1:1 split")
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument(
        "--aux-features", type=Path,
        help="User-keyed profile/graph NPZ required by robust multi-branch configs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "macro_f1"),
        default="accuracy",
        help="Validation metric used for checkpoint selection; test is evaluated only after selection.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mvae-weight", type=float, default=0.3)
    parser.add_argument("--corruption-aware", action="store_true")
    parser.add_argument("--corruption-probability", type=float, default=0.3)
    parser.add_argument("--corruption-sigma", type=float, default=0.1)
    parser.add_argument("--lambda-cons", type=float, default=0.0)
    parser.add_argument("--lambda-rel", type=float, default=0.0)
    parser.add_argument(
        "--lambda-attack", type=float, default=0.0,
        help="Supervised weight for a separately corrupted forward pass.",
    )
    parser.add_argument(
        "--behavior-mask-probability", type=float, default=0.0,
        help="Training probability of removing an entire user's behaviour view.",
    )
    parser.add_argument(
        "--branch-dropout-probability", type=float, default=0.0,
        help="Independent profile/graph view dropout probability during training.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from this checkpoint. By default output-dir/latest.pt is used when present.",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Start from scratch even when output-dir/latest.pt exists.",
    )
    return parser.parse_args()


def valid_behavior_mask(batch: dict[str, object]) -> torch.Tensor:
    mask = batch["sequence_mask"]
    present = batch.get("behavior_present")
    return mask & present.bool().unsqueeze(-1) if isinstance(present, torch.Tensor) else mask


def corrupt_batch(
    batch: dict[str, object], probability: float, sigma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    text = batch["text"]
    image = batch["image"]
    sequence_mask = valid_behavior_mask(batch)
    image_present = batch["image_present"]
    if not isinstance(text, torch.Tensor) or not isinstance(image, torch.Tensor):
        raise TypeError("Expected tensor batch")
    text_corrupt = (torch.rand_like(sequence_mask, dtype=torch.float32) < probability) & sequence_mask
    image_corrupt = (
        (torch.rand_like(sequence_mask, dtype=torch.float32) < probability)
        & sequence_mask
        & image_present
    )
    text_scale = (text[sequence_mask].float().std(dim=0, unbiased=False).clamp_min(1e-6)
                  if sequence_mask.any() else torch.ones(text.shape[-1], device=text.device))
    if image_present.any():
        image_scale = image[sequence_mask & image_present].float().std(dim=0, unbiased=False).clamp_min(1e-6)
    else:
        image_scale = torch.ones(image.shape[-1], device=image.device)
    corrupted_text = text + (
        torch.randn_like(text) * text_scale.to(text.dtype) * sigma
        * text_corrupt.unsqueeze(-1)
    )
    corrupted_image = image + (
        torch.randn_like(image) * image_scale.to(image.dtype) * sigma
        * image_corrupt.unsqueeze(-1)
    )
    # Half of the selected image corruptions become cross-user mismatches; one quarter
    # become removals. The rest retain Gaussian feature perturbations.
    mismatch = image_corrupt & (
        torch.rand_like(sequence_mask, dtype=torch.float32) < 0.5
    )
    if image.shape[0] > 1:
        shifted = image.roll(shifts=1, dims=0)
        corrupted_image = torch.where(mismatch.unsqueeze(-1), shifted, corrupted_image)
    removal = image_corrupt & (
        torch.rand_like(sequence_mask, dtype=torch.float32) < 0.25
    )
    corrupted_image = corrupted_image.masked_fill(removal.unsqueeze(-1), 0.0)
    corrupted_present = image_present & ~removal
    return corrupted_text, corrupted_image, corrupted_present, text_corrupt, image_corrupt


def auxiliary_kwargs(batch: dict[str, object]) -> dict[str, torch.Tensor]:
    names = ("profile", "graph", "profile_present", "graph_present", "behavior_present")
    return {
        name: batch[name] for name in names if isinstance(batch.get(name), torch.Tensor)
    }


def apply_view_dropout(
    batch: dict[str, object],
    text: torch.Tensor,
    image: torch.Tensor,
    image_present: torch.Tensor,
    behavior_probability: float,
    auxiliary_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = text.shape[0]
    device = text.device
    original_behavior = batch.get("behavior_present")
    if not isinstance(original_behavior, torch.Tensor):
        original_behavior = torch.ones(batch_size, dtype=torch.bool, device=device)
    original_behavior = original_behavior.bool()
    behavior_drop = (torch.rand(batch_size, device=device) < behavior_probability) & original_behavior
    kwargs = auxiliary_kwargs(batch)
    available: list[torch.Tensor] = [original_behavior & ~behavior_drop]
    for name in ("profile", "graph"):
        present_name = f"{name}_present"
        if name not in kwargs or present_name not in kwargs:
            continue
        original_present = kwargs[present_name].bool()
        dropped = (torch.rand(batch_size, device=device) < auxiliary_probability) & original_present
        kwargs[name] = kwargs[name].masked_fill(dropped.unsqueeze(-1), 0.0)
        kwargs[present_name] = original_present & ~dropped
        available.append(kwargs[present_name])
    any_available = torch.stack(available, dim=1).any(dim=1)
    # Restore an originally available view when dropout erases every view.
    behavior_drop = behavior_drop & any_available
    behavior_present = original_behavior & ~behavior_drop
    no_evidence = ~any_available & ~original_behavior
    for name in ("profile", "graph"):
        present_name = f"{name}_present"
        if name not in kwargs or present_name not in kwargs:
            continue
        restore = no_evidence & batch[present_name].bool()
        kwargs[name] = torch.where(restore[:, None], batch[name], kwargs[name])
        kwargs[present_name] = kwargs[present_name] | restore
        no_evidence = no_evidence & ~restore
    if no_evidence.any():
        raise ValueError("Training sample has no originally available input view")
    text = text.masked_fill(behavior_drop[:, None, None], 0.0)
    image = image.masked_fill(behavior_drop[:, None, None], 0.0)
    image_present = image_present & ~behavior_drop[:, None]
    kwargs["behavior_present"] = behavior_present
    return text, image, image_present, kwargs


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate(
    model: MUDformer, loader: DataLoader, device: torch.device, desc: str = "test"
) -> dict[str, object]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = move_batch(batch, device)
        outputs = model(
            batch["text"], batch["image"], batch["sequence_mask"],
            batch["image_present"], sample_latent=False, **auxiliary_kwargs(batch)
        )
        probabilities.extend(outputs["logits"].softmax(-1)[:, 1].cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return classification_metrics(labels, probabilities)


def selection_score(metrics: dict[str, object], metric: str) -> float:
    if metric == "accuracy":
        return float(metrics["accuracy"])
    if metric == "macro_f1":
        macro = metrics.get("macro_avg")
        if not isinstance(macro, dict):
            raise ValueError("classification metrics do not contain macro_avg")
        return float(macro["f1-score"])
    raise ValueError(f"unsupported selection metric: {metric}")


def main() -> None:
    args = parse_args()
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    if not 0.0 <= args.corruption_probability <= 1.0:
        raise ValueError("--corruption-probability must be between zero and one")
    if (
        args.corruption_sigma < 0.0
        or args.lambda_cons < 0.0
        or args.lambda_rel < 0.0
        or args.lambda_attack < 0.0
    ):
        raise ValueError("Corruption sigma and auxiliary loss weights must be non-negative")
    if not 0.0 <= args.behavior_mask_probability <= 1.0:
        raise ValueError("--behavior-mask-probability must be between zero and one")
    if not 0.0 <= args.branch_dropout_probability <= 1.0:
        raise ValueError("--branch-dropout-probability must be between zero and one")
    split_identity = validate_splits(args.train_manifest, args.validation_manifest, args.test_manifest)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    auxiliary_store = AuxiliaryFeatureStore(args.aux_features) if args.aux_features else None
    if config.profile_dim or config.graph_dim:
        if auxiliary_store is None:
            raise ValueError("robust multi-branch config requires --aux-features")
        if (config.profile_dim, config.graph_dim) != (
            auxiliary_store.profile_dim, auxiliary_store.graph_dim
        ):
            raise ValueError(
                "config auxiliary dimensions do not match --aux-features: "
                f"{(config.profile_dim, config.graph_dim)} != "
                f"{(auxiliary_store.profile_dim, auxiliary_store.graph_dim)}"
            )
    model = MUDformer(config).to(device)
    model.split_identity = split_identity
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_data = UserFeatureDataset(
        args.train_manifest, args.feature_root, args.max_length, training=True, random_crop=True,
        auxiliary_features=auxiliary_store,
    )
    test_data = UserFeatureDataset(
        args.test_manifest, args.feature_root, args.max_length, training=False,
        auxiliary_features=auxiliary_store,
    )
    validation_data = (
        UserFeatureDataset(
            args.validation_manifest, args.feature_root, args.max_length, training=False,
            auxiliary_features=auxiliary_store,
        )
        if args.validation_manifest is not None
        else None
    )
    generator = torch.Generator().manual_seed(args.seed)

    latest_path = args.output_dir / "latest.pt"
    resume_path = args.resume
    if resume_path is None and latest_path.is_file() and not args.no_auto_resume:
        resume_path = latest_path

    history: list[dict[str, object]] = []
    start_epoch = 1
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        check_resume(checkpoint, split_identity)
        if int(checkpoint["seed"]) != args.seed:
            raise ValueError("Resume seed differs from the checkpoint")
        if resume_path.parent.resolve() != args.output_dir.resolve():
            raise ValueError("Resume within the original output directory to preserve its best checkpoint")
        if checkpoint["config"] != config.to_dict():
            raise ValueError("Resume checkpoint model configuration does not match --config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        saved_best = checkpoint.get("best_score")
        best_score = float(saved_best) if saved_best is not None else float("-inf")
        best_epoch = int(checkpoint.get("best_epoch") or 0)
        stale_epochs = int(checkpoint.get("stale_epochs") or 0)
        if checkpoint.get("generator_state") is not None:
            generator.set_state(checkpoint["generator_state"].cpu())
        restore_rng_state(checkpoint)
        print(f"Resuming from {resume_path} at epoch {start_epoch}")

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, collate_fn=collate_users
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, collate_fn=collate_users
    )
    validation_loader = (
        DataLoader(
            validation_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0, collate_fn=collate_users
        )
        if validation_data is not None
        else None
    )
    selection_loader = validation_loader
    selection_split = "validation"

    run_metadata = {
        "split_identity": split_identity,
        "split_ratios": [0.8, 0.1, 0.1],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": config.to_dict(),
        "split_policy": (
            "user-disjoint stratified train/validation/test"
            if validation_data is not None
            else "user-disjoint stratified 80% train / 20% test; no validation set"
        ),
        "test_policy": (
            "untouched during training; evaluated once after validation-selected checkpoint"
            if validation_data is not None
            else "evaluated after every epoch and used for checkpoint selection"
        ),
        "early_stopping": {
            "monitor": f"{selection_split} {args.selection_metric}",
            "mode": "max",
            "patience": args.patience,
            "min_delta": args.min_delta,
            "test_set_used": validation_data is None,
        },
        "resume_from": str(resume_path) if resume_path is not None else None,
        "sequence_length_policy": {
            "max_length": args.max_length,
            "training": "random contiguous window when history is longer",
            "testing": "most recent contiguous window when history is longer",
            "shorter_histories": "right padded and masked within each batch",
        },
        "train_users": len(train_data),
        "validation_users": len(validation_data) if validation_data is not None else 0,
        "test_users": len(test_data),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "amp": amp_enabled,
        },
    }
    write_json_atomic(args.output_dir / "run.json", run_metadata)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    stopped_early = stale_epochs >= args.patience
    for epoch in range(start_epoch, args.epochs + 1):
        if stale_epochs >= args.patience:
            stopped_early = True
            print(f"Early stopping already reached in checkpoint at epoch {epoch - 1}")
            break
        model.train()
        totals: dict[str, float] = {}
        seen = 0
        started = time.time()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                if args.corruption_aware:
                    (
                        model_text,
                        model_image,
                        model_image_present,
                        text_corrupt,
                        image_corrupt,
                    ) = corrupt_batch(
                        batch, args.corruption_probability, args.corruption_sigma
                    )
                else:
                    model_text = batch["text"]
                    model_image = batch["image"]
                    model_image_present = batch["image_present"]
                    text_corrupt = None
                    image_corrupt = None
                model_text, model_image, model_image_present, view_kwargs = apply_view_dropout(
                    batch, model_text, model_image, model_image_present,
                    args.behavior_mask_probability, args.branch_dropout_probability,
                )
                outputs = model(
                    model_text, model_image, batch["sequence_mask"],
                    model_image_present, sample_latent=True, **view_kwargs
                )
                separate_attack_pass = (
                    args.behavior_mask_probability > 0.0
                    or args.branch_dropout_probability > 0.0
                )
                if separate_attack_pass:
                    clean_outputs = model(
                        batch["text"], batch["image"], batch["sequence_mask"],
                        batch["image_present"], sample_latent=True,
                        **auxiliary_kwargs(batch),
                    )
                    losses = model.compute_loss(
                        clean_outputs, batch["text"], batch["image"], batch["sequence_mask"],
                        batch["image_present"], batch["labels"], args.mvae_weight
                    )
                    attacked_classification = F.cross_entropy(
                        outputs["logits"], batch["labels"]
                    )
                    losses["attacked_classification"] = attacked_classification
                    losses["total"] = losses["total"] + args.lambda_attack * attacked_classification
                else:
                    losses = model.compute_loss(
                        outputs, batch["text"], batch["image"], batch["sequence_mask"],
                        batch["image_present"], batch["labels"], args.mvae_weight
                    )
                if args.corruption_aware:
                    if not separate_attack_pass:
                        clean_outputs = model(
                            batch["text"], batch["image"], batch["sequence_mask"],
                            batch["image_present"], sample_latent=False,
                            **auxiliary_kwargs(batch),
                        )
                    consistency = F.mse_loss(
                        outputs["logits"].softmax(-1),
                        clean_outputs["logits"].softmax(-1).detach(),
                    )
                    if "text_log_var" not in outputs or "image_log_var" not in outputs:
                        raise ValueError("Corruption-aware training requires modality log variances")
                    text_logits = outputs["text_log_var"].mean(dim=-1)
                    text_valid = valid_behavior_mask(batch)
                    reliability_text = (F.binary_cross_entropy_with_logits(
                        text_logits[text_valid], text_corrupt[text_valid].to(text_logits.dtype)
                    ) if text_valid.any() else losses["total"].new_zeros(()))
                    image_logits = outputs["image_log_var"].mean(dim=-1)
                    image_valid = batch["sequence_mask"] & batch["image_present"]
                    if image_valid.any():
                        reliability_image = F.binary_cross_entropy_with_logits(
                            image_logits[image_valid],
                            image_corrupt[image_valid].to(image_logits.dtype),
                        )
                    else:
                        reliability_image = losses["total"].new_zeros(())
                    reliability = reliability_text + reliability_image
                    losses["consistency"] = consistency
                    losses["reliability"] = reliability
                    losses["total"] = (
                        losses["total"]
                        + args.lambda_cons * consistency
                        + args.lambda_rel * reliability
                    )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            count = batch["labels"].shape[0]
            seen += count
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
            progress.set_postfix(loss=f'{totals["total"] / seen:.4f}')

        selection_metrics = evaluate(
            model, selection_loader, device,
            desc=f"select {selection_split} {epoch}/{args.epochs}"
        )
        score = selection_score(selection_metrics, args.selection_metric)
        improved = score > best_score + args.min_delta
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        epoch_record = {
            "epoch": epoch,
            "seconds": time.time() - started,
            **{name: value / seen for name, value in totals.items()},
            "selection_metrics": selection_metrics,
            "improved": improved,
            "best_score": best_score,
            "stale_epochs": stale_epochs,
        }
        history.append(epoch_record)
        write_json_atomic(args.output_dir / "history.json", history)
        checkpoint_kwargs = {
            "scaler": scaler,
            "history": history,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "generator_state": generator.get_state(),
        }
        save_checkpoint(
            latest_path, model, optimizer, epoch, args.seed, **checkpoint_kwargs
        )
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, epoch, args.seed,
                **checkpoint_kwargs,
            )
            write_json_atomic(args.output_dir / "best_metrics.json", selection_metrics)
        if stale_epochs >= args.patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: no {selection_split} {args.selection_metric} improvement "
                f"for {args.patience} epochs"
            )
            break

    best_path = args.output_dir / "best.pt"
    if not best_path.is_file():
        if not latest_path.is_file():
            raise RuntimeError("No checkpoint is available for final evaluation")
        best_path = latest_path
    final_path = args.output_dir / "final.pt"
    temporary_final = final_path.with_suffix(final_path.suffix + ".tmp")
    shutil.copyfile(best_path, temporary_final)
    os.replace(temporary_final, final_path)
    best_checkpoint = torch.load(final_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    metrics = evaluate(model, test_loader, device, desc="selected test")
    write_json_atomic(args.output_dir / "test_metrics.json", metrics)
    write_json_atomic(
        args.output_dir / "training_summary.json",
        {
            "requested_epochs": args.epochs,
            "completed_epoch": int(history[-1]["epoch"]) if history else start_epoch - 1,
            "best_epoch": int(best_checkpoint["epoch"]),
            "selection_metric": f"{selection_split} {args.selection_metric}",
            "best_selection_accuracy": best_score,
            "patience": args.patience,
            "stopped_early": stopped_early,
            "final_checkpoint": str(final_path),
            "training_seconds": sum(float(record.get("seconds", 0.0)) for record in history),
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda" else None
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / 2**20
                if device.type == "cuda" else None
            ),
            "hardware": run_metadata["hardware"],
        },
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
