#!/usr/bin/env python3
"""Train the full model audit head using training donors and validation selection.
The test partition is evaluated only after checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mudformer.data import AuxiliaryFeatureStore, UserFeatureDataset, collate_users
from audit_data import SyntheticRiskBlockDataset
from mudformer.protocol import validate_splits, check_resume
from mudformer.model import MUDformer, ModelConfig
from mudformer.runtime import (
    classification_metrics,
    load_config,
    move_batch,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
)
from mudformer.temporal_localization import NUM_TEMPORAL_BLOCKS, temporal_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--aux-features", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mvae-weight", type=float, default=0.3)
    parser.add_argument("--synthetic-user-weight", type=float, default=1.0)
    parser.add_argument("--risk-block-weight", type=float, default=1.0)
    parser.add_argument("--clean-audit-weight", type=float, default=1.0)
    parser.add_argument("--clean-distillation-weight", type=float, default=0.0)
    parser.add_argument(
        "--attribution-block-weight", type=float, default=0.0,
        help="Train-only second-order loss that aligns signed Gradient x Input with the known injected block.",
    )
    parser.add_argument("--attribution-temperature", type=float, default=1.0)
    parser.add_argument("--selection-accuracy-weight", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument(
        "--initialize-checkpoint", type=Path,
        help="Optional same-backbone checkpoint used to initialize Temporal-MIL.",
    )
    parser.add_argument(
        "--freeze-base", action="store_true",
        help="Train only the new temporal risk head after --initialize-checkpoint.",
    )
    parser.add_argument(
        "--skip-test-eval", action="store_true",
        help="Skip final test evaluation during development.",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def auxiliary_kwargs(batch: dict[str, object]) -> dict[str, torch.Tensor]:
    return {
        name: batch[name]
        for name in ("profile", "graph", "profile_present", "graph_present", "behavior_present")
        if isinstance(batch.get(name), torch.Tensor)
    }


def temporal_block_scores(
    risk_logits: torch.Tensor, sequence_mask: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """Five length-normalized LogSumExp scores aligned to retained behaviours."""
    if risk_logits.ndim != 2 or sequence_mask.shape != risk_logits.shape:
        raise ValueError("risk logits and sequence mask must have matching [batch, length] shapes")
    rows: list[torch.Tensor] = []
    for row in range(risk_logits.shape[0]):
        positions = torch.nonzero(sequence_mask[row], as_tuple=False).flatten()
        if positions.numel() < NUM_TEMPORAL_BLOCKS:
            raise ValueError("every synthetic sequence must retain at least five behaviours")
        values = risk_logits[row, positions]
        pieces = temporal_groups(int(values.numel()))
        scores = []
        for piece in pieces:
            indices = torch.as_tensor(piece, dtype=torch.long, device=values.device)
            block_values = values[indices] / temperature
            scores.append(temperature * (torch.logsumexp(block_values, dim=0) - math.log(len(piece))))
        rows.append(torch.stack(scores))
    return torch.stack(rows)


def temporal_attribution_block_scores(
    contribution: torch.Tensor, sequence_mask: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """Metric-matched five-block signed Gradient x Input scores.

    The construction is intentionally the same block aggregation used by the
    frozen V1 evaluator; only the examples come from the training manifest.
    """
    if contribution.ndim != 2 or sequence_mask.shape != contribution.shape:
        raise ValueError("attribution contribution and sequence mask must have matching shapes")
    rows: list[torch.Tensor] = []
    for row in range(contribution.shape[0]):
        positions = torch.nonzero(sequence_mask[row], as_tuple=False).flatten()
        if positions.numel() < NUM_TEMPORAL_BLOCKS:
            raise ValueError("every synthetic sequence must retain at least five behaviours")
        values = contribution[row, positions]
        blocks = temporal_groups(int(values.numel()))
        rows.append(torch.stack([values[torch.as_tensor(block, device=values.device)].sum() / temperature for block in blocks]))
    return torch.stack(rows)


def base_logits_from_temporal_output(
    outputs: dict[str, torch.Tensor], config: ModelConfig,
) -> torch.Tensor:
    """Recover the unchanged user-classification logits from a MIL output."""
    del config
    return outputs.get("user_logits", outputs["logits"])


def audit_logits_from_output(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    audit = outputs.get("audit_logits")
    if audit is None:
        raise ValueError("Temporal-MIL output does not contain audit logits")
    return audit


def initialize_temporal_mil(
    model: MUDformer, checkpoint_path: Path, config: ModelConfig, freeze_base: bool,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    check_resume(checkpoint, model.split_identity)
    source_config = ModelConfig.from_dict(checkpoint["config"])
    source_values = source_config.to_dict()
    target_values = config.to_dict()
    for key in ("temporal_mil", "temporal_mil_scale", "temporal_mil_temperature"):
        source_values.pop(key, None)
        target_values.pop(key, None)
    if source_values != target_values:
        raise ValueError("initial checkpoint backbone configuration does not match Temporal-MIL config")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    expected_missing = (
        set()
        if source_config.temporal_mil
        else {"temporal_risk_head.0.weight", "temporal_risk_head.0.bias", "temporal_risk_head.1.weight", "temporal_risk_head.1.bias"}
    )
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "initial checkpoint must have the same backbone; "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    if freeze_base:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("temporal_risk_head."))
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "freeze_base": freeze_base,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


@torch.inference_mode()
def evaluate_user_classification(
    model: MUDformer, loader: DataLoader, device: torch.device, description: str,
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for batch in tqdm(loader, desc=description, leave=False):
        batch = move_batch(batch, device)
        outputs = model(
            batch["text"], batch["image"], batch["sequence_mask"], batch["image_present"],
            sample_latent=False, **auxiliary_kwargs(batch),
        )
        user_logits = outputs.get("user_logits", outputs["logits"])
        values = user_logits.softmax(-1)[:, 1].detach().cpu().tolist()
        labels.extend(batch["labels"].cpu().tolist())
        probabilities.extend(values)
        rows.extend(
            {"user_id": str(user_id), "spammer_probability": float(probability)}
            for user_id, probability in zip(batch["user_ids"], values, strict=True)
        )
        audit_logits = outputs.get("audit_logits", user_logits)
        audit_values = audit_logits.softmax(-1)[:, 1].detach().cpu().tolist()
        audit_rows.extend(
            {"user_id": str(user_id), "spammer_probability": float(probability)}
            for user_id, probability in zip(batch["user_ids"], audit_values, strict=True)
        )
    return classification_metrics(labels, probabilities), rows, audit_rows


@torch.inference_mode()
def evaluate_synthetic_localization(
    model: MUDformer,
    clean_loader: DataLoader,
    injected_loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> dict[str, float | int]:
    """Internal-only checkpoint selector; it never reads the fixed test set."""
    model.eval()
    targets = detections = hits = 0
    for clean, injected in zip(clean_loader, injected_loader, strict=True):
        if clean["user_ids"] != injected["user_ids"]:
            raise ValueError("clean and synthetic internal-validation batches are misaligned")
        clean = move_batch(clean, device)
        injected = move_batch(injected, device)
        clean_outputs = model(
            clean["text"], clean["image"], clean["sequence_mask"], clean["image_present"],
            sample_latent=False, **auxiliary_kwargs(clean),
        )
        injected_outputs = model(
            injected["text"], injected["image"], injected["sequence_mask"],
            injected["image_present"], sample_latent=False, **auxiliary_kwargs(injected),
        )
        clean_logits = audit_logits_from_output(clean_outputs)
        injected_logits = audit_logits_from_output(injected_outputs)
        risk_logits = injected_outputs.get("temporal_risk_logits")
        if risk_logits is None:
            raise ValueError("Temporal-MIL configuration did not emit temporal risk logits")
        block_scores = temporal_block_scores(risk_logits, injected["sequence_mask"], temperature)
        clean_prediction = clean_logits.argmax(dim=-1)
        injected_prediction = injected_logits.argmax(dim=-1)
        top_blocks = block_scores.argmax(dim=-1)
        detected = (clean_prediction == 0) & (injected_prediction == 1)
        targets += int(detected.numel())
        detections += int(detected.sum())
        hits += int((detected & (top_blocks == injected["risk_blocks"])).sum())
    return {
        "targets": targets,
        "injection_detection_successes": detections,
        "risk_head_top1_hits": hits,
        "injection_detection_rate": detections / targets if targets else 0.0,
        "risk_head_localization_recall_at_1_given_detection": hits / detections if detections else 0.0,
        "internal_synthetic_TGLS_at_1": hits / targets if targets else 0.0,
    }


def _loader(
    dataset: UserFeatureDataset,
    *, batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_users,
    )


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience and batch size must be positive")
    if args.max_length < NUM_TEMPORAL_BLOCKS:
        raise ValueError("max length must retain at least five behaviours")
    if min(args.mvae_weight, args.synthetic_user_weight, args.risk_block_weight, args.clean_audit_weight, args.clean_distillation_weight, args.attribution_block_weight) < 0.0:
        raise ValueError("loss weights must be non-negative")
    if args.attribution_temperature <= 0.0:
        raise ValueError("attribution temperature must be positive")
    if not args.skip_test_eval and args.test_manifest is None:
        raise ValueError("--test-manifest is required unless --skip-test-eval is set")
    if args.freeze_base and args.initialize_checkpoint is None:
        raise ValueError("--freeze-base requires --initialize-checkpoint")
    split_identity = validate_splits(args.train_manifest, args.validation_manifest, args.test_manifest)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = load_config(args.config)
    if not config.temporal_mil:
        raise ValueError("--config must set temporal_mil=true")
    auxiliary_store = AuxiliaryFeatureStore(args.aux_features) if args.aux_features else None
    if config.profile_dim or config.graph_dim:
        if auxiliary_store is None:
            raise ValueError("multi-branch Temporal-MIL config requires --aux-features")
        if (config.profile_dim, config.graph_dim) != (
            auxiliary_store.profile_dim, auxiliary_store.graph_dim
        ):
            raise ValueError("config and auxiliary feature dimensions differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = MUDformer(config).to(device)
    model.split_identity = split_identity
    initialization: dict[str, object] | None = None
    if args.initialize_checkpoint is not None:
        initialization = initialize_temporal_mil(
            model, args.initialize_checkpoint, config, args.freeze_base
        )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("no trainable parameters remain")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_data = UserFeatureDataset(
        args.train_manifest, args.feature_root, args.max_length, training=True, random_crop=True,
        auxiliary_features=auxiliary_store,
    )
    train_data.records = [r for r in train_data.records if r.behavior_count >= 50]
    train_synthetic_data = SyntheticRiskBlockDataset(
        args.train_manifest, args.train_manifest, args.feature_root, args.max_length,
        auxiliary_features=auxiliary_store,
    )
    validation_data = (
        UserFeatureDataset(args.validation_manifest, args.feature_root, args.max_length, training=False, auxiliary_features=auxiliary_store)
        if args.validation_manifest is not None else None
    )
    validation_normal_data = None
    validation_synthetic_data = None
    if args.validation_manifest is not None:
        validation_normal_data = UserFeatureDataset(
            args.validation_manifest, args.feature_root, args.max_length, training=False,
            auxiliary_features=auxiliary_store,
        )
        validation_normal_data.records = [record for record in validation_normal_data.records if record.label == 0 and record.behavior_count >= 50]
        validation_synthetic_data = SyntheticRiskBlockDataset(
            args.validation_manifest, args.train_manifest, args.feature_root, args.max_length,
            auxiliary_features=auxiliary_store,
        )
        if [record.user_id for record in validation_normal_data.records] != [record.user_id for record in validation_synthetic_data.records]:
            raise AssertionError("internal normal and synthetic validation users are not aligned")
    test_data = (
        UserFeatureDataset(args.test_manifest, args.feature_root, args.max_length, training=False, auxiliary_features=auxiliary_store)
        if args.test_manifest is not None and not args.skip_test_eval else None
    )
    if test_data is not None:
        test_data.records = [r for r in test_data.records if r.behavior_count >= 50]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = _loader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
                           workers=args.num_workers, device=device)
    synthetic_loader = _loader(train_synthetic_data, batch_size=args.batch_size, shuffle=True, generator=generator,
                               workers=args.num_workers, device=device)
    validation_loader = (
        _loader(validation_data, batch_size=args.batch_size, shuffle=False, generator=None, workers=args.num_workers, device=device)
        if validation_data is not None else None
    )
    validation_normal_loader = (
        _loader(validation_normal_data, batch_size=args.batch_size, shuffle=False, generator=None, workers=args.num_workers, device=device)
        if validation_normal_data is not None else None
    )
    validation_synthetic_loader = (
        _loader(validation_synthetic_data, batch_size=args.batch_size, shuffle=False, generator=None, workers=args.num_workers, device=device)
        if validation_synthetic_data is not None else None
    )
    test_loader = (
        _loader(test_data, batch_size=args.batch_size, shuffle=False, generator=None,
                workers=args.num_workers, device=device)
        if test_data is not None else None
    )

    latest_path = args.output_dir / "latest.pt"
    resume_path = args.resume
    if resume_path is None and latest_path.is_file() and not args.no_auto_resume:
        resume_path = latest_path
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config.to_dict():
            raise ValueError("resume checkpoint model configuration does not match --config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", float("-inf")))
        best_epoch = int(checkpoint.get("best_epoch") or 0)
        stale_epochs = int(checkpoint.get("stale_epochs") or 0)
        if checkpoint.get("generator_state") is not None:
            generator.set_state(checkpoint["generator_state"].cpu())
        restore_rng_state(checkpoint)
        print(f"Resuming from {resume_path} at epoch {start_epoch}", flush=True)

    run_metadata = {
        "split_identity": split_identity,
        "split_ratios": [0.8, 0.1, 0.1],
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": config.to_dict(),
        "fixed_partition": {"train_pool": len(train_data), "fixed_test": len(test_data) if test_data is not None else 0},
        "test_policy": (
            "not loaded during internal-only development"
            if args.skip_test_eval else "not read for checkpoint selection; evaluated once after the selected/fixed-epoch checkpoint"
        ),
        "auxiliary_supervision": {
            "target": "a known 20%-length chronological block copied from a train-pool spammer into a train-pool normal history",
            "source_population": "only train manifest users; internal validation targets are disjoint from their donor pool",
            "loss": "synthetic-spammer user cross-entropy plus five-way temporal risk-block cross-entropy",
            "risk_head": "one score per input behaviour before hierarchy coarsening",
            "attribution_alignment": (
                "second-order train-only Gradient x Input five-block cross-entropy"
                if args.attribution_block_weight > 0.0 else "disabled"
            ),
        },
        "initialization": initialization,
        "checkpoint_selection": (
            "internal validation synthetic TGLS proxy plus 0.10 * internal validation accuracy"
            if validation_data is not None else "fixed epoch count determined before this all-training-pool fit"
        ),
        "train_users": len(train_data),
        "train_synthetic_normal_targets": len(train_synthetic_data),
        "validation_users": len(validation_data) if validation_data is not None else 0,
        "fixed_test_users": len(test_data) if test_data is not None else 0,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "hardware": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "cuda": torch.version.cuda, "amp": amp_enabled,
        },
    }
    write_json_atomic(args.output_dir / "run.json", run_metadata)

    synthetic_iterator = iter(synthetic_loader)
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        if validation_data is not None and stale_epochs >= args.patience:
            stopped_early = True
            break
        model.train()
        if args.freeze_base:
            # Preserve the already selected user-level backbone exactly: no
            # classifier/encoder dropout or batch-stat updates while the new
            # deterministic risk head receives synthetic block supervision.
            model.eval()
            model.temporal_risk_head.train()
        totals: dict[str, float] = {}
        seen = 0
        started = time.time()
        for clean_batch in tqdm(train_loader, desc=f"temporal-mil epoch {epoch}/{args.epochs}"):
            try:
                injected_batch = next(synthetic_iterator)
            except StopIteration:
                synthetic_iterator = iter(synthetic_loader)
                injected_batch = next(synthetic_iterator)
            clean_batch = move_batch(clean_batch, device)
            injected_batch = move_batch(injected_batch, device)
            if args.attribution_block_weight > 0.0:
                injected_batch["text"] = injected_batch["text"].detach().requires_grad_(True)
                injected_batch["image"] = injected_batch["image"].detach().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                clean_outputs = model(
                    clean_batch["text"], clean_batch["image"], clean_batch["sequence_mask"],
                    clean_batch["image_present"], sample_latent=False,
                    **auxiliary_kwargs(clean_batch),
                )
                clean_losses = model.compute_loss(
                    clean_outputs, clean_batch["text"], clean_batch["image"], clean_batch["sequence_mask"],
                    clean_batch["image_present"], clean_batch["labels"], args.mvae_weight,
                )
                audit_clean = F.cross_entropy(audit_logits_from_output(clean_outputs), clean_batch["labels"])
                clean_distillation = clean_losses["total"].new_zeros(())
                if args.clean_distillation_weight > 0.0:
                    teacher_logits = base_logits_from_temporal_output(clean_outputs, config).detach()
                    clean_distillation = F.kl_div(
                        audit_logits_from_output(clean_outputs).log_softmax(dim=-1),
                        teacher_logits.softmax(dim=-1), reduction="batchmean",
                    )
                injected_outputs = model(
                    injected_batch["text"], injected_batch["image"], injected_batch["sequence_mask"],
                    injected_batch["image_present"], sample_latent=False,
                    **auxiliary_kwargs(injected_batch),
                )
                synthetic_user = F.cross_entropy(
                    audit_logits_from_output(injected_outputs), injected_batch["labels"]
                )
                risk_logits = injected_outputs.get("temporal_risk_logits")
                if risk_logits is None:
                    raise AssertionError("temporal-MIL model unexpectedly lacks risk logits")
                block_scores = temporal_block_scores(
                    risk_logits, injected_batch["sequence_mask"], config.temporal_mil_temperature
                )
                risk_block = F.cross_entropy(block_scores, injected_batch["risk_blocks"])
                attribution_block = risk_block.new_zeros(())
                if args.attribution_block_weight > 0.0:
                    # The frozen hierarchy uses PyTorch's memory-efficient
                    # attention kernel, whose second derivative is not
                    # implemented.  Its contribution is parameter-free in
                    # this training mode.  We therefore align the exact V1
                    # signed GxI *block aggregation* for the learnable risk
                    # margin component; that component is added with scale 3
                    # to the final audit margin at evaluation time.
                    audit_margin = (
                        config.temporal_mil_scale
                        * injected_outputs["temporal_mil_evidence"]
                    ).sum()
                    text_gradient, image_gradient = torch.autograd.grad(
                        audit_margin, (injected_batch["text"], injected_batch["image"]),
                        create_graph=True,
                    )
                    contribution = (text_gradient * injected_batch["text"]).sum(dim=-1)
                    contribution = contribution + (
                        (image_gradient * injected_batch["image"]).sum(dim=-1)
                        * injected_batch["image_present"].to(contribution.dtype)
                    )
                    attribution_scores = temporal_attribution_block_scores(
                        contribution, injected_batch["sequence_mask"], args.attribution_temperature
                    )
                    attribution_block = F.cross_entropy(attribution_scores, injected_batch["risk_blocks"])
                total = (
                    clean_losses["total"]
                    + args.clean_audit_weight * audit_clean
                    + args.synthetic_user_weight * synthetic_user
                    + args.risk_block_weight * risk_block
                    + args.clean_distillation_weight * clean_distillation
                    + args.attribution_block_weight * attribution_block
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(clean_batch["labels"].shape[0])
            seen += count
            values = {
                "total": total,
                "clean_total": clean_losses["total"],
                "clean_classification": clean_losses["classification"],
                "clean_audit": audit_clean,
                "synthetic_user": synthetic_user,
                "risk_block": risk_block,
                "clean_distillation": clean_distillation,
                "attribution_block": attribution_block,
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count

        record: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "batch_size": args.batch_size,
            **{name: value / max(seen, 1) for name, value in totals.items()},
        }
        improved = False
        if validation_loader is not None and validation_normal_loader is not None and validation_synthetic_loader is not None:
            validation_metrics, _, _ = evaluate_user_classification(
                model, validation_loader, device, f"internal validation {epoch}/{args.epochs}"
            )
            validation_localization = evaluate_synthetic_localization(
                model, validation_normal_loader, validation_synthetic_loader, device,
                config.temporal_mil_temperature,
            )
            score = float(validation_localization["internal_synthetic_TGLS_at_1"]) + (
                args.selection_accuracy_weight * float(validation_metrics["accuracy"])
            )
            improved = score > best_score
            if improved:
                best_score, best_epoch, stale_epochs = score, epoch, 0
            else:
                stale_epochs += 1
            record.update({
                "internal_validation_metrics": validation_metrics,
                "internal_validation_synthetic_localization": validation_localization,
                "selection_score": score,
                "improved": improved,
                "best_score": best_score,
                "stale_epochs": stale_epochs,
            })
        else:
            best_score, best_epoch, improved = float("nan"), epoch, True
            record.update({"selection_score": None, "improved": True, "best_score": None, "stale_epochs": 0})
        history.append(record)
        write_json_atomic(args.output_dir / "history.json", history)
        checkpoint_kwargs = {
            "scaler": scaler, "history": history, "best_score": best_score,
            "best_epoch": best_epoch, "stale_epochs": stale_epochs,
            "generator_state": generator.get_state(),
        }
        save_checkpoint(latest_path, model, optimizer, epoch, args.seed, **checkpoint_kwargs)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, args.seed, **checkpoint_kwargs)
        print(json.dumps({"epoch": epoch, **{key: record[key] for key in record if key not in {"internal_validation_metrics", "internal_validation_synthetic_localization"}}}, ensure_ascii=False), flush=True)
        if validation_data is not None and stale_epochs >= args.patience:
            stopped_early = True
            break

    source = args.output_dir / "best.pt" if validation_data is not None else latest_path
    if not source.is_file():
        raise RuntimeError("no checkpoint is available for final evaluation")
    final_path = args.output_dir / "final.pt"
    temporary_final = final_path.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary_final)
    os.replace(temporary_final, final_path)
    final_checkpoint = torch.load(final_path, map_location=device, weights_only=False)
    model.load_state_dict(final_checkpoint["model"])
    test_metrics: dict[str, Any] | None = None
    if test_loader is not None:
        test_metrics, user_predictions, audit_predictions = evaluate_user_classification(
            model, test_loader, device, "fixed test once"
        )
        # The behavioural audit output is intentionally stored separately from
        # the user-identification metrics.  It is the only prediction source
        # passed to the controlled temporal localization evaluator.
        write_json_atomic(args.output_dir / "test_metrics.json", test_metrics)
        write_json_atomic(args.output_dir / "predictions.json", user_predictions)
        write_json_atomic(args.output_dir / "audit_predictions.json", audit_predictions)
    write_json_atomic(args.output_dir / "training_summary.json", {
        "requested_epochs": args.epochs,
        "completed_epoch": int(history[-1]["epoch"]) if history else start_epoch - 1,
        "best_epoch": int(final_checkpoint["epoch"]),
        "checkpoint_selection": run_metadata["checkpoint_selection"],
        "stopped_early": stopped_early,
        "final_checkpoint": str(final_path),
        "training_seconds": sum(float(record["seconds"]) for record in history),
        "fixed_test_evaluated_once_after_selection": test_loader is not None,
        "hardware": run_metadata["hardware"],
    })
    print(json.dumps({"test_metrics": test_metrics, "checkpoint": str(final_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
