#!/usr/bin/env python3
"""Controlled injected-risk-block localization for one frozen MUDformer seed.

The evaluator is test-only.  It copies a deterministic 20%-length cached
behaviour block from a fixed training spammer into each fixed-test normal
user's retained sequence, preserving the recipient sequence length and event
timestamps.  A successful result must (1) turn a saved clean normal prediction
into spammer and (2) rank the known injected chronological block first using
signed Gradient x Input on the spammer-minus-normal logit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_io import load_features  # noqa: E402
from mudformer.data import AuxiliaryFeatureStore, ManifestRecord, collate_users, load_manifest  # noqa: E402
from mudformer.runtime import load_model_checkpoint, move_batch, seed_everything  # noqa: E402
from mudformer.temporal_evidence import retained_start  # noqa: E402
from mudformer.temporal_localization import (  # noqa: E402
    NUM_TEMPORAL_BLOCKS,
    InjectionRecord,
    protocol_metadata,
    sequence_injection_records,
    temporal_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clean-predictions", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--aux-features", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decision-calibration", type=Path,
        help="Train-only calibration receipt containing the frozen audit decision threshold.",
    )
    parser.add_argument(
        "--logit-key", default="logits",
        help="Model output used for injected-risk detection and Gradient x Input; default preserves legacy user logits.",
    )
    parser.add_argument("--model-name", default="Final_MUDformer")
    parser.add_argument(
        "--selection-protocol", default="user_stratified_8_1_1_validation_selected",
        help="Recorded provenance only; it does not influence V1 injection or scoring.",
    )
    parser.add_argument(
        "--auxiliary-temporal-training", action="store_true",
        help="Record that the supplied checkpoint was trained with train-only temporal supervision.",
    )
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def saved_probabilities(path: Path, expected_ids: set[str]) -> dict[str, float]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("saved MUDformer predictions must be a JSON list")
    raw_ids = [str(row["user_id"]) for row in values]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("Duplicate users in saved clean predictions")
    result = {str(row["user_id"]): float(row["spammer_probability"]) for row in values if str(row["user_id"]) in expected_ids}
    if set(result) != expected_ids or len(result) != len(expected_ids):
        raise ValueError("saved clean predictions do not cover exactly the test manifest")
    return result


def _resample_indices(source: np.ndarray, count: int) -> np.ndarray:
    if source.size == 0 or count <= 0:
        raise ValueError("cannot inject an empty temporal block")
    offsets = np.rint(np.linspace(0, source.size - 1, count)).astype(np.int64)
    return source[offsets]


@dataclass
class FeatureProvider:
    feature_root: Path
    max_length: int
    records: dict[str, ManifestRecord]
    plan: dict[str, InjectionRecord]
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    auxiliary: AuxiliaryFeatureStore | None = None

    def clean(self, record: ManifestRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if record.user_id not in self.cache:
            text, image, present = load_features(record, self.feature_root, self.max_length)
            self.cache[record.user_id] = (text, image, present)
        text, image, present = self.cache[record.user_id]
        return text.copy(), image.copy(), present.copy()

    def injected(self, record: ManifestRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        text, image, present = self.clean(record)
        item = self.plan.get(record.user_id)
        if item is None:
            return text, image, present, np.empty(0, dtype=np.int64)
        donor = self.records[item.donor_id]
        donor_text, donor_image, donor_present = self.clean(donor)
        target_group = temporal_groups(len(text))[item.block_index]
        donor_group = temporal_groups(len(donor_text))[item.block_index]
        sampled = _resample_indices(donor_group, len(target_group))
        text[target_group] = donor_text[sampled]
        image[target_group] = donor_image[sampled]
        present[target_group] = donor_present[sampled]
        return text, image, present, target_group


def sample(
    record: ManifestRecord, provider: FeatureProvider, *, injected: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    if injected:
        text, image, present, target_group = provider.injected(record)
    else:
        text, image, present = provider.clean(record)
        target_group = np.empty(0, dtype=np.int64)
    result = {
        "text": torch.from_numpy(text), "image": torch.from_numpy(image),
        "image_present": torch.from_numpy(present), "label": record.label, "user_id": record.user_id,
        "behavior_present": record.behavior_count >= 50,
    }
    if provider.auxiliary is not None:
        profile, graph, profile_present, graph_present = provider.auxiliary.get(record.user_id)
        result.update(
            {
                "profile": torch.from_numpy(profile.copy()),
                "graph": torch.from_numpy(graph.copy()),
                "profile_present": profile_present,
                "graph_present": graph_present,
            }
        )
    return result, target_group


def auxiliary_kwargs(batch: dict[str, object]) -> dict[str, torch.Tensor]:
    return {
        name: batch[name]
        for name in ("profile", "graph", "profile_present", "graph_present", "behavior_present")
        if isinstance(batch.get(name), torch.Tensor)
    }


def predict_all(
    model: torch.nn.Module, records: list[ManifestRecord], provider: FeatureProvider,
    *, injected: bool, batch_size: int, device: torch.device, logit_key: str,
    decision_threshold: float, score_space: str = "probability",
) -> dict[str, tuple[int, float]]:
    result: dict[str, tuple[int, float]] = {}
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            values = [sample(record, provider, injected=injected)[0] for record in records[start : start + batch_size]]
            batch = move_batch(collate_users(values), device)
            outputs = model(
                batch["text"], batch["image"], batch["sequence_mask"], batch["image_present"], sample_latent=False,
                **auxiliary_kwargs(batch),
            )
            if logit_key not in outputs:
                raise ValueError(f"model output does not contain requested --logit-key {logit_key}")
            logits = outputs[logit_key]
            probabilities = logits.softmax(dim=-1)[:, 1].detach().cpu().tolist()
            decisions = (logits[:, 1].double() - logits[:, 0].double()).cpu().tolist() if score_space == "margin" else probabilities
            for user_id, probability, decision in zip(batch["user_ids"], probabilities, decisions, strict=True):
                result[str(user_id)] = (int(decision >= decision_threshold), float(probability))
    return result


def temporal_scores(
    model: torch.nn.Module, record: ManifestRecord, provider: FeatureProvider,
    device: torch.device, logit_key: str,
) -> tuple[list[float], np.ndarray]:
    values, target_group = sample(record, provider, injected=True)
    text = values["text"].unsqueeze(0).to(device).requires_grad_(True)
    image = values["image"].unsqueeze(0).to(device).requires_grad_(True)
    present = values["image_present"].unsqueeze(0).to(device)
    sequence_mask = torch.ones((1, text.shape[1]), dtype=torch.bool, device=device)
    aux_batch: dict[str, object] = {}
    for name in ("profile", "graph"):
        if name in values:
            aux_batch[name] = values[name].unsqueeze(0).to(device)
            aux_batch[f"{name}_present"] = torch.as_tensor(
                [values[f"{name}_present"]], dtype=torch.bool, device=device
            )
    model.zero_grad(set_to_none=True)
    outputs = model(
        text, image, sequence_mask, present, sample_latent=False,
        **auxiliary_kwargs(aux_batch),
    )
    if logit_key not in outputs:
        raise ValueError(f"model output does not contain requested --logit-key {logit_key}")
    logits = outputs[logit_key]
    margin = logits[0, 1] - logits[0, 0]
    text_gradient, image_gradient = torch.autograd.grad(margin, (text, image), create_graph=False)
    contribution = (text_gradient * text).sum(dim=-1)[0]
    contribution = contribution + (image_gradient * image).sum(dim=-1)[0] * present[0].to(text.dtype)
    groups = temporal_groups(text.shape[1])
    scores = [float(contribution[group].sum().detach().cpu()) for group in groups]
    return scores, target_group


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.max_length < NUM_TEMPORAL_BLOCKS or args.batch_size <= 0:
        raise ValueError("max length and batch size are invalid")
    seed_everything(args.seed)
    train_records = load_manifest(args.train_manifest)
    test_records = load_manifest(args.test_manifest)
    original_partition_counts = {"train": len(train_records), "test": len(test_records)}
    train_records = [r for r in train_records if r.behavior_count >= 50]
    test_records = [r for r in test_records if r.behavior_count >= 50]
    train_by_id = {record.user_id: record for record in train_records}
    test_by_id = {record.user_id: record for record in test_records}
    if len(train_by_id) != len(train_records) or len(test_by_id) != len(test_records) or set(train_by_id) & set(test_by_id):
        raise ValueError("train/test manifests do not have disjoint unique user IDs")
    eligible_train = [r for r in train_records if r.behavior_count >= 50]
    eligible_test = [r for r in test_records if r.behavior_count >= 50]
    records, protocol = sequence_injection_records(
        [record.user_id for record in eligible_train], [record.label for record in eligible_train],
        [record.user_id for record in eligible_test], [record.label for record in eligible_test],
    )
    plan = {item.user_id: item for item in records}
    all_records = {**train_by_id, **test_by_id}
    auxiliary = AuxiliaryFeatureStore(args.aux_features) if args.aux_features else None
    provider = FeatureProvider(args.feature_root, args.max_length, all_records, plan, {}, auxiliary)
    saved = saved_probabilities(args.clean_predictions, set(test_by_id))
    score_space = "probability"
    decision_threshold = 0.5
    calibration: dict[str, Any] | None = None
    if args.decision_calibration is not None:
        calibration = json.loads(args.decision_calibration.read_text(encoding="utf-8"))
        if calibration.get("protocol", {}).get("test_manifest_accessed") is not False:
            raise ValueError("decision calibration must explicitly attest that the test manifest was not accessed")
        if int(calibration.get("seed")) != args.seed:
            raise ValueError("decision calibration seed does not match evaluator seed")
        score_space = calibration.get("decision_score_space", "probability")
        if score_space not in ("probability", "margin"): raise ValueError("unknown decision score space")
        decision_threshold = float(calibration["decision_threshold"])
        if score_space == "probability" and not 0.0 < decision_threshold < 1.0:
            raise ValueError("calibrated decision threshold must lie strictly between zero and one")
    device = torch.device(args.device)
    model = load_model_checkpoint(args.checkpoint, device).eval()
    if model.config.profile_dim or model.config.graph_dim:
        if auxiliary is None:
            raise ValueError("multi-branch checkpoint requires --aux-features")
        if (model.config.profile_dim, model.config.graph_dim) != (
            auxiliary.profile_dim, auxiliary.graph_dim
        ):
            raise ValueError("checkpoint and auxiliary feature dimensions differ")
    replay = predict_all(
        model, test_records, provider, injected=False, batch_size=args.batch_size,
        device=device, logit_key=args.logit_key, decision_threshold=decision_threshold, score_space=score_space,
    )
    replay_probabilities = {user_id: probability for user_id, (_, probability) in replay.items()}
    if any(abs(replay_probabilities[user_id] - probability) > 1e-5 for user_id, probability in saved.items()):
        raise ValueError("clean replay probabilities do not reproduce the saved MUDformer audit outputs")
    injected = predict_all(
        model, test_records, provider, injected=True, batch_size=args.batch_size,
        device=device, logit_key=args.logit_key, decision_threshold=decision_threshold, score_space=score_space,
    )
    rows: list[dict[str, Any]] = []
    for item in records:
        injected_prediction, injected_probability = injected[item.user_id]
        clean_prediction, clean_probability = replay[item.user_id]
        detected = bool(clean_prediction == 0 and injected_prediction == 1)
        scores: list[float] | None = None
        target_indices: np.ndarray | None = None
        top_block: int | None = None
        if detected:
            scores, target_indices = temporal_scores(model, test_by_id[item.user_id], provider, device, args.logit_key)
            top_block = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        target_record = test_by_id[item.user_id]
        rows.append({
            "user_id": item.user_id,
            "target_block": item.block_index,
            "donor_user_id": item.donor_id,
            "clean_prediction": clean_prediction,
            "clean_spammer_probability": clean_probability,
            "injected_prediction": injected_prediction,
            "injected_spammer_probability": injected_probability,
            "injection_detection_success": detected,
            "top_attributed_block": top_block,
            "localization_hit_at_1": bool(top_block == item.block_index) if top_block is not None else False,
            "temporal_gradient_x_input": scores,
            "retained_behavior_count": min(target_record.behavior_count, args.max_length),
            "retained_global_start_index": retained_start(target_record.behavior_count, args.max_length),
            "target_local_start_index": int(target_indices[0]) if target_indices is not None else None,
            "target_local_end_index": int(target_indices[-1]) if target_indices is not None else None,
        })
    count = len(rows)
    detections = sum(bool(row["injection_detection_success"]) for row in rows)
    hits = sum(bool(row["localization_hit_at_1"]) for row in rows)
    summary = {
        "model": args.model_name,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "clean_predictions": str(args.clean_predictions),
        "protocol": {
            **protocol,
            "historical_representation": "cached per-behaviour text/image representations; recipient length and timestamps retained",
            "input_attribution": "signed Gradient x Input of spammer-minus-normal logit, aggregated into five equal-count chronological blocks",
            "decision_logit_key": args.logit_key,
            "decision_threshold": decision_threshold,
            "decision_score_space": score_space,
            "decision_calibration": str(args.decision_calibration) if args.decision_calibration else None,
            "decision_calibration_protocol": calibration.get("protocol") if calibration else None,
            "detection_requirement": "saved clean prediction normal and injected prediction spammer",
            "primary_metric": "TGLS@1 = mean[controlled injection detection success AND Top-1 block equals the known injected block]",
            "training_or_checkpoint_modification": False,
            "auxiliary_temporal_training_before_evaluation": args.auxiliary_temporal_training,
            "fixed_partition": original_partition_counts,
            "min50_users_evaluated": {"train":len(train_records), "test":len(test_records)},
            "below50_history_users_skipped": True,
            "selection_protocol": args.selection_protocol,
            "max_length": args.max_length,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "metrics": {
            "target_normal_test_users": count,
            "clean_correct_normal_targets": int(sum(row["clean_prediction"] == 0 for row in rows)),
            "clean_normal_false_positive_rate": sum(row["clean_prediction"] != 0 for row in rows) / count,
            "injection_detection_successes": detections,
            "injection_detection_rate": float(detections / count),
            "top1_localization_hits_given_detection": hits,
            "localization_recall_at_1_given_detection": float(hits / detections) if detections else None,
            "TGLS_at_1": float(hits / count),
            "top1_chance_rate": 1.0 / NUM_TEMPORAL_BLOCKS,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_user.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
