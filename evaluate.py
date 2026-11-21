#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mudformer.data import AuxiliaryFeatureStore, UserFeatureDataset, collate_users
from mudformer.runtime import classification_metrics, load_model_checkpoint, move_batch, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen MUDformer checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--aux-features", type=Path)
    parser.add_argument("--logit-key", choices=("logits", "audit_logits"), default="logits")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    model = load_model_checkpoint(args.checkpoint, device).eval()
    auxiliary = AuxiliaryFeatureStore(args.aux_features) if args.aux_features else None
    if model.config.profile_dim or model.config.graph_dim:
        if auxiliary is None or (auxiliary.profile_dim, auxiliary.graph_dim) != (model.config.profile_dim, model.config.graph_dim):
            raise ValueError("Supply --aux-features with dimensions matching the checkpoint")
    dataset = UserFeatureDataset(args.manifest, args.feature_root, args.max_length, training=False, auxiliary_features=auxiliary)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", collate_fn=collate_users
    )
    labels, probabilities, predictions = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate"):
            batch = move_batch(batch, device)
            outputs = model(
                batch["text"], batch["image"], batch["sequence_mask"],
                batch["image_present"], sample_latent=False,
                **{k: batch[k] for k in ("profile", "graph", "profile_present", "graph_present", "behavior_present") if k in batch}
            )
            probability = outputs[args.logit_key].softmax(-1)[:, 1].cpu().tolist()
            labels.extend(batch["labels"].cpu().tolist())
            probabilities.extend(probability)
            predictions.extend(
                {"user_id": uid, "label": int(label), "spammer_probability": float(prob)}
                for uid, label, prob in zip(batch["user_ids"], batch["labels"].cpu(), probability)
            )

    payload = {"metrics": classification_metrics(labels, probabilities), "predictions": predictions}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(payload["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(
            json.dumps(payload["predictions"], ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
