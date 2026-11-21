from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .temporal_localization import (
    InjectionRecord,
    synthetic_training_sequence_injection_records,
    temporal_groups,
)


@dataclass(frozen=True)
class ManifestRecord:
    user_id: str
    label: int
    path: str
    behavior_count: int


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    return [ManifestRecord(**record) for record in records]


class AuxiliaryFeatureStore:
    """In-memory, user-keyed profile and label-free graph feature views."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            user_ids = data["user_ids"].astype(str).tolist()
            profile = np.asarray(data["profile_features"], dtype=np.float32)
            graph = np.asarray(data["graph_features"], dtype=np.float32)
            profile_present = np.asarray(data["profile_present"], dtype=np.bool_)
            graph_present = np.asarray(data["graph_present"], dtype=np.bool_)
        if len(set(user_ids)) != len(user_ids):
            raise ValueError(f"Duplicate user IDs in {self.path}")
        expected = (len(user_ids),)
        if profile.ndim != 2 or graph.ndim != 2:
            raise ValueError("Auxiliary feature matrices must be two-dimensional")
        if profile.shape[0] != len(user_ids) or graph.shape[0] != len(user_ids):
            raise ValueError("Auxiliary feature rows do not match user IDs")
        if profile_present.shape != expected or graph_present.shape != expected:
            raise ValueError("Auxiliary presence masks do not match user IDs")
        if not np.isfinite(profile).all() or not np.isfinite(graph).all():
            raise ValueError("Auxiliary features contain non-finite values")
        self.profile_dim = int(profile.shape[1])
        self.graph_dim = int(graph.shape[1])
        self._rows = {
            user_id: (
                profile[index].copy(), graph[index].copy(),
                bool(profile_present[index]), bool(graph_present[index]),
            )
            for index, user_id in enumerate(user_ids)
        }

    def get(self, user_id: str) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        try:
            return self._rows[user_id]
        except KeyError as error:
            raise KeyError(f"User {user_id} is absent from {self.path}") from error


class UserFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        feature_root: str | Path | None = None,
        max_length: int = 4096,
        training: bool = False,
        random_crop: bool = True,
        auxiliary_features: AuxiliaryFeatureStore | str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.records = load_manifest(self.manifest_path)
        self.feature_root = (
            Path(feature_root) if feature_root is not None else self.manifest_path.parent
        )
        self.max_length = max_length
        self.training = training
        self.random_crop = random_crop
        self.auxiliary_features = (
            auxiliary_features
            if isinstance(auxiliary_features, AuxiliaryFeatureStore)
            else AuxiliaryFeatureStore(auxiliary_features)
            if auxiliary_features is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_record(self, record: ManifestRecord) -> dict[str, Any]:
        feature_path = self.feature_root / record.path
        with np.load(feature_path, allow_pickle=False) as data:
            text = np.asarray(data["text"], dtype=np.float32)
            image = np.asarray(data["image"], dtype=np.float32)
            image_present = np.asarray(data["image_present"], dtype=np.bool_)
            declared_behavior_present = (
                bool(data["behavior_present"].item()) if "behavior_present" in data else None
            )

        if text.ndim != 2 or image.shape != text.shape:
            raise ValueError(f"Invalid feature shapes in {feature_path}: {text.shape}, {image.shape}")
        if image_present.shape != (text.shape[0],):
            raise ValueError(f"Invalid image_present shape in {feature_path}")
        length = text.shape[0]
        behavior_present = (length > 0 if declared_behavior_present is None
                            else declared_behavior_present)
        if length == 0:
            if self.auxiliary_features is None:
                raise ValueError(f"Empty user sequence requires auxiliary views: {feature_path}")
            # A zero numerical slot keeps attention finite. It carries no real
            # behavior and is excluded by the explicit availability flag.
            text = np.zeros((1, text.shape[1]), dtype=np.float32)
            image = np.zeros_like(text)
            image_present = np.zeros(1, dtype=np.bool_)
        if length > self.max_length:
            if self.training and self.random_crop:
                start = random.randint(0, length - self.max_length)
            else:
                start = length - self.max_length
            stop = start + self.max_length
            text, image, image_present = text[start:stop], image[start:stop], image_present[start:stop]

        result = {
            "text": torch.from_numpy(text.copy()),
            "image": torch.from_numpy(image.copy()),
            "image_present": torch.from_numpy(image_present.copy()),
            "label": record.label,
            "user_id": record.user_id,
            "behavior_present": behavior_present,
        }
        if self.auxiliary_features is not None:
            profile, graph, profile_present, graph_present = self.auxiliary_features.get(
                record.user_id
            )
            if not behavior_present and not profile_present and not graph_present:
                raise ValueError(f"User has no available input view: {record.user_id}")
            result.update(
                {
                    "profile": torch.from_numpy(profile.copy()),
                    "graph": torch.from_numpy(graph.copy()),
                    "profile_present": profile_present,
                    "graph_present": graph_present,
                }
            )
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._load_record(self.records[index])


class SyntheticRiskBlockDataset(UserFeatureDataset):
    """Train-only normal histories with one deterministic spammer block copied in.

    A sample is labelled as spammer and carries the known five-way block target.
    Donors and targets are supplied through manifests so validation targets can
    remain disjoint from the donor pool used for checkpoint selection.
    """

    def __init__(
        self,
        target_manifest: str | Path,
        donor_manifest: str | Path,
        feature_root: str | Path | None = None,
        max_length: int = 4096,
        auxiliary_features: AuxiliaryFeatureStore | str | Path | None = None,
    ) -> None:
        super().__init__(
            target_manifest, feature_root, max_length, training=False, random_crop=False,
            auxiliary_features=auxiliary_features,
        )
        # The manifests describe a labelled user population.  This auxiliary
        # loader uses its normal users as recipients and its spammer users as
        # donors; the source manifests themselves may contain both classes.
        self.records = [record for record in self.records if record.label == 0]
        donor_records = [record for record in load_manifest(donor_manifest) if record.label == 1]
        if not self.records or not donor_records:
            raise ValueError("synthetic risk-block data needs normal targets and spammer donors")
        if {record.user_id for record in self.records} & {record.user_id for record in donor_records}:
            raise ValueError("synthetic targets and donors must be user-disjoint")
        self.donors = {record.user_id: record for record in donor_records}
        plan = synthetic_training_sequence_injection_records(
            [record.user_id for record in self.records],
            [record.user_id for record in donor_records],
        )
        self.plan: dict[str, InjectionRecord] = {record.user_id: record for record in plan}
        if set(self.plan) != {record.user_id for record in self.records}:
            raise AssertionError("synthetic injection plan does not cover all targets")

    @staticmethod
    def _resample_indices(source: np.ndarray, count: int) -> np.ndarray:
        if source.size == 0 or count <= 0:
            raise ValueError("cannot inject an empty temporal block")
        offsets = np.rint(np.linspace(0, source.size - 1, count)).astype(np.int64)
        return source[offsets]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target = self._load_record(record)
        plan = self.plan[record.user_id]
        donor = self._load_record(self.donors[plan.donor_id])
        target_groups = temporal_groups(target["text"].shape[0])
        donor_groups = temporal_groups(donor["text"].shape[0])
        target_indices = target_groups[plan.block_index]
        donor_indices = self._resample_indices(
            donor_groups[plan.block_index], len(target_indices)
        )
        target["text"][target_indices] = donor["text"][donor_indices]
        target["image"][target_indices] = donor["image"][donor_indices]
        target["image_present"][target_indices] = donor["image_present"][donor_indices]
        target["label"] = 1
        target["risk_block"] = plan.block_index
        target["donor_user_id"] = plan.donor_id
        return target


def collate_users(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    max_length = max(item["text"].shape[0] for item in samples)
    dim = samples[0]["text"].shape[1]
    batch_size = len(samples)
    text = torch.zeros(batch_size, max_length, dim, dtype=torch.float32)
    image = torch.zeros_like(text)
    sequence_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    image_present = torch.zeros_like(sequence_mask)
    labels = torch.empty(batch_size, dtype=torch.long)
    user_ids: list[str] = []
    risk_blocks = torch.empty(batch_size, dtype=torch.long) if all(
        "risk_block" in item for item in samples
    ) else None
    has_auxiliary = ["profile" in item and "graph" in item for item in samples]
    if any(has_auxiliary) and not all(has_auxiliary):
        raise ValueError("Cannot mix samples with and without auxiliary features")
    profiles = None
    graphs = None
    profile_present = None
    graph_present = None
    if all(has_auxiliary):
        profile_dim = samples[0]["profile"].shape[0]
        graph_dim = samples[0]["graph"].shape[0]
        profiles = torch.empty(batch_size, profile_dim, dtype=torch.float32)
        graphs = torch.empty(batch_size, graph_dim, dtype=torch.float32)
        profile_present = torch.empty(batch_size, dtype=torch.bool)
        graph_present = torch.empty(batch_size, dtype=torch.bool)

    for row, item in enumerate(samples):
        length = item["text"].shape[0]
        text[row, :length] = item["text"]
        image[row, :length] = item["image"]
        sequence_mask[row, :length] = True
        image_present[row, :length] = item["image_present"]
        labels[row] = item["label"]
        user_ids.append(item["user_id"])
        if risk_blocks is not None:
            risk_blocks[row] = int(item["risk_block"])
        if profiles is not None and graphs is not None:
            profiles[row] = item["profile"]
            graphs[row] = item["graph"]
            profile_present[row] = bool(item["profile_present"])
            graph_present[row] = bool(item["graph_present"])

    result: dict[str, Any] = {
        "text": text,
        "image": image,
        "sequence_mask": sequence_mask,
        "image_present": image_present,
        "labels": labels,
        "user_ids": user_ids,
        "behavior_present": torch.tensor(
            [item.get("behavior_present", True) for item in samples], dtype=torch.bool
        ),
    }
    if risk_blocks is not None:
        result["risk_blocks"] = risk_blocks
    if profiles is not None and graphs is not None:
        result.update(
            {
                "profile": profiles,
                "graph": graphs,
                "profile_present": profile_present,
                "graph_present": graph_present,
            }
        )
    return result
