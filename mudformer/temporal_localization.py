"""Shared protocol for controlled temporal-behaviour localization evaluation.

The supervised Weibo-25 labels are user labels, not labels for individual
posts.  This module therefore creates an *evaluation-only* reference: a
contiguous historical representation block from a fixed training spammer is
copied into a fixed normal test user's history.  The copied block and its
position are the known localization target.  No model is retrained, and model
predictions/gradients never choose the target, donor, or position.

The protocol deliberately operates on the representation consumed by a model:
the six text-history baselines receive their frozen time-slice embeddings, and
the DNA baseline receives its chronological DNA cells.  MUDformer copies its
cached per-behaviour text/image representations while retaining the recipient
timestamps and sequence length.  Results must be described as controlled
*injected-risk-block localization*, not human-annotated real-post anomaly
localization.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


PROTOCOL_NAME = "controlled_temporal_risk_block_localization_v1"
INJECTION_SEED = 20260903
NUM_TEMPORAL_BLOCKS = 5


@dataclass(frozen=True)
class InjectionRecord:
    """One fixed normal-test target and its train-spammer donor."""

    user_id: str
    target_index: int
    donor_id: str
    donor_index: int
    block_index: int
    native_indices: tuple[int, ...]


def _digest(*parts: object) -> bytes:
    value = ":".join(str(part) for part in (PROTOCOL_NAME, INJECTION_SEED, *parts))
    return hashlib.sha256(value.encode("utf-8")).digest()


def _ranked(values: Iterable[str], *parts: object) -> list[str]:
    return sorted(
        (str(value) for value in values),
        key=lambda value: (_digest(*parts, value), value),
    )


def temporal_groups(length: int, groups: int = NUM_TEMPORAL_BLOCKS) -> tuple[np.ndarray, ...]:
    """Return ordered, non-empty contiguous groups covering ``range(length)``."""
    if length < groups:
        raise ValueError(f"need at least {groups} temporal units, got {length}")
    result = tuple(np.asarray(values, dtype=np.int64) for values in np.array_split(np.arange(length), groups))
    if any(values.size == 0 for values in result):
        raise AssertionError("temporal grouping unexpectedly created an empty group")
    return result


def target_block(user_id: str, groups: int = NUM_TEMPORAL_BLOCKS) -> int:
    if groups <= 1:
        raise ValueError("at least two temporal blocks are required")
    return int.from_bytes(_digest("target-block", user_id)[:8], "little") % groups


def _cohort(
    user_ids: Sequence[str] | np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[list[str], dict[str, int], list[str]]:
    ids = [str(value) for value in user_ids]
    labels = np.asarray(labels, dtype=np.int64)
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    if len(ids) != len(set(ids)) or labels.shape != (len(ids),):
        raise ValueError("user IDs and labels must be unique and aligned")
    if train_mask.shape != labels.shape or test_mask.shape != labels.shape or np.any(train_mask & test_mask):
        raise ValueError("invalid train/test masks")
    index_by_id = {user_id: index for index, user_id in enumerate(ids)}
    targets = [ids[index] for index in np.flatnonzero(test_mask & (labels == 0))]
    donors = [ids[index] for index in np.flatnonzero(train_mask & (labels == 1))]
    if not targets or not donors:
        raise ValueError("controlled localization requires normal test targets and spammer train donors")
    return targets, index_by_id, donors


def inject_tweet_feature_blocks(payload: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[InjectionRecord], dict[str, object]]:
    """Inject one donor historical text block into every normal test user.

    ``tweet_features`` uses [time, user, embedding] and ``active_mask`` uses
    [time, user].  A block may cover one or two native snapshots when a model
    has eight snapshots; this is the required coarsening to five shared
    chronological evaluation blocks.  Only the target user's history fields
    are changed.
    """
    required = ("user_ids", "labels", "train_mask", "test_mask", "tweet_features", "active_mask")
    if any(key not in payload for key in required):
        raise ValueError("tweet-history payload is missing a required protocol field")
    features = np.asarray(payload["tweet_features"])
    active = np.asarray(payload["active_mask"], dtype=bool)
    if features.ndim != 3 or active.shape != features.shape[:2]:
        raise ValueError("unexpected tweet_features or active_mask shape")
    targets, index_by_id, donor_ids = _cohort(
        payload["user_ids"], payload["labels"], payload["train_mask"], payload["test_mask"],
    )
    groups = temporal_groups(features.shape[0])
    result = dict(payload)
    result["tweet_features"] = features.copy()
    result["active_mask"] = active.copy()
    records: list[InjectionRecord] = []
    for user_id in sorted(targets):
        block = target_block(user_id)
        native = groups[block]
        eligible = [
            donor_id for donor_id in donor_ids
            if active[native, index_by_id[donor_id]].any()
        ]
        if not eligible:
            raise ValueError(f"no active spammer donor for temporal block {block}")
        donor_id = _ranked(eligible, "tweet-donor", user_id, block)[0]
        target_index = index_by_id[user_id]
        donor_index = index_by_id[donor_id]
        result["tweet_features"][native, target_index] = features[native, donor_index]
        result["active_mask"][native, target_index] = active[native, donor_index]
        records.append(InjectionRecord(
            user_id=user_id, target_index=target_index, donor_id=donor_id, donor_index=donor_index,
            block_index=block, native_indices=tuple(int(value) for value in native.tolist()),
        ))
    return result, records, protocol_metadata(targets, records, native_unit_count=features.shape[0])


def inject_dna_blocks(payload: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[InjectionRecord], dict[str, object]]:
    """Inject a contiguous chronological Digital-DNA cell block.

    Digital-DNA cells are built in chronological raster order, so flattened
    cell ranges retain the temporal order used by the TIFS baseline.
    """
    required = ("user_ids", "labels", "train_mask", "test_mask", "dna_images")
    if any(key not in payload for key in required):
        raise ValueError("DNA payload is missing a required protocol field")
    dna = np.asarray(payload["dna_images"])
    if dna.ndim != 4 or dna.shape[1] != 2 or dna.shape[2] != dna.shape[3]:
        raise ValueError("unexpected two-channel square Digital-DNA input")
    targets, index_by_id, donor_ids = _cohort(
        payload["user_ids"], payload["labels"], payload["train_mask"], payload["test_mask"],
    )
    groups = temporal_groups(dna.shape[2] * dna.shape[3])
    result = dict(payload)
    result["dna_images"] = dna.copy()
    records: list[InjectionRecord] = []
    for user_id in sorted(targets):
        block = target_block(user_id)
        native = groups[block]
        donor_id = _ranked(donor_ids, "dna-donor", user_id, block)[0]
        target_index = index_by_id[user_id]
        donor_index = index_by_id[donor_id]
        target_cells = result["dna_images"][target_index].reshape(2, -1)
        donor_cells = dna[donor_index].reshape(2, -1)
        target_cells[:, native] = donor_cells[:, native]
        records.append(InjectionRecord(
            user_id=user_id, target_index=target_index, donor_id=donor_id, donor_index=donor_index,
            block_index=block, native_indices=tuple(int(value) for value in native.tolist()),
        ))
    return result, records, protocol_metadata(targets, records, native_unit_count=dna.shape[2] * dna.shape[3])


def sequence_injection_records(
    train_user_ids: Sequence[str], train_labels: Sequence[int],
    test_user_ids: Sequence[str], test_labels: Sequence[int],
) -> tuple[list[InjectionRecord], dict[str, object]]:
    """Create fixed donor/target assignments for variable-length histories.

    The actual native positions are derived after each recipient sequence is
    loaded because MUDformer retains a user-specific number of recent events.
    ``native_indices`` is consequently empty in this plan and is filled by the
    evaluator's per-user result row.
    """
    train_ids = [str(value) for value in train_user_ids]
    test_ids = [str(value) for value in test_user_ids]
    train_values = np.asarray(train_labels, dtype=np.int64)
    test_values = np.asarray(test_labels, dtype=np.int64)
    if train_values.shape != (len(train_ids),) or test_values.shape != (len(test_ids),):
        raise ValueError("sequence user IDs and labels must be aligned")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids) or set(train_ids) & set(test_ids):
        raise ValueError("sequence train/test IDs must be unique and disjoint")
    donors = [user_id for user_id, label in zip(train_ids, train_values, strict=True) if label == 1]
    targets = [user_id for user_id, label in zip(test_ids, test_values, strict=True) if label == 0]
    if not donors or not targets:
        raise ValueError("sequence protocol requires spammer donors and normal test targets")
    train_index = {user_id: index for index, user_id in enumerate(train_ids)}
    test_index = {user_id: index for index, user_id in enumerate(test_ids)}
    records: list[InjectionRecord] = []
    for user_id in sorted(targets):
        block = target_block(user_id)
        donor_id = _ranked(donors, "sequence-donor", user_id, block)[0]
        records.append(InjectionRecord(
            user_id=user_id, target_index=test_index[user_id], donor_id=donor_id,
            donor_index=train_index[donor_id], block_index=block, native_indices=(),
        ))
    return records, protocol_metadata(targets, records, native_unit_count=NUM_TEMPORAL_BLOCKS)


def synthetic_training_sequence_injection_records(
    target_normal_ids: Sequence[str], donor_spammer_ids: Sequence[str],
) -> list[InjectionRecord]:
    """Return a deterministic train-only synthetic block plan.

    This is intentionally separate from the frozen V1 evaluator: callers pass
    only an internal training/validation population, never a fixed-test ID.
    The block geometry and donor resampling match the V1 sequence operation.
    """
    targets = [str(user_id) for user_id in target_normal_ids]
    donors = [str(user_id) for user_id in donor_spammer_ids]
    if not targets or not donors or len(set(targets)) != len(targets) or len(set(donors)) != len(donors):
        raise ValueError("synthetic temporal training requires unique non-empty targets and donors")
    if set(targets) & set(donors):
        raise ValueError("synthetic temporal training targets and donors must be disjoint")
    result: list[InjectionRecord] = []
    for target_index, user_id in enumerate(sorted(targets)):
        block = target_block(user_id)
        donor_id = _ranked(donors, "synthetic-train-sequence-donor", user_id, block)[0]
        result.append(InjectionRecord(
            user_id=user_id,
            target_index=target_index,
            donor_id=donor_id,
            donor_index=donors.index(donor_id),
            block_index=block,
            native_indices=(),
        ))
    return result


def protocol_metadata(
    target_ids: Iterable[str], records: Iterable[InjectionRecord], *, native_unit_count: int,
) -> dict[str, object]:
    targets = tuple(sorted(str(value) for value in target_ids))
    entries = tuple(records)
    if len(targets) != len(entries) or {entry.user_id for entry in entries} != set(targets):
        raise ValueError("injection records do not cover exactly the target users")
    target_digest = hashlib.sha256("\n".join(targets).encode("utf-8")).hexdigest()
    specification = "\n".join(
        f"{entry.user_id},{entry.donor_id},{entry.block_index},{','.join(map(str, entry.native_indices))}"
        for entry in sorted(entries, key=lambda value: value.user_id)
    )
    return {
        "name": PROTOCOL_NAME,
        "injection_seed": INJECTION_SEED,
        "temporal_blocks": NUM_TEMPORAL_BLOCKS,
        "target_population": "all fixed-test users with ground-truth normal user label",
        "donor_population": "fixed-training users with ground-truth spammer user label",
        "target_or_donor_selection": "SHA256 protocol ranks; independent of model parameters, predictions and gradients",
        "train_or_checkpoint_modification": False,
        "target_user_count": len(targets),
        "target_user_ids_sha256": target_digest,
        "injection_specification_sha256": hashlib.sha256(specification.encode("utf-8")).hexdigest(),
        "native_temporal_unit_count": int(native_unit_count),
        "top1_chance_rate": 1.0 / NUM_TEMPORAL_BLOCKS,
    }
