#!/usr/bin/env python3
"""Build label-free profile and graph views for robust MUDformer.

All fitted transforms use the fixed training partition.  Test labels are copied
only for integrity checks and are never supplied to a feature transformer.
Graph spectral coordinates are transductive but label-free: they use topology
from the fixed cohort graph and no node labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-dim", type=int, default=64)
    parser.add_argument("--spectral-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def standardize(values: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mean = values[train_mask].mean(axis=0, keepdims=True)
    scale = values[train_mask].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return ((values - mean) / scale).astype(np.float32)


def hashed_character_features(texts: list[str], dimensions: int) -> np.ndarray:
    """Deterministic signed feature hashing; no test-fitted vocabulary."""
    result = np.zeros((len(texts), dimensions), dtype=np.float32)
    for row, raw_text in enumerate(texts):
        value = " ".join(raw_text.lower().split())
        for size in (2, 3, 4):
            for start in range(max(0, len(value) - size + 1)):
                token = value[start : start + size].encode("utf-8")
                digest = hashlib.blake2b(token, digest_size=8).digest()
                integer = int.from_bytes(digest, "little")
                column = integer % dimensions
                result[row, column] += 1.0 if integer & (1 << 63) else -1.0
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    result /= np.maximum(norm, 1.0)
    return result


def safe_svd(matrix: sparse.spmatrix, requested_dim: int, seed: int) -> np.ndarray:
    maximum = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
    components = min(requested_dim, maximum)
    if components <= 0 or matrix.nnz == 0:
        return np.zeros((matrix.shape[0], requested_dim), dtype=np.float32)
    rng = np.random.default_rng(seed)
    left, singular, _ = svds(
        matrix.astype(np.float64), k=components, which="LM", v0=rng.normal(size=matrix.shape[1])
    )
    order = np.argsort(singular)[::-1]
    coordinates = left[:, order] * singular[order]
    # Remove the arbitrary SVD sign so regenerated feature files are stable.
    for column in range(coordinates.shape[1]):
        pivot = int(np.argmax(np.abs(coordinates[:, column])))
        if coordinates[pivot, column] < 0:
            coordinates[:, column] *= -1
    return np.pad(coordinates.astype(np.float32), ((0, 0), (0, requested_dim - components)))


def node_structural_features(
    node_count: int,
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edge_weight: np.ndarray,
    relation_count: int,
) -> tuple[np.ndarray, list[str], sparse.csr_matrix]:
    source, target = edge_index
    columns: list[np.ndarray] = []
    names: list[str] = []
    for relation in range(relation_count):
        selected = edge_type == relation
        src = source[selected]
        dst = target[selected]
        weight = np.maximum(edge_weight[selected].astype(np.float64), 1.0)
        for direction, nodes in (("out", src), ("in", dst)):
            degree = np.bincount(nodes, minlength=node_count).astype(np.float64)
            weighted = np.bincount(nodes, weights=weight, minlength=node_count)
            columns.extend((np.log1p(degree), np.log1p(weighted)))
            names.extend(
                (f"relation_{relation}_{direction}_degree", f"relation_{relation}_{direction}_weight")
            )

    binary = sparse.coo_matrix(
        (np.ones(source.shape[0], dtype=np.float32), (source, target)),
        shape=(node_count, node_count),
    ).tocsr()
    binary.data.fill(1.0)
    binary.eliminate_zeros()
    reciprocal = binary.multiply(binary.T).sum(axis=1).A1
    total_degree = np.asarray(binary.sum(axis=0)).ravel() + np.asarray(binary.sum(axis=1)).ravel()
    two_hop = np.asarray((binary @ binary).astype(bool).sum(axis=1)).ravel()
    columns.extend((np.log1p(reciprocal), np.log1p(total_degree), np.log1p(two_hop)))
    names.extend(("reciprocal_degree", "total_degree", "two_hop_reach"))
    return np.column_stack(columns).astype(np.float32), names, binary


def main() -> int:
    args = parse_args()
    if args.text_dim < 0 or args.spectral_dim < 0:
        raise ValueError("feature dimensions must be non-negative")
    with np.load(args.graph, allow_pickle=False) as graph:
        user_ids = graph["user_ids"].astype(str)
        labels = graph["labels"].astype(np.int64)
        train_mask = graph["train_mask"].astype(bool)
        test_mask = graph["test_mask"].astype(bool)
        numeric_profile = graph["profile_features"].astype(np.float64)
        numeric_names = graph["profile_feature_names"].astype(str).tolist()
        edge_index = graph["edge_index"].astype(np.int64)
        edge_type = graph["edge_type"].astype(np.int64)
        edge_weight = graph["edge_weight"].astype(np.int64)
        relation_names = graph["edge_type_names"].astype(str).tolist()

    rows: dict[str, dict[str, str]] = {}
    with args.nodes.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            user_id = str(row["user_id"])
            if user_id in rows:
                raise ValueError(f"duplicate node row: {user_id}")
            rows[user_id] = row
    if set(rows) != set(user_ids):
        raise ValueError("nodes.csv and graph.npz user IDs differ")

    numeric_profile[:, :3] = signed_log1p(numeric_profile[:, :3])
    scaled_numeric = standardize(numeric_profile, train_mask)
    profile_text = [
        " ".join(
            filter(
                None,
                (
                    rows[user_id].get("screen_name", ""),
                    rows[user_id].get("description", ""),
                    rows[user_id].get("location", ""),
                    rows[user_id].get("gender", ""),
                    rows[user_id].get("verified_reason", ""),
                ),
            )
        )
        for user_id in user_ids
    ]
    if args.text_dim:
        text_features = standardize(
            hashed_character_features(profile_text, args.text_dim), train_mask
        )
    else:
        text_features = np.zeros((len(user_ids), 0), dtype=np.float32)
    profile_features = np.concatenate((scaled_numeric, text_features), axis=1)
    profile_names = numeric_names + [f"profile_text_svd_{index}" for index in range(text_features.shape[1])]

    structure, structure_names, adjacency = node_structural_features(
        len(user_ids), edge_index, edge_type, edge_weight, len(relation_names)
    )
    structure = standardize(structure, train_mask)
    symmetric = adjacency.maximum(adjacency.T)
    row_sum = np.asarray(symmetric.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inv_sqrt[nonzero] = 1.0 / np.sqrt(row_sum[nonzero])
    normalized = sparse.diags(inv_sqrt) @ symmetric @ sparse.diags(inv_sqrt)
    spectral = safe_svd(normalized, args.spectral_dim, args.seed)
    if spectral.shape[1]:
        spectral = standardize(spectral, train_mask)
    graph_features = np.concatenate((structure, spectral), axis=1)
    graph_names = structure_names + [f"spectral_{index}" for index in range(spectral.shape[1])]

    profile_present = np.asarray(
        [rows[user_id].get("historical_profile_available") == "1" for user_id in user_ids],
        dtype=np.bool_,
    )
    graph_present = np.ones(len(user_ids), dtype=np.bool_)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        user_ids=user_ids,
        labels=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        profile_features=profile_features,
        profile_feature_names=np.asarray(profile_names),
        profile_present=profile_present,
        graph_features=graph_features,
        graph_feature_names=np.asarray(graph_names),
        graph_present=graph_present,
    )
    os.replace(temporary, args.output)
    summary = {
        "users": len(user_ids),
        "train_users": int(train_mask.sum()),
        "test_users": int(test_mask.sum()),
        "profile_dim": int(profile_features.shape[1]),
        "graph_dim": int(graph_features.shape[1]),
        "profile_present": int(profile_present.sum()),
        "graph_present": int(graph_present.sum()),
        "leakage_guard": "scalers fitted on train; profile text uses stateless hashing; topology embedding is label-free; labels excluded from features",
        "graph_protocol": "transductive fixed-cohort topology without labels",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
