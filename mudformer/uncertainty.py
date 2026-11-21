"""Small, dependency-light utilities for auditable PoE uncertainty experiments."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def stable_seed(*parts: object, base_seed: int = 42) -> int:
    """Derive a reproducible RNG seed that is independent of Python hash randomization."""
    message = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(message, digest_size=8).digest()
    return (int.from_bytes(digest, "little") ^ int(base_seed)) % (2**32)


def select_fraction(
    candidates: np.ndarray, fraction: float, *, seed: int
) -> np.ndarray:
    """Select a deterministic fraction of candidate positions without replacement."""
    if candidates.ndim != 1:
        raise ValueError("candidates must be one-dimensional")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    count = int(round(candidates.size * fraction))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == candidates.size:
        return candidates.astype(np.int64, copy=True)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(candidates, size=count, replace=False).astype(np.int64))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, assigning tied values their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Spearman rho without depending on SciPy; return None for constants."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if x.size < 2:
        return None
    ranked_x = average_ranks(x)
    ranked_y = average_ranks(y)
    centered_x = ranked_x - ranked_x.mean()
    centered_y = ranked_y - ranked_y.mean()
    denominator = np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    if denominator == 0.0:
        return None
    return float(np.dot(centered_x, centered_y) / denominator)


def binary_detection_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Metrics for fixed-threshold corruption detection, with no threshold fitting."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.size != scores.size:
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if labels.size == 0:
        return {"support": 0, "accuracy": None, "precision": None, "recall": None, "f1": None}
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    predictions = scores >= 0.5
    positive = labels == 1
    true_positive = int(np.logical_and(predictions, positive).sum())
    false_positive = int(np.logical_and(predictions, ~positive).sum())
    false_negative = int(np.logical_and(~predictions, positive).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "support": int(labels.size),
        "corrupted_support": int(positive.sum()),
        "clean_support": int((~positive).sum()),
        "accuracy": float((predictions == positive).mean()),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "brier": float(np.mean((scores - labels) ** 2)),
        "threshold": 0.5,
    }
