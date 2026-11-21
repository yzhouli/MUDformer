"""Utilities for auditable, behavior-level temporal evidence.

The classifier is trained with user labels, not labels for individual behaviors.
Consequently, these helpers deliberately describe a behavior as *supporting* or
*countering* the spammer prediction.  They must not be interpreted as an
independently supervised, ground-truth behavior anomaly detector.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


def retained_start(full_length: int, max_length: int) -> int:
    """Return the first global position retained by latest-behavior truncation."""
    if full_length <= 0:
        raise ValueError("full_length must be positive")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    return max(0, full_length - max_length)


def timestamp_fields(timestamp: int | None) -> dict[str, int | str | None]:
    """Serialize an optional Unix timestamp without depending on local timezone."""
    if timestamp is None:
        return {"timestamp_unix": None, "timestamp_utc": None}
    value = int(timestamp)
    return {
        "timestamp_unix": value,
        "timestamp_utc": datetime.fromtimestamp(value, tz=timezone.utc).isoformat(),
    }


def contiguous_risk_segments(
    scores: np.ndarray,
    *,
    global_start: int,
    timestamps: np.ndarray | None,
    quantile: float,
    max_segments: int,
) -> list[dict[str, Any]]:
    """Summarize contiguous, positive high-risk contributions in temporal order.

    A segment is selected only when its contribution is positive and at least
    the requested within-user quantile.  Positive contributions are therefore
    evidence supporting the spammer prediction, rather than a claim that the
    corresponding behavior has a ground-truth anomaly label.
    """
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if max_segments <= 0:
        raise ValueError("max_segments must be positive")
    if timestamps is not None and timestamps.shape != scores.shape:
        raise ValueError("timestamps must have the same shape as scores")

    threshold = max(0.0, float(np.quantile(scores, quantile)))
    selected = (scores > 0.0) & (scores >= threshold)
    segments: list[dict[str, Any]] = []
    start: int | None = None
    for index, is_selected in enumerate(selected):
        if is_selected and start is None:
            start = index
        is_last = index == len(selected) - 1
        if start is not None and (not is_selected or is_last):
            stop = index + 1 if is_selected and is_last else index
            values = scores[start:stop]
            payload: dict[str, Any] = {
                "local_start_index": start,
                "local_end_index": stop - 1,
                "global_start_index": global_start + start,
                "global_end_index": global_start + stop - 1,
                "behavior_count": stop - start,
                "sum_spammer_margin_contribution": float(values.sum()),
                "mean_spammer_margin_contribution": float(values.mean()),
                "max_spammer_margin_contribution": float(values.max()),
            }
            if timestamps is not None:
                payload["start_timestamp"] = timestamp_fields(int(timestamps[start]))
                payload["end_timestamp"] = timestamp_fields(int(timestamps[stop - 1]))
            else:
                payload["start_timestamp"] = timestamp_fields(None)
                payload["end_timestamp"] = timestamp_fields(None)
            segments.append(payload)
            start = None

    return sorted(
        segments,
        key=lambda item: (
            item["sum_spammer_margin_contribution"],
            item["max_spammer_margin_contribution"],
        ),
        reverse=True,
    )[:max_segments]
