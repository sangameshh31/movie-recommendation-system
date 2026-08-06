"""Hybrid fusion + re-ranking layer (Stage 2 of the pipeline).

Candidates produced by the collaborative and content paths are merged and
scored with a weighted blend of CF score, content similarity, popularity and
real-time feedback deltas.
"""

from __future__ import annotations

import numpy as np


def minmax_normalize(values: dict[int, float]) -> dict[int, float]:
    """Min-max normalize a dict of scores into [0, 1] (flat maps to 0.5)."""
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def hybrid_rerank(
    candidates: dict[int, dict],
    weights: dict[str, float],
) -> list[tuple[int, float]]:
    """Rerank the merged candidate dict.

    Each candidate entry maps to a dict of per-path scores, e.g.::

        {movie_id: {"cf": 4.2, "cb": 0.78, "pop": 3.9, "feedback": 0.35}}

    Each path is min-max normalized across the candidate set, weighted, and
    summed. Returns ``(movie_id, final_score)`` sorted descending.
    """
    normalized: dict[str, dict[int, float]] = {}
    for path in weights:
        raw = {mid: entry.get(path, 0.0) for mid, entry in candidates.items()}
        normalized[path] = minmax_normalize(raw)

    finals: dict[int, float] = {}
    for mid in candidates:
        total = sum(weights[path] * normalized[path][mid] for path in weights)
        finals[mid] = total

    return sorted(finals.items(), key=lambda kv: kv[1], reverse=True)
