"""Slow Memory distillation, aggregation, and the single rerank equation."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .types import FastMemoryState, SlowMemoryEpisode, TypedConstraint


def _render_facts(items: Sequence[TypedConstraint]) -> str:
    return ", ".join(f"{item.kind.value}={item.value}" for item in items) or "none"


def distill_summary(state: FastMemoryState) -> str:
    """Render final Fast Memory in one fixed, human-auditable order."""

    return "; ".join((
        f"category={state.category or 'clothing item'}",
        f"intent={state.intent}",
        f"hard facts: {_render_facts(state.hard_constraints)}",
        f"soft facts: {_render_facts(state.soft_preferences)}",
        f"negatives: {_render_facts(state.negatives)}",
    ))


def aggregate_slow_vector(
    episodes: Sequence[SlowMemoryEpisode],
    *,
    user_id: str,
    current_sequence_index: int,
    embedding_space_id: str,
    tau: float,
) -> np.ndarray | None:
    """Return the normalized exponentially weighted visible history vector."""

    compatible = [
        episode
        for episode in episodes
        if episode.user_id == user_id
        and episode.sequence_index < current_sequence_index
        and episode.embedding_space_id == embedding_space_id
        and episode.embedding
    ]
    if not compatible:
        return None

    dimension = len(compatible[0].embedding)
    if dimension == 0:
        return None
    total = np.zeros(dimension, dtype=np.float64)
    used = 0
    for episode in compatible:
        if len(episode.embedding) != dimension:
            continue
        vector = np.asarray(episode.embedding, dtype=np.float64)
        if not np.isfinite(vector).all():
            continue
        age = current_sequence_index - episode.sequence_index
        total += math.exp(-float(age) / tau) * vector
        used += 1
    norm = float(np.linalg.norm(total))
    if used == 0 or norm <= 1e-12:
        return None
    return np.asarray(total / norm, dtype=np.float32)


def rerank_with_slow_memory(
    candidate_ids: Sequence[str],
    candidate_vectors: np.ndarray,
    slow_vector: np.ndarray,
    *,
    lambda_memory: float,
) -> list[str]:
    """Apply reciprocal baseline rank plus cosine Slow Memory affinity."""

    identifiers = [str(value) for value in candidate_ids]
    matrix = np.asarray(candidate_vectors, dtype=np.float32)
    slow = np.asarray(slow_vector, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(identifiers):
        raise ValueError("candidate vectors must align with candidate identifiers")
    if slow.ndim != 1 or matrix.shape[1] != slow.shape[0]:
        raise ValueError("Slow Memory and product vector dimensions differ")
    similarities = np.asarray(matrix @ slow, dtype=np.float64)
    scored = [
        (
            identifier,
            1.0 / baseline_rank + lambda_memory * float(similarities[baseline_rank - 1]),
            baseline_rank,
        )
        for baseline_rank, identifier in enumerate(identifiers, start=1)
    ]
    scored.sort(key=lambda row: (-row[1], row[2], row[0]))
    return [row[0] for row in scored]


__all__ = ["aggregate_slow_vector", "distill_summary", "rerank_with_slow_memory"]
