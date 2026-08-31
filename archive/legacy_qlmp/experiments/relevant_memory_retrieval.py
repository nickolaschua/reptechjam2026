"""Evaluator-private M4 relevant-memory retrieval steering.

M4 is intentionally separate from the frozen M0/M1/M2/M3 implementation.  It
selects whole historical MemoryItems by cosine relevance before taking the
equal-weight mean and passing the softly steered query to the canonical scorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Protocol, Sequence

import numpy as np

from nickolas.memory.qlmp import MemoryItem


M4_TOP_K = 3
M4_LAMBDA_MEMORY = 0.20


class DenseScorer(Protocol):
    def dense_retrieve_vector(self, query_embedding: np.ndarray, top_n: int = 150) -> Any: ...


@dataclass(frozen=True)
class RelevantMemoryDiagnostics:
    eligible_memory_ids: tuple[str, ...]
    eligible_memory_texts: tuple[str, ...]
    eligible_memory_origin_indices: tuple[int, ...]
    eligible_similarity_scores: tuple[float, ...]
    selected_memory_ids: tuple[str, ...]
    selected_memory_texts: tuple[str, ...]
    selected_memory_origin_indices: tuple[int, ...]
    selected_similarity_scores: tuple[float, ...]
    selected_memory_count: int
    aggregate_selected_memory_norm: float
    cosine_q_m_top: float | None
    cosine_q_q_m4: float
    k: int
    lambda_memory: float
    applied: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unit_vector(value: Any, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite, non-empty one-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{name} must be L2-normalized")
    return vector.copy()


def _normalize(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    return np.asarray(value / norm, dtype=np.float64)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def relevant_memory_query(
    query: np.ndarray,
    memories: Sequence[MemoryItem],
    origin_sequence_indices: Sequence[int],
    *,
    current_sequence_index: int,
    query_space_id: str,
    memory_space_id: str,
    k: int = M4_TOP_K,
    lambda_memory: float = M4_LAMBDA_MEMORY,
    buyer_active: bool = True,
) -> tuple[np.ndarray, RelevantMemoryDiagnostics]:
    """Return M4's query and complete memory-level selection diagnostics.

    Equal scores are resolved by earlier origin sequence, then stable memory ID,
    then original candidate position.  Thus selection is deterministic without
    consulting targets or downstream retrieval outcomes.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not math.isfinite(float(lambda_memory)) or float(lambda_memory) < 0.0:
        raise ValueError("lambda_memory must be finite and non-negative")
    if not str(query_space_id).strip() or not str(memory_space_id).strip():
        raise ValueError("query and memory embedding-space IDs must be non-empty")
    if query_space_id != memory_space_id:
        raise ValueError(
            f"query and memory embedding spaces differ: {query_space_id} != {memory_space_id}"
        )
    if len(memories) != len(origin_sequence_indices):
        raise ValueError("memories and origin_sequence_indices must have equal length")
    q = _unit_vector(query, "query")

    candidates = tuple(memories) if buyer_active else ()
    origins = tuple(int(value) for value in origin_sequence_indices) if buyer_active else ()
    if any(not isinstance(item, MemoryItem) for item in candidates):
        raise ValueError("memories must contain only MemoryItem values")
    if any(origin >= int(current_sequence_index) for origin in origins):
        raise ValueError("current/future memory cannot enter M4 candidate selection")

    vectors: list[np.ndarray] = []
    scores: list[float] = []
    for index, item in enumerate(candidates):
        vector = _unit_vector(item.embedding, f"memory[{index}]")
        if vector.shape != q.shape:
            raise ValueError(
                f"query and memory dimensions differ: {q.size} != {vector.size}"
            )
        vectors.append(vector)
        scores.append(_cosine(q, vector))

    order = sorted(
        range(len(candidates)),
        key=lambda index: (-scores[index], origins[index], str(candidates[index].id), index),
    )
    selected_indices = tuple(order[: min(k, len(order))])
    selected = tuple(candidates[index] for index in selected_indices)
    selected_origins = tuple(origins[index] for index in selected_indices)
    selected_scores = tuple(scores[index] for index in selected_indices)

    if not selected_indices or float(lambda_memory) == 0.0:
        m_top = None
        q_m4 = q.copy()
    else:
        raw_mean = np.mean(
            np.stack([vectors[index] for index in selected_indices]),
            axis=0,
            dtype=np.float64,
        )
        m_top = _normalize(raw_mean, "selected-memory mean")
        q_m4 = _normalize(q + float(lambda_memory) * m_top, "M4 query")

    diagnostics = RelevantMemoryDiagnostics(
        eligible_memory_ids=tuple(str(item.id) for item in candidates),
        eligible_memory_texts=tuple(str(item.text) for item in candidates),
        eligible_memory_origin_indices=origins,
        eligible_similarity_scores=tuple(scores),
        selected_memory_ids=tuple(str(item.id) for item in selected),
        selected_memory_texts=tuple(str(item.text) for item in selected),
        selected_memory_origin_indices=selected_origins,
        selected_similarity_scores=selected_scores,
        selected_memory_count=len(selected),
        aggregate_selected_memory_norm=0.0 if m_top is None else float(np.linalg.norm(m_top)),
        cosine_q_m_top=None if m_top is None else _cosine(q, m_top),
        cosine_q_q_m4=_cosine(q, q_m4),
        k=int(k),
        lambda_memory=float(lambda_memory),
        applied=bool(buyer_active and selected_indices and float(lambda_memory) != 0.0),
    )
    return q_m4, diagnostics


def score_relevant_memory_query(
    scorer: DenseScorer,
    query: np.ndarray,
    memories: Sequence[MemoryItem],
    origin_sequence_indices: Sequence[int],
    *,
    current_sequence_index: int,
    query_space_id: str,
    memory_space_id: str,
    top_n: int,
    buyer_active: bool = True,
) -> tuple[Any, np.ndarray, RelevantMemoryDiagnostics]:
    """Pass M4's final vector into the unchanged canonical M0 scorer boundary."""

    q_m4, diagnostics = relevant_memory_query(
        query,
        memories,
        origin_sequence_indices,
        current_sequence_index=current_sequence_index,
        query_space_id=query_space_id,
        memory_space_id=memory_space_id,
        k=M4_TOP_K,
        lambda_memory=M4_LAMBDA_MEMORY,
        buyer_active=buyer_active,
    )
    return scorer.dense_retrieve_vector(q_m4, top_n=top_n), q_m4, diagnostics


def assert_logical_memory_parity(large_sessions: Sequence[Any], small_sessions: Sequence[Any]) -> None:
    """Prove corresponding spaces expose identical logical M4 candidates."""

    if len(large_sessions) != len(small_sessions):
        raise ValueError("large and small session counts differ")
    for large, small in zip(large_sessions, small_sessions):
        if str(large.session_id) != str(small.session_id):
            raise ValueError("large and small session IDs differ")
        if [str(item.id) for item in large.memories] != [str(item.id) for item in small.memories]:
            raise ValueError(f"candidate memory IDs differ for {large.session_id}")
        if [str(item.text) for item in large.memories] != [str(item.text) for item in small.memories]:
            raise ValueError(f"candidate memory texts differ for {large.session_id}")
        if tuple(large.memory_origin_indices) != tuple(small.memory_origin_indices):
            raise ValueError(f"candidate memory timestamps differ for {large.session_id}")


__all__ = [
    "M4_LAMBDA_MEMORY",
    "M4_TOP_K",
    "RelevantMemoryDiagnostics",
    "assert_logical_memory_parity",
    "relevant_memory_query",
    "score_relevant_memory_query",
]
