"""Deterministic M0/M1/M2/M3 longitudinal-memory retrieval experiment.

This module is evaluator-side research code.  It does not alter Agent.respond,
the public response contract, or the canonical M0 dense scorer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from nickolas.memory.qlmp import MemoryItem, MemoryPolarity


DEFAULT_KEEP_RATIO = 0.20
DEFAULT_LAMBDA_MEMORY = 0.20


class SteeringMethod(str, Enum):
    M0 = "M0"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"


@dataclass(frozen=True)
class SteeringConfig:
    method: SteeringMethod | str = SteeringMethod.M3
    keep_ratio: float = DEFAULT_KEEP_RATIO
    lambda_memory: float = DEFAULT_LAMBDA_MEMORY

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", SteeringMethod(self.method))
        if not math.isfinite(float(self.keep_ratio)) or not 0.0 < float(self.keep_ratio) <= 1.0:
            raise ValueError("keep_ratio must be finite and in (0, 1]")
        if not math.isfinite(float(self.lambda_memory)) or float(self.lambda_memory) < 0.0:
            raise ValueError("lambda_memory must be finite and non-negative")


@dataclass(frozen=True)
class SteeringDiagnostics:
    method: str
    keep_ratio: float
    lambda_memory: float
    query_dimension: int
    kept_dimensions: int
    retained_fraction: float
    memory_norm: float
    cleaned_memory_norm: float
    query_memory_cosine: float | None
    query_cleaned_memory_cosine: float | None
    query_steered_cosine: float
    top_interaction_magnitudes: tuple[float, ...]
    memory_item_count: int = 0
    applied: bool = False
    activation_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        result = np.array(value, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.issubdtype(result.dtype, np.floating):
        result = result.astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _normalize(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    return np.asarray(value / norm, dtype=value.dtype)


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _kept_count(dimension: int, keep_ratio: float) -> int:
    if not math.isfinite(float(keep_ratio)) or not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("keep_ratio must be finite and in (0, 1]")
    return min(dimension, max(1, int(math.ceil(dimension * float(keep_ratio)))))


def _masked_memory(signal: np.ndarray, memory: np.ndarray, keep_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    count = _kept_count(signal.size, keep_ratio)
    # Stable tie policy: greater signal first, then lower coordinate index.
    indices = np.lexsort((np.arange(signal.size), -signal))[:count]
    cleaned = np.zeros_like(memory)
    cleaned[indices] = memory[indices]
    return cleaned, indices


def interaction_mask_memory(query: np.ndarray, memory: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Return ``m_clean = topk_mask(abs(q * m)) * m`` without renormalizing it."""

    q = _vector(query, "query")
    m = _vector(memory, "memory")
    if q.shape != m.shape:
        raise ValueError(f"query and memory dimensions differ: {q.size} != {m.size}")
    dtype = np.result_type(q.dtype, m.dtype)
    q, m = q.astype(dtype, copy=False), m.astype(dtype, copy=False)
    return _masked_memory(np.abs(q * m), m, keep_ratio)[0]


def aggregate_user_memory(memories: Iterable[MemoryItem]) -> np.ndarray | None:
    """Equal-weight and normalize existing eligible MemoryItem embeddings.

    Existing QLMP baselines exclude negative-polarity items.  We retain that
    convention because negative memory text already encodes its semantics and
    subtracting its embedding would introduce a second polarity representation.
    """

    items = tuple(memories)
    if any(not isinstance(item, MemoryItem) for item in items):
        raise ValueError("memories must contain only MemoryItem values")
    eligible = tuple(item for item in items if item.polarity is not MemoryPolarity.NEGATIVE)
    if not eligible:
        return None
    dimensions = {item.embedding.size for item in eligible}
    if len(dimensions) != 1:
        raise ValueError("memory embeddings must have equal dimensionality")
    raw = np.mean(np.stack([item.embedding for item in eligible]), axis=0, dtype=np.float64)
    if float(np.linalg.norm(raw)) <= 0.0:
        return None
    return _normalize(raw, "aggregate memory")


def prior_memory_items(
    records: Iterable[Any],
    *,
    user_id: str,
    before_sequence_index: int,
    embedding_space_id: str,
) -> tuple[MemoryItem, ...]:
    """Select same-user, same-space records strictly earlier than the query."""

    if not str(user_id).strip() or not str(embedding_space_id).strip():
        raise ValueError("user_id and embedding_space_id must be non-empty")
    if isinstance(before_sequence_index, bool) or before_sequence_index < 0:
        raise ValueError("before_sequence_index must be a non-negative integer")
    selected: list[MemoryItem] = []
    for record in records:
        if (
            str(getattr(record, "user_id", "")) == str(user_id)
            and str(getattr(record, "embedding_space_id", "")) == str(embedding_space_id)
            and int(getattr(record, "sequence_index", before_sequence_index))
            < before_sequence_index
        ):
            value = getattr(record, "item", None)
            if not isinstance(value, MemoryItem):
                raise ValueError("selected records must contain MemoryItem values")
            selected.append(value)
    return tuple(selected)


def steer_query(
    query: np.ndarray,
    memory: np.ndarray | None,
    method: SteeringMethod | str,
    keep_ratio: float = DEFAULT_KEEP_RATIO,
    lambda_memory: float = DEFAULT_LAMBDA_MEMORY,
) -> np.ndarray:
    """Return normalized M0/M1/M2/M3 q without mutating either input."""

    config = SteeringConfig(method, keep_ratio, lambda_memory)
    q = _normalize(_vector(query, "query"), "query")
    if config.method is SteeringMethod.M0 or memory is None or config.lambda_memory == 0.0:
        return q
    m = _normalize(_vector(memory, "memory"), "memory")
    if q.shape != m.shape:
        raise ValueError(f"query and memory dimensions differ: {q.size} != {m.size}")
    dtype = np.result_type(q.dtype, m.dtype)
    q, m = q.astype(dtype, copy=False), m.astype(dtype, copy=False)
    if config.method is SteeringMethod.M1:
        cleaned = m.copy()
    elif config.method is SteeringMethod.M2:
        cleaned, _ = _masked_memory(np.abs(q), m, config.keep_ratio)
    else:
        cleaned, _ = _masked_memory(np.abs(q * m), m, config.keep_ratio)
    return _normalize(q + float(config.lambda_memory) * cleaned, "steered query")


def steer_query_with_diagnostics(
    query: np.ndarray,
    memory: np.ndarray | None,
    config: SteeringConfig,
    *,
    memory_item_count: int = 0,
    buyer_active: bool = True,
) -> tuple[np.ndarray, SteeringDiagnostics]:
    q = _normalize(_vector(query, "query"), "query")
    apply = buyer_active and config.method is not SteeringMethod.M0 and memory is not None
    effective_method = config.method if apply else SteeringMethod.M0
    m = None if memory is None else _normalize(_vector(memory, "memory"), "memory")
    if m is not None and q.shape != m.shape:
        raise ValueError(f"query and memory dimensions differ: {q.size} != {m.size}")
    if apply and config.lambda_memory != 0.0:
        if effective_method is SteeringMethod.M1:
            cleaned, indices = m.copy(), np.arange(m.size)
            signal = np.abs(q * m)
        elif effective_method is SteeringMethod.M2:
            signal = np.abs(q)
            cleaned, indices = _masked_memory(signal, m, config.keep_ratio)
        else:
            signal = np.abs(q * m)
            cleaned, indices = _masked_memory(signal, m, config.keep_ratio)
        steered = steer_query(q, m, effective_method, config.keep_ratio, config.lambda_memory)
    else:
        cleaned = np.zeros_like(q)
        indices = np.asarray([], dtype=int)
        signal = np.zeros_like(q)
        steered = q.copy()
    top = tuple(float(signal[index]) for index in indices[: min(5, indices.size)])
    diagnostics = SteeringDiagnostics(
        method=config.method.value,
        keep_ratio=float(config.keep_ratio),
        lambda_memory=float(config.lambda_memory),
        query_dimension=q.size,
        kept_dimensions=int(indices.size),
        retained_fraction=float(indices.size / q.size),
        memory_norm=0.0 if m is None else float(np.linalg.norm(m)),
        cleaned_memory_norm=float(np.linalg.norm(cleaned)),
        query_memory_cosine=None if m is None else _cosine(q, m),
        query_cleaned_memory_cosine=_cosine(q, cleaned),
        query_steered_cosine=float(_cosine(q, steered)),
        top_interaction_magnitudes=top,
        memory_item_count=int(memory_item_count),
        applied=bool(apply and config.lambda_memory != 0.0),
        activation_reason=("buyer" if buyer_active else "non_buyer_control"),
    )
    return steered, diagnostics


class DenseScorer(Protocol):
    def dense_retrieve_vector(self, query_embedding: np.ndarray, top_n: int = 150) -> Any: ...


def score_snapshot_variants(
    scorer: DenseScorer,
    snapshot: Any,
    memories: Sequence[MemoryItem],
    *,
    is_buyer: bool,
    keep_ratio: float = DEFAULT_KEEP_RATIO,
    lambda_memory: float = DEFAULT_LAMBDA_MEMORY,
    top_n: int = 150,
) -> dict[str, dict[str, Any]]:
    """Evaluator-side replay through the exact canonical M0 vector scorer."""

    memory = aggregate_user_memory(memories)
    output: dict[str, dict[str, Any]] = {}
    for method in SteeringMethod:
        q_final, diagnostics = steer_query_with_diagnostics(
            snapshot.query_embedding,
            memory,
            SteeringConfig(method, keep_ratio, lambda_memory),
            memory_item_count=len(memories),
            buyer_active=is_buyer,
        )
        result = scorer.dense_retrieve_vector(q_final, top_n=top_n)
        target = getattr(snapshot, "target_product_id", None)
        rank = None if target not in result.product_ids else result.product_ids.index(target) + 1
        output[method.value] = {
            "product_ids": list(result.product_ids),
            "scores": [float(value) for value in result.scores],
            "target_rank": rank,
            "hit": rank is not None,
            "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
            "diagnostics": diagnostics.to_dict(),
        }
    return output


def summarize_variant_sessions(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Preserve per-session results and calculate transparent HR/MRR deltas."""

    by_method: dict[str, list[Mapping[str, Any]]] = {method.value: [] for method in SteeringMethod}
    for session in sessions:
        for method in SteeringMethod:
            by_method[method.value].append(session[method.value])
    metrics = {}
    for method, rows in by_method.items():
        count = len(rows)
        metrics[method] = {
            "session_count": count,
            "hit_rate_at_n": 0.0 if not count else sum(bool(row["hit"]) for row in rows) / count,
            "mrr": 0.0 if not count else sum(float(row["reciprocal_rank"]) for row in rows) / count,
        }
    return {"metrics": metrics, "sessions": list(sessions)}


__all__ = [
    "DEFAULT_KEEP_RATIO", "DEFAULT_LAMBDA_MEMORY", "SteeringConfig",
    "SteeringDiagnostics", "SteeringMethod", "aggregate_user_memory",
    "interaction_mask_memory", "prior_memory_items", "score_snapshot_variants", "steer_query",
    "steer_query_with_diagnostics", "summarize_variant_sessions",
]
