"""Thin numeric/orchestration boundary between canonical M0 and QLMP.

This module owns conversions and composition only.  M0 remains the retrieval
owner and QLMP remains the geometry/steering owner.  Projection is deliberately
exposed only through :func:`run_projector_isolation`; it cannot produce a
retrieval-steering query in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Protocol

import numpy as np

from nickolas.memory.qlmp import (
    BaselineConfig,
    LocalSubspace,
    MemoryBaselineResult,
    MemoryItem,
    MemoryProjection,
    ProjectionConfig,
    QuerySteeringResult,
    SteeringConfig,
    bound_query_shift,
    build_cosine_memory_baseline,
    build_local_subspace,
    build_naive_memory_baseline,
    project_memory_residual,
)

try:
    from .embedding_backends import OPENAI_EMBEDDING_SPACE_ID
except ImportError:  # pragma: no cover - script-style compatibility
    from embedding_backends import OPENAI_EMBEDDING_SPACE_ID


class MemoryMode(str, Enum):
    NONE = "none"
    NAIVE = "naive"
    COSINE = "cosine"
    PROJECTION = "projection"


class CandidateUniverse(str, Enum):
    M0_FULL_CATALOGUE = "m0_full_catalogue"
    POST_CURRENT_HARD_FILTER = "post_current_hard_filter"


class FallbackReason(str, Enum):
    NO_MEMORIES = "no_memories"
    NO_ELIGIBLE_MEMORIES = "no_eligible_memories"
    ALL_WEIGHTS_ZERO = "all_weights_zero"
    AGGREGATE_ZERO = "aggregate_zero"
    LOCAL_NEIGHBOURHOOD_EMPTY = "local_neighbourhood_empty"
    EFFECTIVE_RANK_ZERO = "effective_rank_zero"
    STEERING_ZERO = "steering_zero"


class InvalidReason(str, Enum):
    EMBEDDING_SPACE_MISMATCH = "embedding_space_mismatch"
    DIMENSION_MISMATCH = "dimension_mismatch"
    NON_FINITE_VECTOR = "non_finite_vector"
    Q_MISMATCH = "q_mismatch"
    CANDIDATE_UNIVERSE_UNSUPPORTED = "candidate_universe_unsupported"


class DenseScorer(Protocol):
    """The M0 methods/identity used by this boundary."""

    catalog_embeddings: np.ndarray
    embedding_space_id: str

    def dense_retrieve_vector(
        self, query_embedding: np.ndarray, top_n: int = 150
    ) -> Any: ...


class QLMPIntegrationError(ValueError):
    """Invalid experimental input with a stable machine-readable reason."""

    def __init__(self, reason: InvalidReason | str, message: str) -> None:
        self.reason = reason.value if isinstance(reason, InvalidReason) else str(reason)
        super().__init__(f"{self.reason}: {message}")


class ProjectionSteeringDeferredError(RuntimeError):
    """Raised when a caller tries to use isolation-only projection as B3."""


@dataclass(frozen=True)
class CandidateMemoryBatch:
    """Ordered memory tuple plus the embedding-space envelope QLMP lacks."""

    items: tuple[MemoryItem, ...]
    embedding_space_id: str

    def __post_init__(self) -> None:
        try:
            items = tuple(self.items)
        except TypeError as exc:
            raise ValueError("items must be an iterable of MemoryItem values") from exc
        if any(not isinstance(item, MemoryItem) for item in items):
            raise ValueError("items must contain only MemoryItem values")
        if not isinstance(self.embedding_space_id, str) or not self.embedding_space_id.strip():
            raise ValueError("embedding_space_id must be a non-empty string")
        object.__setattr__(self, "items", items)


@dataclass(frozen=True)
class QLMPIntegrationConfig:
    """One explicit switch/configuration for the component boundary."""

    memory_mode: MemoryMode | str = MemoryMode.NONE
    embedding_space_id: str = OPENAI_EMBEDDING_SPACE_ID
    embedding_dimension: int = 3072
    baseline: BaselineConfig = BaselineConfig()
    steering: SteeringConfig = SteeringConfig()
    projection: ProjectionConfig = ProjectionConfig()
    local_k: int = 500
    candidate_universe: CandidateUniverse | str = CandidateUniverse.M0_FULL_CATALOGUE

    def __post_init__(self) -> None:
        try:
            mode = MemoryMode(self.memory_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("memory_mode must be none, naive, cosine, or projection") from exc
        try:
            universe = CandidateUniverse(self.candidate_universe)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "candidate_universe must be m0_full_catalogue or "
                "post_current_hard_filter"
            ) from exc
        if not isinstance(self.embedding_space_id, str) or not self.embedding_space_id.strip():
            raise ValueError("embedding_space_id must be a non-empty string")
        if (
            isinstance(self.embedding_dimension, bool)
            or not isinstance(self.embedding_dimension, int)
            or self.embedding_dimension <= 0
        ):
            raise ValueError("embedding_dimension must be a positive integer")
        if isinstance(self.local_k, bool) or not isinstance(self.local_k, int) or self.local_k <= 0:
            raise ValueError("local_k must be a positive integer")
        if not isinstance(self.baseline, BaselineConfig):
            raise ValueError("baseline must be a BaselineConfig")
        if not isinstance(self.steering, SteeringConfig):
            raise ValueError("steering must be a SteeringConfig")
        if not isinstance(self.projection, ProjectionConfig):
            raise ValueError("projection must be a ProjectionConfig")
        object.__setattr__(self, "memory_mode", mode)
        object.__setattr__(self, "candidate_universe", universe)


@dataclass(frozen=True)
class QLMPIntegrationResult:
    """Small result for none/B1/B2 component retrieval."""

    memory_mode: MemoryMode
    q_m0: np.ndarray
    q_final: np.ndarray
    final_dense_result: Any
    baseline_result: MemoryBaselineResult | None = None
    steering_result: QuerySteeringResult | None = None
    fallback_reason: str | None = None
    angle64_deg: float | None = None
    angle32_deg: float | None = None


@dataclass(frozen=True)
class ProjectorMemoryResult:
    """One label-free query-memory projection produced by QLMP."""

    memory_id: str
    memory_text: str
    memory_scope: str | None
    memory_source: str
    memory_confidence: float
    projection: MemoryProjection


@dataclass(frozen=True)
class ProjectorIsolationResult:
    """First-pass M0 neighbourhood and projections, intentionally no q_star."""

    q_m0: np.ndarray
    initial_dense_result: Any
    local_subspace: LocalSubspace
    memory_projections: tuple[ProjectorMemoryResult, ...]
    requested_local_k: int
    candidate_universe: CandidateUniverse
    fallback_reason: str | None = None


def _invalid(reason: InvalidReason, message: str) -> QLMPIntegrationError:
    return QLMPIntegrationError(reason, message)


def _as_canonical_q(q_m0: object, dimension: int) -> np.ndarray:
    raw = np.asarray(q_m0)
    if raw.ndim != 1 or raw.shape[0] != dimension:
        raise _invalid(
            InvalidReason.DIMENSION_MISMATCH,
            f"q_m0 must have shape ({dimension},), got {raw.shape}",
        )
    if raw.dtype != np.dtype(np.float32):
        raise _invalid(
            InvalidReason.Q_MISMATCH,
            f"q_m0 must be the canonical float32 vector, got {raw.dtype}",
        )
    if not np.all(np.isfinite(raw)):
        raise _invalid(InvalidReason.NON_FINITE_VECTOR, "q_m0 contains non-finite values")
    norm = float(np.linalg.norm(raw))
    if not np.isclose(norm, 1.0, rtol=1e-5, atol=1e-6):
        raise _invalid(InvalidReason.Q_MISMATCH, f"q_m0 is not M0-normalized: {norm:.8g}")
    return raw


def promote_q_work(q_m0: object, *, dimension: int = 3072) -> np.ndarray:
    """Return an owned, normalized float64 working copy of canonical q."""

    canonical = _as_canonical_q(q_m0, dimension)
    working = np.array(canonical, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(working))
    if not math.isfinite(norm) or norm <= 0.0:
        raise _invalid(InvalidReason.NON_FINITE_VECTOR, "q_work norm is invalid")
    working /= norm
    return working


def promote_local_product_rows(
    products: object, *, dimension: int = 3072
) -> np.ndarray:
    """Promote and row-normalize an owned local product matrix only."""

    try:
        working = np.array(products, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise _invalid(
            InvalidReason.DIMENSION_MISMATCH,
            "local product vectors must be a numeric matrix",
        ) from exc
    if working.ndim != 2 or working.shape[1] != dimension:
        raise _invalid(
            InvalidReason.DIMENSION_MISMATCH,
            f"local products must have shape (K, {dimension}), got {working.shape}",
        )
    if not np.all(np.isfinite(working)):
        raise _invalid(
            InvalidReason.NON_FINITE_VECTOR,
            "local product vectors contain non-finite values",
        )
    if working.shape[0] == 0:
        return working
    norms = np.linalg.norm(working, axis=1)
    zero = np.flatnonzero(~np.isfinite(norms) | (norms <= 0.0))
    if zero.size:
        raise _invalid(
            InvalidReason.NON_FINITE_VECTOR,
            f"local product row {int(zero[0])} has an invalid norm",
        )
    working /= norms[:, None]
    return working


def _validate_owner(scorer: DenseScorer, config: QLMPIntegrationConfig) -> None:
    owner_space = str(getattr(scorer, "embedding_space_id", ""))
    if owner_space != config.embedding_space_id:
        raise _invalid(
            InvalidReason.EMBEDDING_SPACE_MISMATCH,
            "M0 and integration embedding spaces differ",
        )
    catalogue = np.asarray(getattr(scorer, "catalog_embeddings", None))
    if catalogue.ndim != 2 or catalogue.shape[1] != config.embedding_dimension:
        raise _invalid(
            InvalidReason.DIMENSION_MISMATCH,
            "M0 catalogue dimension differs from integration configuration",
        )


def _validate_memory_batch(
    batch: CandidateMemoryBatch,
    config: QLMPIntegrationConfig,
) -> tuple[MemoryItem, ...]:
    if batch.embedding_space_id != config.embedding_space_id:
        raise _invalid(
            InvalidReason.EMBEDDING_SPACE_MISMATCH,
            "candidate memories belong to a different embedding space",
        )
    ids: set[str] = set()
    for item in batch.items:
        vector = item.embedding
        if vector.dtype != np.dtype(np.float64) or vector.shape != (
            config.embedding_dimension,
        ):
            raise _invalid(
                InvalidReason.DIMENSION_MISMATCH,
                f"memory {item.id!r} does not match the float64 "
                f"({config.embedding_dimension},) contract",
            )
        if not np.all(np.isfinite(vector)):
            raise _invalid(
                InvalidReason.NON_FINITE_VECTOR,
                f"memory {item.id!r} contains non-finite values",
            )
        norm = float(np.linalg.norm(vector))
        if not np.isclose(
            norm,
            1.0,
            atol=config.projection.epsilon,
            rtol=config.projection.epsilon,
        ):
            raise _invalid(
                InvalidReason.Q_MISMATCH,
                f"memory {item.id!r} is not QLMP-normalized: {norm:.17g}",
            )
        if item.id in ids:
            raise ValueError(f"candidate memory IDs must be unique; duplicate {item.id!r}")
        ids.add(item.id)
    return batch.items


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    cosine = float(
        np.dot(left64, right64)
        / (float(np.linalg.norm(left64)) * float(np.linalg.norm(right64)))
    )
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _fallback_result(
    scorer: DenseScorer,
    *,
    mode: MemoryMode,
    q_m0: np.ndarray,
    top_n: int,
    reason: FallbackReason | None,
    baseline: MemoryBaselineResult | None = None,
    steering: QuerySteeringResult | None = None,
) -> QLMPIntegrationResult:
    dense = scorer.dense_retrieve_vector(q_m0, top_n=top_n)
    return QLMPIntegrationResult(
        memory_mode=mode,
        q_m0=q_m0,
        q_final=q_m0,
        final_dense_result=dense,
        baseline_result=baseline,
        steering_result=steering,
        fallback_reason=None if reason is None else reason.value,
        angle64_deg=0.0 if steering is not None else None,
        angle32_deg=0.0 if steering is not None else None,
    )


def run_qlmp_integration(
    scorer: DenseScorer,
    *,
    q_m0: np.ndarray,
    candidate_memories: CandidateMemoryBatch | None = None,
    query_scope: str | None = None,
    top_n: int = 150,
    config: QLMPIntegrationConfig | None = None,
) -> QLMPIntegrationResult:
    """Run exact M0, B1, or B2 and score with the unchanged M0 scorer."""

    settings = QLMPIntegrationConfig() if config is None else config
    if not isinstance(settings, QLMPIntegrationConfig):
        raise ValueError("config must be a QLMPIntegrationConfig")
    _validate_owner(scorer, settings)
    canonical = _as_canonical_q(q_m0, settings.embedding_dimension)

    if settings.memory_mode is MemoryMode.PROJECTION:
        raise ProjectionSteeringDeferredError(
            "projection is isolation-only before the stop/go gate; "
            "call run_projector_isolation()"
        )
    if settings.memory_mode is MemoryMode.NONE:
        # Intentionally do not inspect candidate_memories or create q_work.
        return _fallback_result(
            scorer,
            mode=MemoryMode.NONE,
            q_m0=canonical,
            top_n=top_n,
            reason=None,
        )
    if candidate_memories is None or not candidate_memories.items:
        return _fallback_result(
            scorer,
            mode=settings.memory_mode,
            q_m0=canonical,
            top_n=top_n,
            reason=FallbackReason.NO_MEMORIES,
        )

    memories = _validate_memory_batch(candidate_memories, settings)
    q_work = promote_q_work(canonical, dimension=settings.embedding_dimension)
    builder = (
        build_naive_memory_baseline
        if settings.memory_mode is MemoryMode.NAIVE
        else build_cosine_memory_baseline
    )
    baseline = builder(
        q_work,
        memories,
        query_scope=query_scope,
        config=settings.baseline,
    )
    steering = bound_query_shift(
        q_work,
        baseline.aggregate_delta,
        config=settings.steering,
    )

    reason: FallbackReason | None = None
    if not baseline.selected_memory_ids:
        reason = FallbackReason.NO_ELIGIBLE_MEMORIES
    elif settings.memory_mode is MemoryMode.COSINE and not any(
        item.aggregation_weight > 0.0 for item in baseline.memory_diagnostics
    ):
        reason = FallbackReason.ALL_WEIGHTS_ZERO
    elif float(np.linalg.norm(baseline.aggregate_delta)) <= settings.steering.epsilon:
        reason = FallbackReason.AGGREGATE_ZERO
    elif (
        steering.diagnostics.delta_zero
        or steering.diagnostics.applied_tangent_norm <= settings.steering.epsilon
        or steering.diagnostics.actual_shift_deg <= settings.steering.epsilon
    ):
        reason = FallbackReason.STEERING_ZERO
    if reason is not None:
        return _fallback_result(
            scorer,
            mode=settings.memory_mode,
            q_m0=canonical,
            top_n=top_n,
            reason=reason,
            baseline=baseline,
            steering=steering,
        )

    q_star64 = steering.q_star
    if not np.all(np.isfinite(q_star64)) or not np.isclose(
        float(np.linalg.norm(q_star64)), 1.0, atol=1e-12, rtol=1e-12
    ):
        raise _invalid(InvalidReason.NON_FINITE_VECTOR, "QLMP returned an invalid q_star64")
    q_star32 = np.asarray(q_star64, dtype=np.float32)
    angle64 = _angle_degrees(q_work, q_star64)
    angle32 = _angle_degrees(q_work, q_star32)
    if angle32 > settings.steering.max_shift_deg + 1e-4:
        raise _invalid(
            InvalidReason.Q_MISMATCH,
            "float32 q_star materially exceeds the configured angular cap",
        )
    dense = scorer.dense_retrieve_vector(q_star32, top_n=top_n)
    return QLMPIntegrationResult(
        memory_mode=settings.memory_mode,
        q_m0=canonical,
        q_final=q_star32,
        final_dense_result=dense,
        baseline_result=baseline,
        steering_result=steering,
        fallback_reason=None,
        angle64_deg=angle64,
        angle32_deg=angle32,
    )


def run_projector_isolation(
    scorer: DenseScorer,
    *,
    q_m0: np.ndarray,
    candidate_memories: CandidateMemoryBatch,
    config: QLMPIntegrationConfig,
) -> ProjectorIsolationResult:
    """Project memories in a real M0 neighbourhood and stop before steering."""

    if not isinstance(config, QLMPIntegrationConfig):
        raise ValueError("config must be a QLMPIntegrationConfig")
    if config.memory_mode is not MemoryMode.PROJECTION:
        raise ValueError("projector isolation requires memory_mode='projection'")
    if config.candidate_universe is not CandidateUniverse.M0_FULL_CATALOGUE:
        raise _invalid(
            InvalidReason.CANDIDATE_UNIVERSE_UNSUPPORTED,
            "post_current_hard_filter has not been exposed safely by M0",
        )
    _validate_owner(scorer, config)
    canonical = _as_canonical_q(q_m0, config.embedding_dimension)
    memories = _validate_memory_batch(candidate_memories, config)
    q_work = promote_q_work(canonical, dimension=config.embedding_dimension)

    initial = scorer.dense_retrieve_vector(canonical, top_n=config.local_k)
    local_products = promote_local_product_rows(
        initial.product_embeddings,
        dimension=config.embedding_dimension,
    )
    subspace = build_local_subspace(
        q_work,
        local_products,
        rank=config.projection.rank,
        epsilon=config.projection.epsilon,
    )
    projected = tuple(
        ProjectorMemoryResult(
            memory_id=item.id,
            memory_text=item.text,
            memory_scope=item.scope,
            memory_source=item.source.value,
            memory_confidence=item.confidence,
            projection=project_memory_residual(
                q_work,
                item.embedding,
                subspace.basis,
                epsilon=config.projection.epsilon,
            ),
        )
        for item in memories
    )
    fallback: FallbackReason | None = None
    if local_products.shape[0] == 0:
        fallback = FallbackReason.LOCAL_NEIGHBOURHOOD_EMPTY
    elif subspace.effective_rank == 0:
        fallback = FallbackReason.EFFECTIVE_RANK_ZERO
    return ProjectorIsolationResult(
        q_m0=canonical,
        initial_dense_result=initial,
        local_subspace=subspace,
        memory_projections=projected,
        requested_local_k=config.local_k,
        candidate_universe=config.candidate_universe,
        fallback_reason=None if fallback is None else fallback.value,
    )


__all__ = [
    "CandidateMemoryBatch",
    "CandidateUniverse",
    "FallbackReason",
    "InvalidReason",
    "MemoryMode",
    "ProjectorIsolationResult",
    "ProjectorMemoryResult",
    "ProjectionSteeringDeferredError",
    "QLMPIntegrationConfig",
    "QLMPIntegrationError",
    "QLMPIntegrationResult",
    "promote_local_product_rows",
    "promote_q_work",
    "run_projector_isolation",
    "run_qlmp_integration",
]
