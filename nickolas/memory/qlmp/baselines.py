"""Deterministic B1 naive and B2 cosine-gated memory mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .aggregation import (
    aggregate_raw_residuals,
    memory_tangent_residual,
    normalize_nonnegative_weights,
)
from .config import BaselineConfig
from .diagnostics import (
    BaselineMode,
    MemoryBaselineResult,
    MemorySelectionDiagnostic,
)
from .geometry import _require_unit_vector, cosine_similarity
from .models import FloatArray, MemoryItem, MemoryPolarity


def _validate_scope(scope: str | None, name: str) -> str | None:
    if scope is not None and (not isinstance(scope, str) or not scope.strip()):
        raise ValueError(f"{name} must be None or a non-empty string")
    return scope


def is_scope_compatible(
    query_scope: str | None,
    memory_scope: str | None,
    *,
    allow_unscoped_memory: bool = True,
) -> bool:
    """Return whether an exact structured scope check permits a memory.

    With no query scope there is no structured restriction.  With a scoped
    query, an unscoped memory is eligible only when ``allow_unscoped_memory``
    is true (the default).  Two present scopes must match exactly and
    case-sensitively; no taxonomy or semantic inference is attempted.
    """

    current = _validate_scope(query_scope, "query_scope")
    historical = _validate_scope(memory_scope, "memory_scope")
    if not isinstance(allow_unscoped_memory, bool):
        raise ValueError("allow_unscoped_memory must be a bool")
    if current is None:
        return True
    if historical is None:
        return allow_unscoped_memory
    return current == historical


@dataclass(frozen=True)
class _ScoredMemory:
    index: int
    item: MemoryItem
    scope_compatible: bool
    polarity_eligible: bool
    raw_cosine: float | None
    residual: FloatArray | None
    threshold_passed: bool | None


def _inputs(
    q: object,
    memories: Iterable[MemoryItem],
    query_scope: str | None,
    config: BaselineConfig | None,
) -> tuple[FloatArray, tuple[MemoryItem, ...], str | None, BaselineConfig]:
    settings = BaselineConfig() if config is None else config
    if not isinstance(settings, BaselineConfig):
        raise ValueError("config must be a BaselineConfig")
    current_scope = _validate_scope(query_scope, "query_scope")
    query = _require_unit_vector(q, "q", settings.epsilon)
    try:
        items = tuple(memories)
    except TypeError as exc:
        raise ValueError("memories must be an iterable of MemoryItem values") from exc
    if any(not isinstance(item, MemoryItem) for item in items):
        raise ValueError("memories must contain only MemoryItem values")
    return query, items, current_scope, settings


def _score_structurally_eligible(
    query: FloatArray,
    items: tuple[MemoryItem, ...],
    query_scope: str | None,
    settings: BaselineConfig,
    *,
    apply_threshold: bool,
) -> list[_ScoredMemory]:
    records: list[_ScoredMemory] = []
    for index, item in enumerate(items):
        scope_ok = is_scope_compatible(
            query_scope,
            item.scope,
            allow_unscoped_memory=settings.allow_unscoped_memory,
        )
        polarity_ok = item.polarity is not MemoryPolarity.NEGATIVE
        if scope_ok and polarity_ok:
            raw_cosine = cosine_similarity(query, item.embedding, settings.epsilon)
            residual = memory_tangent_residual(
                query, item.embedding, settings.epsilon
            )
            threshold_passed = (
                settings.cosine_threshold is None
                or raw_cosine >= settings.cosine_threshold
            ) if apply_threshold else None
        else:
            raw_cosine = None
            residual = None
            threshold_passed = None
        records.append(
            _ScoredMemory(
                index=index,
                item=item,
                scope_compatible=scope_ok,
                polarity_eligible=polarity_ok,
                raw_cosine=raw_cosine,
                residual=residual,
                threshold_passed=threshold_passed,
            )
        )
    return records


def _diagnostic(
    record: _ScoredMemory,
    *,
    selected: bool,
    selection_rank: int | None,
    weight: float,
) -> MemorySelectionDiagnostic:
    return MemorySelectionDiagnostic(
        memory_id=record.item.id,
        scope=record.item.scope,
        source=record.item.source,
        polarity=record.item.polarity,
        confidence=record.item.confidence,
        scope_compatible=record.scope_compatible,
        polarity_eligible=record.polarity_eligible,
        raw_cosine=record.raw_cosine,
        threshold_passed=record.threshold_passed,
        selected=selected,
        selection_rank=selection_rank,
        raw_tangent_norm=(
            None if record.residual is None else float(np.linalg.norm(record.residual))
        ),
        aggregation_weight=weight,
    )


def build_naive_memory_baseline(
    q: object,
    memories: Iterable[MemoryItem],
    *,
    query_scope: str | None = None,
    config: BaselineConfig | None = None,
) -> MemoryBaselineResult:
    """Build B1 by uniformly aggregating every eligible raw residual.

    Raw cosine is computed for diagnostics only and never affects B1 selection
    or weighting.
    """

    query, items, current_scope, settings = _inputs(
        q, memories, query_scope, config
    )
    records = _score_structurally_eligible(
        query, items, current_scope, settings, apply_threshold=False
    )
    selected = [
        record
        for record in records
        if record.scope_compatible and record.polarity_eligible
    ]
    weights = normalize_nonnegative_weights(np.ones(len(selected)))
    if selected:
        residuals = np.vstack([record.residual for record in selected])
        aggregate = aggregate_raw_residuals(
            residuals, weights, normalize_weights=False
        )
    else:
        aggregate = np.zeros(query.size, dtype=np.float64)
    rank_by_index = {record.index: rank for rank, record in enumerate(selected, 1)}
    weight_by_index = {
        record.index: float(weight) for record, weight in zip(selected, weights)
    }
    diagnostics = tuple(
        _diagnostic(
            record,
            selected=record.index in rank_by_index,
            selection_rank=rank_by_index.get(record.index),
            weight=weight_by_index.get(record.index, 0.0),
        )
        for record in records
    )
    return MemoryBaselineResult(
        mode=BaselineMode.NAIVE,
        aggregate_delta=aggregate,
        selected_memory_ids=tuple(record.item.id for record in selected),
        memory_diagnostics=diagnostics,
    )


def build_cosine_memory_baseline(
    q: object,
    memories: Iterable[MemoryItem],
    *,
    query_scope: str | None = None,
    config: BaselineConfig | None = None,
) -> MemoryBaselineResult:
    """Build B2 with deterministic cosine gating and raw residual steering.

    Candidates meeting the inclusive threshold are ranked by descending raw
    cosine, with original input order as the stable tie-break.  Selected
    cosine scores are clamped at zero and normalized; if all are non-positive,
    every aggregation weight is zero.
    """

    query, items, current_scope, settings = _inputs(
        q, memories, query_scope, config
    )
    records = _score_structurally_eligible(
        query, items, current_scope, settings, apply_threshold=True
    )
    thresholded = [record for record in records if record.threshold_passed]
    ranked = sorted(
        thresholded,
        key=lambda record: (-float(record.raw_cosine), record.index),
    )
    selected = ranked[: settings.memory_top_k]
    raw_weights = np.asarray(
        [max(float(record.raw_cosine), 0.0) for record in selected],
        dtype=np.float64,
    )
    weights = normalize_nonnegative_weights(raw_weights)
    if selected:
        residuals = np.vstack([record.residual for record in selected])
        aggregate = aggregate_raw_residuals(
            residuals, weights, normalize_weights=False
        )
    else:
        aggregate = np.zeros(query.size, dtype=np.float64)
    rank_by_index = {record.index: rank for rank, record in enumerate(selected, 1)}
    weight_by_index = {
        record.index: float(weight) for record, weight in zip(selected, weights)
    }
    diagnostics = tuple(
        _diagnostic(
            record,
            selected=record.index in rank_by_index,
            selection_rank=rank_by_index.get(record.index),
            weight=weight_by_index.get(record.index, 0.0),
        )
        for record in records
    )
    return MemoryBaselineResult(
        mode=BaselineMode.COSINE,
        aggregate_delta=aggregate,
        selected_memory_ids=tuple(record.item.id for record in selected),
        memory_diagnostics=diagnostics,
    )
