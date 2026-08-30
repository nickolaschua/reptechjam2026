"""Inspectable, JSON-safe result models for Phase 2 memory steering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

from .models import (
    FloatArray,
    MemoryPolarity,
    MemorySource,
    _readonly_vector,
)


class BaselineMode(str, Enum):
    """Stable names for the implemented pre-Checkpoint-A baselines."""

    NAIVE = "naive"
    COSINE = "cosine"


def _optional_finite(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be None or finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be None or finite")
    return result


@dataclass(frozen=True)
class MemorySelectionDiagnostic:
    """One memory's honest participation in B1 or B2."""

    memory_id: str
    scope: str | None
    source: MemorySource
    polarity: MemoryPolarity
    confidence: float
    scope_compatible: bool
    polarity_eligible: bool
    raw_cosine: float | None
    threshold_passed: bool | None
    selected: bool
    selection_rank: int | None
    raw_tangent_norm: float | None
    aggregation_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        if self.scope is not None and (
            not isinstance(self.scope, str) or not self.scope.strip()
        ):
            raise ValueError("scope must be None or a non-empty string")
        try:
            source = MemorySource(self.source)
            polarity = MemoryPolarity(self.polarity)
        except (TypeError, ValueError) as exc:
            raise ValueError("source and polarity must be valid enum values") from exc
        for name in ("scope_compatible", "polarity_eligible", "selected"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.threshold_passed is not None and not isinstance(
            self.threshold_passed, bool
        ):
            raise ValueError("threshold_passed must be None or a bool")
        if self.selection_rank is not None and (
            isinstance(self.selection_rank, bool)
            or not isinstance(self.selection_rank, int)
            or self.selection_rank <= 0
        ):
            raise ValueError("selection_rank must be None or a positive integer")
        confidence = _optional_finite(self.confidence, "confidence")
        raw_cosine = _optional_finite(self.raw_cosine, "raw_cosine")
        tangent_norm = _optional_finite(self.raw_tangent_norm, "raw_tangent_norm")
        weight = _optional_finite(self.aggregation_weight, "aggregation_weight")
        assert confidence is not None and weight is not None
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if raw_cosine is not None and not -1.0 <= raw_cosine <= 1.0:
            raise ValueError("raw_cosine must lie in [-1, 1]")
        if tangent_norm is not None and tangent_norm < 0.0:
            raise ValueError("raw_tangent_norm must be non-negative")
        if weight < 0.0:
            raise ValueError("aggregation_weight must be non-negative")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "raw_cosine", raw_cosine)
        object.__setattr__(self, "raw_tangent_norm", tangent_norm)
        object.__setattr__(self, "aggregation_weight", weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "scope": self.scope,
            "source": self.source.value,
            "polarity": self.polarity.value,
            "confidence": self.confidence,
            "scope_compatible": self.scope_compatible,
            "polarity_eligible": self.polarity_eligible,
            "raw_cosine": self.raw_cosine,
            "threshold_passed": self.threshold_passed,
            "selected": self.selected,
            "selection_rank": self.selection_rank,
            "raw_tangent_norm": self.raw_tangent_norm,
            "aggregation_weight": self.aggregation_weight,
        }


@dataclass(frozen=True, eq=False)
class MemoryBaselineResult:
    """Selected memory metadata and aggregate delta, independent of retrieval."""

    mode: BaselineMode
    aggregate_delta: FloatArray
    selected_memory_ids: tuple[str, ...]
    memory_diagnostics: tuple[MemorySelectionDiagnostic, ...]

    def __post_init__(self) -> None:
        try:
            mode = BaselineMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("mode must be 'naive' or 'cosine'") from exc
        aggregate = _readonly_vector(self.aggregate_delta, "aggregate_delta")
        ids = tuple(self.selected_memory_ids)
        diagnostics = tuple(self.memory_diagnostics)
        if any(not isinstance(item, MemorySelectionDiagnostic) for item in diagnostics):
            raise ValueError("memory_diagnostics must contain diagnostic records")
        ranked = sorted(
            (item for item in diagnostics if item.selected),
            key=lambda item: item.selection_rank or 0,
        )
        if ids != tuple(item.memory_id for item in ranked):
            raise ValueError("selected_memory_ids must match diagnostic selection ranks")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "aggregate_delta", aggregate)
        object.__setattr__(self, "selected_memory_ids", ids)
        object.__setattr__(self, "memory_diagnostics", diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "aggregate_delta": self.aggregate_delta.tolist(),
            "selected_memory_ids": list(self.selected_memory_ids),
            "memory_diagnostics": [item.to_dict() for item in self.memory_diagnostics],
        }


@dataclass(frozen=True)
class SteeringDiagnostics:
    """Requested and applied quantities for one bounded query shift."""

    requested_beta: float
    applied_beta: float
    original_delta_norm: float
    corrected_tangent_norm: float
    requested_tangent_norm: float
    applied_tangent_norm: float
    unclipped_angle_deg: float
    max_shift_deg: float
    actual_shift_deg: float
    clipped: bool
    delta_zero: bool
    tangency_corrected: bool

    def __post_init__(self) -> None:
        scalar_names = (
            "requested_beta",
            "applied_beta",
            "original_delta_norm",
            "corrected_tangent_norm",
            "requested_tangent_norm",
            "applied_tangent_norm",
            "unclipped_angle_deg",
            "max_shift_deg",
            "actual_shift_deg",
        )
        for name in scalar_names:
            value = _optional_finite(getattr(self, name), name)
            assert value is not None
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("clipped", "delta_zero", "tangency_corrected"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_beta": self.requested_beta,
            "applied_beta": self.applied_beta,
            "original_delta_norm": self.original_delta_norm,
            "corrected_tangent_norm": self.corrected_tangent_norm,
            "requested_tangent_norm": self.requested_tangent_norm,
            "applied_tangent_norm": self.applied_tangent_norm,
            "unclipped_angle_deg": self.unclipped_angle_deg,
            "max_shift_deg": self.max_shift_deg,
            "actual_shift_deg": self.actual_shift_deg,
            "clipped": self.clipped,
            "delta_zero": self.delta_zero,
            "tangency_corrected": self.tangency_corrected,
        }


@dataclass(frozen=True, eq=False)
class QuerySteeringResult:
    """Normalized personalized query and its steering diagnostics."""

    q_star: FloatArray
    diagnostics: SteeringDiagnostics

    def __post_init__(self) -> None:
        q_star = _readonly_vector(self.q_star, "q_star")
        if not math.isclose(float(sum(q_star * q_star)), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("q_star must be normalized")
        if not isinstance(self.diagnostics, SteeringDiagnostics):
            raise ValueError("diagnostics must be SteeringDiagnostics")
        object.__setattr__(self, "q_star", q_star)

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_star": self.q_star.tolist(),
            "diagnostics": self.diagnostics.to_dict(),
        }
