"""Validated configuration for isolated QLMP geometry and steering."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProjectionConfig:
    """Numerical controls for QLMP.

    The defaults are engineering starting points only.  They have not been
    tuned against a real catalogue or downstream retrieval metric.
    """

    rank: int = 16
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if isinstance(self.epsilon, bool):
            raise ValueError("epsilon must be a positive finite number")
        try:
            epsilon = float(self.epsilon)
        except (TypeError, ValueError) as exc:
            raise ValueError("epsilon must be a positive finite number") from exc
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be a positive finite number")
        object.__setattr__(self, "epsilon", epsilon)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class BaselineConfig:
    """Controls deterministic B1/B2 selection.

    Defaults are untuned engineering starting values, not scientifically
    validated parameter choices.
    """

    memory_top_k: int = 3
    cosine_threshold: float | None = None
    allow_unscoped_memory: bool = True
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if (
            isinstance(self.memory_top_k, bool)
            or not isinstance(self.memory_top_k, int)
            or self.memory_top_k <= 0
        ):
            raise ValueError("memory_top_k must be a positive integer")
        if self.cosine_threshold is not None:
            threshold = _finite_number(self.cosine_threshold, "cosine_threshold")
            if not -1.0 <= threshold <= 1.0:
                raise ValueError("cosine_threshold must lie in [-1, 1]")
            object.__setattr__(self, "cosine_threshold", threshold)
        if not isinstance(self.allow_unscoped_memory, bool):
            raise ValueError("allow_unscoped_memory must be a bool")
        epsilon = _finite_number(self.epsilon, "epsilon")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "epsilon", epsilon)


@dataclass(frozen=True)
class SteeringConfig:
    """Controls the common bounded query-steering operation.

    ``max_shift_deg=10`` and ``beta=1`` are untuned engineering defaults.
    """

    beta: float = 1.0
    max_shift_deg: float = 10.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        beta = _finite_number(self.beta, "beta")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        max_shift = _finite_number(self.max_shift_deg, "max_shift_deg")
        if not 0.0 <= max_shift < 90.0:
            raise ValueError("max_shift_deg must satisfy 0 <= angle < 90")
        epsilon = _finite_number(self.epsilon, "epsilon")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "max_shift_deg", max_shift)
        object.__setattr__(self, "epsilon", epsilon)
