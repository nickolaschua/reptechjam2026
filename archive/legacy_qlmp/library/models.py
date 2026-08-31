"""Small, validated data models for Phase 1 QLMP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class MemorySource(str, Enum):
    """Origin of a memory item, independent of its semantic polarity."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    EXPLICIT_PREFERENCE = "explicit_preference"
    PURCHASE_EPISODE = "purchase_episode"
    BEHAVIORAL_INFERENCE = "behavioral_inference"
    CLICK = "click"
    RECOMMENDATION_SHOWN = "recommendation_shown"


class MemoryPolarity(str, Enum):
    """Whether a memory expresses positive, negative, or neutral evidence."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


def _readonly_vector(value: Any, name: str, *, allow_empty: bool = False) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a float64 vector") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_matrix(value: Any, name: str) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a float64 matrix") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _basis_contract_tolerance(embedding_dimension: int, basis_rank: int) -> float:
    """Return the dimension-scaled float64 tolerance for a tangent basis."""

    return (
        32.0
        * max(1, embedding_dimension, basis_rank)
        * float(np.finfo(np.float64).eps)
    )


def _validate_basis_contract(
    basis: FloatArray, *, query: FloatArray | None = None
) -> float:
    """Require spectral-norm orthonormality and, when supplied, tangency."""

    embedding_dimension, basis_rank = basis.shape
    tolerance = _basis_contract_tolerance(embedding_dimension, basis_rank)
    if basis_rank == 0:
        return tolerance

    gram_error = float(
        np.linalg.norm(basis.T @ basis - np.eye(basis_rank), ord=2)
    )
    if gram_error > tolerance:
        raise ValueError(
            "basis columns must be orthonormal: "
            f"||B.T @ B - I||_2={gram_error:.17g} exceeds tau={tolerance:.17g}"
        )
    if query is not None:
        tangent_error = float(np.linalg.norm(query @ basis, ord=2))
        if tangent_error > tolerance:
            raise ValueError(
                "basis columns must be tangent (orthogonal) to q: "
                f"||q.T @ B||_2={tangent_error:.17g} exceeds tau={tolerance:.17g}"
            )
    return tolerance


@dataclass(frozen=True, eq=False)
class MemoryItem:
    """A dense, labelled memory record used by later QLMP phases."""

    id: str
    text: str
    embedding: FloatArray
    source: MemorySource
    polarity: MemoryPolarity
    scope: str | None = None
    timestamp: datetime | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if self.scope is not None and (
            not isinstance(self.scope, str) or not self.scope.strip()
        ):
            raise ValueError("scope must be None or a non-empty string")
        if self.timestamp is not None and not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be None or a datetime")
        if isinstance(self.confidence, (bool, np.bool_)):
            raise ValueError("confidence must be a finite number in [0, 1]")
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be a finite number in [0, 1]") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite number in [0, 1]")
        try:
            source = MemorySource(self.source)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source must be a valid MemorySource, got {self.source!r}") from exc
        try:
            polarity = MemoryPolarity(self.polarity)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"polarity must be a valid MemoryPolarity, got {self.polarity!r}"
            ) from exc
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "embedding", _readonly_vector(self.embedding, "embedding"))


@dataclass(frozen=True, eq=False)
class LocalSubspace:
    """SVD-derived tangent basis and the diagnostics needed to audit it."""

    basis: FloatArray
    singular_values: FloatArray
    requested_rank: int
    effective_rank: int
    embedding_dimension: int
    local_product_count: int

    def __post_init__(self) -> None:
        basis = _readonly_matrix(self.basis, "basis")
        singular_values = _readonly_vector(
            self.singular_values, "singular_values", allow_empty=True
        )
        counts = {
            "requested_rank": self.requested_rank,
            "effective_rank": self.effective_rank,
            "embedding_dimension": self.embedding_dimension,
            "local_product_count": self.local_product_count,
        }
        for name, value in counts.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.requested_rank <= 0:
            raise ValueError("requested_rank must be positive")
        if self.effective_rank < 0:
            raise ValueError("effective_rank must be non-negative")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.local_product_count < 0:
            raise ValueError("local_product_count must be non-negative")
        expected_spectrum = min(self.local_product_count, self.embedding_dimension)
        if basis.shape != (self.embedding_dimension, self.effective_rank):
            raise ValueError(
                "basis shape must equal (embedding_dimension, effective_rank), "
                f"got {basis.shape}"
            )
        if singular_values.shape != (expected_spectrum,):
            raise ValueError(
                f"singular_values must have shape ({expected_spectrum},), "
                f"got {singular_values.shape}"
            )
        if self.effective_rank > min(
            self.requested_rank, self.local_product_count, self.embedding_dimension
        ):
            raise ValueError("effective_rank exceeds its requested or dimensional cap")
        if np.any(singular_values < 0.0):
            raise ValueError("singular_values must be non-negative")
        _validate_basis_contract(basis)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "singular_values", singular_values)


@dataclass(frozen=True, eq=False)
class MemoryProjection:
    """Projection of a query-memory tangent residual into a local subspace."""

    residual: FloatArray
    coefficients: FloatArray
    projected_residual: FloatArray
    raw_query_memory_cosine: float
    tangent_norm: float
    projected_norm: float
    projection_fraction: float

    def __post_init__(self) -> None:
        residual = _readonly_vector(self.residual, "residual")
        coefficients = _readonly_vector(self.coefficients, "coefficients", allow_empty=True)
        projected = _readonly_vector(self.projected_residual, "projected_residual")
        if projected.shape != residual.shape:
            raise ValueError("projected_residual must have the same shape as residual")
        scalar_names = (
            "raw_query_memory_cosine",
            "tangent_norm",
            "projected_norm",
            "projection_fraction",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not -1.0 <= self.raw_query_memory_cosine <= 1.0:
            raise ValueError("raw_query_memory_cosine must lie in [-1, 1]")
        if self.tangent_norm < 0.0 or self.projected_norm < 0.0:
            raise ValueError("projection norms must be non-negative")
        if not 0.0 <= self.projection_fraction <= 1.0:
            raise ValueError("projection_fraction must lie in [0, 1]")
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "projected_residual", projected)

    @property
    def raw_cosine(self) -> float:
        """Concise alias for ``raw_query_memory_cosine``."""

        return self.raw_query_memory_cosine
