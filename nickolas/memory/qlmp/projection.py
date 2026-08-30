"""Projection of memory residuals into query-local tangent subspaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import (
    _require_unit_vector,
    _validate_epsilon,
    cosine_similarity,
    tangent_residual,
)
from .models import FloatArray, MemoryProjection, _validate_basis_contract


def _as_basis(value: Any, dimension: int) -> FloatArray:
    try:
        basis = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("basis must be convertible to a float64 matrix") from exc
    if basis.ndim != 2:
        raise ValueError(f"basis must be two-dimensional, got shape {basis.shape}")
    if basis.shape[0] != dimension:
        raise ValueError(
            f"basis must have embedding dimension {dimension}, got {basis.shape[0]}"
        )
    if not np.all(np.isfinite(basis)):
        raise ValueError("basis must contain only finite values")
    return basis


def _roundoff_clamp_fraction(value: float, basis_tolerance: float) -> float:
    roundoff = (
        2.0 * basis_tolerance
        + basis_tolerance**2
        + 32.0 * float(np.finfo(np.float64).eps)
    )
    if 1.0 < value <= 1.0 + roundoff:
        return 1.0
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"projection fraction escaped [0, 1]: {value:.17g}")
    return value


def project_memory_residual(
    q: Any,
    memory: Any,
    basis: Any,
    epsilon: float = 1e-8,
) -> MemoryProjection:
    """Project a normalized memory's tangent residual onto ``basis``.

    The reported fraction is ``||projection||^2 / (||residual||^2 + epsilon)``.
    """

    tolerance = _validate_epsilon(epsilon)
    query = _require_unit_vector(q, "q", tolerance)
    memory_vector = _require_unit_vector(memory, "memory", tolerance)
    if query.shape != memory_vector.shape:
        raise ValueError(
            f"q and memory must have equal dimensions, got {query.size} and "
            f"{memory_vector.size}"
        )
    local_basis = _as_basis(basis, query.size)
    basis_tolerance = _validate_basis_contract(local_basis, query=query)

    residual = tangent_residual(query, memory_vector, tolerance)
    coefficients = local_basis.T @ residual
    projected = local_basis @ coefficients
    tangent_norm = float(np.linalg.norm(residual))
    projected_norm = float(np.linalg.norm(projected))
    raw_cosine = cosine_similarity(query, memory_vector, tolerance)
    fraction = projected_norm**2 / (tangent_norm**2 + tolerance)
    fraction = _roundoff_clamp_fraction(float(fraction), basis_tolerance)
    return MemoryProjection(
        residual=residual,
        coefficients=coefficients,
        projected_residual=projected,
        raw_query_memory_cosine=raw_cosine,
        tangent_norm=tangent_norm,
        projected_norm=projected_norm,
        projection_fraction=fraction,
    )
