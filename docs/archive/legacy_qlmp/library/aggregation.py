"""Reusable weighted aggregation for raw and projected memory residuals."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import tangent_residual
from .models import FloatArray


def memory_tangent_residual(
    q: Any, memory_embedding: Any, epsilon: float = 1e-8
) -> FloatArray:
    """Return the Phase 1 tangent residual for one normalized memory."""

    return tangent_residual(q, memory_embedding, epsilon=epsilon)


def normalize_nonnegative_weights(
    weights: Any, *, normalize: bool = True
) -> FloatArray:
    """Validate finite non-negative weights and optionally make them sum to one.

    A zero-total vector remains all zero.  The returned array is an owned
    ``float64`` copy, and the caller's input is never mutated.
    """

    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a bool")
    try:
        result = np.array(weights, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be convertible to a float64 vector") from exc
    if result.ndim != 1:
        raise ValueError(f"weights must be one-dimensional, got shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("weights must contain only finite values")
    if np.any(result < 0.0):
        raise ValueError("weights must be non-negative")
    if normalize:
        total = float(np.sum(result, dtype=np.float64))
        if total > 0.0:
            result /= total
        else:
            result.fill(0.0)
    return result


def _as_residual_matrix(residuals: Any) -> FloatArray:
    try:
        matrix = np.array(residuals, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("residuals must be convertible to a float64 matrix") from exc
    if matrix.ndim != 2:
        raise ValueError(f"residuals must be two-dimensional, got shape {matrix.shape}")
    if matrix.shape[1] == 0:
        raise ValueError("residuals must have a positive embedding dimension")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("residuals must contain only finite values")
    return matrix


def aggregate_residuals(
    residuals: Any,
    weights: Any,
    *,
    normalize_weights: bool = True,
) -> FloatArray:
    """Return ``sum(alpha_i * residual_i)`` without selecting or steering."""

    matrix = _as_residual_matrix(residuals)
    alphas = normalize_nonnegative_weights(weights, normalize=normalize_weights)
    if alphas.shape != (matrix.shape[0],):
        raise ValueError(
            "weights length must equal residual count, "
            f"got {alphas.size} and {matrix.shape[0]}"
        )
    return np.asarray(alphas @ matrix, dtype=np.float64)


def aggregate_raw_residuals(
    residuals: Any,
    weights: Any,
    *,
    normalize_weights: bool = True,
) -> FloatArray:
    """Aggregate already-computed raw tangent residuals."""

    return aggregate_residuals(
        residuals, weights, normalize_weights=normalize_weights
    )


def aggregate_projected_residuals(
    projected_residuals: Any,
    weights: Any,
    *,
    normalize_weights: bool = True,
) -> FloatArray:
    """Aggregate already-projected Phase 1 residuals."""

    return aggregate_residuals(
        projected_residuals, weights, normalize_weights=normalize_weights
    )
