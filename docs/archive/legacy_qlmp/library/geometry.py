"""Query-centred tangent geometry for QLMP."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import FloatArray, LocalSubspace


def _validate_epsilon(epsilon: float) -> float:
    if isinstance(epsilon, bool):
        raise ValueError("epsilon must be a positive finite number")
    try:
        value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("epsilon must be a positive finite number")
    return value


def _validate_rank(rank: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    return rank


def _as_vector(value: Any, name: str) -> FloatArray:
    try:
        vector = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a float64 vector") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _require_unit_vector(value: Any, name: str, epsilon: float) -> FloatArray:
    vector = _as_vector(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= epsilon:
        raise ValueError(f"{name} norm must be greater than epsilon ({epsilon:g})")
    if not np.isclose(norm, 1.0, atol=epsilon, rtol=epsilon):
        raise ValueError(f"{name} must be normalized to unit length; got norm {norm:.17g}")
    return vector


def normalize(vector: Any, epsilon: float = 1e-8) -> FloatArray:
    """Return a finite, one-dimensional ``float64`` unit vector copy."""

    tolerance = _validate_epsilon(epsilon)
    result = _as_vector(vector, "vector")
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm):
        raise ValueError("vector norm must be finite")
    if norm <= tolerance:
        raise ValueError(f"vector norm must be greater than epsilon ({tolerance:g})")
    result /= norm
    return result


def cosine_similarity(left: Any, right: Any, epsilon: float = 1e-8) -> float:
    """Return cosine similarity while applying the package zero-vector policy."""

    tolerance = _validate_epsilon(epsilon)
    left_vector = _as_vector(left, "left")
    right_vector = _as_vector(right, "right")
    if left_vector.shape != right_vector.shape:
        raise ValueError(
            f"left and right must have equal dimensions, got {left_vector.size} and "
            f"{right_vector.size}"
        )
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm <= tolerance:
        raise ValueError(f"left norm must be greater than epsilon ({tolerance:g})")
    if right_norm <= tolerance:
        raise ValueError(f"right norm must be greater than epsilon ({tolerance:g})")
    cosine = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
    return float(np.clip(cosine, -1.0, 1.0))


def tangent_residual(q: Any, vector: Any, epsilon: float = 1e-8) -> FloatArray:
    """Remove the component of normalized ``vector`` parallel to normalized ``q``."""

    tolerance = _validate_epsilon(epsilon)
    query = _require_unit_vector(q, "q", tolerance)
    candidate = _require_unit_vector(vector, "vector", tolerance)
    if query.shape != candidate.shape:
        raise ValueError(
            f"q and vector must have equal dimensions, got {query.size} and {candidate.size}"
        )
    residual = candidate - float(np.dot(query, candidate)) * query
    # A second subtraction removes the tiny query component left by roundoff.
    residual -= float(np.dot(query, residual)) * query
    return residual


def _as_product_matrix(products: Any, dimension: int) -> FloatArray:
    try:
        matrix = np.array(products, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("products must be convertible to a float64 matrix") from exc
    if matrix.size == 0 and matrix.ndim == 1:
        return np.empty((0, dimension), dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"products must be two-dimensional, got shape {matrix.shape}")
    if matrix.shape[1] != dimension:
        raise ValueError(
            f"products must have embedding dimension {dimension}, got {matrix.shape[1]}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("products must contain only finite values")
    return matrix


def build_tangent_matrix(q: Any, products: Any, epsilon: float = 1e-8) -> FloatArray:
    """Build the ``K x d`` matrix of query-centred local-product residuals."""

    tolerance = _validate_epsilon(epsilon)
    query = _require_unit_vector(q, "q", tolerance)
    matrix = _as_product_matrix(products, query.size)
    if matrix.shape[0] == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1)
    small = np.flatnonzero(norms <= tolerance)
    if small.size:
        raise ValueError(
            f"products row {int(small[0])} norm must be greater than epsilon ({tolerance:g})"
        )
    non_unit = np.flatnonzero(~np.isclose(norms, 1.0, atol=tolerance, rtol=tolerance))
    if non_unit.size:
        index = int(non_unit[0])
        raise ValueError(
            f"products row {index} must be normalized to unit length; got norm "
            f"{float(norms[index]):.17g}"
        )
    residuals = matrix - np.outer(matrix @ query, query)
    residuals -= np.outer(residuals @ query, query)
    return residuals


def _numerical_rank_tolerance(matrix: FloatArray, sigma_max: float) -> float:
    """Return the conventional SVD numerical-rank tolerance for ``matrix``."""

    return (
        float(sigma_max)
        * max(matrix.shape)
        * float(np.finfo(np.float64).eps)
    )


def build_local_subspace(
    q: Any,
    products: Any,
    rank: int,
    epsilon: float = 1e-8,
) -> LocalSubspace:
    """Fit a numerically ranked tangent basis with query-centred SVD.

    No ordinary mean-centering is performed: every row is a residual relative
    to the current query direction.
    """

    requested_rank = _validate_rank(rank)
    tolerance = _validate_epsilon(epsilon)
    query = _require_unit_vector(q, "q", tolerance)
    tangent_matrix = build_tangent_matrix(query, products, tolerance)
    product_count, dimension = tangent_matrix.shape
    if product_count == 0:
        return LocalSubspace(
            basis=np.empty((dimension, 0), dtype=np.float64),
            singular_values=np.empty((0,), dtype=np.float64),
            requested_rank=requested_rank,
            effective_rank=0,
            embedding_dimension=dimension,
            local_product_count=0,
        )

    _, singular_values, right_vectors = np.linalg.svd(
        tangent_matrix, full_matrices=False
    )
    if singular_values.size:
        rank_tolerance = _numerical_rank_tolerance(
            tangent_matrix, float(singular_values[0])
        )
        numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    else:
        numerical_rank = 0
    effective_rank = min(requested_rank, numerical_rank)
    basis = right_vectors[:effective_rank].T.copy()
    return LocalSubspace(
        basis=basis,
        singular_values=singular_values,
        requested_rank=requested_rank,
        effective_rank=effective_rank,
        embedding_dimension=dimension,
        local_product_count=product_count,
    )
