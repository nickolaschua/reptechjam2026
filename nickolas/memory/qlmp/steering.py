"""Shared bounded angular query steering for memory baselines."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import SteeringConfig
from .diagnostics import QuerySteeringResult, SteeringDiagnostics
from .geometry import _as_vector, _require_unit_vector


def bound_query_shift(
    q: Any,
    delta: Any,
    *,
    config: SteeringConfig | None = None,
) -> QuerySteeringResult:
    """Apply ``q + beta * delta`` with a strict tangent angular bound.

    The supplied delta is reprojected into ``q``'s tangent space before the
    tangent formula is used.  This makes the angular guarantee robust to
    caller error and records whether a material correction occurred.
    """

    settings = SteeringConfig() if config is None else config
    if not isinstance(settings, SteeringConfig):
        raise ValueError("config must be a SteeringConfig")
    query = _require_unit_vector(q, "q", settings.epsilon)
    original_delta = _as_vector(delta, "delta")
    if original_delta.shape != query.shape:
        raise ValueError(
            "q and delta must have equal dimensions, "
            f"got {query.size} and {original_delta.size}"
        )

    original_norm = float(np.linalg.norm(original_delta))
    parallel = float(np.dot(query, original_delta))
    tangent = original_delta - parallel * query
    tangent -= float(np.dot(query, tangent)) * query
    tangent_norm = float(np.linalg.norm(tangent))
    corrected = abs(parallel) > settings.epsilon
    max_tangent_norm = math.tan(math.radians(settings.max_shift_deg))

    if tangent_norm <= settings.epsilon:
        diagnostics = SteeringDiagnostics(
            requested_beta=settings.beta,
            applied_beta=0.0,
            original_delta_norm=original_norm,
            corrected_tangent_norm=tangent_norm,
            requested_tangent_norm=0.0,
            applied_tangent_norm=0.0,
            unclipped_angle_deg=0.0,
            max_shift_deg=settings.max_shift_deg,
            actual_shift_deg=0.0,
            clipped=False,
            delta_zero=True,
            tangency_corrected=corrected,
        )
        return QuerySteeringResult(q_star=query, diagnostics=diagnostics)

    requested_norm = settings.beta * tangent_norm
    unclipped_angle = math.degrees(math.atan(requested_norm))
    boundary_tolerance = (
        64.0
        * float(np.finfo(np.float64).eps)
        * max(1.0, requested_norm, max_tangent_norm)
    )
    clipped = requested_norm > max_tangent_norm + boundary_tolerance
    applied_norm = max_tangent_norm if clipped else requested_norm
    applied_beta = applied_norm / tangent_norm
    candidate = query + applied_beta * tangent
    q_star = candidate / float(np.linalg.norm(candidate))
    actual_angle = math.degrees(
        math.acos(float(np.clip(np.dot(query, q_star), -1.0, 1.0)))
    )
    diagnostics = SteeringDiagnostics(
        requested_beta=settings.beta,
        applied_beta=applied_beta,
        original_delta_norm=original_norm,
        corrected_tangent_norm=tangent_norm,
        requested_tangent_norm=requested_norm,
        applied_tangent_norm=applied_norm,
        unclipped_angle_deg=unclipped_angle,
        max_shift_deg=settings.max_shift_deg,
        actual_shift_deg=actual_angle,
        clipped=clipped,
        delta_zero=False,
        tangency_corrected=corrected,
    )
    return QuerySteeringResult(q_star=q_star, diagnostics=diagnostics)
