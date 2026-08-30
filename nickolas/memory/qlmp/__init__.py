"""NumPy-only geometry and bounded memory steering for QLMP."""

from .aggregation import (
    aggregate_projected_residuals,
    aggregate_raw_residuals,
    aggregate_residuals,
    memory_tangent_residual,
    normalize_nonnegative_weights,
)
from .baselines import (
    build_cosine_memory_baseline,
    build_naive_memory_baseline,
    is_scope_compatible,
)
from .config import BaselineConfig, ProjectionConfig, SteeringConfig
from .diagnostics import (
    BaselineMode,
    MemoryBaselineResult,
    MemorySelectionDiagnostic,
    QuerySteeringResult,
    SteeringDiagnostics,
)
from .geometry import (
    build_local_subspace,
    build_tangent_matrix,
    cosine_similarity,
    normalize,
    tangent_residual,
)
from .models import (
    LocalSubspace,
    MemoryItem,
    MemoryPolarity,
    MemoryProjection,
    MemorySource,
)
from .projection import project_memory_residual
from .steering import bound_query_shift


__all__ = [
    "BaselineConfig",
    "BaselineMode",
    "LocalSubspace",
    "MemoryBaselineResult",
    "MemoryItem",
    "MemoryPolarity",
    "MemoryProjection",
    "MemorySelectionDiagnostic",
    "MemorySource",
    "ProjectionConfig",
    "QuerySteeringResult",
    "SteeringConfig",
    "SteeringDiagnostics",
    "aggregate_projected_residuals",
    "aggregate_raw_residuals",
    "aggregate_residuals",
    "bound_query_shift",
    "build_cosine_memory_baseline",
    "build_local_subspace",
    "build_naive_memory_baseline",
    "build_tangent_matrix",
    "cosine_similarity",
    "is_scope_compatible",
    "memory_tangent_residual",
    "normalize",
    "normalize_nonnegative_weights",
    "project_memory_residual",
    "tangent_residual",
]
