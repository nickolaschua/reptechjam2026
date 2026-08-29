"""Fast/Slow longitudinal memory for conversational product search."""

from .config import MINILM_MODEL_ID, MemoryConfig, memory_config_for_mode
from .embeddings import (
    CatalogEmbeddingIndex,
    DeterministicLexicalEmbedder,
    EmbeddingProvider,
    EmbeddingService,
    MiniLMEmbeddingProvider,
)
from .fast_memory import (
    FastMemoryUpdate,
    SemanticParser,
    classify_constraint,
    override_intent,
    update_state,
)
from .integration import MemorySystem
from .metrics import memory_harm_rate, rank_uplift, reciprocal_rank, reciprocal_rank_uplift
from .slow_memory import aggregate_slow_vector, distill_summary, rerank_with_slow_memory
from .store import ActiveSession, InMemoryMemoryStore
from .types import (
    ConstraintKind,
    FastMemoryState,
    MemoryDebugTrace,
    SlowMemoryEpisode,
    TypedConstraint,
)

__all__ = [
    "ActiveSession",
    "CatalogEmbeddingIndex",
    "ConstraintKind",
    "DeterministicLexicalEmbedder",
    "EmbeddingProvider",
    "EmbeddingService",
    "FastMemoryState",
    "FastMemoryUpdate",
    "InMemoryMemoryStore",
    "MINILM_MODEL_ID",
    "MemoryConfig",
    "MemoryDebugTrace",
    "MemorySystem",
    "MiniLMEmbeddingProvider",
    "SemanticParser",
    "SlowMemoryEpisode",
    "TypedConstraint",
    "aggregate_slow_vector",
    "classify_constraint",
    "distill_summary",
    "memory_config_for_mode",
    "memory_harm_rate",
    "override_intent",
    "rank_uplift",
    "reciprocal_rank",
    "reciprocal_rank_uplift",
    "rerank_with_slow_memory",
    "update_state",
]
