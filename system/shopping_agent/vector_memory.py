"""Gated vector-memory scoring and deterministic evidence serialization."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping
import numpy as np

try:
    from .config import (
        BROWSING_CURRENT_WEIGHT,
        BROWSING_MEMORY_WEIGHT,
        BUYING_CURRENT_WEIGHT,
        BUYING_MEMORY_WEIGHT,
        EWMA_ALPHA,
        RELEVANCE_THRESHOLD,
    )
except ImportError:
    from config import (
        BROWSING_CURRENT_WEIGHT,
        BROWSING_MEMORY_WEIGHT,
        BUYING_CURRENT_WEIGHT,
        BUYING_MEMORY_WEIGHT,
        EWMA_ALPHA,
        RELEVANCE_THRESHOLD,
    )

class BuyerMode(str, Enum):
    BUYING = "buying"
    BROWSING = "browsing"

@dataclass(frozen=True)
class VectorMemoryConfig:
    relevance_threshold: float = RELEVANCE_THRESHOLD
    buying_current_weight: float = BUYING_CURRENT_WEIGHT
    buying_memory_weight: float = BUYING_MEMORY_WEIGHT
    browsing_current_weight: float = BROWSING_CURRENT_WEIGHT
    browsing_memory_weight: float = BROWSING_MEMORY_WEIGHT
    ewma_alpha: float = EWMA_ALPHA

DEFAULT_VECTOR_MEMORY_CONFIG = VectorMemoryConfig()
_POSITIVE_MARKERS = {"true", "yes", "affirmative", "required", "included"}
_NEGATIVE_MARKERS = {"false", "no", "none", "n/a", "null", "other", ""}

def positive_slot_text(disclosed_slots: Mapping[object, object]) -> str:
    fragments: list[str] = []
    for raw_attr in sorted(disclosed_slots, key=lambda value: str(value).casefold()):
        attr = " ".join(str(raw_attr).strip().lower().split())
        raw_values = disclosed_slots[raw_attr]
        values = raw_values if isinstance(raw_values, (set, list, tuple)) else [raw_values]
        for raw_value in sorted(values, key=lambda value: str(value).casefold()):
            value = " ".join(str(raw_value).strip().lower().split())
            if value in _POSITIVE_MARKERS: value = attr
            elif value in _NEGATIVE_MARKERS: continue
            if attr and value: fragments.append(f"{attr}: {value}")
    return "; ".join(fragments)

def score_catalog(catalog_embeddings: np.ndarray, v1: np.ndarray, v2: np.ndarray | None,
                  buyer_mode: BuyerMode | None, config: VectorMemoryConfig = DEFAULT_VECTOR_MEMORY_CONFIG
                  ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float | None, bool, float, float]:
    matrix, current = np.asarray(catalog_embeddings, dtype=np.float32), np.asarray(v1, dtype=np.float32)
    s1 = matrix @ current
    if v2 is None: return s1, None, s1.copy(), None, False, 1.0, 0.0
    memory = np.asarray(v2, dtype=np.float32)
    if memory.shape != current.shape: raise ValueError("long-term vector dimension does not match current query")
    gate, s2 = float(current @ memory), matrix @ memory
    passed = gate >= config.relevance_threshold
    if not passed: return s1, s2, s1.copy(), gate, False, 1.0, 0.0
    if buyer_mode is BuyerMode.BUYING: a, b = config.buying_current_weight, config.buying_memory_weight
    elif buyer_mode is BuyerMode.BROWSING: a, b = config.browsing_current_weight, config.browsing_memory_weight
    else: raise ValueError("buyer_mode is required when long-term memory exists")
    return s1, s2, a*s1+b*s2, gate, True, a, b

__all__ = ["BuyerMode", "DEFAULT_VECTOR_MEMORY_CONFIG", "VectorMemoryConfig", "positive_slot_text", "score_catalog"]
