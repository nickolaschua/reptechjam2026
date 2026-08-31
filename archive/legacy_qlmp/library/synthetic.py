"""Locked deterministic geometry and controlled text fixtures for Phase 1."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

from .geometry import normalize


EMBEDDING_DIMENSION = 8


def _frozen(vector: np.ndarray) -> np.ndarray:
    result = np.array(vector, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _axis(index: int) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIMENSION, dtype=np.float64)
    vector[index] = 1.0
    return _frozen(vector)


QUERY_VECTOR = _axis(0)
SUPPORTED_MEMORY_VECTOR = _axis(1)
UNSUPPORTED_MEMORY_VECTOR = _axis(5)
REDUNDANT_MEMORY_VECTOR = QUERY_VECTOR.copy()
REDUNDANT_MEMORY_VECTOR.setflags(write=False)
SUPPORTED_COMBINATION_MEMORY_VECTOR = _frozen(
    0.6 * _axis(1) + 0.8 * _axis(2)
)
MIXED_MEMORY_VECTOR = _frozen(0.6 * _axis(1) + 0.8 * _axis(5))

_LOCAL_COEFFICIENTS = (
    (0.8, 0.2),
    (-0.7, 0.5),
    (0.3, -0.9),
    (-0.4, -0.6),
)
LOCAL_PRODUCT_VECTORS = np.vstack(
    [
        normalize(QUERY_VECTOR + a * _axis(1) + b * _axis(2))
        for a, b in _LOCAL_COEFFICIENTS
    ]
)
LOCAL_PRODUCT_VECTORS.setflags(write=False)


CURRENT_QUERY_TEXT = (
    "I need lightweight waterproof hiking shoes with reliable grip for a rainy trail."
)
USEFUL_MEMORY_TEXT = (
    "The user prefers lightweight outdoor footwear with strong wet-weather traction."
)
REDUNDANT_MEMORY_TEXT = "The user is currently shopping for hiking shoes."
SAME_CATEGORY_NEGATIVE_TEXT = "The user does not want heavy leather hiking boots."
CROSS_DOMAIN_NEGATIVE_TEXT = "The user does not want a countertop blender."

ENTANGLED_LEVEL_0_TEXT = "The user prefers lightweight hiking shoes."
ENTANGLED_LEVEL_1_TEXT = (
    "The user prefers lightweight hiking shoes and usually chooses dark colours."
)
ENTANGLED_LEVEL_2_TEXT = (
    "The user prefers lightweight hiking shoes, usually chooses dark colours, "
    "and avoids noisy kitchen appliances."
)
ENTANGLED_LEVEL_3_TEXT = (
    "The user prefers lightweight hiking shoes, usually chooses dark colours, "
    "avoids noisy kitchen appliances, and plans to replace a cracked phone case."
)
ENTANGLED_TEXT_FIXTURES = (
    ENTANGLED_LEVEL_0_TEXT,
    ENTANGLED_LEVEL_1_TEXT,
    ENTANGLED_LEVEL_2_TEXT,
    ENTANGLED_LEVEL_3_TEXT,
)

CONTROLLED_TEXT_FIXTURES = MappingProxyType(
    {
        "current_query": CURRENT_QUERY_TEXT,
        "useful": USEFUL_MEMORY_TEXT,
        "redundant": REDUNDANT_MEMORY_TEXT,
        "same_category_negative": SAME_CATEGORY_NEGATIVE_TEXT,
        "cross_domain_negative": CROSS_DOMAIN_NEGATIVE_TEXT,
        "entangled_level_0": ENTANGLED_LEVEL_0_TEXT,
        "entangled_level_1": ENTANGLED_LEVEL_1_TEXT,
        "entangled_level_2": ENTANGLED_LEVEL_2_TEXT,
        "entangled_level_3": ENTANGLED_LEVEL_3_TEXT,
    }
)
