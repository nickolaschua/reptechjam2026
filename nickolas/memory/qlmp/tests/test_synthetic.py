from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import unittest

import numpy as np

from nickolas.memory.qlmp import MemoryItem, MemoryPolarity, MemorySource
from nickolas.memory.qlmp.synthetic import (
    CONTROLLED_TEXT_FIXTURES,
    CROSS_DOMAIN_NEGATIVE_TEXT,
    CURRENT_QUERY_TEXT,
    EMBEDDING_DIMENSION,
    ENTANGLED_LEVEL_0_TEXT,
    ENTANGLED_LEVEL_1_TEXT,
    ENTANGLED_LEVEL_2_TEXT,
    ENTANGLED_LEVEL_3_TEXT,
    ENTANGLED_TEXT_FIXTURES,
    LOCAL_PRODUCT_VECTORS,
    MIXED_MEMORY_VECTOR,
    QUERY_VECTOR,
    REDUNDANT_MEMORY_TEXT,
    SAME_CATEGORY_NEGATIVE_TEXT,
    SUPPORTED_COMBINATION_MEMORY_VECTOR,
    SUPPORTED_MEMORY_VECTOR,
    UNSUPPORTED_MEMORY_VECTOR,
    USEFUL_MEMORY_TEXT,
)


class SyntheticFixtureTests(unittest.TestCase):
    def test_locked_geometry_has_expected_shapes_norms_and_coordinates(self) -> None:
        self.assertEqual(EMBEDDING_DIMENSION, 8)
        self.assertEqual(QUERY_VECTOR.shape, (8,))
        self.assertEqual(LOCAL_PRODUCT_VECTORS.shape, (4, 8))
        np.testing.assert_allclose(np.linalg.norm(LOCAL_PRODUCT_VECTORS, axis=1), 1.0)
        np.testing.assert_allclose(SUPPORTED_MEMORY_VECTOR, np.eye(8)[1])
        np.testing.assert_allclose(UNSUPPORTED_MEMORY_VECTOR, np.eye(8)[5])
        np.testing.assert_allclose(
            SUPPORTED_COMBINATION_MEMORY_VECTOR, 0.6 * np.eye(8)[1] + 0.8 * np.eye(8)[2]
        )
        np.testing.assert_allclose(
            MIXED_MEMORY_VECTOR, 0.6 * np.eye(8)[1] + 0.8 * np.eye(8)[5]
        )

    def test_required_controlled_text_fixtures_are_present_exactly(self) -> None:
        expected = {
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
        self.assertEqual(dict(CONTROLLED_TEXT_FIXTURES), expected)
        self.assertEqual(
            ENTANGLED_TEXT_FIXTURES,
            (
                ENTANGLED_LEVEL_0_TEXT,
                ENTANGLED_LEVEL_1_TEXT,
                ENTANGLED_LEVEL_2_TEXT,
                ENTANGLED_LEVEL_3_TEXT,
            ),
        )
        self.assertTrue(all(text.strip() for text in expected.values()))

    def test_memory_item_is_validated_frozen_and_owns_embedding_copy(self) -> None:
        caller_embedding = np.asarray([1.0, 0.0], dtype=np.float32)
        item = MemoryItem(
            id="memory-001",
            text="useful memory",
            embedding=caller_embedding,
            source="user",  # type: ignore[arg-type]
            polarity="positive",  # type: ignore[arg-type]
        )
        caller_embedding[0] = 0.0
        np.testing.assert_array_equal(item.embedding, [1.0, 0.0])
        self.assertEqual(item.embedding.dtype, np.float64)
        self.assertFalse(item.embedding.flags.writeable)
        self.assertFalse(np.shares_memory(item.embedding, caller_embedding))
        self.assertIs(item.source, MemorySource.USER)
        self.assertIs(item.polarity, MemoryPolarity.POSITIVE)
        self.assertIsNone(item.scope)
        self.assertIsNone(item.timestamp)
        self.assertEqual(item.confidence, 1.0)
        with self.assertRaises(FrozenInstanceError):
            item.text = "changed"  # type: ignore[misc]

    def test_memory_item_preserves_optional_metadata_and_explicit_confidence(self) -> None:
        timestamp = datetime(2026, 8, 29, 12, 30, tzinfo=timezone.utc)
        item = MemoryItem(
            id="purchase-17",
            text="Bought waterproof trail shoes.",
            embedding=[0.0, 1.0],
            source=MemorySource.PURCHASE_EPISODE,
            polarity=MemoryPolarity.POSITIVE,
            scope="footwear",
            timestamp=timestamp,
            confidence=0.75,
        )
        self.assertEqual(item.scope, "footwear")
        self.assertIs(item.timestamp, timestamp)
        self.assertEqual(item.confidence, 0.75)

    def test_memory_item_accepts_every_declared_source_and_polarity(self) -> None:
        expected_sources = {
            "user",
            "assistant",
            "system",
            "explicit_preference",
            "purchase_episode",
            "behavioral_inference",
            "click",
            "recommendation_shown",
        }
        self.assertEqual({source.value for source in MemorySource}, expected_sources)
        for index, source in enumerate(MemorySource):
            for polarity in MemoryPolarity:
                with self.subTest(source=source, polarity=polarity):
                    item = MemoryItem(
                        id=f"memory-{index}-{polarity.value}",
                        text="memory",
                        embedding=[1.0, 0.0],
                        source=source,
                        polarity=polarity,
                    )
                    self.assertIs(item.source, source)
                    self.assertIs(item.polarity, polarity)

    def test_memory_item_requires_id(self) -> None:
        with self.assertRaises(TypeError):
            MemoryItem(  # type: ignore[call-arg]
                text="memory",
                embedding=[1.0, 0.0],
                source=MemorySource.USER,
                polarity=MemoryPolarity.NEUTRAL,
            )

    def test_memory_item_rejects_invalid_core_and_optional_fields(self) -> None:
        kwargs = {
            "id": "memory-001",
            "text": "memory",
            "embedding": [1.0, 0.0],
            "source": MemorySource.USER,
            "polarity": MemoryPolarity.NEUTRAL,
        }
        invalid_overrides = (
            {"id": " "},
            {"id": 7},
            {"text": " "},
            {"embedding": []},
            {"embedding": [[1.0]]},
            {"embedding": [np.inf]},
            {"source": "catalogue"},
            {"polarity": "mixed"},
            {"scope": " "},
            {"scope": 7},
            {"timestamp": "2026-08-29"},
            {"confidence": True},
            {"confidence": np.bool_(False)},
            {"confidence": np.nan},
            {"confidence": np.inf},
            {"confidence": -0.01},
            {"confidence": 1.01},
        )
        for override in invalid_overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                MemoryItem(**(kwargs | override))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
