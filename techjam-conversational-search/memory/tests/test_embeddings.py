from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from memory import CatalogEmbeddingIndex, DeterministicLexicalEmbedder, MemoryConfig
from memory.embeddings import sha256_file
from memory.tests.helpers import KeywordEmbeddingProvider, PRODUCTS, TemporaryCatalog, catalog_embeddings


class EmbeddingTests(unittest.TestCase):
    def test_lexical_fallback_is_normalized_and_deterministic(self) -> None:
        left = DeterministicLexicalEmbedder(32).embed(["red hiking shoes", ""])
        right = DeterministicLexicalEmbedder(32).embed(["red hiking shoes", ""])
        np.testing.assert_array_equal(left, right)
        self.assertAlmostEqual(float(np.linalg.norm(left[0])), 1.0, places=6)
        self.assertEqual(float(np.linalg.norm(left[1])), 0.0)

    def test_catalog_index_rejects_non_normalized_or_misaligned_values(self) -> None:
        with self.assertRaises(ValueError):
            CatalogEmbeddingIndex(["one"], np.ones((2, 2), dtype=np.float32), space_id="s")
        with self.assertRaises(ValueError):
            CatalogEmbeddingIndex(["one"], np.asarray([[2.0, 0.0]], dtype=np.float32), space_id="s")

    def test_cache_validation_checks_hash_rows_dimension_dtype_and_norms(self) -> None:
        catalog = TemporaryCatalog()
        directory = tempfile.TemporaryDirectory()
        try:
            cache = Path(directory.name) / "dense.npy"
            matrix = catalog_embeddings(KeywordEmbeddingProvider()).astype(np.float32)
            np.save(cache, matrix)
            ids = [str(product["parent_asin"]) for product in PRODUCTS]
            patches = (
                mock.patch("memory.embeddings.discover_cache_path", return_value=cache),
                mock.patch("memory.embeddings.EXPECTED_CATALOG_ROWS", len(ids)),
                mock.patch("memory.embeddings.MINILM_DIMENSION", matrix.shape[1]),
                mock.patch("memory.embeddings.EXPECTED_CATALOG_SHA256", sha256_file(catalog.path)),
                mock.patch("memory.embeddings.EXPECTED_CACHE_SHA256", sha256_file(cache)),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = CatalogEmbeddingIndex.load_validated(
                    catalog.path, ids, MemoryConfig()
                )
            self.assertIsNotNone(result)
            self.assertEqual(result.space_id, "sentence-transformers/all-MiniLM-L6-v2:seq128:normalized")

            with mock.patch("memory.embeddings.discover_cache_path", return_value=cache), mock.patch(
                "memory.embeddings.EXPECTED_CATALOG_ROWS", len(ids)
            ), mock.patch(
                "memory.embeddings.EXPECTED_CATALOG_SHA256", "0" * 64
            ):
                self.assertIsNone(CatalogEmbeddingIndex.load_validated(catalog.path, ids))
            mmap = getattr(result.matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()
            del result
        finally:
            directory.cleanup()
            catalog.close()


if __name__ == "__main__":
    unittest.main()
