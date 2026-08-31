"""Production BGE embeddings plus deterministic backend injection contracts.

This module is deliberately side-effect free: importing it never constructs a
model, downloads weights, or embeds catalog data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import numpy as np

BGE_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PRODUCT_TEXT_VERSION = "patch2-product-text-v1"


def make_embedding_space_id(
    backend_id: str,
    model_id: str,
    vector_dimension: int,
    *,
    normalized: bool = True,
) -> str:
    """Return the stable identity of mutually compatible catalog/query vectors."""
    if backend_id.startswith("bge-"):
        query_convention = "bge-search-prefix-v1"
    else:
        query_convention = "backend-defined-v1"
    normalization = "l2" if normalized else "none"
    return (
        f"{backend_id}:{model_id}:dimensions={int(vector_dimension)}:"
        f"normalization={normalization}:query={query_convention}"
    )


BGE_EMBEDDING_SPACE_ID = make_embedding_space_id(
    "bge-base-en-v1.5", BGE_MODEL, 768
)
class EmbeddingError(RuntimeError):
    """Base error for embedding failures."""


class CatalogCacheMissError(EmbeddingError):
    """Raised when explicit catalog generation is required but not allowed."""


class CacheValidationError(EmbeddingError):
    """Raised when an embedding cache belongs to a different embedding space."""


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise EmbeddingError(f"Expected a non-empty 2D embedding matrix, got {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise EmbeddingError("Expected a non-empty embedding vector")
    return array / max(float(np.linalg.norm(array)), 1e-12)


def fingerprint_texts(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def fingerprint_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_filename(backend_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", backend_id)
    return f"catalog_cache_{safe}.npz"


def production_product_text(product: dict[str, Any]) -> str:
    """Build the exact product passage embedded by the production runtime."""

    title = product.get("title") or ""
    categories = ", ".join(product.get("categories") or [])
    features = "; ".join((product.get("features") or [])[:3])
    return (
        f"Product: {title}. Categories: {categories}. Features: {features}."
    ).strip()


def production_product_texts(products: Sequence[dict[str, Any]]) -> list[str]:
    """Build production passages in caller-supplied catalogue row order."""

    return [production_product_text(product) for product in products]


@dataclass(frozen=True)
class CacheExpectation:
    backend_id: str
    model_id: str
    embedding_space_id: str
    catalog_ids: Sequence[str]
    product_text_fingerprint: str
    catalog_fingerprint: str
    vector_dimension: int | None = None
    product_text_version: str = PRODUCT_TEXT_VERSION
    normalized: bool = True


def save_embedding_cache(
    path: str | Path,
    embeddings: np.ndarray,
    expectation: CacheExpectation,
) -> None:
    matrix = np.asarray(embeddings, dtype=np.float32)
    ids = [str(item) for item in expectation.catalog_ids]
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise CacheValidationError("Embedding rows must match catalog ID rows")
    if expectation.vector_dimension is not None and matrix.shape[1] != expectation.vector_dimension:
        raise CacheValidationError(
            f"Expected vector dimension {expectation.vector_dimension}, got {matrix.shape[1]}"
        )
    metadata = {
        "schema_version": 2,
        "backend_id": expectation.backend_id,
        "model_id": expectation.model_id,
        "embedding_space_id": expectation.embedding_space_id,
        "row_count": len(ids),
        "vector_dimension": int(matrix.shape[1]),
        "normalized": bool(expectation.normalized),
        "product_text_version": expectation.product_text_version,
        "product_text_fingerprint": expectation.product_text_fingerprint,
        "catalog_fingerprint": expectation.catalog_fingerprint,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=matrix,
        ids=np.asarray(ids, dtype=np.str_),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    )
    temporary.replace(destination)


def load_embedding_cache(path: str | Path, expectation: CacheExpectation) -> np.ndarray:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as data:
            if not {"embeddings", "ids", "metadata_json"}.issubset(data.files):
                raise CacheValidationError("Cache is missing required arrays or metadata")
            matrix = np.asarray(data["embeddings"], dtype=np.float32)
            ids = [str(item) for item in data["ids"].tolist()]
            metadata = json.loads(str(data["metadata_json"].item()))
    except CacheValidationError:
        raise
    except Exception as exc:
        raise CacheValidationError(f"Could not read cache {source}: {exc}") from exc

    expected_ids = [str(item) for item in expectation.catalog_ids]
    cached_space_id = metadata.get("embedding_space_id")
    if cached_space_id is None and metadata.get("vector_dimension") is not None:
        # Phase-3 caches predate the explicit field. Derive the same stable ID
        # from metadata so a valid cache is reused rather than regenerated.
        cached_space_id = make_embedding_space_id(
            str(metadata.get("backend_id", "")),
            str(metadata.get("model_id", "")),
            int(metadata["vector_dimension"]),
            normalized=bool(metadata.get("normalized", False)),
        )

    expected_metadata = {
        "backend_id": expectation.backend_id,
        "model_id": expectation.model_id,
        "embedding_space_id": expectation.embedding_space_id,
        "row_count": len(expected_ids),
        "normalized": bool(expectation.normalized),
        "product_text_version": expectation.product_text_version,
        "product_text_fingerprint": expectation.product_text_fingerprint,
        "catalog_fingerprint": expectation.catalog_fingerprint,
    }
    actual_metadata = {**metadata, "embedding_space_id": cached_space_id}
    problems = [
        key for key, value in expected_metadata.items()
        if actual_metadata.get(key) != value
    ]
    if metadata.get("schema_version") not in {1, 2}:
        problems.append("schema_version")
    if matrix.ndim != 2:
        problems.append("embedding_rank")
    else:
        if matrix.shape[0] != len(expected_ids):
            problems.append("embedding_row_count")
        if metadata.get("vector_dimension") != int(matrix.shape[1]):
            problems.append("vector_dimension")
        if (
            expectation.vector_dimension is not None
            and int(matrix.shape[1]) != expectation.vector_dimension
        ):
            problems.append("backend_vector_dimension")
    if ids != expected_ids:
        problems.append("catalog_ids_exact_row_order")
    if matrix.ndim == 2:
        if not np.all(np.isfinite(matrix)):
            problems.append("embedding_finiteness")
        elif not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-4, atol=1e-5):
            problems.append("embedding_normalization")
    if problems:
        raise CacheValidationError(
            f"Cache {source.name} failed validation: {', '.join(sorted(set(problems)))}"
        )
    return matrix


class EmbeddingBackend:
    backend_id: str
    model_id: str
    embedding_space_id: str

    def embed_catalog(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def usage_snapshot(self) -> dict[str, Any]:
        return {"request_count": 0, "input_tokens": 0, "request_latencies_seconds": []}


class BGEEmbeddingBackend(EmbeddingBackend):
    backend_id = "bge-base-en-v1.5"
    model_id = BGE_MODEL
    vector_dimension = 768
    embedding_space_id = BGE_EMBEDDING_SPACE_ID

    def __init__(
        self,
        model: Any = None,
        model_factory: Callable[[str], Any] | None = None,
        batch_size: int = 256,
    ) -> None:
        self._model = model
        self._model_factory = model_factory
        self.batch_size = int(batch_size)

    @property
    def model(self) -> Any:
        if self._model is None:
            if self._model_factory is None:
                from sentence_transformers import SentenceTransformer

                self._model_factory = SentenceTransformer
            self._model = self._model_factory(self.model_id)
        return self._model

    def embed_catalog(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return normalize_rows(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        # The prefix stays backend-internal so both variants receive identical
        # canonical retrieval query text from Agent.
        vector = self.model.encode(
            BGE_QUERY_PREFIX + str(text),
            convert_to_numpy=True,
        )
        return normalize_vector(vector)


__all__ = [
    "BGEEmbeddingBackend",
    "BGE_EMBEDDING_SPACE_ID",
    "BGE_MODEL",
    "BGE_QUERY_PREFIX",
    "CacheExpectation",
    "CacheValidationError",
    "CatalogCacheMissError",
    "EmbeddingBackend",
    "EmbeddingError",
    "PRODUCT_TEXT_VERSION",
    "cache_filename",
    "fingerprint_file",
    "fingerprint_texts",
    "load_embedding_cache",
    "make_embedding_space_id",
    "normalize_rows",
    "normalize_vector",
    "production_product_text",
    "production_product_texts",
    "save_embedding_cache",
]
