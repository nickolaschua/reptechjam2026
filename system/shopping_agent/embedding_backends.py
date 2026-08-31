"""Minimal, swappable embedding backends for the Phase 3 bake-off.

This module is deliberately side-effect free: importing it never constructs a
hosted client, downloads a model, or embeds catalog data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Sequence

import numpy as np

try:
    from .config import (
        OPENAI_EMBEDDING_DIMENSIONS,
        OPENAI_EMBEDDING_MODEL,
        OPENAI_SMALL_EMBEDDING_DIMENSIONS,
        OPENAI_SMALL_EMBEDDING_MODEL,
    )
except ImportError:
    from config import (
        OPENAI_EMBEDDING_DIMENSIONS,
        OPENAI_EMBEDDING_MODEL,
        OPENAI_SMALL_EMBEDDING_DIMENSIONS,
        OPENAI_SMALL_EMBEDDING_MODEL,
    )


BGE_MODEL = "BAAI/bge-base-en-v1.5"
OPENAI_MODEL = OPENAI_EMBEDDING_MODEL
OPENAI_SMALL_MODEL = OPENAI_SMALL_EMBEDDING_MODEL
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
    if backend_id == "bge-base-en-v1.5":
        query_convention = "bge-search-prefix-v1"
    elif backend_id == "openai-text-embedding-3-large":
        query_convention = "symmetric-v1"
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
OPENAI_EMBEDDING_SPACE_ID = make_embedding_space_id(
    "openai-text-embedding-3-large", OPENAI_MODEL, OPENAI_EMBEDDING_DIMENSIONS
)
OPENAI_SMALL_EMBEDDING_SPACE_ID = make_embedding_space_id(
    "openai-text-embedding-3-small",
    OPENAI_SMALL_MODEL,
    OPENAI_SMALL_EMBEDDING_DIMENSIONS,
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
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", backend_id)
    return f"catalog_cache_{safe}.npz"


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


class OpenAIEmbeddingBackend(EmbeddingBackend):
    backend_id = "openai-text-embedding-3-large"
    model_id = OPENAI_MODEL
    vector_dimension = OPENAI_EMBEDDING_DIMENSIONS
    embedding_space_id = OPENAI_EMBEDDING_SPACE_ID

    def __init__(
        self,
        client: Any = None,
        api_key: str | None = None,
        batch_size: int = 1000,
        *,
        model_id: str = OPENAI_MODEL,
        backend_id: str | None = None,
        vector_dimension: int | None = 3072,
    ) -> None:
        self.model_id = str(model_id)
        self.backend_id = str(backend_id or f"openai-{self.model_id}")
        self.vector_dimension = (
            None if vector_dimension is None else int(vector_dimension)
        )
        if self.vector_dimension is not None and self.vector_dimension <= 0:
            raise ValueError("vector_dimension must be positive when supplied")
        self.embedding_space_id = (
            ""
            if self.vector_dimension is None
            else make_embedding_space_id(
                self.backend_id, self.model_id, self.vector_dimension
            )
        )
        self._client = client
        self._api_key = api_key
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._request_count = 0
        self._input_tokens = 0
        self._request_latencies: list[float] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise EmbeddingError("OPENAI_API_KEY is required for OpenAI embeddings")
            from openai import OpenAI

            self._client = OpenAI(api_key=key)
        return self._client

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        started = time.perf_counter()
        try:
            response = self.client.embeddings.create(model=self.model_id, input=list(texts))
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding request failed: {exc}") from exc
        finally:
            self._request_latencies.append(time.perf_counter() - started)
            self._request_count += 1

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "total_tokens", 0)
        self._input_tokens += int(input_tokens or 0)

        data = list(response.data)
        if data and all(getattr(item, "index", None) is not None for item in data):
            data.sort(key=lambda item: int(item.index))
        if len(data) != len(texts):
            raise EmbeddingError(
                f"OpenAI returned {len(data)} embeddings for {len(texts)} inputs"
            )
        matrix = np.asarray([item.embedding for item in data], dtype=np.float32)
        actual_dimension = int(matrix.shape[1])
        if self.vector_dimension is None:
            # Used by explicit cache-freeze commands: derive the space from the
            # provider response instead of assuming a model dimension.
            self.vector_dimension = actual_dimension
            self.embedding_space_id = make_embedding_space_id(
                self.backend_id, self.model_id, actual_dimension
            )
        elif actual_dimension != self.vector_dimension:
            raise EmbeddingError(
                f"OpenAI returned dimension {actual_dimension}, expected {self.vector_dimension}"
            )
        return matrix

    def embed_catalog(self, texts: Sequence[str]) -> np.ndarray:
        ordered = list(texts)
        batches = [
            self._embed_batch(ordered[start : start + self.batch_size])
            for start in range(0, len(ordered), self.batch_size)
        ]
        if not batches:
            raise EmbeddingError("Cannot embed an empty catalog")
        return normalize_rows(np.vstack(batches))

    def embed_query(self, text: str) -> np.ndarray:
        # No BGE retrieval instruction is added here.
        return normalize_vector(self._embed_batch([str(text)])[0])

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "request_count": self._request_count,
            "input_tokens": self._input_tokens,
            "request_latencies_seconds": list(self._request_latencies),
        }


def embedding_backend_for_mode(test_mode: bool) -> EmbeddingBackend:
    """Select the canonical embedder for the main workflow mode."""
    if bool(test_mode):
        return OpenAIEmbeddingBackend(
            model_id=OPENAI_SMALL_MODEL,
            backend_id="openai-text-embedding-3-small",
            vector_dimension=OPENAI_SMALL_EMBEDDING_DIMENSIONS,
        )
    return BGEEmbeddingBackend()
