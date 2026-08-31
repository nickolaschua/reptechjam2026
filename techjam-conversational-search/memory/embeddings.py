"""Episode embeddings and strictly validated MiniLM product-cache reuse."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .config import MINILM_MODEL_ID, MemoryConfig


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
MINILM_DIMENSION = 384
MINILM_MAX_SEQUENCE_LENGTH = 128
MINILM_SPACE_ID = f"{MINILM_MODEL_ID}:seq{MINILM_MAX_SEQUENCE_LENGTH}:normalized"
EXPECTED_CATALOG_ROWS = 50_000
EXPECTED_CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
EXPECTED_CACHE_SHA256 = "ef51a7c67a9cceaa39d6cd1930c2093563b7e689a97961ac10876cfc54a19b31"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a vector or matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.float32(1e-12))


@runtime_checkable
class EmbeddingProvider(Protocol):
    dimension: int
    space_id: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding row for every input text."""


class DeterministicLexicalEmbedder:
    """Network-free signed feature hashing for Slow Memory episodes."""

    def __init__(self, dimension: int = MINILM_DIMENSION) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.space_id = f"lexical-sha256-v1-{self.dimension}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = [token.lower() for token in TOKEN_RE.findall(str(text))]
            features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "little") % self.dimension
                output[row, index] += 1.0 if digest[8] & 1 else -1.0
        return normalize_rows(output)


class MiniLMEmbeddingProvider:
    """Lazy, local-only wrapper around the fixed MiniLM experiment model."""

    dimension = MINILM_DIMENSION

    def __init__(self, config: MemoryConfig, model_path: str | Path | None = None) -> None:
        self.config = config
        self.model_path = str(model_path or config.embedding_model_id)
        self.space_id = (
            f"{config.embedding_model_id}:seq{MINILM_MAX_SEQUENCE_LENGTH}:normalized"
        )
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                model = SentenceTransformer(self.model_path, local_files_only=True)
            except TypeError:  # Older sentence-transformers releases.
                model = SentenceTransformer(self.model_path)
            model.max_seq_length = MINILM_MAX_SEQUENCE_LENGTH
            self._model = model
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        values = self._load().encode(
            list(texts),
            batch_size=min(64, max(1, len(texts))),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = normalize_rows(np.asarray(values, dtype=np.float32))
        if matrix.shape != (len(texts), self.dimension):
            raise ValueError(
                f"MiniLM emitted shape {matrix.shape}, expected {(len(texts), self.dimension)}"
            )
        return matrix


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_cache_path(catalog_path: Path) -> Path | None:
    name = (
        f"dense_{EXPECTED_CATALOG_SHA256[:16]}_minilm_"
        f"seq{MINILM_MAX_SEQUENCE_LENGTH}.npy"
    )
    for ancestor in catalog_path.resolve().parents:
        candidate = ancestor / "docs" / "archive" / "research_evaluation" / "cache" / name
        if candidate.is_file():
            return candidate
    return None


def discover_local_model(catalog_path: Path, config: MemoryConfig) -> Path | None:
    if config.embedding_model_id != MINILM_MODEL_ID:
        return None
    relative = Path(
        "archive/research_evaluation/cache/models/"
        "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
    )
    for ancestor in catalog_path.resolve().parents:
        snapshots = ancestor / relative
        if snapshots.is_dir():
            candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
            if candidates:
                return candidates[-1]
    return None


class CatalogEmbeddingIndex:
    """Cosine vectors for the frozen product catalog."""

    def __init__(self, ids: Sequence[str], matrix: np.ndarray, *, space_id: str) -> None:
        values = np.asarray(matrix)
        if values.ndim != 2 or values.shape[0] != len(ids):
            raise ValueError("catalog embedding rows must match catalog identifiers")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("catalog embeddings must be floating point")
        if not np.isfinite(values).all():
            raise ValueError("catalog embeddings must be finite")
        norms = np.linalg.norm(values, axis=1)
        if not np.all(np.isclose(norms, 1.0, atol=2e-3) | np.isclose(norms, 0.0, atol=1e-9)):
            raise ValueError("catalog embeddings must be normalized")
        self.ids = tuple(str(value) for value in ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("catalog identifiers must be unique")
        self.id_to_index = {identifier: index for index, identifier in enumerate(self.ids)}
        self.matrix = values
        self.dimension = int(values.shape[1])
        self.space_id = str(space_id)

    @classmethod
    def load_validated(
        cls,
        catalog_path: str | Path,
        ids: Sequence[str],
        config: MemoryConfig | None = None,
    ) -> "CatalogEmbeddingIndex | None":
        catalog = Path(catalog_path)
        if config is not None and config.embedding_model_id != MINILM_MODEL_ID:
            return None
        cache = discover_cache_path(catalog)
        if cache is None:
            return None
        try:
            if len(ids) != EXPECTED_CATALOG_ROWS:
                return None
            if sha256_file(catalog) != EXPECTED_CATALOG_SHA256:
                return None
            if sha256_file(cache) != EXPECTED_CACHE_SHA256:
                return None
            values = np.load(cache, mmap_mode="r")
            if values.shape != (EXPECTED_CATALOG_ROWS, MINILM_DIMENSION):
                return None
            if values.dtype != np.float32:
                return None
            for start in range(0, len(values), 4096):
                block = np.asarray(values[start : start + 4096])
                if not np.isfinite(block).all():
                    return None
                if not np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=2e-3):
                    return None
            return cls(ids, values, space_id=MINILM_SPACE_ID)
        except (OSError, ValueError):
            return None

    def vectors_for(self, candidate_ids: Sequence[str]) -> np.ndarray | None:
        indices: list[int] = []
        for identifier in candidate_ids:
            index = self.id_to_index.get(str(identifier))
            if index is None:
                return None
            indices.append(index)
        return np.asarray(self.matrix[indices], dtype=np.float32)


class EmbeddingService:
    """Embeds completed episodes and exposes compatible cached products."""

    def __init__(
        self,
        config: MemoryConfig,
        catalog_path: str | Path,
        ids: Sequence[str],
        provider: EmbeddingProvider | None = None,
        catalog_embeddings: np.ndarray | None = None,
    ) -> None:
        self.config = config
        fallback_dimension = int(provider.dimension) if provider is not None else MINILM_DIMENSION
        self.fallback = DeterministicLexicalEmbedder(fallback_dimension)
        if provider is not None:
            self.provider: EmbeddingProvider = provider
        else:
            local_model = discover_local_model(Path(catalog_path), config)
            self.provider = MiniLMEmbeddingProvider(config, local_model) if local_model else self.fallback
        if catalog_embeddings is not None:
            self.catalog = CatalogEmbeddingIndex(
                ids, catalog_embeddings, space_id=self.provider.space_id
            )
        else:
            self.catalog = CatalogEmbeddingIndex.load_validated(catalog_path, ids, config)

    @property
    def space_id(self) -> str:
        return str(self.provider.space_id)

    def embed_once(self, text: str) -> np.ndarray:
        """Embed one completed-session summary, with deterministic fallback."""

        try:
            matrix = normalize_rows(self.provider.embed([text]))
            expected = (1, int(self.provider.dimension))
            if matrix.shape != expected or not np.isfinite(matrix).all():
                raise ValueError(f"embedding provider emitted {matrix.shape}, expected {expected}")
            return matrix[0]
        except Exception:
            self.provider = self.fallback
            return self.fallback.embed([text])[0]


__all__ = [
    "CatalogEmbeddingIndex",
    "DeterministicLexicalEmbedder",
    "EmbeddingProvider",
    "EmbeddingService",
    "MiniLMEmbeddingProvider",
    "MINILM_DIMENSION",
    "MINILM_MAX_SEQUENCE_LENGTH",
    "MINILM_SPACE_ID",
    "normalize_rows",
    "sha256_file",
]
