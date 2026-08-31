"""Thin OpenAI-embedding configuration of the canonical Patch-2 Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .agent import Agent as CanonicalAgent
    from .embedding_backends import OpenAIEmbeddingBackend
except ImportError:
    from agent import Agent as CanonicalAgent
    from embedding_backends import OpenAIEmbeddingBackend


class Agent(CanonicalAgent):
    def __init__(
        self,
        catalog_path: str | Path | None = None,
        embedding_backend: Any = None,
        *,
        allow_catalog_embedding: bool = False,
        **kwargs: Any,
    ) -> None:
        backend = embedding_backend or OpenAIEmbeddingBackend()
        super().__init__(
            catalog_path,
            embedding_backend=backend,
            allow_catalog_embedding=allow_catalog_embedding,
            **kwargs,
        )
