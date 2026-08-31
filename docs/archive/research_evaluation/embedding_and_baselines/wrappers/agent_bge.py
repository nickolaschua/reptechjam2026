"""Thin vanilla-BGE configuration of the canonical Patch-2 Agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from .agent import Agent as CanonicalAgent, SentenceTransformer
    from .embedding_backends import BGEEmbeddingBackend
except ImportError:
    from agent import Agent as CanonicalAgent, SentenceTransformer
    from embedding_backends import BGEEmbeddingBackend


class Agent(CanonicalAgent):
    def __init__(
        self,
        catalog_path: str | Path | None = None,
        embedding_backend: Any = None,
        **kwargs: Any,
    ) -> None:
        batch_size = int(os.environ.get("BGE_EMBEDDING_BATCH_SIZE", "256"))
        backend = embedding_backend or BGEEmbeddingBackend(
            model_factory=SentenceTransformer,
            batch_size=batch_size,
        )
        super().__init__(catalog_path, embedding_backend=backend, **kwargs)
