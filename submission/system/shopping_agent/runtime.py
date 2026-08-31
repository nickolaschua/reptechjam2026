"""One process-start factory for the selected chat and embedding providers."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .config import RUNTIME_CONFIG, RuntimeConfig
    from .embedding_backends import BGEEmbeddingBackend, EmbeddingBackend
    from .model_client import ModelClient
    from .ollama_client import OllamaClient
except ImportError:  # direct-module compatibility used by legacy evaluation harnesses
    from config import RUNTIME_CONFIG, RuntimeConfig
    from embedding_backends import BGEEmbeddingBackend, EmbeddingBackend
    from model_client import ModelClient
    from ollama_client import OllamaClient


@dataclass(frozen=True)
class RuntimeProviders:
    config: RuntimeConfig
    llm_client: ModelClient
    embedding_backend: EmbeddingBackend

    @property
    def provider(self) -> str: return self.config.provider


_DEFAULT_PROVIDERS: RuntimeProviders | None = None


def create_runtime_providers(config: RuntimeConfig = RUNTIME_CONFIG) -> RuntimeProviders:
    """Construct the frozen BGE/Ollama providers without making a request."""
    if config.test_mode:
        raise ValueError("this submission is frozen to BGE/Ollama; TEST_MODE must be false")
    llm = OllamaClient(host=config.ollama_host, model=config.ollama_model,
                       timeout_seconds=config.ollama_timeout_seconds)
    embeddings = BGEEmbeddingBackend()
    return RuntimeProviders(config, llm, embeddings)


def get_runtime_providers() -> RuntimeProviders:
    global _DEFAULT_PROVIDERS
    if _DEFAULT_PROVIDERS is None: _DEFAULT_PROVIDERS = create_runtime_providers()
    return _DEFAULT_PROVIDERS


__all__ = ["RuntimeProviders", "create_runtime_providers", "get_runtime_providers"]
