"""One process-start factory for the selected chat and embedding providers."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .config import RUNTIME_CONFIG, RuntimeConfig
    from .embedding_backends import BGEEmbeddingBackend, EmbeddingBackend, OpenAIEmbeddingBackend
    from .model_client import ModelClient
    from .ollama_client import OllamaClient
except ImportError:  # direct-module compatibility used by legacy evaluation harnesses
    from config import RUNTIME_CONFIG, RuntimeConfig
    from embedding_backends import BGEEmbeddingBackend, EmbeddingBackend, OpenAIEmbeddingBackend
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
    """Construct clients without making a model-provider network request."""
    if config.test_mode:
        assert config.openai_api_key is not None
        try: from .openai_client import OpenAIClient
        except ImportError: from openai_client import OpenAIClient
        llm = OpenAIClient(api_key=config.openai_api_key, model=config.openai_chat_model,
                           timeout_seconds=config.openai_timeout_seconds)
        embeddings = OpenAIEmbeddingBackend(api_key=config.openai_api_key,
            model=config.openai_embedding_model,
            vector_dimension=config.openai_embedding_dimensions,
            timeout_seconds=config.openai_timeout_seconds)
    else:
        llm = OllamaClient(host=config.ollama_host, model=config.ollama_model,
                           timeout_seconds=config.ollama_timeout_seconds)
        embeddings = BGEEmbeddingBackend()
    return RuntimeProviders(config, llm, embeddings)


def get_runtime_providers() -> RuntimeProviders:
    global _DEFAULT_PROVIDERS
    if _DEFAULT_PROVIDERS is None: _DEFAULT_PROVIDERS = create_runtime_providers()
    return _DEFAULT_PROVIDERS


__all__ = ["RuntimeProviders", "create_runtime_providers", "get_runtime_providers"]
