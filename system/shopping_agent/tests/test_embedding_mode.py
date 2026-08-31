import pytest

from system.shopping_agent.config import OPENAI_SMALL_EMBEDDING_DIMENSIONS, _env_bool
from system.shopping_agent.demo import build_parser
from system.shopping_agent.embedding_backends import (
    BGEEmbeddingBackend,
    OpenAIEmbeddingBackend,
    embedding_backend_for_mode,
)


def test_production_mode_selects_bge_without_loading_the_model():
    backend = embedding_backend_for_mode(False)

    assert isinstance(backend, BGEEmbeddingBackend)
    assert backend.backend_id == "bge-base-en-v1.5"
    assert backend._model is None


def test_test_mode_selects_openai_small():
    backend = embedding_backend_for_mode(True)

    assert isinstance(backend, OpenAIEmbeddingBackend)
    assert backend.backend_id == "openai-text-embedding-3-small"
    assert backend.model_id == "text-embedding-3-small"
    assert backend.vector_dimension == OPENAI_SMALL_EMBEDDING_DIMENSIONS
    assert "query=backend-defined-v1" in backend.embedding_space_id


def test_demo_cli_has_no_embedding_mode_flags():
    parser = build_parser()

    args = parser.parse_args([])
    assert not hasattr(args, "test_mode")
    assert not hasattr(args, "build_embedding_cache")


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_env_bool_accepts_true_values(monkeypatch, value):
    monkeypatch.setenv("TEST_MODE", value)
    assert _env_bool("TEST_MODE") is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", "off"])
def test_env_bool_accepts_false_values(monkeypatch, value):
    monkeypatch.setenv("TEST_MODE", value)
    assert _env_bool("TEST_MODE") is False
