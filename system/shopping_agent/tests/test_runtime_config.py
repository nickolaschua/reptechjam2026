from __future__ import annotations

import pytest

from system.shopping_agent.config import load_runtime_config


CONFIG_KEYS = (
    "TEST_MODE", "OPENAI_API_KEY", "OPENAI_CHAT_MODEL", "OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_DIMENSIONS", "OPENAI_TIMEOUT_SECONDS", "OLLAMA_HOST",
    "OLLAMA_MODEL", "OLLAMA_TIMEOUT_SECONDS", "ALLOW_CATALOG_EMBEDDING",
    "CONFIDENCE_SIMILARITY_THRESHOLD",
)


def clear(monkeypatch):
    for key in CONFIG_KEYS: monkeypatch.delenv(key, raising=False)


def test_unset_and_false_select_local_without_openai_key(monkeypatch):
    clear(monkeypatch)
    assert load_runtime_config().test_mode is False
    monkeypatch.setenv("TEST_MODE", "false")
    assert load_runtime_config().provider == "ollama"


def test_true_requires_key_and_honors_model_overrides(monkeypatch):
    clear(monkeypatch); monkeypatch.setenv("TEST_MODE", "true")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_runtime_config()
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "chat-override")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "embed-override")
    cfg = load_runtime_config()
    assert (cfg.provider, cfg.openai_chat_model, cfg.openai_embedding_model) == (
        "openai", "chat-override", "embed-override")


def test_invalid_boolean_is_rejected(monkeypatch):
    clear(monkeypatch); monkeypatch.setenv("TEST_MODE", "sometimes")
    with pytest.raises(ValueError, match="TEST_MODE must be one of"):
        load_runtime_config()


def test_confidence_similarity_threshold_defaults_and_is_validated(monkeypatch):
    clear(monkeypatch)
    assert load_runtime_config().confidence_similarity_threshold == 0.40
    monkeypatch.setenv("CONFIDENCE_SIMILARITY_THRESHOLD", "-0.25")
    assert load_runtime_config().confidence_similarity_threshold == -0.25
    for invalid in ("nan", "1.01", "not-a-number"):
        monkeypatch.setenv("CONFIDENCE_SIMILARITY_THRESHOLD", invalid)
        with pytest.raises(ValueError, match="CONFIDENCE_SIMILARITY_THRESHOLD"):
            load_runtime_config()
