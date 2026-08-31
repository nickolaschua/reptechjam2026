from __future__ import annotations

import inspect
import os
from pathlib import Path

import starter.agent as entry


def test_entrypoint_forces_bge_ollama_provider() -> None:
    assert os.environ["TEST_MODE"] == "false"
    assert os.environ["ALLOW_CATALOG_EMBEDDING"] == "false"


def test_official_constructor_routes_catalog_and_cache(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(entry._CoreAgent, "__init__", fake_init)
    cache_dir = tmp_path / "cache"
    agent = entry.Agent(tmp_path / "catalog.jsonl", embedding_cache_dir=cache_dir)
    assert isinstance(agent, entry.Agent)
    assert captured["catalog_path"] == tmp_path / "catalog.jsonl"
    assert captured["embedding_cache_dir"] == cache_dir
    assert captured["allow_catalog_embedding"] is False
    assert captured["explicit_cache_build"] is False


def test_provider_injection_is_rejected() -> None:
    try:
        entry.Agent("catalog.jsonl", test_mode=True)
    except TypeError as exc:
        assert "provider configuration is frozen" in str(exc)
    else:
        raise AssertionError("provider override unexpectedly accepted")


def test_required_contract_methods_are_inherited() -> None:
    reset = inspect.signature(entry.Agent.reset)
    respond = inspect.signature(entry.Agent.respond)
    assert list(reset.parameters)[:3] == ["self", "session_id", "user_profile"]
    assert list(respond.parameters)[:5] == [
        "self", "session_id", "user_message", "turn", "top_k",
    ]
