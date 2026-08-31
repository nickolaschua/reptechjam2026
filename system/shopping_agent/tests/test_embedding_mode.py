from __future__ import annotations

import inspect

import pytest

from system.shopping_agent.agent import Agent
from system.shopping_agent.demo import DemoApplication, build_parser
from system.shopping_agent.embedding_backends import BGEEmbeddingBackend, cache_filename
from system.shopping_agent.visualizer.server import BrowserApplication, run_server


def test_production_backend_is_lazy_bge():
    backend = BGEEmbeddingBackend()

    assert backend.backend_id == "bge-base-en-v1.5"
    assert backend.model_id == "BAAI/bge-base-en-v1.5"
    assert backend.vector_dimension == 768
    assert backend._model is None
    assert cache_filename(backend.backend_id) == "catalog_cache_bge-base-en-v1.5.npz"


def test_agent_test_mode_is_a_deprecated_noop_selecting_bge(monkeypatch):
    monkeypatch.setattr(Agent, "_build_category_index", lambda self: None)
    monkeypatch.setattr(Agent, "_build_vector_index", lambda self: None)
    monkeypatch.setattr(Agent, "catalogue", object(), raising=False)

    with pytest.warns(DeprecationWarning, match="environment selection remains active"):
        agent = Agent(test_mode=True, embedding_backend=BGEEmbeddingBackend())

    assert isinstance(agent.embedding_backend, BGEEmbeddingBackend)


def test_cli_and_server_configuration_have_no_embedding_mode_switch():
    args = build_parser().parse_args([])
    assert not hasattr(args, "test_mode")
    assert not hasattr(args, "build_embedding_cache")
    assert "test_mode" not in inspect.signature(DemoApplication).parameters
    assert "test_mode" not in inspect.signature(BrowserApplication).parameters
    assert "test_mode" not in inspect.signature(run_server).parameters
