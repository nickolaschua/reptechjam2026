from __future__ import annotations

import json
from copy import deepcopy

import pytest

from system.shopping_agent.agent import Agent, ExperimentConfig
from system.shopping_agent.ollama_client import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    OllamaClient,
    OllamaRequestError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from system.shopping_agent.visualizer.simulator import call_shopper_llm


def response(content: str, *, model: str = "llama3.1:8b") -> dict:
    return {"model": model, "message": {"content": content}}


def test_default_shared_configuration(monkeypatch):
    for name in (
        "OLLAMA_HOST",
        "OLLAMA_MODEL",
        "OLLAMA_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    client = OllamaClient(transport=lambda *_: response("ok"))

    assert client.host == DEFAULT_OLLAMA_HOST
    assert client.model == DEFAULT_OLLAMA_MODEL == "llama3.1:8b"
    assert client.timeout_seconds == DEFAULT_OLLAMA_TIMEOUT_SECONDS == 30.0
    assert client.retries == 1


def test_chat_payload_timeout_and_actual_model_instrumentation():
    calls = []

    def transport(url, body, timeout):
        calls.append((url, json.loads(body), timeout))
        return response("hello", model="llama3.1:8b")

    client = OllamaClient(
        host="ollama.test:11434",
        model="llama3.1:8b",
        timeout_seconds=7,
        transport=transport,
    )
    result = client.chat_result(
        [{"role": "user", "content": "hi"}],
        options={"temperature": 0.4},
        role="assistant",
    )

    assert result.content == "hello"
    assert calls == [
        (
            "http://ollama.test:11434/api/chat",
            {
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"temperature": 0.4},
            },
            7.0,
        )
    ]
    assert client.instrumentation()[-1] | {"latency_seconds": 0} == {
        "role": "assistant",
        "model": "llama3.1:8b",
        "latency_seconds": 0,
        "attempts": 1,
        "retry_count": 0,
        "success": True,
        "error_type": None,
        "cause_type": None,
    }


def test_transport_retry_then_success_records_retry_count():
    calls = 0

    def transport(*_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("connection reset")
        return response("ok")

    client = OllamaClient(model="llama3.1:8b", transport=transport)
    result = client.chat_result([{"role": "user", "content": "hi"}])

    assert result.retry_count == 1
    assert client.instrumentation()[-1]["retry_count"] == 1


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (lambda: TimeoutError("timed out"), OllamaTimeoutError),
        (lambda: OSError("offline"), OllamaRequestError),
    ],
)
def test_exhausted_transport_failures_are_typed_and_instrumented(failure, error_type):
    def transport(*_):
        raise failure()

    client = OllamaClient(model="llama3.1:8b", transport=transport)
    with pytest.raises(error_type) as captured:
        client.chat([{"role": "user", "content": "hi"}], role="assistant")

    assert captured.value.retry_count == 1
    assert client.instrumentation()[-1]["error_type"] == error_type.__name__


def test_exhausted_invalid_output_is_typed():
    client = OllamaClient(
        model="llama3.1:8b",
        transport=lambda *_: {"model": "llama3.1:8b", "message": {}},
    )

    with pytest.raises(OllamaResponseError, match="2 attempts"):
        client.chat([{"role": "user", "content": "hi"}])


def test_state_editor_assistant_and_shopper_share_the_same_model_and_transport():
    payloads = []

    def transport(_url, body, _timeout):
        payload = json.loads(body)
        payloads.append(payload)
        return response("local response")

    client = OllamaClient(model="llama3.1:8b", transport=transport)
    agent = Agent.__new__(Agent)
    agent.experiment_config = ExperimentConfig()
    agent.ollama_client = client
    agent.instrumentation = {"llm_calls": []}
    agent._sessions = {"s": {"debug_info": {}}}
    assert agent._call_llm("help", "system", session_id="s") == "local response"
    assert call_shopper_llm("reply", "shopper", client=client) == "local response"

    assert [payload["model"] for payload in payloads] == ["llama3.1:8b"] * 2
    assert [item["role"] for item in client.instrumentation()] == [
        "assistant",
        "shopper",
    ]


def test_assistant_failure_triggers_full_turn_rollback_and_records_provider_error():
    client = OllamaClient(
        model="llama3.1:8b",
        transport=lambda *_: (_ for _ in ()).throw(OSError("offline")),
    )
    agent = Agent.__new__(Agent)
    agent.experiment_config = ExperimentConfig()
    agent.ollama_client = client
    agent._sessions = {"s": Agent._new_session_state()}
    agent._active_lifecycle = {
        "s": {"user_id": "u", "sequence_index": 0, "visible_state": None}
    }
    agent._forensic_ranking_snapshots = {}
    agent.instrumentation = {
        "semantic_queries": [],
        "turns": [],
        "agent_errors": [],
        "llm_calls": [],
    }
    before = deepcopy(agent._sessions["s"])

    def fail_after_mutation(session_id, *_args, **_kwargs):
        agent._sessions[session_id]["category"] = "phantom"
        return agent._call_llm("reply", "system", session_id=session_id)

    agent._respond_custom = fail_after_mutation
    with pytest.raises(OllamaRequestError):
        agent.respond("s", "boots", 1, 5)

    assert agent._sessions["s"] == before
    assert agent.instrumentation["llm_calls"][-1]["rolled_back"] is True
    assert agent.instrumentation["llm_calls"][-1]["retry_count"] == 1
    error = agent.instrumentation["agent_errors"][-1]
    assert error["model"] == "llama3.1:8b"
    assert error["provider_error_type"] == "OllamaRequestError"
    assert error["rollback"] is True
