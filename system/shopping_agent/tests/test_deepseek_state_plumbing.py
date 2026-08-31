"""State updates prefer DeepSeek and fall back to Ollama; chat never routes."""
from system.shopping_agent.agent import Agent
from system.shopping_agent.model_client import ModelCall


def _bare():
    agent = Agent.__new__(Agent)
    agent._sessions = {}
    agent.instrumentation = {}
    return agent


class _StubOllama:
    provider = "ollama"
    model = "stub"

    def __init__(self):
        self.calls = 0

    def chat_result(self, messages, **kwargs):
        self.calls += 1
        return ModelCall('{"ok": true}', self.model, 0.0, 1, "assistant", self.provider)


_DS_TRACE = {"provider": "deepseek", "role": "assistant", "model": "deepseek-chat",
             "latency_seconds": 0.0, "attempts": 1, "retry_count": 0, "success": True,
             "error_type": None, "cause_type": None, "usage": {}}


def test_missing_or_placeholder_key_disables_deepseek(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert Agent._deepseek_state_call("p", "s") is None
    monkeypatch.setenv("DEEPSEEK_API_KEY", "your_key_here")
    assert Agent._deepseek_state_call("p", "s") is None


def test_state_updates_prefer_deepseek_then_fall_back(monkeypatch):
    agent = _bare()
    agent.llm_client = _StubOllama()
    monkeypatch.setattr(Agent, "_deepseek_state_call",
                        staticmethod(lambda p, s: ('{"via": "deepseek"}', dict(_DS_TRACE))))
    assert agent._call_llm("p", "s", response_json=True) == '{"via": "deepseek"}'
    assert agent.llm_client.calls == 0
    monkeypatch.setattr(Agent, "_deepseek_state_call", staticmethod(lambda p, s: None))
    assert agent._call_llm("p", "s", response_json=True) == '{"ok": true}'
    assert agent.llm_client.calls == 1


def test_chat_replies_never_route_to_deepseek(monkeypatch):
    agent = _bare()
    agent.llm_client = _StubOllama()

    def _boom(prompt, system_prompt):
        raise AssertionError("chat call must not reach DeepSeek")

    monkeypatch.setattr(Agent, "_deepseek_state_call", staticmethod(_boom))
    assert agent._call_llm("p", "s", response_json=False) == '{"ok": true}'
