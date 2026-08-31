"""The deployed agent + bolt-on, as a module the visualizer can load directly:

    python techjam-conversational-search/visualizer/trace_agents.py \
        --candidate-agent winston/lab/bench/bolt_on_agent.py

Template input never reaches the LLM (BoltOnParser returns None -> stock path);
messy input is parsed by local qwen. If Ollama is unreachable the turn falls
back to the stock path instead of failing - a demo must not die mid-session.
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
for p in (BENCH, BENCH.parent, BENCH.parent.parent, BENCH.parent.parent.parent / "techjam-conversational-search"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from plug_check import BoltOnAgent          # noqa: E402
from bolt_on import BoltOnParser            # noqa: E402


class _SafeParser(BoltOnParser):
    def parse(self, message: str, turn: int):
        try:
            return super().parse(message, turn)
        except Exception:                    # noqa: BLE001 - Ollama down: stock path, not a crash
            return None


class Agent(BoltOnAgent):
    def __init__(self, catalog_path="data/catalog.jsonl") -> None:
        super().__init__(catalog_path, _SafeParser(resolver=False))
