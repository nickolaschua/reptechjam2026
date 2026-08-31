"""Compare the post-ranking confidence gate on the released public set."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from . import agent as agent_module
from .agent import Agent
from .config import CONFIDENCE_SIMILARITY_THRESHOLD, CATALOG_PATH, PROJECT_ROOT


def _load_evaluator():
    path = PROJECT_ROOT / "techjam-conversational-search" / "evaluator" / "local_evaluator.py"
    sys.path.insert(0, str(path.parent.parent))
    spec = importlib.util.spec_from_file_location("techjam_public_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deterministic_response(_prompt: str, system_prompt: str, **_kwargs: Any) -> str:
    selected = re.search(r"(?:entropy|fixed-priority)-selected attribute listed here: (.*?)\. If", system_prompt)
    attributes = "another preference" if selected is None else selected.group(1).replace("'", "")
    return f"Could you clarify your preference for {attributes}?"


class _TelemetryAgent:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.evaluated = 0
        self.passed = 0
        self.rejected = 0
        self.empty_turns = 0
        self.turns = 0
        self.session_traces: list[dict[str, Any]] = []
        self._trace_by_session: dict[str, dict[str, Any]] = {}

    def reset(self, session_id: str, profile: dict) -> None:
        self.agent.reset(session_id, profile)
        trace = {"runtime_session_id": session_id, "turns": []}
        self.session_traces.append(trace)
        self._trace_by_session[session_id] = trace

    def respond(self, *args: Any, **kwargs: Any) -> dict:
        kwargs["debug"] = True
        response = self.agent.respond(*args, **kwargs)
        memory_trace = response.get("debug", {}).get("memory_trace", {})
        gate = dict(memory_trace.get("confidence_gate", {}))
        gate_threshold = gate.get("threshold")
        if isinstance(gate_threshold, (int, float)) and not math.isfinite(gate_threshold):
            gate["threshold"] = None
            gate["disabled"] = True
        self.turns += 1
        self.evaluated += len(gate.get("evaluated_products", []))
        self.passed += int(gate.get("pass_count", 0))
        self.rejected += int(gate.get("reject_count", 0))
        self.empty_turns += int(bool(gate.get("empty_result", False)))
        session_id = str(args[0])
        self._trace_by_session[session_id]["turns"].append({
            "turn": int(args[2]),
            "user_message": str(args[1]),
            "ask_attribute": response.get("ask_attribute"),
            "retrieval_route": memory_trace.get("retrieval_route"),
            "ranking_method": memory_trace.get("ranking_method"),
            "current_intent": memory_trace.get("current_intent"),
            "confidence_gate": gate,
            "returned": memory_trace.get("returned", []),
            "final_asins": memory_trace.get("final_asins", []),
        })
        return response

    def summary(self) -> dict[str, int]:
        return {
            "turns": self.turns,
            "evaluated_products": self.evaluated,
            "survivors": self.passed,
            "rejections": self.rejected,
            "empty_result_turns": self.empty_turns,
        }


def _attach_session_context(
    evaluator: Any,
    samples: list[dict[str, Any]],
    result_sessions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not (len(samples) == len(result_sessions) == len(traces)):
        raise RuntimeError("evaluation session records and telemetry are misaligned")
    enriched: list[dict[str, Any]] = []
    for sample, result, trace in zip(samples, result_sessions, traces):
        if sample["sample_id"] != result["sample_id"]:
            raise RuntimeError("evaluation sample order changed unexpectedly")
        target = str(sample["ground_truth"]["parent_asin"])
        product = products[target]
        intent_card, behavior = evaluator.materialize_hidden_fields(sample, products)
        enriched.append({
            **result,
            "target_parent_asin": target,
            "target_product": {
                "title": product.get("title"),
                "categories": product.get("categories") or [],
                "store": product.get("store"),
                "price": product.get("price"),
            },
            "intent_card": intent_card,
            "behavior": behavior,
            "turns": trace["turns"],
        })
    return enriched


def _compare_sessions(
    baseline_sessions: list[dict[str, Any]],
    gated_sessions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = {item["sample_id"]: item for item in baseline_sessions}
    comparisons: list[dict[str, Any]] = []
    for gated in gated_sessions:
        baseline = baseline_by_id[gated["sample_id"]]
        comparisons.append({
            "sample_id": gated["sample_id"],
            "scenario_type": gated["scenario_type"],
            "pre_gate": {
                "hit": baseline["hit"],
                "first_hit_turn": baseline["first_hit_turn"],
                "best_rank": baseline["best_rank"],
                "reciprocal_rank": baseline["reciprocal_rank"],
            },
            "confidence_gate": {
                "hit": gated["hit"],
                "first_hit_turn": gated["first_hit_turn"],
                "best_rank": gated["best_rank"],
                "reciprocal_rank": gated["reciprocal_rank"],
            },
            "reciprocal_rank_delta": round(
                float(gated["reciprocal_rank"]) - float(baseline["reciprocal_rank"]), 12
            ),
            "first_hit_turn_delta": (
                None
                if baseline["first_hit_turn"] is None or gated["first_hit_turn"] is None
                else int(gated["first_hit_turn"]) - int(baseline["first_hit_turn"])
            ),
        })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="evaluate only the first N public sessions")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evaluator = _load_evaluator()
    dataset_path = PROJECT_ROOT / "techjam-conversational-search" / "data" / "public_set.jsonl"
    samples = evaluator.load_jsonl(dataset_path)
    if args.limit is not None:
        samples = samples[:max(0, args.limit)]
    catalog_ids, categories, products = evaluator.catalog_index(CATALOG_PATH)

    active_agent = Agent()
    active_agent._call_llm = _deterministic_response
    runs: dict[str, Any] = {}
    original_threshold = agent_module.CONFIDENCE_SIMILARITY_THRESHOLD
    try:
        for name, threshold in (
            ("pre_gate", float("-inf")),
            ("confidence_gate_0_40", CONFIDENCE_SIMILARITY_THRESHOLD),
        ):
            agent_module.CONFIDENCE_SIMILARITY_THRESHOLD = threshold
            instrumented = _TelemetryAgent(active_agent)
            result = evaluator.evaluate(instrumented, samples, catalog_ids, categories, products)
            metrics = {
                key: result[key]
                for key in (
                    "sample_count", "hit_rate_at_10", "mrr", "mttc",
                    "efficiency", "recommended_technical_score",
                )
            }
            runs[name] = {
                "metrics": metrics,
                "confidence_gate": instrumented.summary(),
                "sessions": _attach_session_context(
                    evaluator, samples, result["sessions"], instrumented.session_traces, products
                ),
            }
    finally:
        agent_module.CONFIDENCE_SIMILARITY_THRESHOLD = original_threshold
        active_agent.close()

    baseline = runs["pre_gate"]["metrics"]
    gated = runs["confidence_gate_0_40"]["metrics"]
    comparisons = _compare_sessions(
        runs["pre_gate"]["sessions"], runs["confidence_gate_0_40"]["sessions"]
    )
    runs["delta"] = {
        "hit_rate_at_10": round(gated["hit_rate_at_10"] - baseline["hit_rate_at_10"], 6),
        "mrr": round(gated["mrr"] - baseline["mrr"], 6),
        "mttc": round(gated["mttc"] - baseline["mttc"], 6),
        "efficiency": round(gated["efficiency"] - baseline["efficiency"], 6),
        "recommended_technical_score": round(
            gated["recommended_technical_score"] - baseline["recommended_technical_score"], 6
        ),
    }
    payload = {
        "threshold": CONFIDENCE_SIMILARITY_THRESHOLD,
        "sample_limit": args.limit,
        **runs,
        "session_comparison": comparisons,
        "confidence_gate_misses": [
            item for item in runs["confidence_gate_0_40"]["sessions"] if not item["hit"]
        ],
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({
            "threshold": payload["threshold"],
            "sample_limit": payload["sample_limit"],
            "pre_gate": {
                "metrics": payload["pre_gate"]["metrics"],
                "confidence_gate": payload["pre_gate"]["confidence_gate"],
            },
            "confidence_gate_0_40": {
                "metrics": payload["confidence_gate_0_40"]["metrics"],
                "confidence_gate": payload["confidence_gate_0_40"]["confidence_gate"],
            },
            "delta": payload["delta"],
            "confidence_gate_miss_ids": [
                item["sample_id"] for item in payload["confidence_gate_misses"]
            ],
            "output": str(args.output),
        }, indent=2))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
