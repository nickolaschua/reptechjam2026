"""Controlled retrieval and shared-evaluator embedding bake-off runner.

Examples:
  python nickolas/shopping_agent/compare_embeddings.py fixture --samples 200
  python nickolas/shopping_agent/compare_embeddings.py smoke-openai
  python nickolas/shopping_agent/compare_embeddings.py retrieval --samples 200 --allow-openai-catalog-build
  python nickolas/shopping_agent/compare_embeddings.py end-to-end --samples 200 --repeats 1

The OpenAI catalog is never generated unless --allow-openai-catalog-build is
present on a command that constructs the OpenAI Agent.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiment_1"
SHARED_REPO = PROJECT_ROOT / "techjam-conversational-search"
DEFAULT_RESULTS_DIR = CURRENT_DIR / "benchmark_results"
DEFAULT_CACHE_DIR = CURRENT_DIR / "embedding_cache"

for path in (CURRENT_DIR, PROJECT_ROOT, EXPERIMENT_DIR, SHARED_REPO):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from agent import Agent as CanonicalAgent, _state_to_retrieval_query
from agent_bge import Agent as BGEAgent
from agent_openai import Agent as OpenAIAgent
from embedding_backends import OpenAIEmbeddingBackend


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
    }


def calculate_retrieval_metrics(ranks: Sequence[int | None]) -> dict[str, Any]:
    known = list(ranks)
    if not known:
        return {
            "query_count": 0,
            "recall_at_10": None,
            "recall_at_50": None,
            "recall_at_150": None,
            "mrr": None,
            "median_target_rank": None,
            "failures": 0,
        }
    numeric = [int(rank) for rank in known if rank is not None]
    denominator = len(known)
    return {
        "query_count": denominator,
        "recall_at_10": sum(rank <= 10 for rank in numeric) / denominator,
        "recall_at_50": sum(rank <= 50 for rank in numeric) / denominator,
        "recall_at_150": sum(rank <= 150 for rank in numeric) / denominator,
        "mrr": sum(1.0 / rank for rank in numeric) / denominator,
        "median_target_rank": statistics.median(numeric) if numeric else None,
        "failures": denominator - len(numeric),
    }


def load_evaluator_data(max_samples: int) -> tuple[list[dict], list[str], dict, dict]:
    from evaluator.local_evaluator import catalog_index

    catalog_path = SHARED_REPO / "data" / "catalog.jsonl"
    dataset_path = SHARED_REPO / "data" / "public_set.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    samples = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:max_samples]
    return samples, catalog_ids, categories, products


def build_fixture(max_samples: int) -> list[dict[str, Any]]:
    """Build label-separated, deterministic semantic queries.

    The target ASIN is stored only in the offline scoring field. It is never
    inserted into the query, Fast Memory, or Agent.respond().
    """
    from evaluator.local_evaluator import coarse_category
    from experiment_1.shopper_agent import materialize_hidden_fields

    samples, _catalog_ids, categories, products = load_evaluator_data(max_samples)
    fixture: list[dict[str, Any]] = []
    for sample in samples:
        target_asin = str(sample["ground_truth"]["parent_asin"])
        card, _behavior = materialize_hidden_fields(sample, products)
        constraints = [
            str(value).strip()
            for value in card.get("hard_constraints", []) + card.get("soft_preferences", [])
            if str(value).strip()
        ]
        state = CanonicalAgent._new_session_state()
        state["category"] = coarse_category(categories.get(target_asin, []))
        state["department"] = ""
        state["disclosed_slots"] = {"feature": set(constraints)}
        query_text = _state_to_retrieval_query(state)
        if target_asin.casefold() in query_text.casefold():
            raise ValueError(f"Fixture query unexpectedly contains target ASIN for {target_asin}")
        serialized_state = {
            **state,
            "disclosed_slots": {
                key: sorted(values) for key, values in state["disclosed_slots"].items()
            },
            "seen_asins": [],
            "negated_terms": [],
            "asked_attributes": [],
        }
        fixture.append(
            {
                "sample_id": str(sample.get("sample_id", len(fixture))),
                "scenario_type": str(sample.get("scenario_type", "")),
                "query_text": query_text,
                "canonical_state": serialized_state,
                "target_asin": target_asin,
            }
        )
    return fixture


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_fixture(path: Path | None, max_samples: int) -> list[dict[str, Any]]:
    if path is None:
        return build_fixture(max_samples)
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    return list(rows)[:max_samples]


def create_agent(backend: str, allow_openai_catalog_build: bool, cache_dir: Path):
    common = {"embedding_cache_dir": cache_dir}
    if backend == "bge":
        return BGEAgent(**common)
    return OpenAIAgent(
        allow_catalog_embedding=allow_openai_catalog_build,
        **common,
    )


def run_retrieval_backend(agent: Any, fixture: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    ranks: list[int | None] = []
    for item in fixture:
        target = str(item["target_asin"])
        before = len(agent.instrumentation["semantic_queries"])
        try:
            indices = agent._dense_retrieve(item["query_text"], top_n=len(agent.catalog_ids))
            ranked_ids = [agent.catalog_ids[int(index)] for index in indices]
            rank = ranked_ids.index(target) + 1 if target in ranked_ids else None
            failure = None
        except Exception as exc:
            rank = None
            failure = {"type": type(exc).__name__, "message": str(exc)}
        ranks.append(rank)
        timing = (
            agent.instrumentation["semantic_queries"][before]
            if len(agent.instrumentation["semantic_queries"]) > before
            else {}
        )
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "scenario_type": item.get("scenario_type"),
                "query_text": item["query_text"],
                "target_asin": target,
                "target_rank": rank,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "query_embedding_seconds": timing.get("query_embedding_seconds"),
                "dense_search_seconds": timing.get("dense_search_seconds"),
                "failure": failure,
            }
        )
    telemetry = agent.get_instrumentation()
    successful_timings = [row for row in rows if row["failure"] is None]
    return {
        "backend_id": agent.embedding_backend_id,
        "model_id": agent.embedding_model_id,
        "metrics": calculate_retrieval_metrics(ranks),
        "query_embedding_latency_seconds": distribution(
            row["query_embedding_seconds"] for row in successful_timings
        ),
        "dense_search_latency_seconds": distribution(
            row["dense_search_seconds"] for row in successful_timings
        ),
        "initialization": telemetry["initialization"],
        "embedding_api": telemetry["embedding_api"],
        "queries": rows,
    }


def summarize_turns(telemetry: dict[str, Any]) -> dict[str, Any]:
    turns = telemetry["turns"]
    return {
        "fast_path_turn_count": sum(item["route"] == "fast" for item in turns),
        "full_path_turn_count": sum(item["route"] == "full" for item in turns),
        "dense_retrieval_invocation_count": sum(bool(item["dense_invoked"]) for item in turns),
        "respond_latency_seconds": distribution(item["respond_seconds"] for item in turns),
        "agent_errors": telemetry["agent_errors"],
        "baseline_fallback_count": telemetry["baseline_fallback_count"],
    }


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_end_to_end_backend(
    backend: str,
    samples: Sequence[dict],
    catalog_ids: Sequence[str],
    categories: dict,
    products: dict,
    shopper_model: str,
    cache_dir: Path,
    output_dir: Path,
    allow_openai_catalog_build: bool,
) -> dict[str, Any]:
    from experiment_1 import run_eval_v2

    started = time.perf_counter()
    agent = create_agent(backend, allow_openai_catalog_build, cache_dir)
    with working_directory(output_dir):
        metrics = run_eval_v2.evaluate_v2(
            agent,
            list(samples),
            set(catalog_ids),
            categories,
            products,
            model_name=shopper_model,
            max_samples=len(samples),
        )
    telemetry = agent.get_instrumentation()
    return {
        "backend_id": agent.embedding_backend_id,
        "model_id": agent.embedding_model_id,
        "evaluator_metrics": metrics,
        "turn_metrics": summarize_turns(telemetry),
        "initialization": telemetry["initialization"],
        "embedding_api": telemetry["embedding_api"],
        "total_evaluation_wall_seconds": time.perf_counter() - started,
        "stochasticity_note": (
            "The shared LLM shopper is stochastic. A single end-to-end run is not a "
            "perfectly paired embedding experiment; controlled retrieval is primary."
        ),
    }


def nested(payload: dict[str, Any] | None, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def catalog_cache_or_build_time(payload: dict[str, Any] | None) -> float | None:
    initialization = nested(payload, "initialization")
    if not isinstance(initialization, dict):
        return None
    return float(initialization.get("embedding_cache_load_seconds", 0.0)) + float(
        initialization.get("catalog_embedding_generation_seconds", 0.0)
    )


def comparison_summary(
    retrieval: dict[str, dict[str, Any]] | None = None,
    end_to_end: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieval = retrieval or {}
    end_to_end = end_to_end or {}
    columns: dict[str, dict[str, Any]] = {}
    for short_name in ("bge", "openai"):
        dense = retrieval.get(short_name)
        extended = end_to_end.get(short_name)
        columns[short_name] = {
            "Dense Recall@10": nested(dense, "metrics", "recall_at_10"),
            "Dense Recall@50": nested(dense, "metrics", "recall_at_50"),
            "Dense Recall@150": nested(dense, "metrics", "recall_at_150"),
            "Dense MRR": nested(dense, "metrics", "mrr"),
            "Median target rank": nested(dense, "metrics", "median_target_rank"),
            "Query embedding p50 (s)": nested(dense, "query_embedding_latency_seconds", "p50"),
            "Query embedding p95 (s)": nested(dense, "query_embedding_latency_seconds", "p95"),
            "Dense-search p50 (s)": nested(dense, "dense_search_latency_seconds", "p50"),
            "Dense-search p95 (s)": nested(dense, "dense_search_latency_seconds", "p95"),
            "Agent respond p50 (s)": nested(extended, "turn_metrics", "respond_latency_seconds", "p50"),
            "Agent respond p95 (s)": nested(extended, "turn_metrics", "respond_latency_seconds", "p95"),
            "Extended HR@10": nested(extended, "evaluator_metrics", "hit_rate_at_10"),
            "Extended MRR": nested(extended, "evaluator_metrics", "mrr"),
            "Mean turns": nested(extended, "evaluator_metrics", "mttc"),
            "Catalog cache/build time (s)": catalog_cache_or_build_time(dense or extended),
            "Total evaluation time (s)": nested(extended, "total_evaluation_wall_seconds"),
            "Embedding API requests": nested(dense or extended, "embedding_api", "request_count"),
            "Embedding API input tokens": nested(dense or extended, "embedding_api", "input_tokens"),
            "Failures": (
                nested(dense, "metrics", "failures")
                if dense is not None
                else len(nested(extended, "turn_metrics", "agent_errors") or [])
            ),
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_evidence": "controlled_retrieval",
        "winner_declared": False,
        "stochasticity_note": (
            "End-to-end shopper runs are stochastic and are supporting, not perfectly paired, evidence."
        ),
        "columns": columns,
    }


def write_summary(summary: dict[str, Any], results_dir: Path) -> None:
    save_json(results_dir / "comparison_summary.json", summary)
    bge = summary["columns"]["bge"]
    openai = summary["columns"]["openai"]
    lines = [
        "# Embedding bake-off summary",
        "",
        "The controlled retrieval benchmark is the primary embedding-quality evidence. "
        "No winner is declared automatically.",
        "",
        "| Metric | BGE | OpenAI |",
        "|---|---:|---:|",
    ]
    for metric in bge:
        lines.append(
            f"| {metric} | {format_value(bge[metric])} | {format_value(openai[metric])} |"
        )
    lines.extend(["", summary["stochasticity_note"], ""])
    (results_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def command_fixture(args: argparse.Namespace) -> None:
    fixture = build_fixture(args.samples)
    destination = args.output or (args.results_dir / "retrieval_fixture.json")
    save_json(destination, fixture)
    print(f"Wrote {len(fixture)} controlled queries to {destination}")


def command_smoke_openai(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not present; smoke test was not run")
    backend = OpenAIEmbeddingBackend(batch_size=2)
    vector = backend.embed_query("lightweight waterproof walking boots")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend_id": backend.backend_id,
        "dimension": int(vector.shape[0]),
        "norm": float(np.linalg.norm(vector)),
        "usage": backend.usage_snapshot(),
    }
    save_json(args.results_dir / f"openai_smoke_{timestamp()}.json", result)
    print(json.dumps(result, indent=2))


def command_retrieval(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    fixture = load_fixture(args.fixture, args.samples)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.results_dir / "retrieval_fixture.json", fixture)
    results: dict[str, dict[str, Any]] = {}
    for backend in args.backends:
        agent = create_agent(backend, args.allow_openai_catalog_build, args.cache_dir)
        result = run_retrieval_backend(agent, fixture)
        results[backend] = result
        save_json(args.results_dir / f"retrieval_{backend}_{timestamp()}.json", result)
    summary = comparison_summary(retrieval=results)
    write_summary(summary, args.results_dir)
    return results


def command_end_to_end(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    samples, catalog_ids, categories, products = load_evaluator_data(args.samples)
    latest: dict[str, dict[str, Any]] = {}
    for repeat in range(1, args.repeats + 1):
        for backend in args.backends:
            run_id = f"e2e_{backend}_repeat{repeat}_{timestamp()}"
            result = run_end_to_end_backend(
                backend,
                samples,
                catalog_ids,
                categories,
                products,
                args.shopper_model,
                args.cache_dir,
                args.results_dir / run_id,
                args.allow_openai_catalog_build,
            )
            result["repeat"] = repeat
            latest[backend] = result
            save_json(args.results_dir / f"{run_id}.json", result)
    summary = comparison_summary(end_to_end=latest)
    write_summary(summary, args.results_dir)
    return latest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Phase 3 controlled embedding bake-off")
    root.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    root.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    subparsers = root.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--samples", type=int, default=200)
    fixture.add_argument("--output", type=Path)

    subparsers.add_parser("smoke-openai")

    for name in ("retrieval", "end-to-end", "all"):
        command = subparsers.add_parser(name)
        command.add_argument("--samples", type=int, default=200)
        command.add_argument("--backends", nargs="+", choices=("bge", "openai"), default=["bge", "openai"])
        command.add_argument(
            "--allow-openai-catalog-build",
            action="store_true",
            help="Explicitly permit batched embedding of a missing/rejected OpenAI catalog cache",
        )
        if name in ("retrieval", "all"):
            command.add_argument("--fixture", type=Path)
        if name in ("end-to-end", "all"):
            command.add_argument("--shopper-model", default="llama3.1")
            command.add_argument("--repeats", type=int, default=1)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "fixture":
        command_fixture(args)
    elif args.command == "smoke-openai":
        command_smoke_openai(args)
    elif args.command == "retrieval":
        command_retrieval(args)
    elif args.command == "end-to-end":
        command_end_to_end(args)
    else:
        retrieval = command_retrieval(args)
        end_to_end = command_end_to_end(args)
        write_summary(comparison_summary(retrieval, end_to_end), args.results_dir)


if __name__ == "__main__":
    main()
