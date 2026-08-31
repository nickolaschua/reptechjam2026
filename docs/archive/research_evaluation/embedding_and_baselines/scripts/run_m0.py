"""Frozen Fast-Memory-only M0_OPENAI baseline runner.

Importing this module does not instantiate an Agent, construct an OpenAI
client, or issue a remote request. Catalog generation requires the explicit
``--allow-openai-catalog-build`` command-line flag.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiment_1"
SHARED_REPO = PROJECT_ROOT / "techjam-conversational-search"
CONFIG_PATH = CURRENT_DIR / "configs" / "m0_openai.json"
CACHE_DIR = CURRENT_DIR / "embedding_cache"
RESULTS_DIR = CURRENT_DIR / "baseline_results" / "m0_openai"

for path in (CURRENT_DIR, PROJECT_ROOT, EXPERIMENT_DIR, SHARED_REPO):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from agent import Agent
from embedding_backends import (
    OPENAI_EMBEDDING_SPACE_ID,
    OPENAI_MODEL,
    OpenAIEmbeddingBackend,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": float(np.mean(data)) if data else None,
        "p50": float(np.percentile(data, 50)) if data else None,
        "p95": float(np.percentile(data, 95)) if data else None,
    }


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    embedding = config.get("embedding", {})
    memory = config.get("memory", {})
    expected = {
        "baseline_name": "M0_OPENAI",
        "backend": "openai",
        "model": OPENAI_MODEL,
        "embedding_space_id": OPENAI_EMBEDDING_SPACE_ID,
        "dimensions": OpenAIEmbeddingBackend.vector_dimension,
        "normalized": True,
        "fallback_enabled": False,
        "fast_memory_enabled": True,
        "slow_memory_enabled": False,
    }
    actual = {
        "baseline_name": config.get("baseline_name"),
        "backend": embedding.get("backend"),
        "model": embedding.get("model"),
        "embedding_space_id": embedding.get("embedding_space_id"),
        "dimensions": embedding.get("dimensions"),
        "normalized": embedding.get("normalized"),
        "fallback_enabled": embedding.get("fallback_enabled"),
        "fast_memory_enabled": memory.get("fast_memory_enabled"),
        "slow_memory_enabled": memory.get("slow_memory_enabled"),
    }
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(f"M0 configuration failed validation: {', '.join(mismatches)}")
    return config


def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not present; no remote M0 work was attempted.\n"
            "After setting it, run:\n"
            "python nickolas/shopping_agent/run_m0.py --smoke\n"
            "python nickolas/shopping_agent/run_m0.py --allow-openai-catalog-build"
        )


def run_test_preflight() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "nickolas/shopping_agent/tests",
        "-v",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    result = {
        "command": "python -m unittest discover -s nickolas/shopping_agent/tests -v",
        "status": "passed" if completed.returncode == 0 else "failed",
        "test_count": int(match.group(1)) if match else None,
        "return_code": completed.returncode,
    }
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError("M0 test preflight failed")
    return result


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


def validate_evaluator_config(config: dict[str, Any]) -> None:
    from experiment_1 import run_eval_v2

    evaluator = config["evaluator"]
    mismatches = []
    if evaluator["max_turns"] != run_eval_v2.MAX_TURNS:
        mismatches.append("max_turns")
    if evaluator["top_k"] != run_eval_v2.TOP_K:
        mismatches.append("top_k")
    if mismatches:
        raise ValueError(
            "Frozen M0 config differs from the shared evaluator: " + ", ".join(mismatches)
        )


def failed_session_count(metrics: dict[str, Any]) -> int:
    sample_count = int(metrics.get("sample_count", 0))
    hits = int(round(sample_count * float(metrics.get("hit_rate_at_10", 0.0))))
    return sample_count - hits


def git_metadata() -> dict[str, Any]:
    def output(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"

    status = output("status", "--porcelain", "--", "nickolas/shopping_agent")
    return {
        "commit_sha": output("rev-parse", "HEAD"),
        "shopping_agent_has_uncommitted_changes": bool(status and status != "unavailable"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_smoke() -> None:
    config = load_config()
    require_openai_key()
    backend = OpenAIEmbeddingBackend(batch_size=2)
    started = time.perf_counter()
    vector = backend.embed_query("lightweight waterproof walking boots")
    payload = {
        "baseline_name": config["baseline_name"],
        "generated_at": utc_now(),
        "backend_id": backend.backend_id,
        "model_id": backend.model_id,
        "embedding_space_id": backend.embedding_space_id,
        "dimension": int(vector.shape[0]),
        "norm": float(np.linalg.norm(vector)),
        "wall_seconds": time.perf_counter() - started,
        "embedding_api": backend.usage_snapshot(),
        "catalog_embedding_attempted": False,
    }
    write_json(RESULTS_DIR / "smoke.json", payload)
    print(json.dumps(payload, indent=2))


def run_full(allow_catalog_build: bool) -> None:
    config = load_config()
    validate_evaluator_config(config)
    test_result = run_test_preflight()
    require_openai_key()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_DIR / "config.json", config)
    evaluator_config = config["evaluator"]
    samples, catalog_ids, categories, products = load_evaluator_data(
        int(evaluator_config["max_samples"])
    )

    run_started = time.perf_counter()
    agent = Agent(
        embedding_cache_dir=CACHE_DIR,
        allow_catalog_embedding=allow_catalog_build,
    )
    if agent.embedding_space_id != config["embedding"]["embedding_space_id"]:
        raise RuntimeError("M0 Agent resolved to the wrong embedding space")

    from experiment_1 import run_eval_v2

    previous_directory = Path.cwd()
    evaluator_started = time.perf_counter()
    try:
        os.chdir(RESULTS_DIR)
        evaluator_metrics = run_eval_v2.evaluate_v2(
            agent,
            samples,
            set(catalog_ids),
            categories,
            products,
            model_name=str(evaluator_config["shopper_model"]),
            max_samples=int(evaluator_config["max_samples"]),
        )
    finally:
        os.chdir(previous_directory)
    evaluator_wall = time.perf_counter() - evaluator_started
    total_run_wall = time.perf_counter() - run_started
    telemetry = agent.get_instrumentation()

    turns = telemetry["turns"]
    semantic_queries = telemetry["semantic_queries"]
    metrics = {
        "baseline_name": config["baseline_name"],
        "generated_at": utc_now(),
        "hit_rate_at_10": evaluator_metrics.get("hit_rate_at_10"),
        "mrr": evaluator_metrics.get("mrr"),
        "mean_turns_to_hit": evaluator_metrics.get("mttc"),
        "scenario_metrics": evaluator_metrics.get("scenario_metrics", {}),
        "failed_session_count": failed_session_count(evaluator_metrics),
        "agent_error_count": len(telemetry["agent_errors"]),
        "evaluator_metrics": evaluator_metrics,
    }
    initialization = telemetry["initialization"]
    cache_status = initialization["cache_status"]
    embedding_api = telemetry["embedding_api"]
    latency = {
        "baseline_name": config["baseline_name"],
        "routing": {
            "fast_path_turns": sum(item["route"] == "fast" for item in turns),
            "full_path_turns": sum(item["route"] == "full" for item in turns),
            "dense_retrieval_invocations": len(semantic_queries),
            "baseline_fallback_count": telemetry["baseline_fallback_count"],
        },
        "agent_respond_seconds": distribution(item["respond_seconds"] for item in turns),
        "query_embedding_seconds": distribution(
            item["query_embedding_seconds"] for item in semantic_queries
        ),
        "dense_search_seconds": distribution(
            item["dense_search_seconds"] for item in semantic_queries
        ),
        "total_evaluator_wall_seconds": evaluator_wall,
        "total_run_wall_seconds": total_run_wall,
        "initialization": initialization,
        "openai_embeddings": {
            "embedding_space_id": telemetry["embedding_space_id"],
            "request_count": embedding_api["request_count"],
            "input_tokens": embedding_api["input_tokens"],
            "request_latency_seconds": distribution(
                embedding_api["request_latencies_seconds"]
            ),
            "cache_status": cache_status,
            "cache_hit": cache_status == "hit",
            "cache_miss": cache_status in {"built", "rejected"},
            "catalog_build_seconds": initialization[
                "catalog_embedding_generation_seconds"
            ],
        },
    }
    write_json(RESULTS_DIR / "metrics.json", metrics)
    write_json(RESULTS_DIR / "latency.json", latency)

    summary = f"""# M0_OPENAI run summary

- Fast Memory: enabled (Patch-2 semantics)
- Slow Memory: disabled and not implemented
- Embedding backend: OpenAI `{OPENAI_MODEL}` for both product and query vectors
- HR@10: {metrics['hit_rate_at_10']}
- MRR: {metrics['mrr']}
- Mean turns to hit/conversion: {metrics['mean_turns_to_hit']}
- Failed sessions: {metrics['failed_session_count']}
- Agent errors: {metrics['agent_error_count']}
- Fast/full turns: {latency['routing']['fast_path_turns']} / {latency['routing']['full_path_turns']}
- Dense retrieval invocations: {latency['routing']['dense_retrieval_invocations']}

OpenAI embeddings are used intentionally as the development baseline. Phase 3
found BGE stronger on the controlled embedding benchmark, so this run does not
claim OpenAI is the final or competition-best embedder. BGE remains available
for later final-system reruns.
"""
    (RESULTS_DIR / "run_summary.md").write_text(summary, encoding="utf-8")

    git = git_metadata()
    source_files = [Path(__file__), CURRENT_DIR / "agent.py", CURRENT_DIR / "embedding_backends.py", CONFIG_PATH]
    fingerprints = {path.name: sha256(path) for path in source_files}
    freeze = f"""# M0_OPENAI freeze manifest

- Baseline: M0_OPENAI
- Fast Memory: enabled
- Slow Memory: disabled
- Embedding backend: OpenAI
- Embedding model: `{OPENAI_MODEL}`
- Embedding space: `{OPENAI_EMBEDDING_SPACE_ID}`
- Evaluator: existing `experiment_1.run_eval_v2` evaluator v2
- Test result: {test_result['status']} ({test_result['test_count']} tests)
- Timestamp/run ID: {metrics['generated_at']}
- Git commit: `{git['commit_sha']}`
- `nickolas/shopping_agent` has uncommitted changes: {str(git['shopping_agent_has_uncommitted_changes']).lower()}
- Result files: `config.json`, `metrics.json`, `latency.json`, `results_v2.json`, `run_summary.md`, `eval_sessions/`

## Source SHA-256

""" + "\n".join(f"- `{name}`: `{value}`" for name, value in fingerprints.items()) + "\n"
    (RESULTS_DIR / "FREEZE.md").write_text(freeze, encoding="utf-8")
    print(f"M0_OPENAI complete: {RESULTS_DIR}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the frozen M0_OPENAI baseline")
    result.add_argument(
        "--smoke",
        action="store_true",
        help="Embed one query only; never initialize or embed the catalog",
    )
    result.add_argument(
        "--allow-openai-catalog-build",
        action="store_true",
        help="Permit a missing/rejected OpenAI catalog cache to be built during the full run",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_full(args.allow_openai_catalog_build)


if __name__ == "__main__":
    main()
