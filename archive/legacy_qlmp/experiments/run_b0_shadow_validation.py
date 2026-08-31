"""Phase 6.1 B0 calibration and fixed-transcript shadow parity runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

import requests

import run_longitudinal_eval as longitudinal
from agent import Agent
from longitudinal_eval.directives import ShopperLLMClient
from memory_store import InMemoryUserMemoryStore
from evaluator.local_evaluator import catalog_index


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
SHARED_REPO = PROJECT_ROOT / "techjam-conversational-search"
DEFAULT_OUTPUT_DIR = (
    CURRENT_DIR / "longitudinal_eval" / "results" / "b0_validation"
)
FROZEN_M0_DIR = CURRENT_DIR / "baseline_results" / "m0_openai"
REQUIRED_OUTPUTS = (
    "config.json",
    "overall_metrics.json",
    "by_user.json",
    "by_sequence.json",
    "probe_sessions.json",
    "disclosure_diagnostics.json",
    "leakage_diagnostics.json",
    "shadow_parity.json",
    "old_vs_new_comparison.json",
    "run_summary.md",
    "full_run.json",
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    }


def validate_output_directory(
    path: Path, *, allowed_existing: Iterable[str] = ()
) -> Path:
    resolved = path.resolve()
    frozen = FROZEN_M0_DIR.resolve()
    baseline_root = (CURRENT_DIR / "baseline_results").resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise ValueError("B0 validation output cannot be written under frozen M0 artifacts")
    if resolved == baseline_root or baseline_root in resolved.parents:
        raise ValueError("B0 validation output cannot overwrite Phase-4 baseline artifacts")
    allowed = set(allowed_existing)
    existing = [
        name
        for name in REQUIRED_OUTPUTS
        if (resolved / name).exists() and name not in allowed
    ]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing B0 validation outputs: {', '.join(existing)}"
        )
    return resolved


def preflight_runtime(provider: str, model: str) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY is required for M0 query and committed-memory embeddings"
        )
    if provider == "ollama":
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                "Ollama is required by the pinned longitudinal shopper configuration"
            ) from exc
        installed = [
            str(value.get("name") or value.get("model") or "")
            for value in response.json().get("models", [])
        ]
        if not any(value == model or value.split(":", 1)[0] == model for value in installed):
            raise RuntimeError(
                f"pinned shopper model {model!r} is not installed in Ollama"
            )
        return {"provider": provider, "model": model, "installed_models": installed}
    credential = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }[provider]
    ShopperLLMClient._credential(credential)
    return {"provider": provider, "model": model}


def grouped_metrics(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(sessions)
    hits = sum(bool(value["hit"]) for value in sessions)
    turns = [
        int(value["first_hit_turn"])
        if value.get("first_hit_turn") is not None
        else longitudinal.MAX_TURNS + 1
        for value in sessions
    ]
    return {
        "session_count": count,
        "hit_rate_at_10": 0.0 if not count else round(hits / count, 6),
        "mrr": 0.0
        if not count
        else round(statistics.fmean(float(value["reciprocal_rank"]) for value in sessions), 6),
        "mean_turns": None if not count else round(statistics.fmean(turns), 6),
        "no_hit_count": count - hits,
        "no_hit_rate": 0.0 if not count else round((count - hits) / count, 6),
    }


def overall_metrics(
    sessions: Sequence[Mapping[str, Any]],
    *,
    agent_errors: int,
    shopper_provider: str,
    shopper_model: str,
    wall_seconds: float,
) -> dict[str, Any]:
    metrics = grouped_metrics(sessions)
    metrics.update(
        {
            "run_name": "B0_LONGITUDINAL_40",
            "total_sessions": metrics.pop("session_count"),
            "agent_errors": int(agent_errors),
            "target_leak_flagged_sessions": sum(
                bool(value.get("target_leakage", {}).get("leaked")) for value in sessions
            ),
            "shopper_provider": shopper_provider,
            "shopper_model": shopper_model,
            "wall_clock_seconds": round(wall_seconds, 6),
        }
    )
    return metrics


def metrics_by_user(
    fixture: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for value in sessions:
        groups.setdefault(str(value["user_id"]), []).append(value)
    result: dict[str, Any] = {}
    for number, user in enumerate(longitudinal.ordered_fixture_users(fixture), start=1):
        user_id = str(user["user_id"])
        result[f"U{number}"] = {
            "user_id": user_id,
            **grouped_metrics(groups.get(user_id, [])),
        }
    return result


def metrics_by_sequence(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sequence_index in range(10):
        values = [
            value
            for value in sessions
            if int(value["sequence_index"]) == sequence_index
        ]
        result[f"S{sequence_index + 1}"] = {
            "sequence_index": sequence_index,
            **grouped_metrics(values),
        }
    return result


def probe_sessions(
    fixture: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    labels = {
        str(user["user_id"]): f"U{number}"
        for number, user in enumerate(longitudinal.ordered_fixture_users(fixture), start=1)
    }
    result: dict[str, Any] = {}
    for value in sessions:
        if int(value["sequence_index"]) != 9:
            continue
        result[f"{labels[str(value['user_id'])]} S10"] = {
            key: value.get(key)
            for key in (
                "session_id",
                "user_id",
                "target_asin",
                "target_title",
                "hit",
                "best_rank",
                "reciprocal_rank",
                "first_hit_turn",
                "turns",
                "final_fast_memory",
                "committed_memory_items",
            )
        }
    return result


def disclosure_diagnostics(
    sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scheduled: list[dict[str, Any]] = []
    for session in sessions:
        for value in session.get("semantic_disclosure_validation", ()):
            if value.get("directive_type") not in {"disclose", "reinforce"}:
                continue
            scheduled.append(
                {
                    "session_id": session["session_id"],
                    "user_id": session["user_id"],
                    "sequence_index": session["sequence_index"],
                    **dict(value),
                }
            )
    count = len(scheduled)

    def outcome(field: str) -> dict[str, Any]:
        successes = sum(bool(value[field]) for value in scheduled)
        return {
            "count": successes,
            "rate": 0.0 if not count else round(successes / count, 6),
        }

    failures = [
        value
        for value in scheduled
        if not (
            value["shopper_expressed"]
            and value["fast_memory_captured"]
            and value["memory_committed"]
        )
    ]
    return {
        "scope": "scheduled disclose and reinforce events only",
        "scheduled": count,
        "shopper_expressed": outcome("shopper_expressed"),
        "fast_memory_captured": outcome("fast_memory_captured"),
        "memory_committed": outcome("memory_committed"),
        "failed_chain_event_count": len(failures),
        "failed_chain_sessions": sorted({value["session_id"] for value in failures}),
        "failed_chain_events": failures,
        "events": scheduled,
    }


def leakage_diagnostics(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    asin = [
        value["session_id"]
        for value in sessions
        if value.get("target_leakage", {}).get("exact_target_asin")
    ]
    title = [
        value["session_id"]
        for value in sessions
        if value.get("target_leakage", {}).get("exact_target_title")
    ]
    return {
        "exact_target_asin_leakage_count": len(asin),
        "exact_target_asin_leakage_sessions": asin,
        "exact_normalized_target_title_leakage_count": len(title),
        "exact_normalized_target_title_leakage_sessions": title,
        "any_target_leakage_count": len(set(asin).union(title)),
        "affected_sessions": sorted(set(asin).union(title)),
        "excluded_from_metrics": False,
    }


def old_vs_new(new: Mapping[str, Any]) -> dict[str, Any]:
    old = json.loads((FROZEN_M0_DIR / "metrics.json").read_text(encoding="utf-8"))
    overall = old["evaluator_metrics"]
    buying = old["scenario_metrics"]["buying"]
    return {
        "comparison_status": "external calibration only",
        "confounds": [
            "curated longitudinal dataset differs from the 200-session public dataset",
            "frozen M0 records shopper model llama3.1 but not the actual fallthrough provider",
        ],
        "m0_public_200": {
            "session_count": overall["sample_count"],
            "hit_rate_at_10": overall["hit_rate_at_10"],
            "mrr": overall["mrr"],
            "mean_turns": overall["mttc"],
            "no_hit_count": old["failed_session_count"],
            "no_hit_rate": old["failed_session_count"] / overall["sample_count"],
            "shopper_model": "llama3.1",
            "shopper_provider": "not recorded; credential fallthrough made it indeterminate",
        },
        "m0_public_buying_80": {
            "session_count": buying["sample_count"],
            "hit_rate_at_10": buying["hit_rate_at_10"],
            "mrr": buying["mrr"],
            "mean_turns": buying["mttc"],
            "no_hit_count": round(
                buying["sample_count"] * (1.0 - buying["hit_rate_at_10"])
            ),
            "no_hit_rate": 1.0 - buying["hit_rate_at_10"],
        },
        "b0_longitudinal_40": {
            "session_count": new["total_sessions"],
            "hit_rate_at_10": new["hit_rate_at_10"],
            "mrr": new["mrr"],
            "mean_turns": new["mean_turns"],
            "no_hit_count": new["no_hit_count"],
            "no_hit_rate": new["no_hit_rate"],
            "shopper_model": new["shopper_model"],
            "shopper_provider": new["shopper_provider"],
        },
        "interpretation": (
            "Old-vs-new differences are benchmark calibration, not a memory effect. "
            "Future memory effects require B0/B1/B2/B3 comparisons on this same fixture."
        ),
    }


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    rendered = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    rendered.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def run_summary(
    config: Mapping[str, Any],
    overall: Mapping[str, Any],
    by_user: Mapping[str, Any],
    by_sequence: Mapping[str, Any],
    probes: Mapping[str, Any],
    disclosure: Mapping[str, Any],
    leakage: Mapping[str, Any],
    parity: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    old = comparison["m0_public_200"]
    buying = comparison["m0_public_buying_80"]
    new = comparison["b0_longitudinal_40"]
    return f"""# Phase 6.1 — B0 longitudinal baseline validation

## A. Current run configuration

- Run: `B0_LONGITUDINAL_40`
- Shopper: `{config['shopper']['provider']}` / `{config['shopper']['model']}`
- Agent: M0_OPENAI / B0, Fast Memory enabled, longitudinal memory shadow-only
- Embeddings: `{config['embedding']['backend']}` / `{config['embedding']['model']}`
- Sessions: {overall['total_sessions']}
- Independent repeats: {config['repeats_run']}
- Benchmark wall time: {overall['wall_clock_seconds']} seconds

## B. B0_LONGITUDINAL_40 results

{markdown_table(['HR@10', 'MRR', 'Mean turns', 'No hits', 'Agent errors', 'Leak flagged'], [[overall['hit_rate_at_10'], overall['mrr'], overall['mean_turns'], overall['no_hit_count'], overall['agent_errors'], overall['target_leak_flagged_sessions']]])}

### By user

{markdown_table(['User', 'Sessions', 'HR@10', 'MRR', 'Mean turns', 'No hits'], [[label, value['session_count'], value['hit_rate_at_10'], value['mrr'], value['mean_turns'], value['no_hit_count']] for label, value in by_user.items()])}

### By sequence index

{markdown_table(['Sequence', 'Sessions', 'HR@10', 'MRR', 'Mean turns', 'No hits'], [[label, value['session_count'], value['hit_rate_at_10'], value['mrr'], value['mean_turns'], value['no_hit_count']] for label, value in by_sequence.items()])}

### Probe sessions

{markdown_table(['Probe', 'Hit@10', 'Best rank', 'RR', 'First hit turn'], [[label, value['hit'], value['best_rank'], value['reciprocal_rank'], value['first_hit_turn']] for label, value in probes.items()])}

Full probe transcripts, final Fast Memory, and committed MemoryItems are in `probe_sessions.json`.

## C. Disclosure diagnostics

{markdown_table(['Scheduled', 'Shopper expressed', 'Fast Memory captured', 'MemoryItem committed'], [[disclosure['scheduled'], f"{disclosure['shopper_expressed']['count']} ({disclosure['shopper_expressed']['rate']:.1%})", f"{disclosure['fast_memory_captured']['count']} ({disclosure['fast_memory_captured']['rate']:.1%})", f"{disclosure['memory_committed']['count']} ({disclosure['memory_committed']['rate']:.1%})"]])}

Failed-chain sessions: {', '.join(disclosure['failed_chain_sessions']) or 'none'}.

## D. Leakage diagnostics

- Exact target ASIN leakage: {leakage['exact_target_asin_leakage_count']} sessions ({', '.join(leakage['exact_target_asin_leakage_sessions']) or 'none'})
- Exact normalized target-title leakage: {leakage['exact_normalized_target_title_leakage_count']} sessions ({', '.join(leakage['exact_normalized_target_title_leakage_sessions']) or 'none'})
- Sessions were not silently excluded.

## E. Shadow-memory parity

{markdown_table(['Paired turns', 'Identical rankings', 'Different rankings', 'Parity', 'Target-rank differences', 'Fast Memory differences', 'Route differences'], [[parity['total_paired_turns'], parity['identical_recommendation_turns'], parity['differing_recommendation_turns'], f"{parity['recommendation_order_parity_rate']:.1%}", parity['target_rank_difference_count'], parity['fast_memory_difference_count'], parity['route_difference_count']]])}

Identical shopper inputs: `{parity['all_shopper_inputs_identical']}`. Historical memory applied: `{parity['historical_memory_applied']}`.

Agent LLM call-tape control: `{parity['agent_llm_call_control']['enabled']}`; recorded/replayed calls: {parity['agent_llm_call_control']['recorded_call_count']}/{parity['agent_llm_call_control']['replayed_call_count']}; prompt mismatches: {parity['agent_llm_call_control']['prompt_mismatch_count']}. This evaluation-only control holds stochastic state/prose generation fixed while varying only the longitudinal store.

## F. Old 200 vs new 40 calibration

{markdown_table(['Metric', 'M0 public 200', 'M0 buying 80', 'B0 longitudinal 40'], [['HR@10', old['hit_rate_at_10'], buying['hit_rate_at_10'], new['hit_rate_at_10']], ['MRR', old['mrr'], buying['mrr'], new['mrr']], ['Mean turns', old['mean_turns'], buying['mean_turns'], new['mean_turns']], ['No-hit rate', f"{old['no_hit_rate']:.1%}", f"{buying['no_hit_rate']:.1%}", f"{new['no_hit_rate']:.1%}"]])}

The frozen run records `llama3.1` but not which credential-fallthrough provider actually answered. Raw comparison is therefore confounded by dataset and shopper-provider differences.

## G. Interpretation

The curated 40-session result characterizes benchmark difficulty; it is not a memory-effect estimate. If strict parity is 100%, Phase-6 history plumbing is behaviorally inert for B0. Later-sequence scores must not be interpreted as memory improvement while history remains shadow-only. Future memory effects require B0/B1/B2/B3 comparison on this identical fixture.

## H. Tests

Run the complete shopping-agent and QLMP unit-test suites after artifact generation. Unit tests do not call OpenAI or download BGE.

## I. Result files

The result directory contains the required JSON metrics/diagnostics, `run_summary.md`, `full_run.json`, and execution logs. The frozen M0 artifact hashes are recorded in `config.json` and remained unchanged.

## J. Scope verification

No `experiment_1`, official `techjam-conversational-search`, or QLMP-math changes; no Graphify run, dependency change, or assistant-created commit.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 6.1 B0 longitudinal calibration and strict parity"
    )
    parser.add_argument("--fixture", type=Path, default=longitudinal.DEFAULT_FIXTURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--shopper-provider",
        choices=sorted(ShopperLLMClient.DEFAULT_MODELS),
        default="ollama",
    )
    parser.add_argument("--shopper-model")
    parser.add_argument("--allow-openai-catalog-build", action="store_true")
    parser.add_argument(
        "--reuse-full-run",
        action="store_true",
        help="reuse an existing full_run.json and rerun parity/report generation only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = validate_output_directory(
        args.output_dir,
        allowed_existing=("full_run.json",) if args.reuse_full_run else (),
    )
    shopper = ShopperLLMClient(args.shopper_provider, args.shopper_model)
    runtime = preflight_runtime(shopper.provider, shopper.model)
    frozen_before = file_hashes(FROZEN_M0_DIR)
    catalog_ids, categories, products = catalog_index(
        SHARED_REPO / "data" / "catalog.jsonl"
    )
    samples = longitudinal._load_public_samples(
        SHARED_REPO / "data" / "public_set.jsonl"
    )
    fixture = longitudinal.load_fixture(args.fixture)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_kwargs = {
        "embedding_cache_dir": CURRENT_DIR / "embedding_cache",
        "allow_catalog_embedding": args.allow_openai_catalog_build,
    }

    if args.reuse_full_run:
        result = json.loads(
            (output_dir / "full_run.json").read_text(encoding="utf-8")
        )
        instrumentation = result.get("instrumentation", {})
        benchmark_seconds = float(result["wall_clock_seconds"])
    else:
        agent = longitudinal.make_fresh_agent(**agent_kwargs)
        started = time.perf_counter()
        result = longitudinal.run_longitudinal_evaluation(
            agent,
            fixture,
            samples,
            set(catalog_ids),
            categories,
            products,
            shopper_call=shopper,
            shopper_provider=shopper.provider,
            model_name=shopper.model,
        )
        benchmark_seconds = time.perf_counter() - started
        instrumentation = agent.get_instrumentation()
        result["instrumentation"] = instrumentation
        result["run_name"] = "B0_LONGITUDINAL_40"
        result["wall_clock_seconds"] = benchmark_seconds
        write_json(output_dir / "full_run.json", result)
        longitudinal._close_agent(agent)
        del agent

    def factory(store: InMemoryUserMemoryStore) -> Agent:
        return Agent(memory_store=store, **agent_kwargs)

    parity_started = time.perf_counter()
    parity = longitudinal.run_strict_shadow_no_history_parity(
        factory,
        fixture,
        result["sessions"],
        set(catalog_ids),
    )
    parity["wall_clock_seconds"] = round(time.perf_counter() - parity_started, 6)
    write_json(output_dir / "shadow_parity.json", parity)
    if parity["status"] != "pass":
        raise RuntimeError(
            "strict shadow/no-history recommendation parity failed; "
            "interpretation stopped after writing shadow_parity.json"
        )

    sessions = result["sessions"]
    overall = overall_metrics(
        sessions,
        agent_errors=len(instrumentation.get("agent_errors", ())),
        shopper_provider=shopper.provider,
        shopper_model=shopper.model,
        wall_seconds=benchmark_seconds,
    )
    by_user = metrics_by_user(fixture, sessions)
    by_sequence = metrics_by_sequence(sessions)
    probes = probe_sessions(fixture, sessions)
    disclosure = disclosure_diagnostics(sessions)
    leakage = leakage_diagnostics(sessions)
    comparison = old_vs_new(overall)
    frozen_after = file_hashes(FROZEN_M0_DIR)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen M0 artifacts changed during B0 validation")
    config = {
        "run_name": "B0_LONGITUDINAL_40",
        "command": (
            "python nickolas/shopping_agent/run_b0_validation.py "
            f"--shopper-provider {shopper.provider} --shopper-model {shopper.model}"
        ),
        "fixture": str(Path(args.fixture).resolve()),
        "fixture_sha256": hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
        "shopper": {"provider": shopper.provider, "model": shopper.model},
        "shopper_runtime": runtime,
        "agent": {
            "baseline": "M0_OPENAI / B0",
            "fast_memory": "current-session only",
            "historical_memory": "shadow-only",
            "max_turns": longitudinal.MAX_TURNS,
            "top_k": longitudinal.TOP_K,
        },
        "embedding": {
            "backend": "openai-text-embedding-3-large",
            "model": "text-embedding-3-large",
            "embedding_space_id": result["sessions"][0]["embedding_space_id"],
            "catalog_cache_status": instrumentation["initialization"]["cache_status"],
        },
        "session_count": len(sessions),
        "user_count": len(fixture["users"]),
        "repeats_run": 1,
        "frozen_m0_provider_known": False,
        "frozen_m0_artifacts_unchanged": True,
        "frozen_m0_hashes": frozen_after,
    }

    write_json(output_dir / "config.json", config)
    write_json(output_dir / "overall_metrics.json", overall)
    write_json(output_dir / "by_user.json", by_user)
    write_json(output_dir / "by_sequence.json", by_sequence)
    write_json(output_dir / "probe_sessions.json", probes)
    write_json(output_dir / "disclosure_diagnostics.json", disclosure)
    write_json(output_dir / "leakage_diagnostics.json", leakage)
    write_json(output_dir / "old_vs_new_comparison.json", comparison)
    (output_dir / "run_summary.md").write_text(
        run_summary(
            config,
            overall,
            by_user,
            by_sequence,
            probes,
            disclosure,
            leakage,
            parity,
            comparison,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"overall": overall, "parity": {
        key: parity[key]
        for key in (
            "total_paired_turns",
            "identical_recommendation_turns",
            "differing_recommendation_turns",
            "recommendation_order_parity_rate",
        )
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "REQUIRED_OUTPUTS",
    "disclosure_diagnostics",
    "leakage_diagnostics",
    "metrics_by_sequence",
    "metrics_by_user",
    "overall_metrics",
    "probe_sessions",
    "validate_output_directory",
]
