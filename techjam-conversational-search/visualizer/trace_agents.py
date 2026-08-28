from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAX_TURNS = 10
TOP_K = 10


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_catalog(path: Path) -> tuple[set[str], dict[str, list[str]], dict[str, dict[str, Any]]]:
    identifiers: set[str] = set()
    categories: dict[str, list[str]] = {}
    products: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            identifiers.add(parent_asin)
            categories[parent_asin] = [str(value) for value in product.get("categories") or []]
            products[parent_asin] = product
    return identifiers, categories, products


def normalize_recommendations(payload: object, catalog_ids: set[str]) -> list[str]:
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= TOP_K:
            break
    return result


def product_preview(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": str(product.get("parent_asin", "")),
        "title": product.get("title") or "Untitled product",
        "store": product.get("store"),
        "price": product.get("price"),
        "average_rating": product.get("average_rating"),
        "rating_number": product.get("rating_number"),
        "categories": product.get("categories") or [],
    }


def metric_summary(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    if not sessions:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
            "efficiency": 0.0,
            "technical_score": 0.0,
        }
    count = len(sessions)
    hit_rate = sum(int(item["hit"]) for item in sessions) / count
    mrr = sum(float(item["reciprocal_rank"]) for item in sessions) / count
    mttc = sum(
        item["first_hit_turn"] if item["first_hit_turn"] is not None else MAX_TURNS + 1
        for item in sessions
    ) / count
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": count,
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(technical_score, 6),
    }


def summarize_agent_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    overall = metric_summary(sessions)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "scenario_metrics": {
            scenario: metric_summary(rows)
            for scenario, rows in sorted(grouped.items())
        },
    }


def safe_delta(candidate: float | int | None, baseline: float | int | None) -> float | int | None:
    if candidate is None or baseline is None:
        return None
    return round(candidate - baseline, 6)


def comparison_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "hit_rate_at_10": safe_delta(candidate["hit_rate_at_10"], baseline["hit_rate_at_10"]),
        "mrr": safe_delta(candidate["mrr"], baseline["mrr"]),
        # Lower MTTC is better, so improvement is baseline minus candidate.
        "mttc_improvement": safe_delta(baseline["mttc"], candidate["mttc"]),
        "technical_score": safe_delta(candidate["technical_score"], baseline["technical_score"]),
    }


def run_session(
    *,
    agent: object,
    label: str,
    sample: dict[str, Any],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict[str, Any]],
    evaluator: object,
) -> dict[str, Any]:
    session_id = f"trace_{label}_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = evaluator.materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = evaluator.initial_message(
        effective_sample,
        evaluator.coarse_category(categories.get(target, [])),
        disclosed,
    )
    hit_turn: int | None = None
    best_rank: int | None = None
    turns: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for turn in range(1, MAX_TURNS + 1):
        error: str | None = None
        try:
            response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception as exc:  # Match the official evaluator's fail-closed behavior.
            error = f"{type(exc).__name__}: {exc}"
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            error = error or "Invalid response: expected an object with a string message"
            response = {"message": "", "ask_attribute": None, "recommendations": []}

        usage = response.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                total_prompt_tokens += prompt_tokens
            if isinstance(completion_tokens, int) and completion_tokens >= 0:
                total_completion_tokens += completion_tokens

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        target_rank = ranked.index(target) + 1 if target in ranked else None
        eligible_hit = bool(override_applied and target_rank is not None)
        recommendation_rows = []
        for rank, parent_asin in enumerate(ranked, start=1):
            recommendation_rows.append({
                "rank": rank,
                "is_target": parent_asin == target,
                **product_preview(products[parent_asin]),
            })

        debug = response.get("debug") if isinstance(response.get("debug"), dict) else None
        if debug is None and hasattr(agent, "sessions"):
            agent_sessions = getattr(agent, "sessions")
            agent_state = agent_sessions.get(session_id) if isinstance(agent_sessions, dict) else None
            if isinstance(agent_state, dict):
                debug = {
                    "mode": getattr(agent, "mode", type(agent).__name__),
                    "category": agent_state.get("category"),
                    "constraints": list(agent_state.get("constraints") or []),
                }
        turn_row: dict[str, Any] = {
            "turn": turn,
            "user_message": user_message,
            "disclosed_constraints_before_response": sorted(disclosed),
            "override_applied": override_applied,
            "agent_message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": recommendation_rows,
            "target_rank": target_rank,
            "eligible_hit": eligible_hit,
            "usage": usage if isinstance(usage, dict) else None,
            "debug": debug,
            "error": error,
            "next_user_message": None,
        }
        turns.append(turn_row)

        if eligible_hit:
            hit_turn = turn
            best_rank = target_rank
            break
        if turn == MAX_TURNS:
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = evaluator.customer_reply(
                effective_sample,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )
        turn_row["next_user_message"] = user_message

    return {
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else round(1.0 / best_rank, 6),
        "reported_token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "turns": turns,
    }


def session_status(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    if candidate["hit"] and not baseline["hit"]:
        return "candidate_win"
    if baseline["hit"] and not candidate["hit"]:
        return "baseline_win"
    if candidate["hit"] and baseline["hit"]:
        candidate_key = (candidate["first_hit_turn"], candidate["best_rank"])
        baseline_key = (baseline["first_hit_turn"], baseline["best_rank"])
        if candidate_key < baseline_key:
            return "candidate_better"
        if candidate_key > baseline_key:
            return "baseline_better"
    return "same"


def choose_example(sessions: list[dict[str, Any]]) -> str | None:
    priority = ("candidate_win", "candidate_better", "baseline_win", "baseline_better")
    for status in priority:
        match = next((row for row in sessions if row["comparison"]["status"] == status), None)
        if match:
            return str(match["sample_id"])
    shared_hits = [row for row in sessions if row["baseline"]["hit"] and row["candidate"]["hit"]]
    if shared_hits:
        best = min(
            shared_hits,
            key=lambda row: (
                row["candidate"]["first_hit_turn"],
                row["candidate"]["best_rank"],
                row["sample_id"],
            ),
        )
        return str(best["sample_id"])
    return str(sessions[0]["sample_id"]) if sessions else None


def dataset_summary(samples: list[dict[str, Any]], catalog_count: int) -> dict[str, Any]:
    return {
        "session_count": len(samples),
        "catalog_product_count": catalog_count,
        "scenarios": dict(sorted(Counter(str(row.get("scenario_type", "unknown")) for row in samples).items())),
        "difficulty_buckets": dict(sorted(Counter(str(row.get("difficulty_bucket", "unknown")) for row in samples).items())),
        "category_buckets": dict(sorted(Counter(str(row.get("category_bucket", "unknown")) for row in samples).items())),
    }


def main() -> None:
    visualizer_dir = Path(__file__).resolve().parent
    project_dir = visualizer_dir.parent
    workspace_dir = project_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate a turn-by-turn JSON comparison of the supplied baseline and the current agent."
    )
    parser.add_argument("--catalog", type=Path, default=project_dir / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=project_dir / "data" / "public_set.jsonl")
    parser.add_argument(
        "--baseline-agent",
        type=Path,
        default=workspace_dir / "techjam-conversational-search-participant-kit" / "starter" / "agent.py",
    )
    parser.add_argument("--candidate-agent", type=Path, default=project_dir / "starter" / "agent.py")
    parser.add_argument("--output", type=Path, default=visualizer_dir / "comparison.json")
    parser.add_argument("--sample-id", help="Generate just one named public session instead of the full dataset")
    parser.add_argument("--limit", type=int, help="Generate only the first N sessions")
    args = parser.parse_args()

    for required in (args.catalog, args.dataset, args.baseline_agent, args.candidate_agent):
        if not required.exists():
            parser.error(f"Required file does not exist: {required}")

    # The official evaluator helpers define the deterministic conversation simulator.
    sys.path.insert(0, str(project_dir))
    evaluator = load_module(project_dir / "evaluator" / "local_evaluator.py", "trace_official_evaluator")
    baseline_module = load_module(args.baseline_agent.resolve(), "trace_baseline_agent")
    candidate_module = load_module(args.candidate_agent.resolve(), "trace_candidate_agent")

    print(f"Loading catalog: {args.catalog}")
    catalog_ids, categories, products = load_catalog(args.catalog)
    samples = load_jsonl(args.dataset)
    if args.sample_id:
        samples = [row for row in samples if row.get("sample_id") == args.sample_id]
        if not samples:
            parser.error(f"Unknown sample id: {args.sample_id}")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        samples = samples[: args.limit]

    print(f"Initializing baseline agent: {args.baseline_agent}")
    baseline_agent = baseline_module.Agent(args.catalog)
    print(f"Initializing current agent: {args.candidate_agent}")
    candidate_agent = candidate_module.Agent(args.catalog)

    session_rows: list[dict[str, Any]] = []
    baseline_outcomes: list[dict[str, Any]] = []
    candidate_outcomes: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        target_asin = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = evaluator.materialize_hidden_fields(sample, products)
        baseline = run_session(
            agent=baseline_agent,
            label="baseline",
            sample=sample,
            catalog_ids=catalog_ids,
            categories=categories,
            products=products,
            evaluator=evaluator,
        )
        candidate = run_session(
            agent=candidate_agent,
            label="candidate",
            sample=sample,
            catalog_ids=catalog_ids,
            categories=categories,
            products=products,
            evaluator=evaluator,
        )
        baseline_outcomes.append(baseline)
        candidate_outcomes.append(candidate)
        status = session_status(baseline, candidate)
        session_rows.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "category_bucket": sample.get("category_bucket"),
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "user_profile": sample.get("user_profile") or {},
            "target": product_preview(products[target_asin]),
            "simulator": {"intent_card": intent_card, "behavior": behavior},
            "baseline": baseline,
            "candidate": candidate,
            "comparison": {
                "status": status,
                "hit_delta": int(candidate["hit"]) - int(baseline["hit"]),
                "turn_delta": safe_delta(candidate["first_hit_turn"], baseline["first_hit_turn"]),
                "rank_delta": safe_delta(candidate["best_rank"], baseline["best_rank"]),
            },
        })
        if index == len(samples) or index % 25 == 0:
            print(f"Traced {index}/{len(samples)} sessions")

    baseline_metrics = summarize_agent_sessions(baseline_outcomes)
    candidate_metrics = summarize_agent_sessions(candidate_outcomes)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "dataset": str(args.dataset.resolve()),
            "catalog": str(args.catalog.resolve()),
            "baseline_agent": {
                "label": "Provided agent",
                "path": str(args.baseline_agent.resolve()),
            },
            "candidate_agent": {
                "label": "Current system",
                "path": str(args.candidate_agent.resolve()),
            },
            "max_turns": MAX_TURNS,
            "top_k": TOP_K,
        },
        "dataset_summary": dataset_summary(samples, len(catalog_ids)),
        "aggregate": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": comparison_delta(candidate_metrics, baseline_metrics),
            "session_status_counts": dict(sorted(Counter(row["comparison"]["status"] for row in session_rows).items())),
        },
        "example_session_id": choose_example(session_rows),
        "sessions": session_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / (1024 * 1024):.2f} MiB)")
    example = next(
        (row for row in session_rows if row["sample_id"] == payload["example_session_id"]),
        None,
    )
    example_output = args.output.with_name(f"{args.output.stem}.example.json")
    if example is not None:
        example_output.write_text(
            json.dumps({
                "schema_version": payload["schema_version"],
                "generated_at": payload["generated_at"],
                "metadata": payload["metadata"],
                "aggregate": payload["aggregate"],
                "example": example,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote example {example_output}")
    print(json.dumps({
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta": payload["aggregate"]["delta"],
        "example_session_id": payload["example_session_id"],
        "example_output": str(example_output) if example is not None else None,
    }, indent=2))


if __name__ == "__main__":
    main()
