from __future__ import annotations

import importlib.util
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiment_07_residual_failure_analysis import load_frozen_split
from .experiment_11_candidate_agent import CleanFTSAgent
from .harness import Harness, experiment_logger, sha256, write_csv, write_json


EXPERIMENT_NUMBER = 11
EXPERIMENT_SLUG = "clean_fts5_candidate"
EXPERIMENT_DIRECTORY = f"experiment_{EXPERIMENT_NUMBER:02d}_{EXPERIMENT_SLUG}"
CURRENT_CONTROL = "current_submission_agent"
YANG_CONTROL = "yang_experiment_1_original"
CONFIGURATIONS = {
    "clean_specific_global_pagination": {"question_policy": "specific", "pagination_mode": "global"},
    "clean_specific_query_pagination": {"question_policy": "specific", "pagination_mode": "query"},
    "clean_other_query_pagination": {"question_policy": "other", "pagination_mode": "query"},
    "clean_specific_no_pagination": {"question_policy": "specific", "pagination_mode": "none"},
}


def _load_agent_class(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def _metrics(official: Any, sessions: Sequence[dict], selected_ids: set[str]) -> dict:
    selected = [row for row in sessions if row["sample_id"] in selected_ids]
    result = official.metric_summary(selected)
    efficiency = max(0.0, min(1.0, (11.0 - float(result["mttc"])) / 10.0))
    result["efficiency"] = round(efficiency, 6)
    result["technical_score"] = round(
        0.50 * result["hit_rate_at_10"] + 0.30 * result["mrr"] + 0.20 * efficiency,
        6,
    )
    result["scenario_metrics"] = {
        scenario: official.metric_summary([row for row in selected if row["scenario_type"] == scenario])
        for scenario in sorted({row["scenario_type"] for row in selected})
    }
    return result


def _metric_sets(official: Any, sessions: Sequence[dict], split_sets: Mapping[str, set[str]]) -> dict:
    return {name: _metrics(official, sessions, ids) for name, ids in split_sets.items()}


def _comparison(baseline: Sequence[dict], candidate: Sequence[dict], ids: set[str]) -> dict:
    left = {row["sample_id"]: row for row in baseline if row["sample_id"] in ids}
    right = {row["sample_id"]: row for row in candidate if row["sample_id"] in ids}
    rescues = sorted(sid for sid in left if not left[sid]["hit"] and right[sid]["hit"])
    regressions = sorted(sid for sid in left if left[sid]["hit"] and not right[sid]["hit"])
    accelerated = sorted(
        sid for sid in left
        if left[sid]["hit"] and right[sid]["hit"] and right[sid]["first_hit_turn"] < left[sid]["first_hit_turn"]
    )
    delayed = sorted(
        sid for sid in left
        if left[sid]["hit"] and right[sid]["hit"] and right[sid]["first_hit_turn"] > left[sid]["first_hit_turn"]
    )
    return {
        "rescues": len(rescues),
        "rescue_sample_ids": rescues,
        "regressions": len(regressions),
        "regression_sample_ids": regressions,
        "accelerated": len(accelerated),
        "accelerated_sample_ids": accelerated,
        "delayed": len(delayed),
        "delayed_sample_ids": delayed,
    }


def _evaluate(official: Any, agent: Any, h: Harness) -> dict:
    return official.evaluate(agent, h.samples, set(h.ids), h.categories, h.product_by_id)


def _latency(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)
    at = lambda quantile: ordered[int(round((len(ordered) - 1) * quantile))]
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 6),
        "p50_ms": round(at(0.50), 6),
        "p95_ms": round(at(0.95), 6),
        "p99_ms": round(at(0.99), 6),
        "max_ms": round(max(values), 6),
    }


def _plot(path: Path, metrics: Mapping[str, Mapping[str, dict]], selected: str) -> None:
    methods = [CURRENT_CONTROL, YANG_CONTROL, *CONFIGURATIONS]
    labels = ["Current", "Yang", "Global/specific", "Query/specific", "Query/other", "No page/specific"]
    calibration = [metrics[method]["calibration"]["technical_score"] for method in methods]
    evaluation = [metrics[method]["evaluation"]["technical_score"] for method in methods]
    x = list(range(len(methods)))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar([value - 0.2 for value in x], calibration, width=0.4, label="Calibration")
    ax.bar([value + 0.2 for value in x], evaluation, width=0.4, label="Evaluation")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("TechnicalScore")
    ax.set_ylim(0.75, max(calibration + evaluation) + 0.02)
    ax.set_title(f"Experiment 11 retrospective comparison (calibration selected: {selected})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_11(h: Harness, root_logger: logging.Logger) -> dict:
    directory = h.results_dir / EXPERIMENT_DIRECTORY
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("RETROSPECTIVE AGENT EVALUATION: clean FTS5 candidate")

    split_path = h.results_dir / "experiment_06_slate_width_counterfactuals" / "metrics.json"
    calibration, evaluation, split = load_frozen_split(split_path, [sample["sample_id"] for sample in h.samples])
    all_ids = {sample["sample_id"] for sample in h.samples}
    split_sets = {"full": all_ids, "calibration": calibration, "evaluation": evaluation}

    current_source = h.repo / "techjam-conversational-search" / "starter" / "agent.py"
    yang_source = h.repo / "experiment_1" / "agent.py"
    CurrentAgent = _load_agent_class(current_source, "experiment11_current_agent")
    YangAgent = _load_agent_class(yang_source, "experiment11_yang_agent")

    logger.info("Evaluating current submission control")
    current_result = _evaluate(h.official, CurrentAgent(h.catalog_path), h)
    logger.info("Evaluating Yang Experiment 1 control")
    yang_result = _evaluate(h.official, YangAgent(h.catalog_path), h)

    candidate = CleanFTSAgent(h.catalog_path)
    candidate_results: dict[str, dict] = {}
    candidate_latency: dict[str, dict] = {}
    for name, config in CONFIGURATIONS.items():
        logger.info("Evaluating clean candidate configuration %s", name)
        candidate.configure(**config)
        result = _evaluate(h.official, candidate, h)
        candidate_results[name] = result
        candidate_latency[name] = _latency(candidate.response_latencies_ms)

    results = {CURRENT_CONTROL: current_result, YANG_CONTROL: yang_result, **candidate_results}
    sessions = {method: result["sessions"] for method, result in results.items()}
    method_metrics = {
        method: _metric_sets(h.official, method_sessions, split_sets)
        for method, method_sessions in sessions.items()
    }
    selected = sorted(
        CONFIGURATIONS,
        key=lambda name: (
            -method_metrics[name]["calibration"]["technical_score"],
            -method_metrics[name]["calibration"]["mrr"],
            name,
        ),
    )[0]
    comparisons = {
        method: {
            split_name: _comparison(sessions[CURRENT_CONTROL], method_sessions, ids)
            for split_name, ids in split_sets.items()
        }
        for method, method_sessions in sessions.items()
        if method != CURRENT_CONTROL
    }
    selected_eval = method_metrics[selected]["evaluation"]
    current_eval = method_metrics[CURRENT_CONTROL]["evaluation"]
    selected_comparison = comparisons[selected]["evaluation"]
    gates = {
        "technical_score_beats_current": selected_eval["technical_score"] > current_eval["technical_score"],
        "hit_rate_not_lower": selected_eval["hit_rate_at_10"] >= current_eval["hit_rate_at_10"],
        "mrr_not_lower": selected_eval["mrr"] >= current_eval["mrr"],
        "regressions_do_not_exceed_rescues": selected_comparison["regressions"] <= selected_comparison["rescues"],
    }

    expected_current_path = h.repo / "techjam-conversational-search" / "current_agent_results.json"
    expected_current = json.loads(expected_current_path.read_text(encoding="utf-8"))
    current_full = method_metrics[CURRENT_CONTROL]["full"]
    current_reproduction = {
        "passed": all(
            current_full[key] == expected_current[expected_key]
            for key, expected_key in (
                ("hit_rate_at_10", "hit_rate_at_10"),
                ("mrr", "mrr"),
                ("mttc", "mttc"),
                ("technical_score", "recommended_technical_score"),
            )
        ),
        "expected_path": str(expected_current_path.relative_to(h.repo)),
        "expected_sha256": sha256(expected_current_path),
    }
    if not current_reproduction["passed"]:
        raise RuntimeError("Current submission control did not reproduce current_agent_results.json")

    session_rows = []
    by_method = {method: {row["sample_id"]: row for row in values} for method, values in sessions.items()}
    for sample in h.samples:
        sid = sample["sample_id"]
        row = {
            "sample_id": sid,
            "split": "calibration" if sid in calibration else "evaluation",
            "oracle_scenario_type": sample["scenario_type"],
        }
        for method in results:
            outcome = by_method[method][sid]
            row[f"{method}_hit"] = outcome["hit"]
            row[f"{method}_first_hit_turn"] = outcome["first_hit_turn"]
            row[f"{method}_best_rank"] = outcome["best_rank"]
        session_rows.append(row)

    source_path = Path(__file__).resolve()
    candidate_source = Path(__file__).with_name("experiment_11_candidate_agent.py")
    metrics = {
        "experiment": EXPERIMENT_NUMBER,
        "slug": EXPERIMENT_SLUG,
        "label": "RETROSPECTIVE PUBLIC-SET AGENT EVALUATION",
        "method_metrics": method_metrics,
        "configuration": {
            "candidates": CONFIGURATIONS,
            "popularity_weight": 0.02,
            "field_weights": {"title": 6.0, "categories": 4.0, "features": 2.5, "details": 2.5, "store": 1.5, "description": 1.0},
            "candidate_pool": 1_000,
            "correct_override_invalidation": True,
            "stale_preference_boost": False,
            "deterministic_asin_tie_break": True,
        },
        "split": split,
        "selection": {
            "selected_on_calibration": selected,
            "evaluation_metrics": selected_eval,
            "comparison_to_current": selected_comparison,
            "diagnostic_gates": gates,
            "all_diagnostic_gates_passed": all(gates.values()),
            "production_promotion_authorized": False,
            "reason": "The public set and evaluation partition were inspected during the preceding audit; private or newly generated validation is required.",
            "starter_agent_modified": False,
        },
        "comparisons_to_current": comparisons,
        "latency": candidate_latency,
        "index_build_seconds": candidate.index_build_seconds,
        "current_control_reproduction": current_reproduction,
        "hashes": {
            "catalog": h.catalog_hash,
            "public_set": h.public_hash,
            "runner_source": {"path": str(source_path.relative_to(h.repo)), "sha256": sha256(source_path)},
            "candidate_source": {"path": str(candidate_source.relative_to(h.repo)), "sha256": sha256(candidate_source)},
            "current_agent_source": {"path": str(current_source.relative_to(h.repo)), "sha256": sha256(current_source)},
            "yang_agent_source": {"path": str(yang_source.relative_to(h.repo)), "sha256": sha256(yang_source)},
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    _plot(directory / "agent_comparison.png", method_metrics, selected)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "sessions.json", sessions)
    write_csv(directory / "rows.csv", session_rows)
    write_json(directory / "comparisons.json", comparisons)
    write_json(directory / "latency.json", candidate_latency)
    (directory / "source_snapshot.py").write_bytes(source_path.read_bytes())
    (directory / "candidate_agent_snapshot.py").write_bytes(candidate_source.read_bytes())

    selected_full = method_metrics[selected]["full"]
    yang_full = method_metrics[YANG_CONTROL]["full"]
    summary = f"""# Experiment 11 — Clean FTS5 candidate

> **RETROSPECTIVE PUBLIC-SET EVALUATION.** The candidate uses only observable dialogue messages and catalog metadata. The preceding investigation inspected the public set and its frozen partition, so these numbers are diagnostic rather than an unbiased promotion test.

The current submission control reproduced its saved official score exactly at **{current_full['technical_score']:.6f}**. Yang's original agent scored **{yang_full['technical_score']:.6f}**. Calibration selected **{selected}**; it scored **{selected_full['technical_score']:.6f}** on all 200 sessions and **{selected_eval['technical_score']:.6f}** on the 140-session evaluation partition.

The clean candidate removes stale-preference boosting, scopes override removal to the revoked preference, uses robust case-insensitive parsing, and applies deterministic ASIN tie-breaking. All diagnostic score, Hit@10, MRR, and rescue/regression gates passed: **{all(gates.values())}**. The starter agent was not modified because private or newly generated validation is still required.
"""
    (directory / "summary.md").write_text(summary, encoding="utf-8")
    logger.info(
        "Completed %s in %.2fs; selected=%s full=%.6f evaluation=%.6f",
        EXPERIMENT_DIRECTORY,
        metrics["elapsed_seconds"],
        selected,
        selected_full["technical_score"],
        selected_eval["technical_score"],
    )
    return metrics

