from __future__ import annotations

import itertools
import json
import logging
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    BM25_RELATIVE_CUTOFF, DENSE_ABSOLUTE_CUTOFF, DENSE_RELATIVE_CUTOFF,
    MAX_TURNS, RRF_DEPTH, SEED, TOP_K,
)
from .harness import (
    FIELDS, Harness, TurnState, experiment_logger, normalize, percentile_summary,
    rank_metrics, replay_policy, write_csv, write_json,
)


def _groups(rows: Sequence[dict], keys: Sequence[str], value: str) -> dict:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(float(row[value]))
    output = {}
    for group, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        name = " / ".join(f"{key}={val}" for key, val in zip(keys, group))
        output[name] = {
            **percentile_summary(values),
            "uniqueness_rate": round(sum(v == 1 for v in values) / len(values), 6),
            "proportion_le_1": round(sum(v <= 1 for v in values) / len(values), 6),
            "proportion_le_10": round(sum(v <= 10 for v in values) / len(values), 6),
            "proportion_le_100": round(sum(v <= 100 for v in values) / len(values), 6),
        }
    return output


def _save_plot(path: Path, title: str, xlabel: str, ylabel: str, series: dict[str, tuple[Sequence, Sequence]], *, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, (x, y) in series.items():
        ax.plot(x, y, marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=.25)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bar_plot(path: Path, title: str, labels: Sequence[str], values: Sequence[float], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(labels, values, color="#277da1")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _finish(directory: Path, metrics: dict, rows: list[dict], summary: str, logger: logging.Logger, started: float) -> dict:
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "rows.csv", rows)
    (directory / "summary.md").write_text(summary.rstrip() + "\n", encoding="utf-8")
    logger.info("Completed %s in %.2fs", directory.name, metrics["elapsed_seconds"])
    return metrics


def run_01(h: Harness, root_logger: logging.Logger) -> dict:
    name = "experiment_01_constraint_uniqueness"
    directory = h.results_dir / name
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("ORACLE DIAGNOSTIC: beginning constraint uniqueness")
    rows: list[dict] = []
    for sample in h.samples:
        sid = sample["sample_id"]
        target = str(sample["ground_truth"]["parent_asin"])
        category = h.official.coarse_category(h.categories[target])
        constraints = [*h.cards[sid].get("hard_constraints", []), *h.cards[sid].get("soft_preferences", [])][:4]
        for count in range(0, len(constraints) + 1):
            phrases = [category, *constraints[:count]]
            exact, overlap, all_token = h.exact_scores(phrases)
            exact_all = exact == len(phrases)
            overlap_half = overlap >= 0.5 * len(phrases)
            rows.append({
                "sample_id": sid, "scenario_type": sample["scenario_type"], "constraint_count": count,
                "constraint_position": "category_only" if count == 0 else count,
                "exact_phrase_candidates": int(exact_all.sum()),
                "all_token_candidates": int(all_token.sum()),
                "token_overlap_candidates": int(overlap_half.sum()),
                "target_exact_phrase_match": bool(exact_all[h.id_to_idx[target]]),
                "target_all_token_match": bool(all_token[h.id_to_idx[target]]),
                "target_token_overlap_match": bool(overlap_half[h.id_to_idx[target]]),
                "oracle_only": True,
            })
    definitions = ["exact_phrase_candidates", "all_token_candidates", "token_overlap_candidates"]
    aggregates = {}
    for definition in definitions:
        aggregates[definition] = {
            "overall": percentile_summary([r[definition] for r in rows]),
            "uniqueness_rate": round(sum(r[definition] == 1 for r in rows) / len(rows), 6),
            "by_position": _groups(rows, ["constraint_position"], definition),
            "by_scenario": _groups(rows, ["scenario_type", "constraint_position"], definition),
        }
    series = {}
    for definition in definitions:
        x, y = [], []
        for count in range(5):
            vals = [r[definition] for r in rows if r["constraint_count"] == count]
            if vals:
                x.append(count); y.append(float(np.median(vals)))
        series[definition.replace("_candidates", "")] = (x, y)
    _save_plot(directory / "candidate_uniqueness.png", "Oracle candidate counts as constraints accumulate", "Disclosed constraint count", "Median candidates", series, log_y=True)
    metrics = {"experiment": 1, "label": "ORACLE DIAGNOSTIC", "rows": len(rows), "aggregates": aggregates}
    last = [r for r in rows if r["constraint_count"] == 4]
    summary = f"""# Experiment 1 — Constraint uniqueness

> **ORACLE DIAGNOSTIC.** This analysis reconstructs hidden intent cards. It is not an agent-realistic score.

Across {len(h.samples)} sessions, exact normalized phrases leave a median of {np.median([r['exact_phrase_candidates'] for r in last]):.0f} candidates after four constraints; all-token matching leaves {np.median([r['all_token_candidates'] for r in last]):.0f}. Token-overlap is deliberately looser and leaves {np.median([r['token_overlap_candidates'] for r in last]):.0f}.

The raw table reports category-only and 1–4 accumulated constraints for every session. A zero is meaningful: generated constraints can be normalized or synthesized strings that no catalog document contains verbatim.
"""
    return _finish(directory, metrics, rows, summary, logger, started)


def _rankers(h: Harness) -> dict[str, Callable[[TurnState], tuple[np.ndarray, np.ndarray]]]:
    return {
        "latest_message_bm25": lambda s: h.lexical.ranked(s.message, "bm25"),
        "stateful_bm25": lambda s: h.lexical.ranked(s.state_query, "bm25"),
        "exact_phrase": lambda s: h.exact_ranked(s.phrases),
        "field_aware_bm25": lambda s: h.lexical.ranked(s.state_query, "field_aware"),
        "dense": lambda s: h.dense.ranked(s.state_query),
        "hybrid_rrf": lambda s: h.hybrid_ranked(s.state_query),
    }


def run_02(h: Harness, root_logger: logging.Logger) -> dict:
    name = "experiment_02_target_rank_curves"
    directory = h.results_dir / name
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("AGENT-REALISTIC EVALUATION: beginning target rank curves")
    h.dense.preload_queries(state.state_query for state in h.traces)
    rankers = _rankers(h)
    rows: list[dict] = []
    for number, state in enumerate(h.traces, 1):
        row = {
            "sample_id": state.sample_id, "scenario_type": state.scenario_type, "turn": state.turn,
            "disclosed_constraint_count": len(state.disclosed_constraints), "override_applied": state.override_applied,
            "agent_query": state.state_query, "latest_message": state.message, "target_asin": state.target_asin,
            "evaluation_mode": "agent_realistic",
        }
        for method, ranker in rankers.items():
            order, _ = ranker(state)
            loc = np.flatnonzero(order == h.id_to_idx[state.target_asin])
            row[f"{method}_rank"] = int(loc[0] + 1) if loc.size else "not_retrieved"
        rows.append(row)
        if number % 100 == 0:
            logger.info("Ranked %d/%d trace turns", number, len(h.traces))
    policy_metrics, policy_sessions = {}, {}
    for method, ranker in rankers.items():
        sessions, metrics = replay_policy(h, ranker)
        policy_sessions[method] = sessions
        policy_metrics[method] = metrics
    series = {}
    for method in rankers:
        medians = []
        for turn in range(1, MAX_TURNS + 1):
            values = [r[f"{method}_rank"] for r in rows if r["turn"] == turn and isinstance(r[f"{method}_rank"], int)]
            medians.append(float(np.median(values)) if values else np.nan)
        series[method] = (range(1, MAX_TURNS + 1), medians)
    _save_plot(directory / "target_rank_curves.png", "Agent-realistic target-rank curves", "Turn", "Median retrieved target rank", series, log_y=True)
    write_json(directory / "early_termination_sessions.json", policy_sessions)
    best = max(policy_metrics, key=lambda key: policy_metrics[key]["technical_score"])
    metrics = {"experiment": 2, "label": "AGENT-REALISTIC EVALUATION", "rows": len(rows), "early_termination": policy_metrics, "best_method": best}
    summary = f"""# Experiment 2 — Target-rank curves

> **AGENT-REALISTIC EVALUATION.** Queries contain only the current message or category plus constraints disclosed by that turn. Target ASINs and undisclosed intent fields never enter a query.

The best early-termination policy is **{best}**, with technical score **{policy_metrics[best]['technical_score']:.6f}**, HitRate@10 **{policy_metrics[best]['hit_rate_at_10']:.3f}**, and MRR **{policy_metrics[best]['mrr']:.3f}**. Full ten-turn traces are retained for diagnostic curves even when a normal evaluation would have stopped on a hit.

`not_retrieved` means the lexical or truncated-RRF ranker did not retrieve the target; it is not silently converted to rank 50,001. Dense ranks cover the full catalog.
"""
    return _finish(directory, metrics, rows, summary, logger, started)


def _rank_from_scores(h: Harness, scores: np.ndarray, target: str) -> int | None:
    positive = np.flatnonzero(scores > 0)
    target_idx = h.id_to_idx[target]
    if target_idx not in positive:
        return None
    order = positive[np.lexsort((h.lexical.ids[positive], -scores[positive]))]
    return int(np.flatnonzero(order == target_idx)[0] + 1)


def run_03(h: Harness, root_logger: logging.Logger) -> dict:
    name = "experiment_03_field_signal"
    directory = h.results_dir / name
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("ORACLE DIAGNOSTIC: beginning field attribution")
    rows: list[dict] = []
    for sample in h.samples:
        sid, scenario = sample["sample_id"], sample["scenario_type"]
        target = str(sample["ground_truth"]["parent_asin"])
        product = h.product_by_id[target]
        category = h.official.coarse_category(h.categories[target])
        constraints = [*h.cards[sid].get("hard_constraints", []), *h.cards[sid].get("soft_preferences", [])]
        for position, constraint in enumerate(constraints, 1):
            field_hits = [field for field in FIELDS if normalize(constraint) in normalize(h.field_texts[field][h.id_to_idx[target]])]
            for field in FIELDS:
                base = h.lexical.scores(category, f"field:{field}")
                added = h.lexical.scores(f"{category} {constraint}", f"field:{field}")
                base_rank, added_rank = _rank_from_scores(h, base, target), _rank_from_scores(h, added, target)
                rows.append({
                    "sample_id": sid, "scenario_type": scenario, "constraint_position": position,
                    "constraint": constraint, "field": field, "occurs_exactly_in_target_field": field in field_hits,
                    "all_matching_fields": field_hits, "matching_field_count": len(field_hits),
                    "category_only_target_rank": base_rank if base_rank is not None else "not_retrieved",
                    "with_constraint_target_rank": added_rank if added_rank is not None else "not_retrieved",
                    "rank_improvement": (base_rank - added_rank) if base_rank is not None and added_rank is not None else None,
                    "oracle_only": True,
                })
    aggregates = {}
    for field in FIELDS:
        values = [r for r in rows if r["field"] == field]
        improvements = [r["rank_improvement"] for r in values if r["rank_improvement"] is not None]
        ranks = [r["with_constraint_target_rank"] for r in values if isinstance(r["with_constraint_target_rank"], int)]
        aggregates[field] = {
            "constraints": len(values), "exact_match_frequency": round(sum(r["occurs_exactly_in_target_field"] for r in values) / len(values), 6),
            "retrieval_coverage": round(len(ranks) / len(values), 6), "target_rank": percentile_summary(ranks),
            "incremental_rank_improvement": percentile_summary(improvements),
            "by_scenario": {
                scenario: {
                    "exact_match_frequency": round(sum(r["occurs_exactly_in_target_field"] for r in values if r["scenario_type"] == scenario) / max(1, sum(r["scenario_type"] == scenario for r in values)), 6),
                    "with_constraint_rank": percentile_summary([r["with_constraint_target_rank"] for r in values if r["scenario_type"] == scenario and isinstance(r["with_constraint_target_rank"], int)]),
                }
                for scenario in sorted({r["scenario_type"] for r in values})
            },
        }
    overlap_constraints = len({(r["sample_id"], r["constraint_position"]) for r in rows if r["matching_field_count"] > 1})
    _bar_plot(directory / "field_coverage.png", "Exact constraint occurrence in target fields", list(FIELDS), [aggregates[f]["exact_match_frequency"] for f in FIELDS], "Fraction of constraints")
    metrics = {"experiment": 3, "label": "ORACLE DIAGNOSTIC", "rows": len(rows), "field_metrics": aggregates, "multi_field_constraints": overlap_constraints}
    top_field = max(FIELDS, key=lambda f: aggregates[f]["exact_match_frequency"])
    summary = f"""# Experiment 3 — Field signal

> **ORACLE DIAGNOSTIC.** Field attribution inspects the target product and is never used as an agent-visible query feature.

The highest exact constraint coverage is in **{top_field}** ({aggregates[top_field]['exact_match_frequency']:.1%}). {overlap_constraints} constraints occur in multiple target fields; the raw output preserves every overlap instead of assigning one forced origin.

Single-field ranks compare category-only retrieval with category-plus-constraint retrieval. Price is represented explicitly as a field, including evaluator-generated `budget around $…` text.
"""
    return _finish(directory, metrics, rows, summary, logger, started)


def _rule_matches(value: str) -> list[str]:
    lowered = value.lower()
    rules = []
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered): rules.append("budget")
    if any(v in lowered for v in ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")): rules.append("material")
    if any(v in lowered for v in ("color", "black", "white", "blue", "red", "pink", "green")): rules.append("color")
    if any(v in lowered for v in ("size", "sizing", "width", "wide", "narrow")): rules.append("size")
    if any(v in lowered for v in ("department", "style", "fit", "sleeve", "neck")): rules.append("style")
    if any(v in lowered for v in ("hiking", "running", "gym", "winter", "outdoor", "work")): rules.append("use_case")
    return rules or ["feature"]


def run_04(h: Harness, root_logger: logging.Logger) -> dict:
    name = "experiment_04_constraint_classification"
    directory = h.results_dir / name
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("ORACLE DIAGNOSTIC: beginning evaluator classifier analysis")
    first_disclosure: dict[tuple[str, str], int | None] = {}
    for state in h.traces:
        for value in state.disclosed_constraints:
            first_disclosure.setdefault((state.sample_id, value), state.turn)
    rows: list[dict] = []
    for sample in h.samples:
        sid = sample["sample_id"]
        hard = list(h.cards[sid].get("hard_constraints", []))
        soft = list(h.cards[sid].get("soft_preferences", []))
        for kind, values in (("hard", hard), ("soft", soft)):
            for position, value in enumerate(values, 1):
                matches = _rule_matches(str(value))
                assigned = h.official.classify_constraint(str(value))
                full_color = bool(h.official.COLOR_RE.search(str(value)))
                rows.append({
                    "sample_id": sid, "scenario_type": sample["scenario_type"], "constraint_kind": kind,
                    "position_within_kind": position, "constraint": value, "assigned_type": assigned,
                    "matching_rules_in_order": matches, "multi_rule": len(matches) > 1,
                    "precedence_winner": matches[0], "disclosure_turn": first_disclosure.get((sid, str(value))),
                    "full_color_regex_match": full_color, "color_classification_mismatch": full_color != (assigned == "color"),
                    "revealed_by_category": False, "revealed_by_brand": False, "revealed_by_other": True,
                    "oracle_only": True,
                })
    counts = Counter(r["assigned_type"] for r in rows)
    scenario_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows: scenario_counts[row["scenario_type"]][row["assigned_type"]] += 1
    metrics = {
        "experiment": 4, "label": "ORACLE DIAGNOSTIC", "rows": len(rows), "type_counts": dict(sorted(counts.items())),
        "scenario_type_counts": {k: dict(sorted(v.items())) for k, v in sorted(scenario_counts.items())},
        "hard_soft_position_counts": dict(Counter(f"{r['constraint_kind']}_{r['position_within_kind']}:{r['assigned_type']}" for r in rows)),
        "disclosure_turn_counts": dict(Counter(str(r["disclosure_turn"]) for r in rows)),
        "category_unrevealed": len(rows), "brand_unrevealed": len(rows), "other_revealed": len(rows),
        "color_classification_mismatches": sum(r["color_classification_mismatch"] for r in rows),
        "multi_rule_precedence_resolutions": sum(r["multi_rule"] for r in rows),
    }
    _bar_plot(directory / "classification_distribution.png", "Ordered evaluator constraint classes", list(counts), list(counts.values()), "Constraints")
    summary = f"""# Experiment 4 — Constraint classification distribution

> **ORACLE DIAGNOSTIC.** This applies the evaluator's exact ordered classifier to reconstructed hidden constraints.

The classifier assigns {len(rows)} generated constraints across {len(counts)} emitted types. `other` can reveal all {len(rows)} constraints, while `category` and `brand` reveal none because the evaluator's classifier never emits those labels. There are {metrics['multi_rule_precedence_resolutions']} multi-rule strings resolved by precedence and {metrics['color_classification_mismatches']} mismatches between the broader color regex used during card construction and the narrower ordered color classifier.
"""
    return _finish(directory, metrics, rows, summary, logger, started)


def run_05(h: Harness, root_logger: logging.Logger) -> dict:
    name = "experiment_05_candidate_set_shrinkage"
    directory = h.results_dir / name
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("ORACLE DIAGNOSTIC: beginning candidate-set shrinkage")
    dense_queries = []
    for sample in h.samples:
        sid = sample["sample_id"]
        target = str(sample["ground_truth"]["parent_asin"])
        category = h.official.coarse_category(h.categories[target])
        constraints = [*h.cards[sid].get("hard_constraints", []), *h.cards[sid].get("soft_preferences", [])][:4]
        dense_queries.extend(" ".join([category, *constraints[:count]]) for count in range(len(constraints) + 1))
    h.dense.preload_queries(dense_queries)
    rows: list[dict] = []
    for number, sample in enumerate(h.samples, 1):
        sid = sample["sample_id"]
        target = str(sample["ground_truth"]["parent_asin"])
        category = h.official.coarse_category(h.categories[target])
        constraints = [*h.cards[sid].get("hard_constraints", []), *h.cards[sid].get("soft_preferences", [])][:4]
        for count in range(len(constraints) + 1):
            phrases = [category, *constraints[:count]]
            query = " ".join(phrases)
            exact, overlap, all_token = h.exact_scores(phrases)
            exact_count = int(np.count_nonzero(exact == len(phrases)))
            bm25_count = h.lexical.cutoff_count(query, "field_aware")
            dense_count = h.dense.cutoff_count(query)
            _, hybrid_scores = h.hybrid_ranked(query)
            hybrid_count = int(np.count_nonzero(hybrid_scores >= hybrid_scores[0] * .50)) if hybrid_scores.size else 0
            rows.append({
                "sample_id": sid, "scenario_type": sample["scenario_type"], "constraint_count": count,
                "exact_phrase_candidates": exact_count, "all_token_candidates": int(all_token.sum()),
                "bm25_cutoff_candidates": bm25_count, "dense_nn_candidates": dense_count,
                "hybrid_candidates": hybrid_count, "oracle_only": True,
            })
        if number % 25 == 0: logger.info("Processed %d/%d sessions", number, len(h.samples))
    definitions = ["exact_phrase_candidates", "all_token_candidates", "bm25_cutoff_candidates", "dense_nn_candidates", "hybrid_candidates"]
    aggregates = {}
    for definition in definitions:
        aggregates[definition] = {"by_position": _groups(rows, ["constraint_count"], definition), "by_scenario": _groups(rows, ["scenario_type", "constraint_count"], definition)}
        for threshold in (1, 10, 100):
            aggregates[definition][f"proportion_le_{threshold}"] = round(sum(r[definition] <= threshold for r in rows) / len(rows), 6)
    series = {}
    for definition in definitions:
        x = range(5)
        y = [float(np.median([r[definition] for r in rows if r["constraint_count"] == count])) for count in x]
        series[definition.replace("_candidates", "")] = (list(x), y)
    _save_plot(directory / "shrinkage_curves.png", "Candidate-set shrinkage", "Accumulated constraint count", "Median candidates", series, log_y=True)
    metrics = {
        "experiment": 5, "label": "ORACLE DIAGNOSTIC", "rows": len(rows), "aggregates": aggregates,
        "cutoffs": {"bm25_relative_to_top": BM25_RELATIVE_CUTOFF, "dense_absolute": DENSE_ABSOLUTE_CUTOFF, "dense_relative_to_top": DENSE_RELATIVE_CUTOFF, "hybrid_relative_to_top": .50},
    }
    summary = """# Experiment 5 — Candidate-set shrinkage

> **ORACLE DIAGNOSTIC.** The accumulated hidden constraints are used only to measure dataset structure. This is separate from agent-realistic ranking.

Hard exact-phrase and all-token filters are reported separately from BM25, dense-neighbor, and hybrid soft candidate definitions. Soft counts use fixed, documented score cutoffs; they are diagnostic candidate sets, not claims that low-score items are impossible matches.
"""
    return _finish(directory, metrics, rows, summary, logger, started)


def _metric_for_subset(session_rows: list[dict], subset: set[str]) -> dict:
    selected = [r for r in session_rows if r["sample_id"] in subset]
    result = rank_metrics(selected)
    result["scenario_metrics"] = {
        scenario: rank_metrics([r for r in selected if r["scenario_type"] == scenario])
        for scenario in sorted({r["scenario_type"] for r in selected})
    }
    return result


def run_06(h: Harness, root_logger: logging.Logger) -> dict:
    name = "experiment_06_slate_width_counterfactuals"
    directory = h.results_dir / name
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("AGENT-REALISTIC EVALUATION: beginning slate counterfactuals")
    h.dense.preload_queries(state.state_query for state in h.traces)
    rng = random.Random(SEED)
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for sample in h.samples: by_scenario[sample["scenario_type"]].append(sample["sample_id"])
    calibration: set[str] = set()
    for scenario, ids in by_scenario.items():
        shuffled = sorted(ids); rng.shuffle(shuffled)
        calibration.update(shuffled[:max(1, round(len(shuffled) * .30))])
    evaluation = {s["sample_id"] for s in h.samples} - calibration

    def hybrid(state: TurnState): return h.hybrid_ranked(state.state_query)
    fixed_sessions: dict[int, list[dict]] = {}
    fixed_metrics: dict[str, dict] = {}
    for width in (1, 3, 5, 10):
        sessions, _ = replay_policy(h, hybrid, width=width)
        fixed_sessions[width] = sessions
        fixed_metrics[str(width)] = {
            "calibration": _metric_for_subset(sessions, calibration),
            "evaluation": _metric_for_subset(sessions, evaluation),
        }

    trace_lookup = h.trace_by_session()
    def replay_adaptive(high: float, medium: float, inclusion: float, abstain: float) -> list[dict]:
        sessions = []
        for sid, turns in trace_lookup.items():
            first_hit = best_rank = None
            for state in turns:
                order, scores = hybrid(state)
                if not scores.size: continue
                top = float(scores[0]); second = float(scores[1]) if scores.size > 1 else 0.0
                norm_top = min(1.0, top * 61.0 / 2.0)
                margin = (top - second) / max(abs(top), 1e-12)
                confidence = .65 * norm_top + .35 * max(0.0, margin)
                if confidence < abstain: width = 0
                elif confidence >= high: width = 1
                elif confidence >= medium: width = 3
                else: width = 10
                if width:
                    relative_count = int(np.count_nonzero(scores[:10] >= top * inclusion))
                    width = min(width, max(1, relative_count))
                shown = order[:width]
                loc = np.flatnonzero(shown == h.id_to_idx[state.target_asin])
                if state.override_applied and loc.size:
                    first_hit, best_rank = state.turn, int(loc[0] + 1); break
            sessions.append({"sample_id": sid, "scenario_type": turns[0].scenario_type, "hit": first_hit is not None, "first_hit_turn": first_hit, "best_rank": best_rank, "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank})
        return sessions

    candidates = []
    for high, medium, inclusion, abstain in itertools.product((.55, .65, .75), (.25, .35, .45), (.0, .50, .70), (0.0, .15)):
        if high <= medium: continue
        sessions = replay_adaptive(high, medium, inclusion, abstain)
        calibration_metrics = _metric_for_subset(sessions, calibration)
        candidates.append((calibration_metrics["technical_score"], high, medium, inclusion, abstain, sessions, calibration_metrics))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3], row[4]))
    _, high, medium, inclusion, abstain, adaptive_sessions, calibration_metrics = candidates[0]
    adaptive_eval = _metric_for_subset(adaptive_sessions, evaluation)
    baseline = {r["sample_id"]: r for r in fixed_sessions[10]}
    adaptive = {r["sample_id"]: r for r in adaptive_sessions}
    rows = []
    all_ids = calibration | evaluation
    for sid in sorted(all_ids):
        base, actual = baseline[sid], adaptive[sid]
        rows.append({
            "sample_id": sid, "scenario_type": base["scenario_type"], "split": "calibration" if sid in calibration else "evaluation",
            "top10_hit": base["hit"], "adaptive_hit": actual["hit"],
            "hit_gained": actual["hit"] and not base["hit"], "hit_lost": base["hit"] and not actual["hit"],
            "top10_first_hit_turn": base["first_hit_turn"], "adaptive_first_hit_turn": actual["first_hit_turn"],
            "turn_delay": (actual["first_hit_turn"] - base["first_hit_turn"]) if actual["first_hit_turn"] is not None and base["first_hit_turn"] is not None else None,
            "top10_rank": base["best_rank"], "adaptive_rank": actual["best_rank"],
        })
    heldout_rows = [r for r in rows if r["split"] == "evaluation"]
    comparisons = {
        "hits_gained": sum(r["hit_gained"] for r in heldout_rows), "hits_lost": sum(r["hit_lost"] for r in heldout_rows),
        "delayed_conversions": sum((r["turn_delay"] or 0) > 0 for r in heldout_rows),
        "rank_changes": sum(r["top10_rank"] != r["adaptive_rank"] for r in heldout_rows),
        "misses_created": sum(r["hit_lost"] for r in heldout_rows),
        "by_scenario": {
            scenario: {
                "hits_gained": sum(r["hit_gained"] for r in heldout_rows if r["scenario_type"] == scenario),
                "hits_lost": sum(r["hit_lost"] for r in heldout_rows if r["scenario_type"] == scenario),
                "delayed_conversions": sum((r["turn_delay"] or 0) > 0 for r in heldout_rows if r["scenario_type"] == scenario),
            }
            for scenario in sorted({r["scenario_type"] for r in heldout_rows})
        },
        "top10_first_hit_turn_counts": dict(Counter(str(r["top10_first_hit_turn"]) for r in heldout_rows)),
        "adaptive_first_hit_turn_counts": dict(Counter(str(r["adaptive_first_hit_turn"]) for r in heldout_rows)),
    }
    eval_scores = [fixed_metrics[str(w)]["evaluation"]["technical_score"] for w in (1,3,5,10)] + [adaptive_eval["technical_score"]]
    _bar_plot(directory / "slate_comparison.png", "Held-out technical score by slate policy", ["width 1","width 3","width 5","width 10","adaptive"], eval_scores, "Technical score")
    metrics = {
        "experiment": 6, "label": "AGENT-REALISTIC EVALUATION", "seed": SEED,
        "split": {"calibration_count": len(calibration), "evaluation_count": len(evaluation), "calibration_ids": sorted(calibration), "evaluation_ids": sorted(evaluation)},
        "fixed_widths": fixed_metrics,
        "adaptive": {"thresholds": {"high": high, "medium": medium, "relative_inclusion": inclusion, "abstain_below": abstain}, "calibration": calibration_metrics, "evaluation": adaptive_eval},
        "comparison_to_top10_on_evaluation": comparisons,
    }
    summary = f"""# Experiment 6 — Slate-width counterfactuals

> **AGENT-REALISTIC EVALUATION.** Policies use hybrid scores from information disclosed by each simulated turn. The target is used only for scoring.

Thresholds were selected on a fixed, scenario-stratified {len(calibration)}-session calibration split and evaluated once on {len(evaluation)} held-out sessions. The chosen adaptive policy scored **{adaptive_eval['technical_score']:.6f}** versus **{fixed_metrics['10']['evaluation']['technical_score']:.6f}** for full Top-10. It gained {comparisons['hits_gained']} hits, lost {comparisons['hits_lost']}, and delayed {comparisons['delayed_conversions']} conversions relative to Top-10.

The adaptive confidence combines a normalized RRF top score and the top-two margin; relative candidate inclusion and optional low-confidence abstention were tuned only on calibration sessions.
"""
    return _finish(directory, metrics, rows, summary, logger, started)


RUNNERS = {1: run_01, 2: run_02, 3: run_03, 4: run_04, 5: run_05, 6: run_06}
