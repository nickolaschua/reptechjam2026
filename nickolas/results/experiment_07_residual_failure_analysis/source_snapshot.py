from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import MAX_TURNS, MODEL_ID, RRF_DEPTH, RRF_K, TOP_K
from .harness import (
    FIELDS,
    TEXT_FIELDS,
    Harness,
    TurnState,
    experiment_logger,
    normalize,
    percentile_summary,
    product_field_text,
    rank_metrics,
    sha256,
    tokens,
    write_csv,
    write_json,
)


EXPERIMENT_NUMBER = 7
EXPERIMENT_SLUG = "residual_failure_analysis"
EXPERIMENT_DIRECTORY = f"experiment_{EXPERIMENT_NUMBER:02d}_{EXPERIMENT_SLUG}"

EXACT_METHOD = "exact_only"
CASCADE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "exact_stateful_bm25_rrf": ("exact", "bm25"),
    "exact_field_aware_bm25_rrf": ("exact", "field_aware"),
    "exact_dense_rrf": ("exact", "dense"),
    "exact_generic_field_dense_rrf": ("exact", "bm25", "field_aware", "dense"),
}
METHODS = (EXACT_METHOD, *CASCADE_COMPONENTS)
FAILURE_TAGS = (
    "no_exact_match",
    "ambiguous_exact_evidence",
    "ranking_failure",
    "cross_field_partial_match",
    "normalization_problem",
    "dialogue_state_problem",
    "insufficient_information",
    "semantic_retrieval_opportunity",
)
EXPECTED_EXACT_BASELINE = {
    "sample_count": 200,
    "hit_rate_at_10": 0.930000,
    "mrr": 0.623058,
    "mttc": 2.750000,
    "efficiency": 0.825000,
    "technical_score": 0.816917,
}
REQUIRED_ARTIFACTS = (
    "summary.md",
    "metrics.json",
    "rows.csv",
    "sessions.json",
    "baseline_reproduction.json",
    "residual_turns.csv",
    "residual_turns.json",
    "hard_failures.csv",
    "hard_failures.json",
    "weak_successes.csv",
    "weak_successes.json",
    "failure_category_counts.csv",
    "failure_category_counts.json",
    "rescue_by_category.csv",
    "rescue_by_category.json",
    "rescue_comparisons.csv",
    "rescue_comparisons.json",
    "rank_distributions.csv",
    "rank_distributions.json",
    "rescue_comparison.png",
    "rank_distributions.png",
    "source_snapshot.py",
    "source_snapshot.json",
    "run.log",
)


@dataclass(frozen=True)
class RetrievalInput:
    """The complete and deliberately narrow input visible to every ranker."""

    category: str
    active_constraints: tuple[str, ...]

    @property
    def phrases(self) -> tuple[str, ...]:
        return (self.category, *self.active_constraints)

    @property
    def query(self) -> str:
        return " ".join(self.phrases).strip()


@dataclass(frozen=True)
class FrozenTurnRanking:
    """Oracle-free retrieval output, frozen before labels are joined."""

    retrieval_input: RetrievalInput
    fallback_activated: bool
    fallback_reasons: tuple[str, ...]
    all_phrases_exact_candidate_count: int
    highest_exact_match_count: int
    highest_exact_match_tier_count: int
    component_top_10: Mapping[str, tuple[int, ...]]
    method_top_10: Mapping[str, tuple[int, ...]]


def robust_normalize(text: object) -> str:
    """NFKC/casefold normalization with Unicode punctuation mapped to spaces."""

    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    value = "".join(" " if unicodedata.category(char).startswith("P") else char for char in value)
    return " ".join(value.split())


def is_normalization_problem(current_exact: bool, robust_exact: bool) -> bool:
    return not current_exact and robust_exact


def fallback_reasons(
    active_constraints: Sequence[str],
    all_phrases_exact_candidate_count: int,
    highest_exact_match_tier_count: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not active_constraints:
        reasons.append("no_active_constraint")
    if all_phrases_exact_candidate_count == 0:
        reasons.append("no_catalog_product_matches_all_active_phrases")
    if highest_exact_match_tier_count > TOP_K:
        reasons.append("highest_exact_match_tier_exceeds_top_10")
    return tuple(reasons)


def deterministic_rrf(
    rankings: Sequence[Sequence[int] | np.ndarray],
    ids: Sequence[str] | np.ndarray,
    *,
    k: int = RRF_K,
    depth: int = RRF_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight RRF with ascending-ASIN tie-breaking."""

    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[int] = set()
        for rank, raw_idx in enumerate(ranking[:depth], 1):
            idx = int(raw_idx)
            if idx in seen:
                continue
            seen.add(idx)
            scores[idx] += 1.0 / (k + rank)
    if not scores:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    candidates = np.fromiter(scores, dtype=np.int64)
    values = np.fromiter((scores[int(idx)] for idx in candidates), dtype=np.float64)
    id_array = np.asarray(ids)
    order = np.lexsort((id_array[candidates], -values))
    return candidates[order], values[order]


def corrected_active_evidence(
    active: Sequence[str], disclosed: Sequence[str], superseded: Sequence[str]
) -> tuple[str, ...]:
    """Remove obsolete evidence and restore any disclosed, non-obsolete evidence."""

    obsolete = set(superseded)
    corrected: list[str] = []
    for value in (*active, *disclosed):
        if value and value not in obsolete and value not in corrected:
            corrected.append(value)
    return tuple(corrected)


def failure_category_counts(rows: Sequence[dict]) -> list[dict]:
    """Count non-exclusive labels; one session can contribute to many rows."""

    output = []
    for tag in FAILURE_TAGS:
        hard = sum(tag in row.get("diagnostic_failure_tags", []) and row.get("diagnostic_residual_type") == "hard_failure" for row in rows)
        weak = sum(tag in row.get("diagnostic_failure_tags", []) and row.get("diagnostic_residual_type") == "weak_success" for row in rows)
        output.append(
            {
                "failure_category": tag,
                "hard_failure_sessions": hard,
                "weak_success_sessions": weak,
                "session_count": hard + weak,
                "non_exclusive": True,
            }
        )
    return output


def load_frozen_split(path: Path, public_ids: Iterable[str]) -> tuple[set[str], set[str], dict]:
    if not path.exists():
        raise RuntimeError(f"Frozen Experiment 6 split is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("split")
    if not isinstance(split, dict):
        raise RuntimeError("Experiment 6 metrics do not contain a split")
    calibration = set(split.get("calibration_ids", []))
    evaluation = set(split.get("evaluation_ids", []))
    current = set(public_ids)
    if len(calibration) != 60 or len(evaluation) != 140:
        raise RuntimeError(f"Frozen split size mismatch: {len(calibration)}/{len(evaluation)}")
    if calibration & evaluation or calibration | evaluation != current:
        raise RuntimeError("Experiment 6 split does not partition the current public set")
    if split.get("calibration_count") != 60 or split.get("evaluation_count") != 140:
        raise RuntimeError("Experiment 6 split count fields are inconsistent")
    return calibration, evaluation, {
        "source": str(path),
        "source_sha256": sha256(path),
        "calibration_count": 60,
        "evaluation_count": 140,
        "calibration_ids": sorted(calibration),
        "evaluation_ids": sorted(evaluation),
        "validated_partition": True,
    }


def assert_exact_baseline_identity(actual: Sequence[dict], expected: Sequence[dict]) -> None:
    fields = ("sample_id", "scenario_type", "hit", "first_hit_turn", "best_rank", "reciprocal_rank")
    canonical_actual = [{field: row.get(field) for field in fields} for row in actual]
    canonical_expected = [{field: row.get(field) for field in fields} for row in expected]
    if canonical_actual != canonical_expected:
        mismatches = [
            i for i, pair in enumerate(zip(canonical_actual, canonical_expected)) if pair[0] != pair[1]
        ]
        if len(canonical_actual) != len(canonical_expected):
            mismatches.append(min(len(canonical_actual), len(canonical_expected)))
        raise RuntimeError(f"Exact baseline session identity failed at rows {mismatches[:10]}")


def validate_artifact_payloads(
    metrics: dict,
    rows: Sequence[dict],
    sessions: dict,
    residual_turns: Sequence[dict],
    hard_failures: Sequence[dict],
    weak_successes: Sequence[dict],
) -> None:
    if metrics.get("experiment") != EXPERIMENT_NUMBER or set(metrics.get("method_metrics", {})) != set(METHODS):
        raise RuntimeError("Experiment 7 metrics schema is incomplete")
    if len(rows) != 200 * MAX_TURNS:
        raise RuntimeError(f"rows.csv schema/count mismatch: {len(rows)}")
    if set(sessions) != set(METHODS) or any(len(value) != 200 for value in sessions.values()):
        raise RuntimeError("sessions.json schema/count mismatch")
    expected_residual_turns = (len(hard_failures) + len(weak_successes)) * MAX_TURNS
    if len(residual_turns) != expected_residual_turns:
        raise RuntimeError("Residual trace artifact does not contain all ten turns per residual session")
    if any("oracle_target_asin" not in row or "diagnostic_failure_tags" not in row for row in (*hard_failures, *weak_successes)):
        raise RuntimeError("Residual artifact schema is missing oracle/diagnostic fields")


def _rank_input(h: Harness, retrieval_input: RetrievalInput) -> FrozenTurnRanking:
    exact_counts, _, _ = h.exact_scores(retrieval_input.phrases)
    exact_order, _ = h.exact_ranked(retrieval_input.phrases)
    highest = int(exact_counts.max()) if exact_counts.size else 0
    highest_tier_count = int(np.count_nonzero(exact_counts == highest)) if highest else 0
    all_exact_count = int(np.count_nonzero(exact_counts == len(retrieval_input.phrases)))
    reasons = fallback_reasons(retrieval_input.active_constraints, all_exact_count, highest_tier_count)

    components: dict[str, np.ndarray] = {"exact": exact_order[:RRF_DEPTH]}
    components["bm25"], _ = h.lexical.ranked(retrieval_input.query, "bm25", RRF_DEPTH)
    components["field_aware"], _ = h.lexical.ranked(retrieval_input.query, "field_aware", RRF_DEPTH)
    components["dense"], _ = h.dense.ranked(retrieval_input.query, RRF_DEPTH)
    method_orders: dict[str, tuple[int, ...]] = {EXACT_METHOD: tuple(int(v) for v in exact_order[:TOP_K])}
    for method, names in CASCADE_COMPONENTS.items():
        if not reasons:
            order = exact_order
        else:
            order, _ = deterministic_rrf([components[name] for name in names], h.ids)
        method_orders[method] = tuple(int(v) for v in order[:TOP_K])
    return FrozenTurnRanking(
        retrieval_input=retrieval_input,
        fallback_activated=bool(reasons),
        fallback_reasons=reasons,
        all_phrases_exact_candidate_count=all_exact_count,
        highest_exact_match_count=highest,
        highest_exact_match_tier_count=highest_tier_count,
        component_top_10={name: tuple(int(v) for v in order[:TOP_K]) for name, order in components.items()},
        method_top_10=method_orders,
    )


def freeze_agent_rankings(h: Harness, logger: logging.Logger) -> list[FrozenTurnRanking]:
    inputs = [RetrievalInput(state.category, tuple(state.active_constraints)) for state in h.traces]
    h.dense.preload_queries(item.query for item in inputs)
    cache: dict[RetrievalInput, FrozenTurnRanking] = {}
    frozen: list[FrozenTurnRanking] = []
    for number, retrieval_input in enumerate(inputs, 1):
        ranking = cache.get(retrieval_input)
        if ranking is None:
            ranking = _rank_input(h, retrieval_input)
            cache[retrieval_input] = ranking
        frozen.append(ranking)
        if number % 100 == 0:
            logger.info("Frozen %d/%d oracle-free turn rankings", number, len(inputs))
    if len(frozen) != len(h.traces):
        raise RuntimeError("Agent-realistic ranking freeze is incomplete")
    for ranking in frozen:
        exact = ranking.method_top_10[EXACT_METHOD]
        if not ranking.fallback_activated and any(ranking.method_top_10[method] != exact for method in CASCADE_COMPONENTS):
            raise RuntimeError("A cascade changed exact ordering while fallback was inactive")
    return frozen


def _rank_in_top(order: Sequence[int], target_index: int) -> int | None:
    try:
        return order.index(target_index) + 1  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        values = list(order)
        return values.index(target_index) + 1 if target_index in values else None


def _replay_frozen(h: Harness, frozen: Sequence[FrozenTurnRanking], method: str) -> list[dict]:
    sessions: list[dict] = []
    start = 0
    for sample in h.samples:
        turns = h.traces[start : start + MAX_TURNS]
        rankings = frozen[start : start + MAX_TURNS]
        start += MAX_TURNS
        first_hit = best_rank = None
        for state, ranking in zip(turns, rankings):
            target_index = h.id_to_idx[state.target_asin]
            rank = _rank_in_top(ranking.method_top_10[method], target_index)
            if state.override_applied and rank is not None:
                first_hit, best_rank = state.turn, rank
                break
        sessions.append(
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "hit": first_hit is not None,
                "first_hit_turn": first_hit,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )
    return sessions


def _metrics_for_subset(session_rows: Sequence[dict], subset: set[str]) -> dict:
    selected = [row for row in session_rows if row["sample_id"] in subset]
    result = rank_metrics(selected)
    result["scenario_metrics"] = {
        scenario: rank_metrics([row for row in selected if row["scenario_type"] == scenario])
        for scenario in sorted({row["scenario_type"] for row in selected})
    }
    return result


def _comparison(
    baseline: Sequence[dict], candidate: Sequence[dict], subset: set[str], weak_ids: set[str]
) -> dict:
    base = {row["sample_id"]: row for row in baseline if row["sample_id"] in subset}
    actual = {row["sample_id"]: row for row in candidate if row["sample_id"] in subset}

    def summarize(ids: set[str]) -> dict:
        hard = {sid for sid in ids if not base[sid]["hit"]}
        rescues = {sid for sid in hard if actual[sid]["hit"]}
        regressions = {sid for sid in ids if base[sid]["hit"] and not actual[sid]["hit"]}
        delays = [
            int(actual[sid]["first_hit_turn"] - base[sid]["first_hit_turn"])
            for sid in ids
            if base[sid]["hit"] and actual[sid]["hit"]
        ]
        return {
            "session_count": len(ids),
            "baseline_hard_failures": len(hard),
            "hard_failure_rescues": len(rescues),
            "rescue_percentage": round(len(rescues) / len(hard), 6) if hard else 0.0,
            "rescued_sample_ids": sorted(rescues),
            "regressions": len(regressions),
            "regressed_sample_ids": sorted(regressions),
            "conversion_delays": {
                "comparable_sessions": len(delays),
                "delayed": sum(value > 0 for value in delays),
                "accelerated": sum(value < 0 for value in delays),
                "unchanged": sum(value == 0 for value in delays),
                "mean_turn_delta": round(statistics.fmean(delays), 6) if delays else None,
                "turn_delta_distribution": dict(sorted(Counter(str(value) for value in delays).items())),
            },
            "weak_success_sessions": len(weak_ids & ids),
        }

    result = summarize(set(base))
    result["by_scenario"] = {
        scenario: summarize({sid for sid, row in base.items() if row["scenario_type"] == scenario})
        for scenario in sorted({row["scenario_type"] for row in base.values()})
    }
    return result


def _observable_disclosed(h: Harness, state: TurnState) -> tuple[str, ...]:
    values: list[str] = []
    if state.scenario_type == "intent_override":
        old = str(h.behaviors[state.sample_id].get("override", {}).get("old_value", ""))
        if old:
            values.append(old)
    for value in state.disclosed_constraints:
        if value and value not in values:
            values.append(value)
    for value in state.active_constraints:
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _superseded(h: Harness, state: TurnState) -> tuple[str, ...]:
    if state.scenario_type != "intent_override" or not state.override_applied:
        return ()
    old = str(h.behaviors[state.sample_id].get("override", {}).get("old_value", ""))
    return (old,) if old else ()


def _attribute_evidence(h: Harness, state: TurnState, evidence: Sequence[str]) -> list[dict]:
    product = h.product_by_id[state.target_asin]
    output: list[dict] = []
    for value in evidence:
        query_tokens = set(tokens(value))
        fields: dict[str, dict] = {}
        for field_name in (*TEXT_FIELDS, "price"):
            field_text = product_field_text(product, field_name)
            field_tokens = set(tokens(field_text))
            matched = sorted(query_tokens & field_tokens)
            fields[field_name] = {
                "source": "synthetic_price" if field_name == "price" else "catalog_metadata",
                "current_exact": bool(normalize(value) and normalize(value) in normalize(field_text)),
                "nfkc_casefold_punctuation_exact": bool(
                    robust_normalize(value) and robust_normalize(value) in robust_normalize(field_text)
                ),
                "matched_token_count": len(matched),
                "evidence_token_count": len(query_tokens),
                "token_coverage": round(len(matched) / len(query_tokens), 6) if query_tokens else 0.0,
                "matched_tokens": matched,
            }
        output.append({"evidence": value, "fields": fields})
    return output


def _diagnostic_exact_rank(h: Harness, category: str, active: Sequence[str], target: str) -> int | None:
    order, _ = h.exact_ranked((category, *active))
    return _rank_in_top(tuple(int(v) for v in order[:TOP_K]), h.id_to_idx[target])


def _diagnose_residual(
    h: Harness,
    state: TurnState,
    ranking: FrozenTurnRanking,
    residual_type: str,
    method_sessions: Mapping[str, Mapping[str, dict]],
    sample: dict,
) -> dict:
    target_index = h.id_to_idx[state.target_asin]
    evidence = _observable_disclosed(h, state)
    superseded = _superseded(h, state)
    corrected = corrected_active_evidence(state.active_constraints, evidence, superseded)
    attribution = _attribute_evidence(h, state, evidence)
    exact_counts, _, _ = h.exact_scores(ranking.retrieval_input.phrases)
    target_exact_count = int(exact_counts[target_index])
    highest_count = int(exact_counts.max())
    highest_tier_count = int(np.count_nonzero(exact_counts == highest_count))
    exact_top_rank = _rank_in_top(ranking.method_top_10[EXACT_METHOD], target_index)
    exact_full_rank = h.exact_rank(ranking.retrieval_input.phrases, state.target_asin)

    tags: list[str] = []
    if target_exact_count < len(ranking.retrieval_input.phrases):
        tags.append("no_exact_match")
    if target_exact_count == highest_count and highest_tier_count > TOP_K:
        tags.append("ambiguous_exact_evidence")
    if exact_full_rank is not None and exact_full_rank >= 6:
        tags.append("ranking_failure")

    evidence_tokens = set(token for value in evidence for token in tokens(value))
    target_field_token_sets = {
        field_name: set(tokens(product_field_text(h.product_by_id[state.target_asin], field_name)))
        for field_name in TEXT_FIELDS
    }
    target_metadata_tokens = set().union(*target_field_token_sets.values()) if target_field_token_sets else set()
    at_least_half = bool(evidence_tokens) and len(evidence_tokens & target_metadata_tokens) / len(evidence_tokens) >= 0.5
    complete_across_fields = bool(evidence_tokens) and evidence_tokens <= target_metadata_tokens and not any(
        evidence_tokens <= field_tokens for field_tokens in target_field_token_sets.values()
    )
    no_exact_text_field = not any(
        report["current_exact"]
        for item in attribution
        for field_name, report in item["fields"].items()
        if field_name in TEXT_FIELDS
    )
    if at_least_half or (complete_across_fields and no_exact_text_field):
        tags.append("cross_field_partial_match")

    normalization_strings = []
    for item in attribution:
        current_any = any(report["current_exact"] for report in item["fields"].values())
        robust_any = any(report["nfkc_casefold_punctuation_exact"] for report in item["fields"].values())
        if is_normalization_problem(current_any, robust_any):
            normalization_strings.append(item["evidence"])
    if normalization_strings:
        tags.append("normalization_problem")

    missing_non_superseded = [value for value in evidence if value not in superseded and value not in state.active_constraints]
    superseded_active = [value for value in superseded if value in state.active_constraints]
    corrected_rank = _diagnostic_exact_rank(h, state.category, corrected, state.target_asin)
    corrected_rescue = exact_top_rank is None and corrected_rank is not None
    if missing_non_superseded or superseded_active or corrected_rescue:
        tags.append("dialogue_state_problem")

    oracle_constraints = list(
        dict.fromkeys(
            [
                *h.cards[state.sample_id].get("hard_constraints", []),
                *h.cards[state.sample_id].get("soft_preferences", []),
            ]
        )
    )
    undisclosed = [value for value in oracle_constraints if value not in evidence and value not in superseded]
    undisclosed_rescues: list[dict] = []
    if exact_top_rank is None:
        for value in undisclosed:
            counterfactual = (*corrected, value)
            rank = _diagnostic_exact_rank(h, state.category, counterfactual, state.target_asin)
            if rank is not None:
                undisclosed_rescues.append({"oracle_constraint": value, "diagnostic_rank": rank})
        if undisclosed_rescues:
            tags.append("insufficient_information")

    complete_active = corrected_active_evidence(corrected, oracle_constraints, superseded)
    complete_rank = _diagnostic_exact_rank(h, state.category, complete_active, state.target_asin)
    intrinsic_ambiguity = complete_rank is None

    component_ranks = {
        component: _rank_in_top(order, target_index) for component, order in ranking.component_top_10.items()
    }
    semantic_opportunity = exact_top_rank is None and component_ranks["dense"] is not None
    dense_only = semantic_opportunity and component_ranks["bm25"] is None and component_ranks["field_aware"] is None
    if semantic_opportunity:
        tags.append("semantic_retrieval_opportunity")

    split = "calibration" if method_sessions[EXACT_METHOD][state.sample_id]["split"] == "calibration" else "evaluation"
    route_outcomes = {
        method: {
            "hit": method_sessions[method][state.sample_id]["hit"],
            "first_hit_turn": method_sessions[method][state.sample_id]["first_hit_turn"],
            "best_rank": method_sessions[method][state.sample_id]["best_rank"],
        }
        for method in METHODS
    }
    return {
        "sample_id": state.sample_id,
        "split": split,
        "oracle_scenario_type": state.scenario_type,
        "oracle_target_asin": state.target_asin,
        "diagnostic_residual_type": residual_type,
        "diagnostic_turn": state.turn,
        "retrieval_category": state.category,
        "active_evidence": list(state.active_constraints),
        "disclosed_evidence": list(evidence),
        "superseded_evidence": list(superseded),
        "diagnostic_corrected_active_evidence": list(corrected),
        "fallback_activated": ranking.fallback_activated,
        "fallback_reasons": list(ranking.fallback_reasons),
        "diagnostic_exact_target_phrase_matches": target_exact_count,
        "diagnostic_exact_highest_phrase_matches": highest_count,
        "exact_all_phrases_candidate_count": ranking.all_phrases_exact_candidate_count,
        "exact_highest_tier_candidate_count": highest_tier_count,
        "diagnostic_exact_target_full_rank": exact_full_rank,
        "diagnostic_component_top_10_ranks": component_ranks,
        "diagnostic_dense_only_rescue": dense_only,
        "diagnostic_normalization_strings": normalization_strings,
        "diagnostic_missing_non_superseded_evidence": missing_non_superseded,
        "diagnostic_superseded_evidence_still_active": superseded_active,
        "diagnostic_corrected_state_rank": corrected_rank,
        "diagnostic_corrected_state_rescue": corrected_rescue,
        "oracle_undisclosed_constraints": undisclosed,
        "diagnostic_undisclosed_constraint_rescues": undisclosed_rescues,
        "diagnostic_complete_oracle_card_rank": complete_rank,
        "diagnostic_intrinsic_ambiguity": intrinsic_ambiguity,
        "diagnostic_cross_field_half_token_coverage": at_least_half,
        "diagnostic_complete_token_coverage_spans_fields": complete_across_fields,
        "diagnostic_evidence_attribution": attribution,
        "diagnostic_failure_tags": [tag for tag in FAILURE_TAGS if tag in tags],
        "diagnostic_route_outcomes": route_outcomes,
        "audit_user_profile": sample.get("user_profile", {}),
    }


def _joined_rows(h: Harness, frozen: Sequence[FrozenTurnRanking], calibration: set[str]) -> list[dict]:
    rows: list[dict] = []
    for state, ranking in zip(h.traces, frozen):
        target_idx = h.id_to_idx[state.target_asin]
        row = {
            "sample_id": state.sample_id,
            "split": "calibration" if state.sample_id in calibration else "evaluation",
            "oracle_scenario_type": state.scenario_type,
            "turn": state.turn,
            "retrieval_category": ranking.retrieval_input.category,
            "active_evidence": list(ranking.retrieval_input.active_constraints),
            "agent_query": ranking.retrieval_input.query,
            "fallback_activated": ranking.fallback_activated,
            "fallback_reasons": list(ranking.fallback_reasons),
            "exact_all_phrases_candidate_count": ranking.all_phrases_exact_candidate_count,
            "exact_highest_match_count": ranking.highest_exact_match_count,
            "exact_highest_tier_candidate_count": ranking.highest_exact_match_tier_count,
            "oracle_target_asin": state.target_asin,
        }
        for method in METHODS:
            order = ranking.method_top_10[method]
            row[f"{method}_top_10"] = [h.ids[idx] for idx in order]
            row[f"diagnostic_{method}_target_rank"] = _rank_in_top(order, target_idx)
        rows.append(row)
    return rows


def _residual_turn_rows(
    h: Harness,
    frozen: Sequence[FrozenTurnRanking],
    residual_types: Mapping[str, str],
    calibration: set[str],
) -> list[dict]:
    rows: list[dict] = []
    histories: dict[str, list[str]] = defaultdict(list)
    for state, ranking in zip(h.traces, frozen):
        histories[state.sample_id].append(state.message)
        if state.sample_id not in residual_types:
            continue
        target_idx = h.id_to_idx[state.target_asin]
        superseded = _superseded(h, state)
        row = {
            "sample_id": state.sample_id,
            "split": "calibration" if state.sample_id in calibration else "evaluation",
            "oracle_scenario_type": state.scenario_type,
            "diagnostic_residual_type": residual_types[state.sample_id],
            "turn": state.turn,
            "message": state.message,
            "message_history": list(histories[state.sample_id]),
            "previous_other_questions": [
                {"after_turn": turn, "ask_attribute": "other"} for turn in range(1, state.turn)
            ],
            "retrieval_category": state.category,
            "active_evidence": list(state.active_constraints),
            "disclosed_evidence": list(_observable_disclosed(h, state)),
            "superseded_evidence": list(superseded),
            "override_applied": state.override_applied,
            "fallback_activated": ranking.fallback_activated,
            "fallback_reasons": list(ranking.fallback_reasons),
            "exact_all_phrases_candidate_count": ranking.all_phrases_exact_candidate_count,
            "exact_highest_match_count": ranking.highest_exact_match_count,
            "exact_highest_tier_candidate_count": ranking.highest_exact_match_tier_count,
            "exact_top_10_candidates": [h.ids[idx] for idx in ranking.method_top_10[EXACT_METHOD]],
            "oracle_target_asin": state.target_asin,
            "diagnostic_exact_target_rank": _rank_in_top(ranking.method_top_10[EXACT_METHOD], target_idx),
        }
        for method in CASCADE_COMPONENTS:
            row[f"{method}_top_10_candidates"] = [h.ids[idx] for idx in ranking.method_top_10[method]]
            row[f"diagnostic_{method}_target_rank"] = _rank_in_top(ranking.method_top_10[method], target_idx)
        rows.append(row)
    return rows


def _same_turn_weak_improvements(
    h: Harness,
    frozen: Sequence[FrozenTurnRanking],
    weak_sessions: Sequence[dict],
    calibration: set[str],
) -> tuple[list[dict], dict[str, dict]]:
    by_key = {(state.sample_id, state.turn): ranking for state, ranking in zip(h.traces, frozen)}
    target_by_sid = {state.sample_id: state.target_asin for state in h.traces}
    scenario_by_sid = {state.sample_id: state.scenario_type for state in h.traces}
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for method in METHODS:
        for session in weak_sessions:
            sid = session["sample_id"]
            turn = session["first_hit_turn"]
            ranking = by_key[(sid, turn)]
            target = h.id_to_idx[target_by_sid[sid]]
            candidate_rank = _rank_in_top(ranking.method_top_10[method], target)
            capped = candidate_rank if candidate_rank is not None else TOP_K + 1
            improvement = int(session["best_rank"] - capped)
            rows.append(
                {
                    "sample_id": sid,
                    "method": method,
                    "split": "calibration" if sid in calibration else "evaluation",
                    "oracle_scenario_type": scenario_by_sid[sid],
                    "original_first_hit_turn": turn,
                    "exact_original_rank": session["best_rank"],
                    "candidate_same_turn_rank": candidate_rank,
                    "candidate_capped_rank": capped,
                    "capped_rank_improvement": improvement,
                }
            )

        def summarize(selected_rows: Sequence[dict]) -> dict:
            improvements = [row["capped_rank_improvement"] for row in selected_rows]
            return {
                **percentile_summary(improvements),
                "improved": sum(value > 0 for value in improvements),
                "unchanged": sum(value == 0 for value in improvements),
                "regressed": sum(value < 0 for value in improvements),
                "rank_cap_for_misses": TOP_K + 1,
            }

        method_rows = [row for row in rows if row["method"] == method]
        summaries[method] = {}
        for split_name in ("full", "calibration", "evaluation"):
            split_rows = method_rows if split_name == "full" else [row for row in method_rows if row["split"] == split_name]
            split_summary = summarize(split_rows)
            split_summary["scenario_metrics"] = {
                scenario: summarize([row for row in split_rows if row["oracle_scenario_type"] == scenario])
                for scenario in sorted({row["oracle_scenario_type"] for row in split_rows})
            }
            summaries[method][split_name] = split_summary
    return rows, summaries


def _rank_distribution_rows(session_sets: Mapping[str, Sequence[dict]], split_sets: Mapping[str, set[str]]) -> list[dict]:
    output: list[dict] = []
    for method, sessions in session_sets.items():
        for split, ids in split_sets.items():
            selected = [row for row in sessions if row["sample_id"] in ids]
            counts = Counter("miss" if row["best_rank"] is None else str(row["best_rank"]) for row in selected)
            for label in [str(rank) for rank in range(1, TOP_K + 1)] + ["miss"]:
                output.append(
                    {
                        "method": method,
                        "split": split,
                        "rank": label,
                        "session_count": counts[label],
                        "proportion": round(counts[label] / len(selected), 6) if selected else 0.0,
                    }
                )
    return output


def _write_charts(
    directory: Path,
    comparisons: Mapping[str, Mapping[str, dict]],
    rank_distributions: Sequence[dict],
) -> None:
    labels = list(CASCADE_COMPONENTS)
    rescues = [comparisons[method]["evaluation"]["hard_failure_rescues"] for method in labels]
    regressions = [comparisons[method]["evaluation"]["regressions"] for method in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 0.2, rescues, 0.4, label="Hard-failure rescues", color="#2a9d8f")
    ax.bar(x + 0.2, regressions, 0.4, label="Regressions", color="#e76f51")
    ax.set_title("Held-out rescue/regression comparison")
    ax.set_ylabel("Sessions")
    ax.set_xticks(x, [label.replace("exact_", "").replace("_rrf", "").replace("_", "\n") for label in labels])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "rescue_comparison.png", dpi=150)
    plt.close(fig)

    eval_rows = [row for row in rank_distributions if row["split"] == "evaluation"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rank_labels = [str(rank) for rank in range(1, TOP_K + 1)] + ["miss"]
    for method in METHODS:
        values = [next(row["proportion"] for row in eval_rows if row["method"] == method and row["rank"] == label) for label in rank_labels]
        ax.plot(rank_labels, values, marker="o", linewidth=1.5, label=method)
    ax.set_title("Held-out first-hit rank distributions")
    ax.set_xlabel("Rank at conversion")
    ax.set_ylabel("Session proportion")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / "rank_distributions.png", dpi=150)
    plt.close(fig)


def _serialize_sessions(session_sets: Mapping[str, Sequence[dict]], calibration: set[str]) -> dict[str, list[dict]]:
    return {
        method: [
            {
                "sample_id": row["sample_id"],
                "split": "calibration" if row["sample_id"] in calibration else "evaluation",
                "oracle_scenario_type": row["scenario_type"],
                "hit": row["hit"],
                "first_hit_turn": row["first_hit_turn"],
                "best_rank": row["best_rank"],
                "reciprocal_rank": row["reciprocal_rank"],
            }
            for row in rows
        ]
        for method, rows in session_sets.items()
    }


def run_07(h: Harness, root_logger: logging.Logger) -> dict:
    directory = h.results_dir / EXPERIMENT_DIRECTORY
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("AGENT-REALISTIC EVALUATION + ORACLE-AFTER-FREEZE DIAGNOSTIC: beginning residual analysis")

    split_path = h.results_dir / "experiment_06_slate_width_counterfactuals" / "metrics.json"
    public_ids = [sample["sample_id"] for sample in h.samples]
    calibration, evaluation, split = load_frozen_split(split_path, public_ids)
    all_ids = set(public_ids)

    try:
        h.dense.ensure()
        if h.dense.embeddings is None or len(h.dense.embeddings) != len(h.products):
            raise RuntimeError("MiniLM dense embeddings are incomplete")
    except Exception as exc:
        raise RuntimeError(f"Required dense model/index is unavailable: {exc}") from exc

    # No oracle value is passed into _rank_input. All rankings are frozen here first.
    frozen = freeze_agent_rankings(h, logger)
    logger.info("All %d rankings frozen; joining oracle labels for scoring", len(frozen))

    session_sets = {method: _replay_frozen(h, frozen, method) for method in METHODS}
    exact_sessions = session_sets[EXACT_METHOD]
    actual_baseline = rank_metrics(exact_sessions)
    for key, expected in EXPECTED_EXACT_BASELINE.items():
        if actual_baseline.get(key) != expected:
            raise RuntimeError(f"Exact baseline mismatch for {key}: expected {expected}, got {actual_baseline.get(key)}")

    exp2_path = h.results_dir / "experiment_02_target_rank_curves" / "early_termination_sessions.json"
    if not exp2_path.exists():
        raise RuntimeError(f"Frozen Experiment 2 session output is missing: {exp2_path}")
    exp2_payload = json.loads(exp2_path.read_text(encoding="utf-8"))
    expected_sessions = exp2_payload.get("exact_phrase")
    if not isinstance(expected_sessions, list):
        raise RuntimeError("Frozen Experiment 2 output lacks exact_phrase sessions")
    assert_exact_baseline_identity(exact_sessions, expected_sessions)
    baseline_reproduction = {
        "passed": True,
        "expected": EXPECTED_EXACT_BASELINE,
        "actual": actual_baseline,
        "session_identity_passed": True,
        "session_identity_count": len(exact_sessions),
        "source": str(exp2_path.relative_to(h.repo)),
        "source_sha256": sha256(exp2_path),
    }

    sample_split = {sid: "calibration" if sid in calibration else "evaluation" for sid in all_ids}
    for rows in session_sets.values():
        for row in rows:
            row["split"] = sample_split[row["sample_id"]]
    by_method_by_id = {method: {row["sample_id"]: row for row in rows} for method, rows in session_sets.items()}

    hard_sessions = [row for row in exact_sessions if not row["hit"]]
    weak_sessions = [row for row in exact_sessions if row["hit"] and 6 <= int(row["best_rank"]) <= 10]
    hard_ids = {row["sample_id"] for row in hard_sessions}
    weak_ids = {row["sample_id"] for row in weak_sessions}
    residual_types = {**{sid: "hard_failure" for sid in hard_ids}, **{sid: "weak_success" for sid in weak_ids}}

    method_metrics = {
        method: {
            "full": _metrics_for_subset(rows, all_ids),
            "calibration": _metrics_for_subset(rows, calibration),
            "evaluation": _metrics_for_subset(rows, evaluation),
        }
        for method, rows in session_sets.items()
    }
    comparisons = {
        method: {
            "full": _comparison(exact_sessions, session_sets[method], all_ids, weak_ids),
            "calibration": _comparison(exact_sessions, session_sets[method], calibration, weak_ids),
            "evaluation": _comparison(exact_sessions, session_sets[method], evaluation, weak_ids),
        }
        for method in METHODS
    }

    selected_method = sorted(
        CASCADE_COMPONENTS,
        key=lambda method: (
            -method_metrics[method]["calibration"]["technical_score"],
            -comparisons[method]["calibration"]["hard_failure_rescues"],
            comparisons[method]["calibration"]["regressions"],
            -method_metrics[method]["calibration"]["mrr"],
            method,
        ),
    )[0]

    exact_eval = method_metrics[EXACT_METHOD]["evaluation"]
    selected_eval = method_metrics[selected_method]["evaluation"]
    selected_comparison = comparisons[selected_method]["evaluation"]
    gates = {
        "technical_score_at_least_exact": selected_eval["technical_score"] >= exact_eval["technical_score"],
        "rescues_at_least_one_hard_failure": selected_comparison["hard_failure_rescues"] >= 1,
        "regressions_do_not_exceed_rescues": selected_comparison["regressions"] <= selected_comparison["hard_failure_rescues"],
    }
    recommend_selected = all(gates.values())
    recommendation = selected_method if recommend_selected else EXACT_METHOD

    joined_rows = _joined_rows(h, frozen, calibration)
    residual_turns = _residual_turn_rows(h, frozen, residual_types, calibration)
    sample_lookup = {sample["sample_id"]: sample for sample in h.samples}
    trace_lookup = h.trace_by_session()
    frozen_lookup = {
        (state.sample_id, state.turn): ranking for state, ranking in zip(h.traces, frozen)
    }

    hard_failure_rows: list[dict] = []
    for session in hard_sessions:
        state = trace_lookup[session["sample_id"]][-1]
        hard_failure_rows.append(
            _diagnose_residual(
                h,
                state,
                frozen_lookup[(state.sample_id, state.turn)],
                "hard_failure",
                by_method_by_id,
                sample_lookup[state.sample_id],
            )
        )
    weak_success_rows: list[dict] = []
    for session in weak_sessions:
        state = trace_lookup[session["sample_id"]][int(session["first_hit_turn"]) - 1]
        weak_success_rows.append(
            _diagnose_residual(
                h,
                state,
                frozen_lookup[(state.sample_id, state.turn)],
                "weak_success",
                by_method_by_id,
                sample_lookup[state.sample_id],
            )
        )
    diagnostic_rows = [*hard_failure_rows, *weak_success_rows]
    category_counts = failure_category_counts(diagnostic_rows)

    weak_improvement_rows, weak_improvement_summary = _same_turn_weak_improvements(
        h, frozen, weak_sessions, calibration
    )
    for row in weak_success_rows:
        row["diagnostic_same_turn_capped_rank_improvements"] = {
            item["method"]: item["capped_rank_improvement"]
            for item in weak_improvement_rows
            if item["sample_id"] == row["sample_id"]
        }

    rescue_by_category: list[dict] = []
    for method in METHODS:
        for tag in FAILURE_TAGS:
            tagged = [row for row in hard_failure_rows if tag in row["diagnostic_failure_tags"]]
            rescued = [
                row for row in tagged if by_method_by_id[method][row["sample_id"]]["hit"]
            ]
            for split_name, split_ids in (("full", all_ids), ("calibration", calibration), ("evaluation", evaluation)):
                selected_tagged = [row for row in tagged if row["sample_id"] in split_ids]
                selected_rescued = [row for row in rescued if row["sample_id"] in split_ids]
                rescue_by_category.append(
                    {
                        "method": method,
                        "split": split_name,
                        "failure_category": tag,
                        "tagged_hard_failures": len(selected_tagged),
                        "rescued_hard_failures": len(selected_rescued),
                        "rescue_percentage": round(len(selected_rescued) / len(selected_tagged), 6) if selected_tagged else 0.0,
                        "non_exclusive": True,
                    }
                )

    comparison_rows = [
        {"method": method, "split": split_name, **values}
        for method, split_values in comparisons.items()
        for split_name, values in split_values.items()
    ]
    rank_distribution_rows = _rank_distribution_rows(
        session_sets,
        {"full": all_ids, "calibration": calibration, "evaluation": evaluation},
    )
    serialized_sessions = _serialize_sessions(session_sets, calibration)

    fallback_turns = sum(ranking.fallback_activated for ranking in frozen)
    source_path = Path(__file__).resolve()
    source_hash = sha256(source_path)
    metrics = {
        "experiment": EXPERIMENT_NUMBER,
        "slug": EXPERIMENT_SLUG,
        "label": "AGENT-REALISTIC EVALUATION + ORACLE-AFTER-FREEZE DIAGNOSTIC",
        "model_id": MODEL_ID,
        "retrieval_contract": {
            "ranker_input_fields": ["category", "active_constraints"],
            "excluded_from_rankers": [
                "target_asin",
                "scenario_type",
                "sample_id",
                "oracle_card",
                "future_turns",
                "user_profile",
            ],
            "user_profiles_used_for_ranking": False,
            "rankings_frozen_before_oracle_join": True,
        },
        "cascade_config": {
            "conditional": True,
            "rrf_k": RRF_K,
            "depth": RRF_DEPTH,
            "weights": "equal",
            "tie_break": "ascending_parent_asin",
            "components": {method: list(parts) for method, parts in CASCADE_COMPONENTS.items()},
            "fallback_activation": [
                "no active constraint",
                "no catalog product matches all active phrases exactly",
                "highest exact-match tier contains more than ten products",
            ],
            "exact_order_preserved_when_inactive": True,
        },
        "split": split,
        "baseline_reproduction": baseline_reproduction,
        "residuals": {
            "hard_failures": len(hard_failure_rows),
            "weak_successes_rank_6_to_10": len(weak_success_rows),
            "residual_trace_turns": len(residual_turns),
            "fallback_turns": fallback_turns,
            "fallback_turn_percentage": round(fallback_turns / len(frozen), 6),
            "failure_category_counts_are_non_exclusive": True,
            "intrinsic_ambiguity_sessions": sum(row["diagnostic_intrinsic_ambiguity"] for row in diagnostic_rows),
            "dense_only_rescues": sum(row["diagnostic_dense_only_rescue"] for row in diagnostic_rows),
        },
        "method_metrics": method_metrics,
        "comparisons_to_exact": comparisons,
        "weak_success_same_turn_capped_rank_improvement": weak_improvement_summary,
        "failure_category_counts": category_counts,
        "selection": {
            "selected_on_calibration": selected_method,
            "tie_break_order": [
                "higher calibration TechnicalScore",
                "more calibration hard-failure rescues",
                "fewer calibration regressions",
                "higher calibration MRR",
                "method name for determinism",
            ],
            "held_out_gates": gates,
            "production_recommendation": recommendation,
            "recommend_selected_fallback": recommend_selected,
            "held_out_results_did_not_reselect_route": True,
            "starter_agent_modified": False,
        },
        "source": {"path": str(source_path.relative_to(h.repo)), "sha256": source_hash},
    }
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)

    validate_artifact_payloads(
        metrics,
        joined_rows,
        serialized_sessions,
        residual_turns,
        hard_failure_rows,
        weak_success_rows,
    )

    _write_charts(directory, comparisons, rank_distribution_rows)
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "rows.csv", joined_rows)
    write_json(directory / "sessions.json", serialized_sessions)
    write_json(directory / "baseline_reproduction.json", baseline_reproduction)
    write_csv(directory / "residual_turns.csv", residual_turns)
    write_json(directory / "residual_turns.json", residual_turns)
    write_csv(directory / "hard_failures.csv", hard_failure_rows)
    write_json(directory / "hard_failures.json", hard_failure_rows)
    write_csv(directory / "weak_successes.csv", weak_success_rows)
    write_json(directory / "weak_successes.json", weak_success_rows)
    write_csv(directory / "failure_category_counts.csv", category_counts)
    write_json(directory / "failure_category_counts.json", category_counts)
    write_csv(directory / "rescue_by_category.csv", rescue_by_category)
    write_json(directory / "rescue_by_category.json", rescue_by_category)
    write_csv(directory / "rescue_comparisons.csv", comparison_rows)
    write_json(directory / "rescue_comparisons.json", comparison_rows)
    write_csv(directory / "rank_distributions.csv", rank_distribution_rows)
    write_json(directory / "rank_distributions.json", rank_distribution_rows)
    (directory / "source_snapshot.py").write_bytes(source_path.read_bytes())
    snapshot_hash = sha256(directory / "source_snapshot.py")
    if snapshot_hash != source_hash:
        raise RuntimeError("Experiment 7 source snapshot hash mismatch")
    write_json(
        directory / "source_snapshot.json",
        {
            "source": str(source_path.relative_to(h.repo)),
            "source_sha256": source_hash,
            "snapshot": str((directory / "source_snapshot.py").relative_to(h.repo)),
            "snapshot_sha256": snapshot_hash,
            "identical": True,
        },
    )

    decision = (
        f"recommend `{selected_method}`"
        if recommend_selected
        else "retain `exact_only`"
    )
    summary = f"""# Experiment 7 — Residual failure analysis

> **AGENT-REALISTIC RANKING, ORACLE-AFTER-FREEZE DIAGNOSTICS.** Rankers received only category plus active disclosed dialogue evidence. Targets, scenarios, sample IDs, hidden cards, future turns, and user profiles were joined only after all rankings were frozen.

The exact baseline reproduced all 200 frozen Experiment 2 sessions exactly: Hit@10 **{actual_baseline['hit_rate_at_10']:.3f}**, MRR **{actual_baseline['mrr']:.6f}**, MTTC **{actual_baseline['mttc']:.2f}**, Efficiency **{actual_baseline['efficiency']:.3f}**, and TechnicalScore **{actual_baseline['technical_score']:.6f}**. It left **{len(hard_failure_rows)} hard failures** and **{len(weak_success_rows)} weak successes** whose first hit ranked 6–10.

Calibration selected **{selected_method}**. On the frozen 140-session held-out set it scored **{selected_eval['technical_score']:.6f}** versus **{exact_eval['technical_score']:.6f}** for exact-only, rescued **{selected_comparison['hard_failure_rescues']}** hard failures, and introduced **{selected_comparison['regressions']}** regressions. The production decision is to **{decision}** under the preregistered gates; held-out results were not used to switch to another cascade.

Failure-category and rescue-by-category counts are explicitly **non-exclusive**. The synthetic price representation is separated from title, features, details, description, categories, and store in every evidence-attribution record. User profiles appear only in the hard/weak audit artifacts and were never ranker inputs. No cascade was installed in the starter agent.
"""
    (directory / "summary.md").write_text(summary, encoding="utf-8")

    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"Experiment 7 artifact set is incomplete: {missing}")
    logger.info(
        "Completed %s in %.2fs; calibration selected %s; production recommendation=%s",
        EXPERIMENT_DIRECTORY,
        metrics["elapsed_seconds"],
        selected_method,
        recommendation,
    )
    return metrics
