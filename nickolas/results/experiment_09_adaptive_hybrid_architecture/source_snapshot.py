from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from .config import DENSE_MAX_SEQ_LENGTH, MAX_TURNS, MODEL_ID, RRF_DEPTH, RRF_K, TOP_K
from .experiment_07_residual_failure_analysis import load_frozen_split
from .experiment_08_intent_routed_dense_browsing import (
    ObservableRetrievalInput,
    ObservableStateParser,
    _freeze,
    _hash_tree,
    _metric_subset,
    _outcomes,
    catalog_constraint_lexicon,
    compare_sessions,
    exp7_exact_stateful_bm25,
    parse_observable_traces,
)
from .harness import (
    FIELDS,
    Harness,
    experiment_logger,
    normalize,
    rank_metrics,
    sha256,
    write_csv,
    write_json,
)


EXPERIMENT_NUMBER = 9
EXPERIMENT_SLUG = "adaptive_hybrid_architecture"
EXPERIMENT_DIRECTORY = f"experiment_{EXPERIMENT_NUMBER:02d}_{EXPERIMENT_SLUG}"
BASELINE_METHOD = "experiment_07_exact_stateful_bm25_rrf"

STATE_CONFIG = {
    "exploratory": {"exact": 0.0, "bm25": 0.25, "dense": 1.0, "structured": 0.25, "pool": 100},
    "mixed": {"exact": 0.5, "bm25": 0.75, "dense": 0.75, "structured": 0.5, "pool": 75},
    "specific": {"exact": 1.0, "bm25": 1.0, "dense": 0.25, "structured": 1.0, "pool": 50},
}
QUESTION_ATTRIBUTES = ("material", "color", "size", "style", "brand", "budget", "feature", "use_case")
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
USE_CASES = ("hiking", "running", "gym", "winter", "outdoor", "work", "walking", "sports", "casual")
NEGATION_CUES = ("ignore", "forget", "doesn't matter", "doesnt matter", "don't want", "dont want", "instead")
HARD_CUES = ("requirement", "need", "must", "instead", "what i need")
PREFERENCE_CUES = ("prefer", "preference", "would like", "nice to have")


def classify_typed_constraint(value: str) -> str:
    lowered = normalize(value)
    if "budget" in lowered or re.search(r"(?:\$|<=|under|below|ceiling)\s*\d", lowered):
        return "budget"
    if any(word in lowered for word in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", *COLORS)):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck", "casual", "formal")):
        return "style"
    if any(word in lowered for word in ("brand", "store", "maker")):
        return "brand_store"
    if any(word in lowered for word in USE_CASES):
        return "use_case"
    return "feature"


@dataclass
class TypedConstraint:
    value: str
    kind: str
    negated: bool
    source_turn: int
    strength: float
    hard: bool
    explicit: bool
    source: str


@dataclass
class TypedObservableState:
    category: str = ""
    intent_seed: str = "buying"
    constraints: list[TypedConstraint] = field(default_factory=list)
    negations: list[TypedConstraint] = field(default_factory=list)
    profile_tags: tuple[str, ...] = ()
    asked_attributes: list[str] = field(default_factory=list)
    explicit_buying_requirement: bool = False
    turn: int = 0

    @property
    def active_constraints(self) -> tuple[TypedConstraint, ...]:
        return tuple(item for item in self.constraints if not item.negated)

    @property
    def retrieval_input(self) -> ObservableRetrievalInput:
        return ObservableRetrievalInput(
            self.category or "clothing item",
            tuple(item.value for item in self.active_constraints),
            self.intent_seed,
        )

    @property
    def specificity(self) -> str:
        count = len(self.active_constraints)
        if count == 0:
            return "exploratory"
        if count >= 2 or self.explicit_buying_requirement:
            return "specific"
        return "mixed"


class TypedStateParser:
    """Typed observable identity layered over the frozen Experiment 7 parser."""

    def __init__(self, profile_tags: Sequence[str] = (), known_constraints: set[str] | None = None) -> None:
        self.flat = ObservableStateParser(known_constraints)
        self.state = TypedObservableState(profile_tags=tuple(str(value) for value in profile_tags))

    def update(self, message: str, turn: int) -> TypedObservableState:
        before = {item.value: item for item in self.state.active_constraints}
        parsed = self.flat.update(message, turn)
        after_values = {value: value for value in parsed.active_constraints}
        lowered = normalize(message)
        hard = any(cue in lowered for cue in HARD_CUES)
        preference = any(cue in lowered for cue in PREFERENCE_CUES)
        ambiguous = lowered.startswith("for that")
        strength = 1.0 if hard or preference else 0.7 if ambiguous else 0.7

        for key, old in before.items():
            if key not in after_values:
                old.negated = True
                self.state.negations.append(TypedConstraint(
                    old.value, old.kind, True, turn, 1.0, True, True, "negation_or_replacement"
                ))
        retained = [item for item in self.state.constraints if item.value in after_values]
        known = {item.value for item in retained}
        for key, value in after_values.items():
            if key not in known:
                item_hard = hard
                retained.append(TypedConstraint(
                    value=value,
                    kind=classify_typed_constraint(value),
                    negated=False,
                    source_turn=turn,
                    strength=1.0 if item_hard else strength,
                    hard=item_hard,
                    explicit=item_hard or preference,
                    source="explicit_requirement" if item_hard else "preference" if preference else "ambiguous_disclosure",
                ))
        self.state.constraints = retained
        self.state.category = parsed.category
        self.state.intent_seed = parsed.locked_intent
        self.state.explicit_buying_requirement = self.state.explicit_buying_requirement or hard
        self.state.turn = turn
        return self.state

    def mark_asked(self, attribute: str) -> None:
        if attribute not in self.state.asked_attributes:
            self.state.asked_attributes.append(attribute)


def weighted_rrf(
    rankings: Mapping[str, Sequence[int] | np.ndarray],
    weights: Mapping[str, float],
    ids: Sequence[str] | np.ndarray,
    *,
    depth: int = RRF_DEPTH,
    k: int = RRF_K,
) -> tuple[np.ndarray, np.ndarray]:
    scores: dict[int, float] = defaultdict(float)
    for name, ranking in rankings.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        seen: set[int] = set()
        for rank, raw_index in enumerate(ranking[:depth], 1):
            index = int(raw_index)
            if index in seen:
                continue
            seen.add(index)
            scores[index] += weight / (k + rank)
    if not scores:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    candidates = np.fromiter(scores, dtype=np.int64)
    values = np.fromiter((scores[int(index)] for index in candidates), dtype=np.float64)
    id_array = np.asarray(ids)
    order = np.lexsort((id_array[candidates], -values))
    return candidates[order], values[order]


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    low, high = float(np.min(values)), float(np.max(values))
    if math.isclose(low, high):
        return np.ones(values.shape, dtype=np.float64) if high > 0 else np.zeros(values.shape, dtype=np.float64)
    return (values.astype(np.float64) - low) / (high - low)


def deterministic_reranker_score(
    hybrid: np.ndarray,
    dense: np.ndarray,
    hard: np.ndarray,
    soft: np.ndarray,
    category: np.ndarray,
    violations: np.ndarray,
    profile: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the preregistered normalized feature formula element-wise."""

    if profile is None:
        return 0.35 * hybrid + 0.20 * dense + 0.25 * hard + 0.10 * soft + 0.10 * category - 0.30 * violations
    return 0.30 * hybrid + 0.20 * dense + 0.25 * hard + 0.10 * soft + 0.10 * category + 0.05 * profile - 0.30 * violations


class AdaptiveHybridRanker:
    def __init__(self, harness: Harness) -> None:
        self.h = harness
        self.ids = np.asarray(harness.ids)
        self._field_total: dict[tuple[str, ...], sparse.csr_matrix] = {}
        self._evidence_cache: dict[tuple[str, str], np.ndarray] = {}
        self._profile_cache: dict[str, np.ndarray] = {}
        self.constraint_lexicon = catalog_constraint_lexicon(harness)
        def numeric_price(product: dict) -> float:
            try:
                return float(product.get("price"))
            except (TypeError, ValueError):
                return np.nan

        self.prices = np.asarray([numeric_price(product) for product in harness.products], dtype=np.float64)

    @staticmethod
    def _fields(kind: str) -> tuple[str, ...]:
        return {
            "category": ("categories", "title"),
            "material": ("title", "features", "details", "description"),
            "color": ("title", "features", "details", "description"),
            "size": ("title", "features", "details"),
            "style": ("title", "features", "details", "categories"),
            "brand_store": ("store", "title"),
            "budget": ("price",),
            "feature": ("title", "features", "details", "description"),
            "use_case": ("title", "features", "details", "description", "categories"),
        }.get(kind, FIELDS)

    def _matrix(self, fields: tuple[str, ...]) -> sparse.csr_matrix:
        matrix = self._field_total.get(fields)
        if matrix is None:
            matrix = sum(
                (self.h.lexical.field_counts[name] for name in fields),
                start=sparse.csr_matrix(self.h.lexical.bm25.shape),
            ).tocsr()
            self._field_total[fields] = matrix
        return matrix

    @staticmethod
    def _budget(value: str) -> float | None:
        match = re.search(r"(?:\$|under\s*|below\s*|<=\s*|around\s*\$?)\s*(\d+(?:\.\d+)?)", value, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def evidence(self, kind: str, value: str) -> np.ndarray:
        key = (kind, normalize(value))
        cached = self._evidence_cache.get(key)
        if cached is not None:
            return cached
        if kind == "budget":
            ceiling = self._budget(value)
            if ceiling is not None:
                result = np.where(np.isnan(self.prices), 0.0, (self.prices <= ceiling).astype(np.float32))
                self._evidence_cache[key] = result
                return result
        query = self.h.lexical._query_vector(value)
        if not query.nnz:
            result = np.zeros(len(self.ids), dtype=np.float32)
        else:
            counts = (self._matrix(self._fields(kind)) @ query.T).toarray().ravel().astype(np.float32)
            result = np.clip(counts / float(query.nnz), 0.0, 1.0)
        self._evidence_cache[key] = result
        return result

    def category_evidence(self, category: str) -> np.ndarray:
        return self.evidence("category", category)

    def structured(self, state: TypedObservableState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        category = self.category_evidence(state.category)
        total = category.astype(np.float64) * 0.35
        weight = np.full(total.shape, 0.35, dtype=np.float64)
        budget_violation = np.zeros(total.shape, dtype=bool)
        for constraint in state.active_constraints:
            evidence = self.evidence(constraint.kind, constraint.value)
            total += evidence * constraint.strength
            weight += constraint.strength
            if constraint.kind == "budget":
                ceiling = self._budget(constraint.value)
                if ceiling is not None:
                    budget_violation |= ~np.isnan(self.prices) & (self.prices > ceiling)
        scores = total / np.maximum(weight, 1e-9)
        scores[budget_violation] = -np.inf
        candidates = np.flatnonzero(~budget_violation)
        order = candidates[np.lexsort((self.ids[candidates], -scores[candidates]))]
        return order, scores[order], budget_violation

    def adaptive_pool(self, state: TypedObservableState) -> tuple[np.ndarray, np.ndarray, dict]:
        item = state.retrieval_input
        config = STATE_CONFIG[state.specificity]
        exact, _ = self.h.exact_ranked(item.phrases)
        bm25, _ = self.h.lexical.ranked(item.query, "bm25", RRF_DEPTH)
        dense, dense_scores = self.h.dense.ranked(item.query, RRF_DEPTH)
        structured, _, budget_violations = self.structured(state)
        rankings = {"exact": exact, "bm25": bm25, "dense": dense, "structured": structured}
        order, scores = weighted_rrf(rankings, config, self.ids)
        if budget_violations.any():
            keep = ~budget_violations[order]
            order, scores = order[keep], scores[keep]
        pool_size = int(config["pool"])
        return order[:pool_size], scores[:pool_size], {
            "specificity": state.specificity,
            "weights": {name: config[name] for name in ("exact", "bm25", "dense", "structured")},
            "pool_size": pool_size,
            "component_heads": {name: [self.h.ids[int(value)] for value in ranking[:10]] for name, ranking in rankings.items()},
            "numeric_budget_violations": int(budget_violations.sum()),
        }

    def _constraint_features(self, state: TypedObservableState, pool: np.ndarray, *, decay: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hard_values: list[tuple[np.ndarray, float]] = []
        soft_values: list[tuple[np.ndarray, float]] = []
        for item in state.active_constraints:
            evidence = self.evidence(item.kind, item.value)[pool]
            if item.hard:
                hard_values.append((evidence, item.strength))
            else:
                turn_age = max(0, state.turn - item.source_turn)
                strength = item.strength * (0.9 ** turn_age if decay else 1.0)
                soft_values.append((evidence, strength))

        def aggregate(values: list[tuple[np.ndarray, float]]) -> np.ndarray:
            if not values:
                return np.ones(len(pool), dtype=np.float64)
            numerator = sum((array * weight for array, weight in values), start=np.zeros(len(pool), dtype=np.float64))
            return numerator / sum(weight for _, weight in values)

        hard = aggregate(hard_values)
        soft = aggregate(soft_values)
        violations = np.zeros(len(pool), dtype=np.float64)
        explicit = [item for item in state.active_constraints if item.hard or item.explicit]
        if explicit:
            violations += sum((self.evidence(item.kind, item.value)[pool] < 0.5 for item in explicit), start=np.zeros(len(pool))) / len(explicit)
        if state.negations:
            violations += sum((self.evidence(item.kind, item.value)[pool] >= 0.5 for item in state.negations), start=np.zeros(len(pool))) / len(state.negations)
        return hard, soft, np.clip(violations, 0.0, 1.0)

    def profile_evidence(self, state: TypedObservableState, pool: np.ndarray) -> np.ndarray:
        if not state.profile_tags:
            return np.zeros(len(pool), dtype=np.float64)
        arrays = []
        for tag in state.profile_tags:
            cached = self._profile_cache.get(normalize(tag))
            if cached is None:
                cached = self.evidence(classify_typed_constraint(tag), tag)
                self._profile_cache[normalize(tag)] = cached
            arrays.append(cached[pool])
        return np.mean(arrays, axis=0) if arrays else np.zeros(len(pool), dtype=np.float64)

    def rank(self, state: TypedObservableState, stage: str) -> tuple[np.ndarray, np.ndarray, dict]:
        if stage == "structured_identity":
            order, scores, diagnostic = exp7_exact_stateful_bm25(self.h, state.retrieval_input)
            return order, scores, {**diagnostic, "specificity": state.specificity, "pool": list(order)}

        pool, hybrid_scores, diagnostic = self.adaptive_pool(state)
        if stage == "adaptive_hybrid":
            return pool[:TOP_K], hybrid_scores[:TOP_K], {**diagnostic, "pool": list(pool), "hybrid_scores": list(map(float, hybrid_scores))}
        if stage not in {"reranker", "profile_decay"}:
            raise ValueError(f"Unknown Experiment 9 rank stage: {stage}")

        dense_full, dense_full_scores = self.h.dense.ranked(state.retrieval_input.query)
        dense_lookup = np.empty(len(self.ids), dtype=np.float64)
        dense_lookup[dense_full] = dense_full_scores
        hard, soft, violations = self._constraint_features(state, pool, decay=stage == "profile_decay")
        category = self.category_evidence(state.category)[pool]
        normalized_hybrid = _minmax(hybrid_scores)
        dense = _minmax(dense_lookup[pool])
        if stage == "profile_decay":
            profile = self.profile_evidence(state, pool)
            final = deterministic_reranker_score(normalized_hybrid, dense, hard, soft, category, violations, profile)
        else:
            profile = np.zeros(len(pool), dtype=np.float64)
            final = deterministic_reranker_score(normalized_hybrid, dense, hard, soft, category, violations)
        order = np.lexsort((self.ids[pool], -final))
        reranked = pool[order]
        scores = final[order]
        return reranked[:TOP_K], scores[:TOP_K], {
            **diagnostic,
            "pool": list(map(int, reranked)),
            "hybrid_scores": list(map(float, normalized_hybrid[order])),
            "dense_similarity": list(map(float, dense[order])),
            "hard_satisfaction": list(map(float, hard[order])),
            "soft_satisfaction": list(map(float, soft[order])),
            "category_consistency": list(map(float, category[order])),
            "profile_evidence": list(map(float, profile[order])),
            "violations": list(map(float, violations[order])),
            "final_scores": list(map(float, scores)),
        }


def _first_word_value(text: str, vocabulary: Sequence[str]) -> str | None:
    lowered = normalize(text)
    return next((value for value in vocabulary if re.search(rf"\b{re.escape(value)}\b", lowered)), None)


def candidate_attribute_value(product: dict, attribute: str) -> str | None:
    text = " ".join(str(product.get(field) or "") for field in ("title", "features", "details", "description", "categories"))
    if attribute == "material":
        return _first_word_value(text, MATERIALS)
    if attribute == "color":
        return _first_word_value(text, COLORS)
    if attribute == "use_case":
        return _first_word_value(text, USE_CASES)
    if attribute == "brand":
        value = normalize(product.get("store"))
        return value[:60] or None
    if attribute == "budget":
        price = product.get("price")
        if price in (None, ""):
            return None
        try:
            value = float(price)
        except (TypeError, ValueError):
            return None
        return f"under_{math.ceil(value / 25.0) * 25:.0f}"
    details = product.get("details") or {}
    if isinstance(details, dict):
        for key, value in details.items():
            lowered = normalize(key)
            if attribute == "size" and any(word in lowered for word in ("size", "width", "fit")):
                return normalize(value)[:60] or None
            if attribute == "style" and any(word in lowered for word in ("style", "department", "sleeve", "neck", "fit")):
                return normalize(value)[:60] or None
    if attribute == "feature":
        features = product.get("features") or []
        if isinstance(features, list) and features:
            return normalize(features[0])[:60] or None
    return None


def meaningful_attribute_values(h: Harness, pool: Sequence[int], attribute: str) -> list[str | None]:
    return [candidate_attribute_value(h.products[int(index)], attribute) for index in pool]


def entropy_question(h: Harness, pool: Sequence[int], asked: Sequence[str]) -> tuple[str, dict]:
    choices: list[tuple[float, str, Counter]] = []
    for attribute in QUESTION_ATTRIBUTES:
        if attribute in asked:
            continue
        values = meaningful_attribute_values(h, pool, attribute)
        counts = Counter(value for value in values if value)
        if len(counts) < 2:
            continue
        total = sum(counts.values())
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        choices.append((entropy, attribute, counts))
    if not choices:
        return "other", {"fallback": "fewer_than_two_meaningful_values"}
    entropy, attribute, counts = sorted(choices, key=lambda item: (-item[0], item[1]))[0]
    return attribute, {"entropy": entropy, "value_counts": dict(sorted(counts.items()))}


def rank_aware_question(h: Harness, pool: Sequence[int], asked: Sequence[str]) -> tuple[str, dict]:
    pool = list(pool)
    if not pool:
        return "other", {"fallback": "empty_candidate_pool"}
    baseline = statistics_utility = sum((1.0 if rank <= TOP_K else 0.0) + (1.0 / rank) for rank in range(1, len(pool) + 1)) / len(pool)
    choices: list[tuple[float, str, dict]] = []
    for attribute in QUESTION_ATTRIBUTES:
        if attribute in asked:
            continue
        values = meaningful_attribute_values(h, pool, attribute)
        meaningful = {value for value in values if value}
        if len(meaningful) < 2:
            continue
        utility = 0.0
        for index, target_value in enumerate(values):
            if target_value is None:
                simulated_rank = index + 1
            else:
                same_value_before = sum(value == target_value for value in values[: index + 1])
                simulated_rank = same_value_before
            utility += (1.0 if simulated_rank <= TOP_K else 0.0) + 1.0 / simulated_rank
        utility /= len(pool)
        gain = utility - baseline
        choices.append((gain, attribute, {"expected_utility": utility, "baseline_utility": baseline, "meaningful_values": len(meaningful)}))
    if not choices:
        return "other", {"fallback": "fewer_than_two_meaningful_values"}
    gain, attribute, diagnostic = sorted(choices, key=lambda item: (-item[0], item[1]))[0]
    if gain <= 0:
        return "other", {"fallback": "no_positive_expected_gain", "best_gain": gain}
    return attribute, {**diagnostic, "expected_gain": gain}


def choose_question(h: Harness, method: str, state: TypedObservableState, diagnostic: Mapping[str, Any]) -> tuple[str, dict]:
    if method == "fixed_other":
        return "other", {"fixed": True}
    pool = [int(value) for value in diagnostic.get("pool", [])]
    if method == "entropy":
        return entropy_question(h, pool, state.asked_attributes)
    if method == "rank_aware":
        return rank_aware_question(h, pool, state.asked_attributes)
    raise ValueError(f"Unknown clarification method: {method}")


def simulate_policy(
    h: Harness,
    ranker: AdaptiveHybridRanker,
    sample_ids: set[str],
    *,
    stage: str,
    question_method: str,
) -> tuple[list[dict], list[dict]]:
    sessions: list[dict] = []
    turn_rows: list[dict] = []
    lexicon = ranker.constraint_lexicon
    for sample in h.samples:
        sid = sample["sample_id"]
        if sid not in sample_ids:
            continue
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = h.cards[sid], h.behaviors[sid]
        effective = {**sample, "intent_card": card, "behavior": behavior}
        category = h.official.coarse_category(h.categories[target])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = h.official.initial_message(effective, category, disclosed)
        parser = TypedStateParser(sample.get("user_profile", {}).get("preference_tags", ()), lexicon)
        first_hit = best_rank = None
        for turn in range(1, MAX_TURNS + 1):
            state = parser.update(message, turn)
            started = time.perf_counter()
            order, scores, diagnostic = ranker.rank(state, stage)
            latency_ms = (time.perf_counter() - started) * 1000.0
            question, question_diagnostic = choose_question(h, question_method, state, diagnostic)
            parser.mark_asked(question)
            # Target/scenario are joined here, after the ranker and clarification
            # selector have returned their frozen outputs.
            target_index = h.id_to_idx[target]
            values = list(map(int, order[:TOP_K]))
            target_rank = values.index(target_index) + 1 if target_index in values else None
            turn_rows.append({
                "sample_id": sid,
                "turn": turn,
                "stage": stage,
                "clarification": question_method,
                "intent_seed": state.intent_seed,
                "specificity": state.specificity,
                "category": state.category,
                "typed_constraints": [vars(item) for item in state.active_constraints],
                "negations": [vars(item) for item in state.negations],
                "profile_tags": list(state.profile_tags),
                "asked_attributes": list(state.asked_attributes),
                "question": question,
                "question_diagnostic": question_diagnostic,
                "candidate_pool_size": len(diagnostic.get("pool", [])),
                "top_10": [h.ids[value] for value in values],
                "latency_ms": latency_ms,
                "oracle_joined_after_freeze": True,
                "oracle_scenario_type": sample["scenario_type"],
                "oracle_target_asin": target,
                "diagnostic_target_rank": target_rank,
            })
            if override_applied and target_rank is not None:
                first_hit, best_rank = turn, target_rank
                break
            if turn == MAX_TURNS:
                break
            override = behavior.get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = h.official.customer_reply(effective, question, disclosed, boundary_used)
        sessions.append({
            "sample_id": sid,
            "scenario_type": sample["scenario_type"],
            "hit": first_hit is not None,
            "first_hit_turn": first_hit,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
    return sessions, turn_rows


def _selection_key(name: str, metrics: Mapping[str, dict], comparisons: Mapping[str, dict]) -> tuple:
    return (
        -metrics[name]["technical_score"],
        -comparisons[name]["hard_failure_rescues"],
        comparisons[name]["regressions"],
        -metrics[name]["mrr"],
        name,
    )


def _plot(path: Path, calibration: Mapping[str, dict]) -> None:
    labels = list(calibration)
    values = [calibration[name]["technical_score"] for name in labels]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(labels, values, color="#577590")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Calibration TechnicalScore")
    ax.set_title("Experiment 9 cumulative and clarification ablations")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_09(h: Harness, root_logger) -> dict:
    directory = h.results_dir / EXPERIMENT_DIRECTORY
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("AGENT-REALISTIC EVALUATION: beginning adaptive hybrid architecture ablations")

    split_path = h.results_dir / "experiment_06_slate_width_counterfactuals" / "metrics.json"
    exp7_source = h.repo / "nickolas" / "experiments" / "experiment_07_residual_failure_analysis.py"
    submission_agent = h.repo / "techjam-conversational-search" / "starter" / "agent.py"
    calibration, evaluation, split = load_frozen_split(split_path, (sample["sample_id"] for sample in h.samples))
    all_ids = calibration | evaluation
    h.dense.ensure()
    ranker = AdaptiveHybridRanker(h)

    # Ablation 1: typed state must be observationally identical to the frozen
    # flat state and therefore reproduce every Experiment 7 ranking.
    flat_inputs = parse_observable_traces(h)
    baseline_slates, _, _ = _freeze(h, flat_inputs, exp7_exact_stateful_bm25)
    baseline_sessions = _outcomes(h, baseline_slates)
    typed_inputs: list[TypedObservableState] = []
    lexicon = ranker.constraint_lexicon
    for sample in h.samples:
        parser = TypedStateParser(sample.get("user_profile", {}).get("preference_tags", ()), lexicon)
        for trace in h.trace_by_session()[sample["sample_id"]]:
            state = parser.update(trace.message, trace.turn)
            # Copy the mutable state so later turns cannot mutate earlier rows.
            typed_inputs.append(TypedObservableState(
                category=state.category,
                intent_seed=state.intent_seed,
                constraints=[TypedConstraint(**vars(item)) for item in state.constraints],
                negations=[TypedConstraint(**vars(item)) for item in state.negations],
                profile_tags=state.profile_tags,
                asked_attributes=list(state.asked_attributes),
                explicit_buying_requirement=state.explicit_buying_requirement,
                turn=state.turn,
            ))
    identity_slates = []
    identity_latencies: list[float] = []
    for index, state in enumerate(typed_inputs):
        identity_started = time.perf_counter()
        order, _, _ = ranker.rank(state, "structured_identity")
        identity_latencies.append((time.perf_counter() - identity_started) * 1000.0)
        slate = tuple(map(int, order[:TOP_K]))
        identity_slates.append(slate)
        if slate != baseline_slates[index]:
            raise RuntimeError(f"Structured state identity changed Experiment 7 row {index}")

    calibration_sessions: dict[str, list[dict]] = {}
    calibration_rows: dict[str, list[dict]] = {}
    baseline_cal = [row for row in baseline_sessions if row["sample_id"] in calibration]
    calibration_sessions["structured_state_identity_fixed_other"] = baseline_cal
    calibration_rows["structured_state_identity_fixed_other"] = []

    specs = {
        "adaptive_hybrid_fixed_other": ("adaptive_hybrid", "fixed_other"),
        "deterministic_reranker_fixed_other": ("reranker", "fixed_other"),
        "deterministic_reranker_entropy": ("reranker", "entropy"),
        "deterministic_reranker_rank_aware": ("reranker", "rank_aware"),
    }
    for name, (stage, question) in specs.items():
        logger.info("Calibration ablation %s", name)
        sessions, rows = simulate_policy(h, ranker, calibration, stage=stage, question_method=question)
        calibration_sessions[name] = sessions
        calibration_rows[name] = rows

    clarification_names = (
        "deterministic_reranker_fixed_other",
        "deterministic_reranker_entropy",
        "deterministic_reranker_rank_aware",
    )
    interim_metrics = {name: rank_metrics(calibration_sessions[name]) for name in clarification_names}
    interim_comparisons = {name: compare_sessions(baseline_cal, calibration_sessions[name], calibration) for name in clarification_names}
    selected_clarification_name = sorted(clarification_names, key=lambda name: _selection_key(name, interim_metrics, interim_comparisons))[0]
    selected_question = specs[selected_clarification_name][1]

    profile_name = f"profile_decay_{selected_question}"
    logger.info("Final profile/decay ablation %s", profile_name)
    profile_sessions, profile_rows = simulate_policy(h, ranker, calibration, stage="profile_decay", question_method=selected_question)
    calibration_sessions[profile_name] = profile_sessions
    calibration_rows[profile_name] = profile_rows

    calibration_metrics = {name: rank_metrics(rows) for name, rows in calibration_sessions.items()}
    calibration_comparisons = {name: compare_sessions(baseline_cal, rows, calibration) for name, rows in calibration_sessions.items()}
    cumulative_candidates = (
        "structured_state_identity_fixed_other",
        "adaptive_hybrid_fixed_other",
        selected_clarification_name,
        profile_name,
    )
    selected = sorted(cumulative_candidates, key=lambda name: _selection_key(name, calibration_metrics, calibration_comparisons))[0]

    if selected == "structured_state_identity_fixed_other":
        selected_stage, selected_question_method = "structured_identity", "fixed_other"
    elif selected == "adaptive_hybrid_fixed_other":
        selected_stage, selected_question_method = "adaptive_hybrid", "fixed_other"
    elif selected == profile_name:
        selected_stage, selected_question_method = "profile_decay", selected_question
    else:
        selected_stage, selected_question_method = "reranker", specs[selected][1]

    # Held-out labels are evaluated once, only for the calibration-selected
    # treatment. The baseline is the already-frozen Experiment 7 output.
    logger.info("One-shot held-out evaluation of %s", selected)
    if selected_stage == "structured_identity":
        held_sessions = [row for row in baseline_sessions if row["sample_id"] in evaluation]
        held_rows: list[dict] = []
    else:
        held_sessions, held_rows = simulate_policy(
            h, ranker, evaluation, stage=selected_stage, question_method=selected_question_method
        )
    baseline_eval = [row for row in baseline_sessions if row["sample_id"] in evaluation]
    held_metrics = rank_metrics(held_sessions)
    baseline_metrics = {
        "full": rank_metrics(baseline_sessions),
        "calibration": rank_metrics(baseline_cal),
        "evaluation": rank_metrics(baseline_eval),
    }
    held_comparison = compare_sessions(baseline_eval, held_sessions, evaluation)
    gates = {
        "technical_score_at_least_experiment_07": held_metrics["technical_score"] >= baseline_metrics["evaluation"]["technical_score"],
        "mrr_at_least_experiment_07": held_metrics["mrr"] >= baseline_metrics["evaluation"]["mrr"],
        "rescues_at_least_one_hard_failure": held_comparison["hard_failure_rescues"] >= 1,
        "regressions_do_not_exceed_rescues": held_comparison["regressions"] <= held_comparison["hard_failure_rescues"],
    }
    promote = all(gates.values())

    ablation_rows = []
    for name in calibration_sessions:
        ablation_rows.append({
            "method": name,
            **calibration_metrics[name],
            **calibration_comparisons[name],
            "selected_clarification_branch": name == selected_clarification_name,
            "selected_cumulative_configuration": name == selected,
        })

    state_rows = []
    specificity_counts = Counter(state.specificity for state in typed_inputs)
    for trace, state in zip(h.traces, typed_inputs):
        state_rows.append({
            "sample_id": trace.sample_id,
            "split": "calibration" if trace.sample_id in calibration else "evaluation",
            "turn": trace.turn,
            "stage": "structured_identity",
            "clarification": "fixed_other",
            "category": state.category,
            "intent_seed": state.intent_seed,
            "specificity": state.specificity,
            "typed_constraints": [vars(item) for item in state.active_constraints],
            "negations": [vars(item) for item in state.negations],
            "profile_tags": list(state.profile_tags),
            "asked_attributes": list(state.asked_attributes),
            "exp7_identity_top_10": [h.ids[value] for value in identity_slates[len(state_rows)]],
            "oracle_joined_after_freeze": True,
            "oracle_scenario_type": trace.scenario_type,
        })

    all_selected_rows = [*calibration_rows[selected], *held_rows]
    if selected_stage == "structured_identity":
        all_selected_rows = list(state_rows)
        latency_values = identity_latencies
    else:
        latency_values = [row["latency_ms"] for row in all_selected_rows]
    latency = {
        "count": len(latency_values),
        "mean_ms": round(float(np.mean(latency_values)), 6) if latency_values else 0.0,
        "p50_ms": round(float(np.percentile(latency_values, 50)), 6) if latency_values else 0.0,
        "p95_ms": round(float(np.percentile(latency_values, 95)), 6) if latency_values else 0.0,
    }

    dense_cache = h.results_dir / "cache" / f"dense_{h.catalog_hash[:16]}_minilm_seq{DENSE_MAX_SEQ_LENGTH}.npy"
    source_path = Path(__file__).resolve()
    hashes = {
        "experiment_07_baseline_source": {"path": str(exp7_source.relative_to(h.repo)), "sha256": sha256(exp7_source)},
        "submission_agent_baseline": {"path": str(submission_agent.relative_to(h.repo)), "sha256": sha256(submission_agent)},
        "frozen_split": {"path": str(split_path.relative_to(h.repo)), "sha256": sha256(split_path)},
        "catalog": {"path": str(h.catalog_path.relative_to(h.repo)), "sha256": h.catalog_hash},
        "public_set": {"path": str(h.public_path.relative_to(h.repo)), "sha256": h.public_hash},
        "dense_embeddings": {"path": str(dense_cache.relative_to(h.repo)), "sha256": sha256(dense_cache)},
        "dense_model": {"model_id": MODEL_ID, "tree_sha256": _hash_tree(h.results_dir / "cache" / "models")},
        "source": {"path": str(source_path.relative_to(h.repo)), "sha256": sha256(source_path)},
    }

    metrics = {
        "experiment": EXPERIMENT_NUMBER,
        "slug": EXPERIMENT_SLUG,
        "split": split,
        "retrieval_contract": {
            "ranker_input_fields": ["category", "intent_seed", "typed_constraints", "negations", "source_turn", "strength", "profile_tags", "asked_attributes"],
            "excluded_before_freeze": ["sample_id", "scenario_type", "target_asin", "oracle_card", "future_turns"],
            "offline_deterministic_only": True,
            "no_llm_cross_encoder_vector_database_or_new_service": True,
        },
        "state_identity": {
            "turn_rankings_reproduced": len(identity_slates),
            "bit_for_bit_experiment_07_identity": True,
            "specificity_counts": dict(sorted(specificity_counts.items())),
        },
        "adaptive_weights": STATE_CONFIG,
        "reranker_formula": {
            "hybrid_retrieval": 0.35,
            "dense_similarity": 0.20,
            "hard_constraint_satisfaction": 0.25,
            "soft_preference_satisfaction": 0.10,
            "category_consistency": 0.10,
            "explicit_or_negated_violations": -0.30,
            "profile_final_ablation": {"hybrid_retrieval": 0.30, "profile_evidence": 0.05, "soft_decay": "0.9^turn_age"},
        },
        "baseline_metrics": baseline_metrics,
        "calibration_ablation_metrics": calibration_metrics,
        "calibration_comparisons": calibration_comparisons,
        "clarification_selection": {
            "candidates": list(clarification_names),
            "selected": selected_clarification_name,
            "selected_question_method": selected_question,
        },
        "selection": {
            "cumulative_candidates": list(cumulative_candidates),
            "selected_on_calibration": selected,
            "tie_break_order": ["TechnicalScore", "rescues", "fewer regressions", "MRR", "method name"],
            "held_out_evaluated_once": True,
            "held_out_metrics": held_metrics,
            "held_out_comparison_to_experiment_07": held_comparison,
            "promotion_gates": gates,
            "recommend_promotion": promote,
            "production_recommendation": selected if promote else BASELINE_METHOD,
            "submission_agent_modified": False,
        },
        "latency": latency,
        "hashes": hashes,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    _plot(directory / "ablation_comparison.png", calibration_metrics)
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "rows.csv", all_selected_rows)
    write_json(directory / "sessions.json", {
        "calibration": calibration_sessions,
        "held_out_selected": held_sessions,
        "experiment_07_baseline": baseline_sessions,
    })
    write_csv(directory / "state_diagnostics.csv", state_rows)
    write_json(directory / "state_diagnostics.json", state_rows)
    write_csv(directory / "ablation_comparisons.csv", ablation_rows)
    write_json(directory / "ablation_comparisons.json", ablation_rows)
    clarification_rows = [row for name in clarification_names for row in calibration_rows[name]]
    write_csv(directory / "clarification_diagnostics.csv", clarification_rows)
    write_json(directory / "clarification_diagnostics.json", clarification_rows)
    write_json(directory / "latency.json", latency)
    write_json(directory / "hashes.json", hashes)
    (directory / "source_snapshot.py").write_bytes(source_path.read_bytes())
    write_json(directory / "source_snapshot.json", {
        "source": hashes["source"],
        "snapshot_sha256": sha256(directory / "source_snapshot.py"),
        "identical": sha256(directory / "source_snapshot.py") == hashes["source"]["sha256"],
    })

    outcome = f"promote `{selected}`" if promote else "retain the Experiment 7 agent"
    summary = f"""# Experiment 9 — Adaptive hybrid architecture

> **AGENT-REALISTIC RANKING.** Every retrieval and clarification decision used only typed observable dialogue state and candidate metadata. Target and scenario fields were joined after each ranking was frozen.

Typed structured state reproduced all **{len(identity_slates)} Experiment 7 turn rankings** bit-for-bit. Calibration selected clarification branch **{selected_clarification_name}**, then selected cumulative configuration **{selected}** using TechnicalScore, rescues, regressions, MRR, and deterministic method-name ordering.

On its single 140-session held-out evaluation, the selected configuration scored **{held_metrics['technical_score']:.6f}** TechnicalScore and **{held_metrics['mrr']:.6f}** MRR, compared with **{baseline_metrics['evaluation']['technical_score']:.6f}** and **{baseline_metrics['evaluation']['mrr']:.6f}** for Experiment 7. It rescued **{held_comparison['hard_failure_rescues']}** hard failures and caused **{held_comparison['regressions']}** regressions. The preregistered decision is to **{outcome}**.

The submission agent was not modified automatically.
"""
    (directory / "summary.md").write_text(summary, encoding="utf-8")
    logger.info("Completed %s in %.2fs; selected=%s promote=%s", EXPERIMENT_DIRECTORY, metrics["elapsed_seconds"], selected, promote)
    return metrics
