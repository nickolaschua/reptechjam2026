"""Query-conditioned long-term-memory portability component experiment.

This module is deliberately evaluator-local.  It consumes the frozen Phase 3A
fixture and its persisted exact vectors, presents only deployable text fields to
an OpenAI judge, and never constructs a steered retrieval query.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from nickolas.memory.qlmp import BaselineConfig, MemoryPolarity, build_cosine_memory_baseline

try:
    from ..qlmp_integration import promote_q_work
    from .qlmp_component_eval import ProjectorLabel, load_projector_fixture
except ImportError:  # pragma: no cover - direct script compatibility
    from qlmp_integration import promote_q_work
    from longitudinal_eval.qlmp_component_eval import ProjectorLabel, load_projector_fixture


FIXTURE_VERSION = "portability-isolation-v1"
ANNOTATION_VERSION = "portability-label-map-v1-frozen-2026-08-30"
PROMPT_VERSION = "portability-controller-v2-frozen-2026-08-30"
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_TEMPERATURE = 0.0
PRIMARY_NEGATIVES = ("CONTEXTUAL", "CONFLICTING", "IRRELEVANT")


class PortabilityLabel(str, Enum):
    PORTABLE = "PORTABLE"
    REDUNDANT = "REDUNDANT"
    CONTEXTUAL = "CONTEXTUAL"
    CONFLICTING = "CONFLICTING"
    IRRELEVANT = "IRRELEVANT"


class ReasonCode(str, Enum):
    PERSISTENT_PREFERENCE = "persistent_preference"
    ALREADY_EXPLICIT = "already_explicit"
    EPISODE_SPECIFIC = "episode_specific"
    CURRENT_OVERRIDE_CONFLICT = "current_override_conflict"
    SCOPE_MISMATCH = "scope_mismatch"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    OTHER = "other"


class SufficiencyClass(str, Enum):
    ENOUGH_CONTEXT = "ENOUGH_CONTEXT"
    CONTEXT_LOST = "CONTEXT_LOST_DURING_MEMORY_DISTILLATION"
    AMBIGUOUS_LABEL = "AMBIGUOUS_LABEL"
    MODEL_ERROR = "MODEL_ERROR"


@dataclass(frozen=True)
class DeployableMemory:
    memory_id: str
    text: str
    source: str
    scope: str | None
    polarity: str
    embedding: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class PrivateAnnotation:
    expected_label: PortabilityLabel
    prior_projector_label: str
    hard_negative_type: str
    label_reason: str | None
    origin_sequence_index: int
    representation_sufficiency: SufficiencyClass


@dataclass(frozen=True)
class PortabilityQuery:
    fixture_id: str
    user_id: str
    session_id: str
    sequence_index: int
    turn_index: int
    split: str
    product_family: str
    current_request: str
    active_intent_text: str
    current_category: str | None
    query_scope: str | None
    q_m0: np.ndarray = field(repr=False, compare=False)
    memories: tuple[DeployableMemory, ...] = ()
    annotations_by_memory_id: Mapping[str, PrivateAnnotation] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        ids = [memory.memory_id for memory in self.memories]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate memory IDs in portability query")
        if set(ids) != set(self.annotations_by_memory_id):
            raise ValueError("private portability annotations do not align with memories")
        if any(
            value.origin_sequence_index >= self.sequence_index
            for value in self.annotations_by_memory_id.values()
        ):
            raise ValueError("future/current-session memory leakage")


@dataclass(frozen=True)
class PortabilityFixtureSet:
    fixture_version: str
    annotation_version: str
    source_fixture_sha256: str
    source_vector_sha256: str | None
    queries: tuple[PortabilityQuery, ...]


@dataclass(frozen=True)
class ModelPrediction:
    memory_id: str
    classification: PortabilityLabel
    portable_score: float
    confidence: float
    reason_code: ReasonCode


@dataclass(frozen=True)
class JudgeCall:
    predictions: tuple[ModelPrediction, ...]
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    attempts: int
    request_count: int


class PortabilityJudgeError(RuntimeError):
    """Hosted judge failed without provider or model fallback."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        latency_seconds: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.latency_seconds = latency_seconds
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


SYSTEM_PROMPT = """You are a query-conditioned long-term shopping-memory portability controller.

Judge each historical memory only against the CURRENT request and active shopping state. Historical semantic similarity does not imply portability. A catalogue-valid product direction does not imply that an old requirement belongs in today's task.

Use exactly one class per memory:
- PORTABLE: an enduring or reusable preference that legitimately adds new positive information for discriminating among plausible products now.
- REDUNDANT: relevant information already explicit in the current request or active state, so it should add no steering.
- CONTEXTUAL: a legitimate requirement from an earlier shopping episode with insufficient reason to transfer it now.
- CONFLICTING: conflicts with the explicit current request or override. Current intent always wins.
- IRRELEVANT: valid shopping history with no meaningful bearing on the current product decision.

Decision order:
1. If already explicit now, REDUNDANT.
2. If it conflicts with active current intent, CONFLICTING.
3. If it looks like a one-off earlier-task requirement rather than an enduring preference, CONTEXTUAL.
4. If unrelated or scope-mismatched, IRRELEVANT.
5. Otherwise use PORTABLE only when it provides legitimate additional positive preference information now.

A one-off historical requirement must not become permanent merely because it is semantically related. If the stored payload lacks enough origin context to establish persistence, be conservative and use CONTEXTUAL with reason_code insufficient_context. Do not infer or discuss any target product. Return only the requested structured JSON; no chain-of-thought.

portable_score is confidence that the memory should provide POSITIVE ADDITIONAL steering now. It should generally be high only for PORTABLE and low for every other class. confidence is confidence in the categorical judgement.

Return exactly one result for every supplied memory_id, in the supplied order. Never omit, merge, rename, or invent a memory_id.
"""
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


OUTPUT_SCHEMA: dict[str, Any] = {
    "name": "portability_classifications",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "memory_id": {"type": "string"},
                        "classification": {
                            "type": "string",
                            "enum": [value.value for value in PortabilityLabel],
                        },
                        "portable_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason_code": {
                            "type": "string",
                            "enum": [value.value for value in ReasonCode],
                        },
                    },
                    "required": [
                        "memory_id",
                        "classification",
                        "portable_score",
                        "confidence",
                        "reason_code",
                    ],
                },
            }
        },
        "required": ["results"],
    },
}


def output_schema_for_count(candidate_count: int) -> dict[str, Any]:
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count <= 0:
        raise ValueError("candidate_count must be a positive integer")
    schema = json.loads(json.dumps(OUTPUT_SCHEMA))
    results = schema["schema"]["properties"]["results"]
    results["minItems"] = candidate_count
    results["maxItems"] = candidate_count
    return schema


def _mapped_label(projector_label: ProjectorLabel, hard_type: str, reason: str | None) -> PortabilityLabel:
    """Frozen target-independent mapping completed before hosted judgement."""

    if projector_label is ProjectorLabel.USEFUL_ADDITIONAL_STEERING:
        return PortabilityLabel.PORTABLE
    if projector_label is ProjectorLabel.RELEVANT_BUT_REDUNDANT:
        return PortabilityLabel.REDUNDANT
    if projector_label is ProjectorLabel.IRRELEVANT:
        return PortabilityLabel.IRRELEVANT
    if hard_type == "override_conflict":
        return PortabilityLabel.CONFLICTING
    if hard_type == "contextual_requirement":
        return PortabilityLabel.CONTEXTUAL
    if hard_type in {"same_category", "nearby_non_portable"}:
        lowered = (reason or "").casefold()
        if "conflict" in lowered or "should not override" in lowered:
            return PortabilityLabel.CONFLICTING
        return PortabilityLabel.CONTEXTUAL
    if projector_label is ProjectorLabel.CROSS_DOMAIN_DISTRACTOR:
        return PortabilityLabel.IRRELEVANT
    if projector_label is ProjectorLabel.SAME_CATEGORY_HARD_NEGATIVE:
        return PortabilityLabel.IRRELEVANT
    raise ValueError(f"unmapped projector annotation {projector_label.value}/{hard_type}")


def _representation_sufficiency(
    label: PortabilityLabel,
    memory: DeployableMemory,
    hard_type: str,
) -> SufficiencyClass:
    """Preregistered audit of whether the atomic stored payload supports its label."""

    if label is PortabilityLabel.PORTABLE:
        # The adapter stores an atomic slot, not the user's "usually/across purchases"
        # provenance.  Nothing in source=user distinguishes stable from session-only.
        return SufficiencyClass.CONTEXT_LOST
    if label is PortabilityLabel.CONTEXTUAL:
        scope = (memory.scope or "").casefold()
        if "specific" in scope or hard_type == "same_category":
            return SufficiencyClass.ENOUGH_CONTEXT
        return SufficiencyClass.CONTEXT_LOST
    if label in {PortabilityLabel.CONFLICTING, PortabilityLabel.REDUNDANT}:
        return SufficiencyClass.ENOUGH_CONTEXT
    if label is PortabilityLabel.IRRELEVANT:
        return SufficiencyClass.ENOUGH_CONTEXT
    return SufficiencyClass.AMBIGUOUS_LABEL


def load_portability_fixture(path: str | Path) -> PortabilityFixtureSet:
    """Reuse the exact Phase 3A vectors while separating safe and private fields."""

    fixture_path = Path(path)
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    forbidden = {"shopper_private_persona", "constant_profile", "latent_profile", "hidden_profile"}
    for entry in raw.get("fixtures", []):
        leaked = sorted(forbidden.intersection(entry))
        if leaked:
            raise ValueError(f"private profile fields are forbidden: {leaked}")
    projector = load_projector_fixture(fixture_path)
    raw_by_id = {str(value["fixture_id"]): value for value in raw["fixtures"]}
    if len(raw_by_id) != len(projector.fixtures):
        raise ValueError("raw and materialized fixture IDs do not align")

    queries: list[PortabilityQuery] = []
    for fixture in projector.fixtures:
        fixture_id = fixture.snapshot.example_id
        raw_query = raw_by_id[fixture_id]
        memories: list[DeployableMemory] = []
        private: dict[str, PrivateAnnotation] = {}
        for item in fixture.candidate_memories.items:
            annotation = fixture.annotations_by_memory_id[item.id]
            safe_memory = DeployableMemory(
                memory_id=item.id,
                text=item.text,
                source=item.source.value,
                scope=item.scope,
                polarity=item.polarity.value,
                embedding=item.embedding,
            )
            expected = _mapped_label(
                annotation.label, annotation.hard_negative_type, annotation.label_reason
            )
            memories.append(safe_memory)
            private[item.id] = PrivateAnnotation(
                expected_label=expected,
                prior_projector_label=annotation.label.value,
                hard_negative_type=annotation.hard_negative_type,
                label_reason=annotation.label_reason,
                origin_sequence_index=annotation.origin_sequence_index,
                representation_sufficiency=_representation_sufficiency(
                    expected, safe_memory, annotation.hard_negative_type
                ),
            )
        queries.append(
            PortabilityQuery(
                fixture_id=fixture_id,
                user_id=str(fixture.snapshot.user_id),
                session_id=str(fixture.snapshot.session_id),
                sequence_index=fixture.sequence_index,
                turn_index=fixture.turn_index,
                split=fixture.split,
                product_family=fixture.product_family,
                current_request=fixture.snapshot.raw_user_message,
                active_intent_text=fixture.snapshot.effective_query_text,
                current_category=fixture.snapshot.current_category,
                query_scope=fixture.snapshot.current_scope,
                q_m0=fixture.snapshot.q_m0,
                memories=tuple(memories),
                annotations_by_memory_id=private,
            )
        )
    return PortabilityFixtureSet(
        fixture_version=FIXTURE_VERSION,
        annotation_version=ANNOTATION_VERSION,
        source_fixture_sha256=str(projector.source_sha256),
        source_vector_sha256=projector.vector_snapshot_sha256,
        queries=tuple(queries),
    )


def build_user_payload(query: PortabilityQuery, memory_ids: Sequence[str]) -> dict[str, Any]:
    """Construct a deployable-only payload in deterministic candidate order."""

    by_id = {memory.memory_id: memory for memory in query.memories}
    ordered = [str(value) for value in memory_ids]
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate requested memory IDs")
    unknown = [value for value in ordered if value not in by_id]
    if unknown:
        raise ValueError(f"unknown memory ID {unknown[0]!r}")
    return {
        "current_request": query.current_request,
        "current_active_shopping_state": {
            "effective_query_text": query.active_intent_text,
            "category": query.current_category,
            "scope": query.query_scope,
        },
        "historical_candidate_memories": [
            {
                "memory_id": by_id[memory_id].memory_id,
                "text": by_id[memory_id].text,
                "source": by_id[memory_id].source,
                "scope": by_id[memory_id].scope,
            }
            for memory_id in ordered
        ],
    }


def build_messages(query: PortabilityQuery, memory_ids: Sequence[str]) -> list[dict[str, str]]:
    payload = build_user_payload(query, memory_ids)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        },
    ]


def parse_model_output(value: str | Mapping[str, Any], expected_ids: Sequence[str]) -> tuple[ModelPrediction, ...]:
    """Strictly validate schema, range, uniqueness, coverage, and input order."""

    try:
        payload = json.loads(value) if isinstance(value, str) else dict(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON model output") from exc
    if set(payload) != {"results"} or not isinstance(payload["results"], list):
        raise ValueError("model output must contain only a results array")
    expected = [str(value) for value in expected_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected memory IDs must be unique")
    parsed: list[ModelPrediction] = []
    seen: set[str] = set()
    required = {"memory_id", "classification", "portable_score", "confidence", "reason_code"}
    for raw in payload["results"]:
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("each result must contain exactly the structured output fields")
        memory_id = str(raw["memory_id"])
        if memory_id in seen:
            raise ValueError(f"duplicate model memory ID {memory_id!r}")
        seen.add(memory_id)
        if memory_id not in expected:
            raise ValueError(f"unknown model memory ID {memory_id!r}")
        try:
            label = PortabilityLabel(raw["classification"])
            reason = ReasonCode(raw["reason_code"])
            score = float(raw["portable_score"])
            confidence = float(raw["confidence"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid categorical or numeric model output") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("portable_score must be finite and in [0, 1]")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        parsed.append(ModelPrediction(memory_id, label, score, confidence, reason))
    if seen != set(expected):
        missing = sorted(set(expected).difference(seen))
        raise ValueError(f"missing model memory IDs: {missing}")
    by_id = {value.memory_id: value for value in parsed}
    return tuple(by_id[memory_id] for memory_id in expected)


class OpenAIPortabilityJudge:
    """One explicit OpenAI model, strict structured output, and no fallback."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        client: Any | None = None,
    ) -> None:
        if not str(api_key).strip():
            raise ValueError("an explicit OpenAI API key is required")
        if not str(model).strip():
            raise ValueError("an explicit OpenAI model is required")
        if temperature != 0.0:
            raise ValueError("scientific portability runs require temperature=0")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
        self.client = client
        self.model = str(model)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)

    def classify(self, query: PortabilityQuery, memory_ids: Sequence[str]) -> JudgeCall:
        ordered = tuple(str(value) for value in memory_ids)
        messages = build_messages(query, ordered)
        failures: list[str] = []
        total_latency = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        for attempt in range(1, self.max_retries + 2):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": output_schema_for_count(len(ordered)),
                    },
                )
                elapsed = time.perf_counter() - started
                total_latency += elapsed
                usage = getattr(response, "usage", None)
                input_tokens = int(
                    getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0
                )
                output_tokens = int(
                    getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0
                )
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                content = response.choices[0].message.content
                predictions = parse_model_output(content, ordered)
                return JudgeCall(
                    predictions=predictions,
                    latency_seconds=total_latency,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    attempts=attempt,
                    request_count=attempt,
                )
            except Exception as exc:  # explicit retry, same provider/model/prompt only
                total_latency += time.perf_counter() - started
                failures.append(f"{type(exc).__name__}: {exc}")
        raise PortabilityJudgeError(
            f"OpenAI portability judgement failed after {len(failures)} attempt(s): "
            + " | ".join(failures),
            attempts=len(failures),
            latency_seconds=total_latency,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )


def raw_cosine(query: PortabilityQuery, memory: DeployableMemory) -> float:
    q = promote_q_work(query.q_m0, dimension=query.q_m0.size)
    return float(np.dot(q, memory.embedding))


def select_cosine_top_k(query: PortabilityQuery, k: int) -> tuple[str, ...]:
    """Use the unchanged B2 implementation for scope, polarity, ranking, and ties."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    from nickolas.memory.qlmp import MemoryItem, MemorySource

    items = tuple(
        MemoryItem(
            id=memory.memory_id,
            text=memory.text,
            embedding=memory.embedding,
            source=MemorySource(memory.source),
            polarity=MemoryPolarity(memory.polarity),
            scope=memory.scope,
        )
        for memory in query.memories
    )
    result = build_cosine_memory_baseline(
        promote_q_work(query.q_m0, dimension=query.q_m0.size),
        items,
        query_scope=query.query_scope,
        config=BaselineConfig(memory_top_k=k),
    )
    return result.selected_memory_ids


def candidate_recall(fixture: PortabilityFixtureSet, ks: Sequence[int] = (3, 5, 10)) -> dict[str, Any]:
    portable_pairs = [
        (query, memory.memory_id)
        for query in fixture.queries
        for memory in query.memories
        if memory.polarity == MemoryPolarity.POSITIVE.value
        and query.annotations_by_memory_id[memory.memory_id].expected_label
        is PortabilityLabel.PORTABLE
    ]
    result: dict[str, Any] = {"portable_count": len(portable_pairs), "by_k": {}}
    for k in ks:
        selected_by_query = {
            query.fixture_id: set(select_cosine_top_k(query, int(k)))
            for query in fixture.queries
        }
        hits = sum(
            memory_id in selected_by_query[query.fixture_id]
            for query, memory_id in portable_pairs
        )
        result["by_k"][str(k)] = {
            "hits": int(hits),
            "total": len(portable_pairs),
            "recall": (None if not portable_pairs else hits / len(portable_pairs)),
        }
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if not positives or not negatives:
        return None
    ranks = _average_ranks(scores)
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _auprc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.count_nonzero(labels == 1))
    negatives = int(np.count_nonzero(labels == 0))
    if not positives or not negatives:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positive = false_positive = 0
    prior_recall = area = 0.0
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        true_positive += int(np.count_nonzero(group == 1))
        false_positive += int(np.count_nonzero(group == 0))
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - prior_recall) * precision
        prior_recall = recall
        start = end
    return float(area)


def _binary_rows(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        value
        for value in records
        if value["memory_polarity"] == MemoryPolarity.POSITIVE.value
        and value["expected_label"] in {PortabilityLabel.PORTABLE.value, *PRIMARY_NEGATIVES}
    ]


def _binary_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chosen = _binary_rows(records)
    labels = np.asarray(
        [1 if value["expected_label"] == PortabilityLabel.PORTABLE.value else 0 for value in chosen],
        dtype=np.int8,
    )
    result: dict[str, Any] = {"pair_count": len(chosen), "positive_count": int(labels.sum())}
    for field in ("raw_cosine", "portable_score"):
        scores = np.asarray([float(value[field]) for value in chosen], dtype=np.float64)
        result[field] = {"auroc": _auroc(labels, scores), "auprc": _auprc(labels, scores)}
    predicted = np.asarray(
        [1 if value["predicted_class"] == PortabilityLabel.PORTABLE.value else 0 for value in chosen],
        dtype=np.int8,
    )
    tp = int(np.count_nonzero((labels == 1) & (predicted == 1)))
    fp = int(np.count_nonzero((labels == 0) & (predicted == 1)))
    fn = int(np.count_nonzero((labels == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    result["classification"] = {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
    }
    return result


def _multiclass_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chosen = [value for value in records if value["memory_polarity"] == MemoryPolarity.POSITIVE.value]
    labels = [value.value for value in PortabilityLabel]
    matrix = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for value in chosen:
        matrix[value["expected_label"]][value["predicted_class"]] += 1
    per_class: dict[str, Any] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {"support": sum(matrix[label].values()), "precision": precision, "recall": recall, "f1": f1}
        recalls.append(recall)
        f1s.append(f1)
    return {
        "pair_count": len(chosen),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _cluster_bootstrap(
    records: Sequence[Mapping[str, Any]], *, unit_field: str, samples: int, seed: int
) -> dict[str, Any]:
    chosen = _binary_rows(records)
    units = sorted({str(value[unit_field]) for value in chosen})
    if samples <= 0 or len(units) < 2:
        return {"available": False, "unit": unit_field, "unit_count": len(units)}
    grouped = {unit: [value for value in chosen if str(value[unit_field]) == unit] for unit in units}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"raw_auroc": [], "raw_auprc": [], "portable_auroc": [], "portable_auprc": [], "auroc_gain": [], "auprc_gain": []}
    for _ in range(samples):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sample = [row for unit in sampled_units for row in grouped[str(unit)]]
        metrics = _binary_metrics(sample)
        raw = metrics["raw_cosine"]
        portable = metrics["portable_score"]
        if None in (raw["auroc"], raw["auprc"], portable["auroc"], portable["auprc"]):
            continue
        values["raw_auroc"].append(raw["auroc"])
        values["raw_auprc"].append(raw["auprc"])
        values["portable_auroc"].append(portable["auroc"])
        values["portable_auprc"].append(portable["auprc"])
        values["auroc_gain"].append(portable["auroc"] - raw["auroc"])
        values["auprc_gain"].append(portable["auprc"] - raw["auprc"])
    intervals: dict[str, Any] = {}
    for name, observed in values.items():
        if observed:
            low, high = np.quantile(observed, [0.025, 0.975])
            intervals[name] = {"lower_95": float(low), "upper_95": float(high), "successful_samples": len(observed)}
        else:
            intervals[name] = None
    return {"available": bool(values["raw_auroc"]), "unit": unit_field, "unit_count": len(units), "samples_requested": samples, "seed": seed, "intervals": intervals}


def _hard_negative_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chosen = [value for value in records if value["memory_polarity"] == MemoryPolarity.POSITIVE.value]
    result: dict[str, Any] = {}
    for label in PRIMARY_NEGATIVES:
        rows = [value for value in chosen if value["expected_label"] == label]
        rejected = sum(value["predicted_class"] != PortabilityLabel.PORTABLE.value for value in rows)
        result[label] = {"count": len(rows), "rejected": rejected, "rejection_accuracy": None if not rows else rejected / len(rows)}
    nearby_types = {"same_category", "contextual_requirement", "override_conflict", "nearby_non_portable"}
    nearby = [value for value in chosen if value["hard_negative_type"] in nearby_types and value["expected_label"] in PRIMARY_NEGATIVES]
    rejected = sum(value["predicted_class"] != PortabilityLabel.PORTABLE.value for value in nearby)
    result["nearby_same_category"] = {"count": len(nearby), "rejected": rejected, "rejection_accuracy": None if not nearby else rejected / len(nearby)}
    return result


def _no_useful_history(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fixture_ids = sorted({str(value["fixture_id"]) for value in records})
    grouped = {fixture_id: [value for value in records if value["fixture_id"] == fixture_id and value["memory_polarity"] == MemoryPolarity.POSITIVE.value] for fixture_id in fixture_ids}
    no_useful = [fixture_id for fixture_id, rows in grouped.items() if not any(value["expected_label"] == PortabilityLabel.PORTABLE.value for value in rows)]
    considered = [value for fixture_id in no_useful for value in grouped[fixture_id]]
    false_portable = sum(value["predicted_class"] == PortabilityLabel.PORTABLE.value for value in considered)
    return {"query_count": len(no_useful), "fixture_ids": no_useful, "pair_count": len(considered), "false_portable_count": false_portable, "false_portable_rate": None if not considered else false_portable / len(considered)}


def _negative_polarity_diagnostic(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    chosen = [
        value
        for value in records
        if value["memory_polarity"] == MemoryPolarity.NEGATIVE.value
    ]
    correct = sum(value["predicted_class"] == value["expected_label"] for value in chosen)
    predicted_counts = {value.value: 0 for value in PortabilityLabel}
    for value in chosen:
        predicted_counts[value["predicted_class"]] += 1
    return {
        "pair_count": len(chosen),
        "correct": correct,
        "accuracy": None if not chosen else correct / len(chosen),
        "predicted_class_counts": predicted_counts,
        "included_in_primary_metrics": False,
    }


def summarize_predictions(
    records: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = 1000, bootstrap_seed: int = 2603
) -> dict[str, Any]:
    records = list(records)
    primary_records = [
        value
        for value in records
        if value["memory_polarity"] == MemoryPolarity.POSITIVE.value
    ]
    errors = [
        value
        for value in primary_records
        if value["predicted_class"] != value["expected_label"]
    ]
    audit_counts = {value.value: 0 for value in SufficiencyClass}
    for value in errors:
        sufficiency = value["representation_sufficiency"]
        tag = SufficiencyClass.MODEL_ERROR.value if sufficiency == SufficiencyClass.ENOUGH_CONTEXT.value else sufficiency
        audit_counts[tag] += 1
    return {
        "binary": _binary_metrics(records),
        "multiclass": _multiclass_metrics(records),
        "hard_negatives": _hard_negative_metrics(records),
        "no_useful_history": _no_useful_history(records),
        "negative_polarity_diagnostic": _negative_polarity_diagnostic(records),
        "query_bootstrap_95_ci": _cluster_bootstrap(records, unit_field="fixture_id", samples=bootstrap_samples, seed=bootstrap_seed),
        "user_bootstrap_95_ci": _cluster_bootstrap(records, unit_field="user_id", samples=bootstrap_samples, seed=bootstrap_seed + 1),
        "classification_error_count": len(errors),
        "classification_error_audit": audit_counts,
    }


def _prediction_record(
    query: PortabilityQuery,
    memory: DeployableMemory,
    prediction: ModelPrediction | None,
    *,
    evaluation_mode: str,
    candidate_rank: int | None,
    top_k: int | None,
    call: JudgeCall | None,
) -> dict[str, Any]:
    annotation = query.annotations_by_memory_id[memory.memory_id]
    retrieved = candidate_rank is not None or evaluation_mode == "judge_capability_all"
    predicted_class = prediction.classification.value if prediction else PortabilityLabel.IRRELEVANT.value
    return {
        "fixture_id": query.fixture_id,
        "user_id": query.user_id,
        "session_id": query.session_id,
        "sequence_index": query.sequence_index,
        "turn_index": query.turn_index,
        "split": query.split,
        "product_family": query.product_family,
        "evaluation_mode": evaluation_mode,
        "memory_id": memory.memory_id,
        "memory_text": memory.text,
        "memory_source": memory.source,
        "memory_scope": memory.scope,
        "memory_polarity": memory.polarity,
        "expected_label": annotation.expected_label.value,
        "prior_projector_label": annotation.prior_projector_label,
        "hard_negative_type": annotation.hard_negative_type,
        "label_reason": annotation.label_reason,
        "representation_sufficiency": annotation.representation_sufficiency.value,
        "predicted_class": predicted_class,
        "portable_score": prediction.portable_score if prediction else 0.0,
        "confidence": prediction.confidence if prediction else 0.0,
        "reason_code": prediction.reason_code.value if prediction else "candidate_not_retrieved",
        "raw_cosine": raw_cosine(query, memory),
        "candidate_rank": candidate_rank,
        "candidate_top_k": top_k,
        "in_candidate_set": retrieved,
        "prediction_status": "judged" if prediction else "not_retrieved",
        "call_latency_seconds": None if call is None else call.latency_seconds,
        "call_attempts": None if call is None else call.attempts,
        "call_input_tokens": None if call is None else call.input_tokens,
        "call_output_tokens": None if call is None else call.output_tokens,
    }


def evaluate_with_judge(
    fixture: PortabilityFixtureSet,
    judge: OpenAIPortabilityJudge,
    *,
    query_ids: Sequence[str] | None = None,
    deployable_k: int = 10,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    selected_queries = [
        query for query in fixture.queries if query_ids is None or query.fixture_id in set(query_ids)
    ]
    records: list[dict[str, Any]] = []
    query_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    calls: list[JudgeCall] = []
    failed_call_stats: list[dict[str, Any]] = []
    for evaluation_mode in ("judge_capability_all", "deployable_cosine_top10"):
        for query in selected_queries:
            if evaluation_mode == "judge_capability_all":
                candidate_ids = tuple(memory.memory_id for memory in query.memories)
                ranks = {memory_id: rank for rank, memory_id in enumerate(candidate_ids, 1)}
                top_k = None
            else:
                candidate_ids = select_cosine_top_k(query, deployable_k)
                ranks = {memory_id: rank for rank, memory_id in enumerate(candidate_ids, 1)}
                top_k = deployable_k
            call: JudgeCall | None = None
            predictions: dict[str, ModelPrediction] = {}
            if candidate_ids:
                try:
                    call = judge.classify(query, candidate_ids)
                    calls.append(call)
                    predictions = {value.memory_id: value for value in call.predictions}
                except PortabilityJudgeError as exc:
                    failure = {
                        "fixture_id": query.fixture_id,
                        "evaluation_mode": evaluation_mode,
                        "error": str(exc),
                        "attempts": exc.attempts,
                        "latency_seconds": exc.latency_seconds,
                        "input_tokens": exc.input_tokens,
                        "output_tokens": exc.output_tokens,
                    }
                    failures.append(failure)
                    failed_call_stats.append(failure)
            for memory in query.memories:
                records.append(
                    _prediction_record(
                        query,
                        memory,
                        predictions.get(memory.memory_id),
                        evaluation_mode=evaluation_mode,
                        candidate_rank=ranks.get(memory.memory_id),
                        top_k=top_k,
                        call=call if memory.memory_id in ranks else None,
                    )
                )
            query_records.append(
                {
                    "fixture_id": query.fixture_id,
                    "user_id": query.user_id,
                    "evaluation_mode": evaluation_mode,
                    "candidate_count": len(candidate_ids),
                    "memory_count": len(query.memories),
                    "portable_expected_count": sum(
                        query.annotations_by_memory_id[memory.memory_id].expected_label is PortabilityLabel.PORTABLE
                        and memory.polarity == MemoryPolarity.POSITIVE.value
                        for memory in query.memories
                    ),
                    "portable_predicted_count": sum(
                        value.classification is PortabilityLabel.PORTABLE for value in predictions.values()
                    ),
                    "call_succeeded": call is not None or not candidate_ids,
                    "latency_seconds": None if call is None else call.latency_seconds,
                    "input_tokens": None if call is None else call.input_tokens,
                    "output_tokens": None if call is None else call.output_tokens,
                }
            )
    summaries = {
        mode: summarize_predictions(
            [value for value in records if value["evaluation_mode"] == mode],
            bootstrap_samples=bootstrap_samples,
        )
        for mode in ("judge_capability_all", "deployable_cosine_top10")
    }
    latencies = [value.latency_seconds for value in calls] + [
        float(value["latency_seconds"]) for value in failed_call_stats
    ]
    usage = {
        "hosted_calls": sum(value.request_count for value in calls)
        + sum(int(value["attempts"]) for value in failed_call_stats),
        "successful_query_calls": len(calls),
        "input_tokens": sum(value.input_tokens for value in calls)
        + sum(int(value["input_tokens"]) for value in failed_call_stats),
        "output_tokens": sum(value.output_tokens for value in calls)
        + sum(int(value["output_tokens"]) for value in failed_call_stats),
        "mean_latency_seconds": None if not latencies else statistics.fmean(latencies),
        "median_latency_seconds": None if not latencies else statistics.median(latencies),
        "failures": len(failures),
        "retries": sum(max(0, value.attempts - 1) for value in calls)
        + sum(max(0, int(value["attempts"]) - 1) for value in failed_call_stats),
    }
    return {
        "predictions": records,
        "query_summary": query_records,
        "mode_summaries": summaries,
        "failures": failures,
        "usage": usage,
    }


def fixture_coverage(fixture: PortabilityFixtureSet, query_ids: Sequence[str] | None = None) -> dict[str, Any]:
    selected = [query for query in fixture.queries if query_ids is None or query.fixture_id in set(query_ids)]
    counts = {value.value: 0 for value in PortabilityLabel}
    negative = 0
    positive_pairs = 0
    for query in selected:
        for memory in query.memories:
            if memory.polarity == MemoryPolarity.NEGATIVE.value:
                negative += 1
                continue
            positive_pairs += 1
            counts[query.annotations_by_memory_id[memory.memory_id].expected_label.value] += 1
    no_useful = sum(
        not any(
            memory.polarity == MemoryPolarity.POSITIVE.value
            and query.annotations_by_memory_id[memory.memory_id].expected_label is PortabilityLabel.PORTABLE
            for memory in query.memories
        )
        for query in selected
    )
    return {
        "users": len({query.user_id for query in selected}),
        "independent_queries": len(selected),
        "memory_pairs": sum(len(query.memories) for query in selected),
        "positive_polarity_pairs": positive_pairs,
        "label_counts": counts,
        "u4_negative_polarity_diagnostic": negative,
        "no_useful_history_states": no_useful,
    }


def estimated_prompt_tokens(fixture: PortabilityFixtureSet, query_ids: Sequence[str] | None = None, deployable_k: int = 10) -> dict[str, Any]:
    selected = [query for query in fixture.queries if query_ids is None or query.fixture_id in set(query_ids)]
    character_count = 0
    candidate_count = 0
    for mode in ("all", "top_k"):
        for query in selected:
            ids = (
                tuple(memory.memory_id for memory in query.memories)
                if mode == "all"
                else select_cosine_top_k(query, deployable_k)
            )
            candidate_count += len(ids)
            messages = build_messages(query, ids)
            character_count += sum(len(value["content"]) for value in messages)
    return {
        "query_count": len(selected),
        "candidate_memory_presentations": candidate_count,
        "expected_llm_calls": len(selected) * 2,
        "estimated_prompt_tokens_chars_div_4": int(math.ceil(character_count / 4)),
        "estimation_method": "UTF-8 prompt characters / 4; actual provider usage is authoritative",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(value), sort_keys=True, ensure_ascii=False, default=_json_default) + "\n" for value in rows), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for value in rows for key in value})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for value in rows:
            writer.writerow(value)


def scientific_verdict(summary: Mapping[str, Any], coverage: Mapping[str, Any]) -> dict[str, str]:
    capability = summary["judge_capability_all"]
    binary = capability["binary"]
    raw_auc = binary["raw_cosine"]["auroc"]
    portable_auc = binary["portable_score"]["auroc"]
    gain = None if raw_auc is None or portable_auc is None else portable_auc - raw_auc
    audit = capability["classification_error_audit"]
    lost = int(audit.get(SufficiencyClass.CONTEXT_LOST.value, 0))
    query_ci = capability["query_bootstrap_95_ci"].get("intervals", {}).get("auroc_gain")
    if lost > 0:
        return {
            "verdict": "PORTABILITY INCONCLUSIVE",
            "reason": "MEMORY PROVENANCE INSUFFICIENT: atomic MemoryItem text lost persistence/episode provenance for classification errors",
        }
    if coverage["users"] < 3 or not query_ci:
        return {"verdict": "PORTABILITY INCONCLUSIVE", "reason": "sample or query-level uncertainty is insufficient for the harsh gate"}
    hard = capability["hard_negatives"]
    no_history = capability["no_useful_history"]
    broadly_correct = all(
        hard[label]["rejection_accuracy"] is not None and hard[label]["rejection_accuracy"] >= 0.8
        for label in ("CONTEXTUAL", "CONFLICTING")
    )
    if (
        gain is not None
        and gain >= 0.05
        and query_ci["lower_95"] > 0.0
        and broadly_correct
        and (no_history["false_portable_rate"] or 0.0) <= 0.1
    ):
        return {"verdict": "PORTABILITY GO", "reason": "portable score clears cosine, hard-negative, no-history, and clustered-uncertainty gates"}
    return {"verdict": "PORTABILITY STOP", "reason": "deployable portability evidence does not clear the preregistered harsh gate"}


def write_artifacts(
    output_dir: str | Path,
    *,
    fixture: PortabilityFixtureSet,
    evaluation: Mapping[str, Any],
    manifest: Mapping[str, Any],
    query_ids: Sequence[str] | None = None,
    scientific: bool = True,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    expected = {"predictions.jsonl", "predictions.csv", "query_summary.jsonl", "summary.json", "run_manifest.json", "report.md"}
    existing = sorted(name for name in expected if (destination / name).exists())
    if existing:
        raise FileExistsError(f"refusing to overwrite portability artifacts: {existing}")
    coverage = fixture_coverage(fixture, query_ids)
    recall = candidate_recall(
        PortabilityFixtureSet(
            fixture.fixture_version,
            fixture.annotation_version,
            fixture.source_fixture_sha256,
            fixture.source_vector_sha256,
            tuple(query for query in fixture.queries if query_ids is None or query.fixture_id in set(query_ids)),
        )
    )
    summary = {
        "coverage": coverage,
        "candidate_recall": recall,
        "judge_capability_all": evaluation["mode_summaries"]["judge_capability_all"],
        "deployable_cosine_top10": evaluation["mode_summaries"]["deployable_cosine_top10"],
        "usage": evaluation["usage"],
        "failures": evaluation["failures"],
    }
    summary["decision"] = (
        scientific_verdict(summary, coverage)
        if scientific
        else {
            "verdict": "NOT_EVALUATED_DRY_RUN",
            "reason": "engineering dry run only; no scientific interpretation",
        }
    )
    _write_jsonl(destination / "predictions.jsonl", evaluation["predictions"])
    _write_csv(destination / "predictions.csv", evaluation["predictions"])
    _write_jsonl(destination / "query_summary.jsonl", evaluation["query_summary"])
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "run_manifest.json", dict(manifest))
    report = _render_report(summary)
    (destination / "report.md").write_text(report, encoding="utf-8")
    return destination


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _render_report(summary: Mapping[str, Any]) -> str:
    coverage = summary["coverage"]
    capability = summary["judge_capability_all"]
    binary = capability["binary"]
    multi = capability["multiclass"]
    recall = summary["candidate_recall"]["by_k"]
    decision = summary["decision"]
    return f"""# Query-conditioned memory portability isolation

- Fixture: {coverage['independent_queries']} queries / {coverage['memory_pairs']} memory pairs / {coverage['users']} users
- Positive-polarity label counts: {json.dumps(coverage['label_counts'], sort_keys=True)}
- U4 negative-polarity diagnostic pairs: {coverage['u4_negative_polarity_diagnostic']}
- Candidate PORTABLE recall: Top-3 {_fmt(recall['3']['recall'])}, Top-5 {_fmt(recall['5']['recall'])}, Top-10 {_fmt(recall['10']['recall'])}

## Primary binary comparison (judge capability)

- Raw cosine: AUROC {_fmt(binary['raw_cosine']['auroc'])}, AUPRC {_fmt(binary['raw_cosine']['auprc'])}
- Portability score: AUROC {_fmt(binary['portable_score']['auroc'])}, AUPRC {_fmt(binary['portable_score']['auprc'])}
- PORTABLE classification: precision {_fmt(binary['classification']['precision'])}, recall {_fmt(binary['classification']['recall'])}, F1 {_fmt(binary['classification']['f1'])}

## Multiclass

- Macro F1: {_fmt(multi['macro_f1'])}
- Balanced accuracy: {_fmt(multi['balanced_accuracy'])}
- Confusion matrix: `{json.dumps(multi['confusion_matrix'], sort_keys=True)}`

## Hard negatives and no-useful-history states

- Rejection: `{json.dumps(capability['hard_negatives'], sort_keys=True)}`
- No-useful-history: `{json.dumps(capability['no_useful_history'], sort_keys=True)}`
- Error information audit: `{json.dumps(capability['classification_error_audit'], sort_keys=True)}`

## Cost and latency

`{json.dumps(summary['usage'], sort_keys=True)}`

## Verdict

{decision['verdict']}

{decision['reason']}
"""


__all__ = [
    "ANNOTATION_VERSION",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "FIXTURE_VERSION",
    "OpenAIPortabilityJudge",
    "OUTPUT_SCHEMA",
    "PROMPT_SHA256",
    "PROMPT_VERSION",
    "PortabilityFixtureSet",
    "PortabilityJudgeError",
    "PortabilityLabel",
    "PortabilityQuery",
    "ReasonCode",
    "SufficiencyClass",
    "build_messages",
    "build_user_payload",
    "candidate_recall",
    "estimated_prompt_tokens",
    "evaluate_with_judge",
    "fixture_coverage",
    "load_portability_fixture",
    "output_schema_for_count",
    "parse_model_output",
    "raw_cosine",
    "select_cosine_top_k",
    "summarize_predictions",
    "write_artifacts",
]
