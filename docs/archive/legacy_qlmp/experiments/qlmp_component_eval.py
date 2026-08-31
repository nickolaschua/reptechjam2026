"""Exact-q component evaluator for the QLMP projector stop/go experiment.

The loader accepts only already-embedded query and memory vectors.  It never
calls a shopper, state distiller, embedding provider, or target-aware selector.
Private labels are joined to label-free projection results only after QLMP has
finished computing them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nickolas.memory.qlmp import MemoryItem, MemoryPolarity, MemorySource

try:
    from ..agent import DenseQuerySnapshot
    from ..embedding_backends import OPENAI_EMBEDDING_SPACE_ID
    from ..memory_store import StoredMemory
    from ..qlmp_integration import (
        CandidateMemoryBatch,
        CandidateUniverse,
        MemoryMode,
        ProjectorIsolationResult,
        QLMPIntegrationConfig,
        run_projector_isolation,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from agent import DenseQuerySnapshot
    from embedding_backends import OPENAI_EMBEDDING_SPACE_ID
    from memory_store import StoredMemory
    from qlmp_integration import (
        CandidateMemoryBatch,
        CandidateUniverse,
        MemoryMode,
        ProjectorIsolationResult,
        QLMPIntegrationConfig,
        run_projector_isolation,
    )


FIXTURE_VERSION = "projector-isolation-v1"
PRIMARY_POSITIVE = "USEFUL_ADDITIONAL_STEERING"
PRIMARY_NEGATIVES = (
    "IRRELEVANT",
    "SAME_CATEGORY_HARD_NEGATIVE",
    "CROSS_DOMAIN_DISTRACTOR",
)
SCORE_FIELDS = ("raw_cosine", "rho", "projected_norm")
DISTRIBUTION_FIELDS = ("raw_cosine", "tangent_norm", "rho", "projected_norm")
VALID_QUERY_MODES = frozenset({"Buying", "Browsing"})
VALID_SPLITS = frozenset({"development", "held_out"})
VALID_HARD_NEGATIVE_TYPES = frozenset(
    {
        "none",
        "same_category",
        "contextual_requirement",
        "override_conflict",
        "nearby_non_portable",
        "easy_distractor",
    }
)


class ProjectorLabel(str, Enum):
    USEFUL_ADDITIONAL_STEERING = "USEFUL_ADDITIONAL_STEERING"
    RELEVANT_BUT_REDUNDANT = "RELEVANT_BUT_REDUNDANT"
    IRRELEVANT = "IRRELEVANT"
    SAME_CATEGORY_HARD_NEGATIVE = "SAME_CATEGORY_HARD_NEGATIVE"
    CROSS_DOMAIN_DISTRACTOR = "CROSS_DOMAIN_DISTRACTOR"


@dataclass(frozen=True)
class ProjectorMemoryAnnotation:
    """Evaluator-private provenance and label; never passed into QLMP."""

    label: ProjectorLabel | str
    origin_user_id: str
    origin_session_id: str
    origin_sequence_index: int
    label_reason: str | None = None
    hard_negative_type: str = "none"

    def __post_init__(self) -> None:
        try:
            label = ProjectorLabel(self.label)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid projector label") from exc
        if not str(self.origin_user_id).strip() or not str(self.origin_session_id).strip():
            raise ValueError("memory origin user/session IDs are required")
        if (
            isinstance(self.origin_sequence_index, bool)
            or not isinstance(self.origin_sequence_index, int)
            or self.origin_sequence_index < 0
        ):
            raise ValueError("memory origin_sequence_index must be non-negative")
        if self.hard_negative_type not in VALID_HARD_NEGATIVE_TYPES:
            raise ValueError("invalid hard_negative_type")
        object.__setattr__(self, "label", label)


@dataclass(frozen=True)
class ProjectorFixture:
    """Safe component input plus evaluator-private labels/metadata."""

    snapshot: DenseQuerySnapshot
    candidate_memories: CandidateMemoryBatch
    annotations_by_memory_id: Mapping[str, ProjectorMemoryAnnotation]
    sequence_index: int
    turn_index: int
    buying_or_browsing_label: str
    split: str
    product_family: str
    has_entangled_memory: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, DenseQuerySnapshot):
            raise ValueError("snapshot must be a DenseQuerySnapshot")
        if not isinstance(self.candidate_memories, CandidateMemoryBatch):
            raise ValueError("candidate_memories must be a CandidateMemoryBatch")
        if self.snapshot.embedding_space_id != self.candidate_memories.embedding_space_id:
            raise ValueError("query and memory embedding spaces differ")
        annotations: dict[str, ProjectorMemoryAnnotation] = {}
        for memory_id, annotation in self.annotations_by_memory_id.items():
            if not isinstance(annotation, ProjectorMemoryAnnotation):
                raise ValueError(f"invalid projector annotation for {memory_id!r}")
            annotations[str(memory_id)] = annotation
        memory_ids = {item.id for item in self.candidate_memories.items}
        if len(memory_ids) != len(self.candidate_memories.items):
            raise ValueError("memory IDs must be unique within a fixture")
        if set(annotations) != memory_ids:
            missing = sorted(memory_ids.difference(annotations))
            extra = sorted(set(annotations).difference(memory_ids))
            raise ValueError(
                f"annotations must match candidate memory IDs; missing={missing}, extra={extra}"
            )
        for name in ("sequence_index", "turn_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.snapshot.user_id or not self.snapshot.session_id:
            raise ValueError("fixture user_id and session_id are required")
        for annotation in annotations.values():
            if annotation.origin_user_id != self.snapshot.user_id:
                raise ValueError("cross-user memory leakage")
            if annotation.origin_sequence_index >= self.sequence_index:
                raise ValueError("future/current-session memory leakage")
        if self.buying_or_browsing_label not in VALID_QUERY_MODES:
            raise ValueError("buying_or_browsing_label must be Buying or Browsing")
        if self.split not in VALID_SPLITS:
            raise ValueError("split must be development or held_out")
        if not str(self.product_family).strip():
            raise ValueError("product_family is required")
        object.__setattr__(self, "annotations_by_memory_id", annotations)

    @property
    def labels_by_memory_id(self) -> Mapping[str, ProjectorLabel]:
        return {key: value.label for key, value in self.annotations_by_memory_id.items()}


@dataclass(frozen=True)
class ProjectorFixtureSet:
    fixture_version: str
    embedding_space_id: str
    candidate_universe: CandidateUniverse
    fixtures: tuple[ProjectorFixture, ...]
    source_sha256: str | None = None
    vector_snapshot_sha256: str | None = None


@dataclass(frozen=True)
class ProjectorEvaluation:
    pair_records: tuple[dict[str, Any], ...]
    query_diagnostics: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def candidate_batch_from_records(
    records: Iterable[StoredMemory], *, expected_embedding_space_id: str
) -> CandidateMemoryBatch:
    """Reuse chronologically ordered store envelopes without fetching history."""

    ordered = tuple(records)
    if any(not isinstance(record, StoredMemory) for record in ordered):
        raise ValueError("records must contain only StoredMemory values")
    if any(record.embedding_space_id != expected_embedding_space_id for record in ordered):
        raise ValueError("stored memory belongs to a different embedding space")
    return CandidateMemoryBatch(
        items=tuple(record.item for record in ordered),
        embedding_space_id=expected_embedding_space_id,
    )


def _memory_from_payload(value: Mapping[str, Any]) -> MemoryItem:
    timestamp = value.get("timestamp")
    return MemoryItem(
        id=str(value["id"]),
        text=str(value["text"]),
        embedding=np.asarray(value["embedding"], dtype=np.float64),
        source=MemorySource(value.get("source", MemorySource.USER.value)),
        polarity=MemoryPolarity(value.get("polarity", MemoryPolarity.POSITIVE.value)),
        scope=None if value.get("scope") is None else str(value["scope"]),
        timestamp=None if timestamp is None else datetime.fromisoformat(str(timestamp)),
        confidence=float(value.get("confidence", 1.0)),
    )


def load_projector_fixture(path: str | Path) -> ProjectorFixtureSet:
    """Load an exact-q fixture; absent vectors are a hard error, never embedded."""

    fixture_path = Path(path)
    raw_bytes = fixture_path.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))
    if payload.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError(f"fixture_version must be {FIXTURE_VERSION!r}")
    embedding_space_id = str(payload.get("embedding_space_id", ""))
    if not embedding_space_id:
        raise ValueError("fixture embedding_space_id is required")
    if embedding_space_id != OPENAI_EMBEDDING_SPACE_ID:
        raise ValueError("fixture must use the canonical 3072-dimensional OpenAI M0 space")
    try:
        universe = CandidateUniverse(payload["candidate_universe"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fixture candidate_universe must be explicit") from exc
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise ValueError("fixture must contain a non-empty fixtures list")
    vector_map: dict[str, np.ndarray] = {}
    vector_snapshot_sha256: str | None = None
    vector_snapshot = payload.get("vector_snapshot")
    if vector_snapshot is not None:
        if not isinstance(vector_snapshot, Mapping) or not vector_snapshot.get("path"):
            raise ValueError("vector_snapshot must declare a relative path")
        vector_path = fixture_path.parent / str(vector_snapshot["path"])
        vector_bytes_hash = _sha256_file(vector_path)
        expected_hash = str(vector_snapshot.get("sha256", ""))
        if not expected_hash or vector_bytes_hash != expected_hash:
            raise ValueError("vector snapshot SHA-256 mismatch")
        with np.load(vector_path, allow_pickle=False) as data:
            if set(data.files) != {"keys", "vectors"}:
                raise ValueError("vector snapshot must contain keys and vectors")
            keys = [str(value) for value in data["keys"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float64)
        if vectors.shape != (len(keys), 3072) or len(set(keys)) != len(keys):
            raise ValueError("vector snapshot keys/vectors are not aligned and unique")
        vector_map = {key: vectors[index] for index, key in enumerate(keys)}
        vector_snapshot_sha256 = vector_bytes_hash

    fixtures: list[ProjectorFixture] = []
    seen_fixture_ids: set[str] = set()
    for raw in raw_fixtures:
        if not isinstance(raw, Mapping):
            raise ValueError("every fixture entry must be an object")
        if "q_m0" not in raw and "q_m0_key" not in raw:
            raise ValueError("every fixture must persist exact q_m0; embedding is disabled")
        fixture_id = str(raw.get("fixture_id", ""))
        if not fixture_id or fixture_id in seen_fixture_ids:
            raise ValueError("fixture_id values must be unique and non-empty")
        seen_fixture_ids.add(fixture_id)
        forbidden = {"shopper_private_persona", "constant_profile", "latent_profile"}
        leaked = sorted(forbidden.intersection(raw))
        if leaked:
            raise ValueError(f"private profile fields are forbidden: {leaked}")
        raw_memories = raw.get("memories")
        if not isinstance(raw_memories, list):
            raise ValueError(f"fixture {fixture_id!r} memories must be a list")
        if any(not isinstance(item, Mapping) for item in raw_memories):
            raise ValueError("every memory must be an object")
        if any("embedding" not in item and "embedding_key" not in item for item in raw_memories):
            raise ValueError("every memory must persist an embedding; embedding is disabled")
        if any("label" not in item for item in raw_memories):
            raise ValueError("every memory must have an evaluator-private projector label")
        if any(
            item.get("embedding_space_id") != embedding_space_id
            for item in raw_memories
        ):
            raise ValueError("every memory must declare the fixture embedding space")
        materialized_memories: list[dict[str, Any]] = []
        for item in raw_memories:
            materialized = dict(item)
            if "embedding" not in materialized:
                key = str(materialized["embedding_key"])
                if key not in vector_map:
                    raise ValueError(f"missing vector snapshot key {key!r}")
                materialized["embedding"] = vector_map[key]
            materialized_memories.append(materialized)
        memories = tuple(_memory_from_payload(item) for item in materialized_memories)
        for item in memories:
            if item.embedding.shape != (3072,):
                raise ValueError("every memory embedding must have shape [3072]")
            if not np.isclose(
                float(np.linalg.norm(item.embedding)), 1.0, atol=1e-12, rtol=1e-12
            ):
                raise ValueError("every memory embedding must be float64 L2-normalized")
        annotations = {
            item.id: ProjectorMemoryAnnotation(
                label=raw_value["label"],
                label_reason=(
                    None if raw_value.get("label_reason") is None
                    else str(raw_value["label_reason"])
                ),
                origin_user_id=str(raw_value.get("origin_user_id", "")),
                origin_session_id=str(raw_value.get("origin_session_id", "")),
                origin_sequence_index=raw_value.get("origin_sequence_index"),
                hard_negative_type=str(raw_value.get("hard_negative_type", "none")),
            )
            for item, raw_value in zip(memories, raw_memories)
        }
        if "q_m0" in raw:
            frozen_q = raw["q_m0"]
        else:
            q_key = str(raw["q_m0_key"])
            if q_key not in vector_map:
                raise ValueError(f"missing vector snapshot key {q_key!r}")
            frozen_q = vector_map[q_key]
        snapshot = DenseQuerySnapshot(
            example_id=fixture_id,
            raw_user_message=str(raw.get("current_message", "")),
            effective_query_text=str(raw.get("effective_query_text", "")),
            query_embedding=np.asarray(frozen_q, dtype=np.float32),
            target_product_id=(
                None if raw.get("target_product_id") is None
                else str(raw["target_product_id"])
            ),
            current_scope=(
                None if raw.get("query_scope") is None else str(raw["query_scope"])
            ),
            current_category=(
                None if raw.get("current_category") is None
                else str(raw["current_category"])
            ),
            user_id=None if raw.get("user_id") is None else str(raw["user_id"]),
            session_id=(
                None if raw.get("session_id") is None else str(raw["session_id"])
            ),
            embedding_space_id=embedding_space_id,
        )
        if snapshot.q_m0.shape != (3072,):
            raise ValueError("every exact q_m0 must have shape [3072]")
        fixtures.append(
            ProjectorFixture(
                snapshot=snapshot,
                candidate_memories=CandidateMemoryBatch(
                    items=memories,
                    embedding_space_id=embedding_space_id,
                ),
                annotations_by_memory_id=annotations,
                sequence_index=raw.get("sequence_index"),
                turn_index=raw.get("turn_index"),
                buying_or_browsing_label=str(raw.get("buying_or_browsing_label", "")),
                split=str(raw.get("split", "")),
                product_family=str(raw.get("product_family", "")),
                has_entangled_memory=bool(raw.get("has_entangled_memory", False)),
            )
        )
    return ProjectorFixtureSet(
        fixture_version=FIXTURE_VERSION,
        embedding_space_id=embedding_space_id,
        candidate_universe=universe,
        fixtures=tuple(fixtures),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        vector_snapshot_sha256=vector_snapshot_sha256,
    )


def _pair_records_after_projection(
    fixture: ProjectorFixture,
    result: ProjectorIsolationResult,
) -> tuple[dict[str, Any], ...]:
    """The only function that joins private labels to computed projections."""

    annotations = fixture.annotations_by_memory_id
    snapshot = fixture.snapshot
    memory_by_id = {item.id: item for item in fixture.candidate_memories.items}
    records: list[dict[str, Any]] = []
    for value in result.memory_projections:
        projection = value.projection
        records.append(
            {
                "fixture_id": snapshot.fixture_id,
                "user_id": snapshot.user_id,
                "session_id": snapshot.session_id,
                "sequence_index": fixture.sequence_index,
                "turn_index": fixture.turn_index,
                "buying_or_browsing_label": fixture.buying_or_browsing_label,
                "split": fixture.split,
                "product_family": fixture.product_family,
                "has_entangled_memory": fixture.has_entangled_memory,
                "memory_id": value.memory_id,
                "label": annotations[value.memory_id].label.value,
                "label_reason": annotations[value.memory_id].label_reason,
                "origin_user_id": annotations[value.memory_id].origin_user_id,
                "origin_session_id": annotations[value.memory_id].origin_session_id,
                "origin_sequence_index": annotations[value.memory_id].origin_sequence_index,
                "hard_negative_type": annotations[value.memory_id].hard_negative_type,
                "memory_text": value.memory_text,
                "query_scope": snapshot.query_scope,
                "current_category": snapshot.current_category,
                "memory_scope": value.memory_scope,
                "memory_source": value.memory_source,
                "memory_polarity": memory_by_id[value.memory_id].polarity.value,
                "memory_confidence": value.memory_confidence,
                "raw_cosine": projection.raw_query_memory_cosine,
                "tangent_norm": projection.tangent_norm,
                "rho": projection.projection_fraction,
                "projected_norm": projection.projected_norm,
                "requested_local_k": result.requested_local_k,
                "actual_local_k": result.local_subspace.local_product_count,
                "requested_rank": result.local_subspace.requested_rank,
                "effective_rank": result.local_subspace.effective_rank,
                "candidate_universe": result.candidate_universe.value,
                "projection_fallback_reason": result.fallback_reason,
            }
        )
    return tuple(records)


def evaluate_projector_fixtures(
    scorer: Any,
    fixture_set: ProjectorFixtureSet,
    *,
    config: QLMPIntegrationConfig,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 2603,
) -> ProjectorEvaluation:
    """Compute label-free projections, then evaluator-owned records/metrics."""

    if config.memory_mode is not MemoryMode.PROJECTION:
        raise ValueError("component evaluation requires memory_mode='projection'")
    if config.embedding_space_id != fixture_set.embedding_space_id:
        raise ValueError("config and fixture embedding spaces differ")
    if config.candidate_universe is not fixture_set.candidate_universe:
        raise ValueError("config and fixture candidate universes differ")

    pair_records: list[dict[str, Any]] = []
    query_diagnostics: list[dict[str, Any]] = []
    for fixture in fixture_set.fixtures:
        # Targets, labels, scenario type, and private profiles are absent from
        # this call.  Only the frozen q/scope-free memory batch reaches QLMP.
        result = run_projector_isolation(
            scorer,
            q_m0=fixture.snapshot.q_m0,
            candidate_memories=fixture.candidate_memories,
            config=config,
        )
        pair_records.extend(_pair_records_after_projection(fixture, result))
        query_diagnostics.append(
            {
                "fixture_id": fixture.snapshot.fixture_id,
                "user_id": fixture.snapshot.user_id,
                "session_id": fixture.snapshot.session_id,
                "sequence_index": fixture.sequence_index,
                "turn_index": fixture.turn_index,
                "buying_or_browsing_label": fixture.buying_or_browsing_label,
                "split": fixture.split,
                "product_family": fixture.product_family,
                "has_entangled_memory": fixture.has_entangled_memory,
                "raw_current_message": fixture.snapshot.raw_user_message,
                "effective_query_text": fixture.snapshot.effective_query_text,
                "query_scope": fixture.snapshot.query_scope,
                "current_category": fixture.snapshot.current_category,
                "requested_local_k": result.requested_local_k,
                "actual_local_k": result.local_subspace.local_product_count,
                "top_k_product_ids": list(result.initial_dense_result.product_ids),
                "top_k_scores": [
                    float(value) for value in result.initial_dense_result.scores
                ],
                "requested_rank": result.local_subspace.requested_rank,
                "effective_rank": result.local_subspace.effective_rank,
                "singular_values": [
                    float(value) for value in result.local_subspace.singular_values
                ],
                "candidate_universe": result.candidate_universe.value,
                "fallback_reason": result.fallback_reason,
            }
        )
    summary = summarize_projector_records(
        pair_records,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary["query_diagnostics"] = query_diagnostics
    return ProjectorEvaluation(
        pair_records=tuple(pair_records),
        query_diagnostics=tuple(query_diagnostics),
        summary=summary,
    )


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
    true_positive = 0
    false_positive = 0
    prior_recall = 0.0
    area = 0.0
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
    return area


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
    }


def _binary_rows(
    records: Sequence[Mapping[str, Any]], negative_labels: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    chosen = [
        value
        for value in records
        if value.get("memory_polarity", "positive") != MemoryPolarity.NEGATIVE.value
        and (value["label"] == PRIMARY_POSITIVE or value["label"] in negative_labels)
    ]
    labels = np.asarray(
        [1 if value["label"] == PRIMARY_POSITIVE else 0 for value in chosen],
        dtype=np.int8,
    )
    scores = {
        field: np.asarray([float(value[field]) for value in chosen], dtype=np.float64)
        for field in SCORE_FIELDS
    }
    return labels, scores


def _query_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    negative_labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    fixture_ids = sorted({str(value["fixture_id"]) for value in records})
    if samples <= 0 or len(fixture_ids) < 2:
        return {"unit": "query", "samples_requested": samples, "available": False}
    grouped = {
        fixture_id: [value for value in records if str(value["fixture_id"]) == fixture_id]
        for fixture_id in fixture_ids
    }
    rng = np.random.default_rng(seed)
    collected: dict[str, dict[str, list[float]]] = {
        field: {"auroc": [], "auprc": []} for field in SCORE_FIELDS
    }
    gains: dict[str, dict[str, list[float]]] = {
        field: {"auroc": [], "auprc": []}
        for field in SCORE_FIELDS
        if field != "raw_cosine"
    }
    for _ in range(samples):
        sampled_ids = rng.choice(fixture_ids, size=len(fixture_ids), replace=True)
        sampled = [value for fixture_id in sampled_ids for value in grouped[str(fixture_id)]]
        labels, scores = _binary_rows(sampled, negative_labels)
        sample_metrics: dict[str, tuple[float | None, float | None]] = {}
        for field in SCORE_FIELDS:
            auc = _auroc(labels, scores[field])
            ap = _auprc(labels, scores[field])
            sample_metrics[field] = (auc, ap)
            if auc is not None:
                collected[field]["auroc"].append(auc)
            if ap is not None:
                collected[field]["auprc"].append(ap)
        cosine_auc, cosine_ap = sample_metrics["raw_cosine"]
        for field in gains:
            auc, ap = sample_metrics[field]
            if auc is not None and cosine_auc is not None:
                gains[field]["auroc"].append(auc - cosine_auc)
            if ap is not None and cosine_ap is not None:
                gains[field]["auprc"].append(ap - cosine_ap)
    result: dict[str, Any] = {
        "unit": "query",
        "samples_requested": samples,
        "seed": seed,
        "available": True,
        "scores": {},
        "gain_over_raw_cosine": {},
    }
    for field, metrics in collected.items():
        result["scores"][field] = {}
        for metric, values in metrics.items():
            if values:
                low, high = np.quantile(values, [0.025, 0.975])
                result["scores"][field][metric] = {
                    "successful_samples": len(values),
                    "lower_95": float(low),
                    "upper_95": float(high),
                }
            else:
                result["scores"][field][metric] = None
    for field, metrics in gains.items():
        result["gain_over_raw_cosine"][field] = {}
        for metric, values in metrics.items():
            if values:
                low, high = np.quantile(values, [0.025, 0.975])
                result["gain_over_raw_cosine"][field][metric] = {
                    "successful_samples": len(values),
                    "lower_95": float(low),
                    "upper_95": float(high),
                }
            else:
                result["gain_over_raw_cosine"][field][metric] = None
    return result


def _legacy_decision(summary: Mapping[str, Any]) -> dict[str, str]:
    counts = summary["primary_binary"]["counts"]
    queries = summary["coverage"]["query_count"]
    hard_count = summary["label_counts"].get("same_category_hard_negative", 0)
    if queries < 3 or counts["positive"] < 5 or counts["negative"] < 5 or hard_count < 3:
        return {
            "verdict": "PROJECTOR INCONCLUSIVE — EVALUATOR DATA INSUFFICIENT",
            "reason": "requires at least 3 queries, 5 positives, 5 negatives, and 3 same-category hard negatives",
        }
    metrics = summary["primary_binary"]["scores"]
    cosine_auc = metrics["raw_cosine"]["auroc"]
    cosine_ap = metrics["raw_cosine"]["auprc"]
    projection_candidates = [metrics["rho"], metrics["projected_norm"]]
    best = max(projection_candidates, key=lambda value: value["auroc"])
    hard = summary["negative_subgroups"]["same_category_hard_negative"]["scores"]
    hard_gain = max(
        hard["rho"]["auroc"] - hard["raw_cosine"]["auroc"],
        hard["projected_norm"]["auroc"] - hard["raw_cosine"]["auroc"],
    )
    if best["auroc"] >= cosine_auc + 0.05 and best["auprc"] >= cosine_ap + 0.03 and hard_gain >= 0.03:
        return {
            "verdict": "PROJECTOR GO",
            "reason": "projection clears conservative pooled and same-category hard-negative margins over cosine",
        }
    if (
        metrics["rho"]["auroc"] <= cosine_auc + 0.01
        and metrics["projected_norm"]["auroc"] <= cosine_auc + 0.01
        and hard_gain <= 0.01
    ):
        return {
            "verdict": "PROJECTOR STOP",
            "reason": "projection does not materially improve pooled or same-category separation over cosine",
        }
    return {
        "verdict": "PROJECTOR INCONCLUSIVE",
        "reason": "evidence is mixed or below the conservative stop/go margins",
    }


def _binary_summary(
    records: Sequence[Mapping[str, Any]], negative_labels: Sequence[str]
) -> dict[str, Any]:
    labels, scores = _binary_rows(records, negative_labels)
    selected = [
        value
        for value in records
        if value.get("memory_polarity", "positive") != MemoryPolarity.NEGATIVE.value
        and (value["label"] == PRIMARY_POSITIVE or value["label"] in negative_labels)
    ]
    return {
        "counts": {
            "positive": int(np.count_nonzero(labels == 1)),
            "negative": int(np.count_nonzero(labels == 0)),
        },
        "query_count": len({value["fixture_id"] for value in selected}),
        "scores": {
            field: {"auroc": _auroc(labels, value), "auprc": _auprc(labels, value)}
            for field, value in scores.items()
        },
    }


def _gain(block: Mapping[str, Any], field: str, metric: str = "auroc") -> float | None:
    cosine = block["scores"]["raw_cosine"][metric]
    projection = block["scores"][field][metric]
    if cosine is None or projection is None:
        return None
    return float(projection - cosine)


def _decision(summary: Mapping[str, Any]) -> dict[str, str]:
    counts = summary["primary_binary"]["counts"]
    coverage = summary["coverage"]
    hard_counts = summary["hard_negative_counts"]
    insufficiencies: list[str] = []
    if coverage["query_count"] < 12:
        insufficiencies.append("fewer than 12 independent current queries")
    if coverage["user_count"] < 3:
        insufficiencies.append("fewer than 3 users")
    if counts["positive"] < 8 or counts["negative"] < 20:
        insufficiencies.append("fewer than 8 positives or 20 primary negatives")
    if coverage["positive_user_count"] < 2:
        insufficiencies.append("useful positives do not span at least 2 users")
    if hard_counts.get("same_category", 0) < 8:
        insufficiencies.append("fewer than 8 same-category hard negatives")
    if hard_counts.get("contextual_requirement", 0) < 5:
        insufficiencies.append("fewer than 5 contextual requirements")
    if hard_counts.get("override_conflict", 0) < 2:
        insufficiencies.append("fewer than 2 override conflicts")
    if insufficiencies:
        return {"verdict": "PROJECTOR INCONCLUSIVE", "reason": "; ".join(insufficiencies)}

    primary = summary["primary_binary"]
    hard = summary["hard_negative_binary"]
    primary_gains = {
        field: (_gain(primary, field), _gain(primary, field, "auprc"))
        for field in ("rho", "projected_norm")
    }
    best_field = max(
        primary_gains,
        key=lambda field: -np.inf
        if primary_gains[field][0] is None
        else primary_gains[field][0],
    )
    auc_gain, ap_gain = primary_gains[best_field]
    hard_gain = _gain(hard, best_field)
    contextual_gain = _gain(summary["hard_negative_types"]["contextual_requirement"], best_field)
    override_gain = _gain(summary["hard_negative_types"]["override_conflict"], best_field)
    user_directional = sum(
        1
        for block in summary["by_user"].values()
        if block["counts"]["positive"] >= 2
        and block["counts"]["negative"] >= 2
        and (_gain(block, best_field) or 0.0) > 0.0
    )
    split_gains = [
        _gain(block, best_field)
        for block in summary["by_split"].values()
        if block["counts"]["positive"] and block["counts"]["negative"]
    ]
    bootstrap_gain = (
        summary["query_bootstrap_95_ci"]
        .get("gain_over_raw_cosine", {})
        .get(best_field, {})
        .get("auroc")
    )
    if (
        auc_gain is not None
        and ap_gain is not None
        and hard_gain is not None
        and contextual_gain is not None
        and override_gain is not None
        and auc_gain >= 0.05
        and ap_gain >= 0.03
        and hard_gain >= 0.03
        and contextual_gain >= 0.0
        and override_gain >= 0.0
        and user_directional >= 2
        and len(split_gains) >= 2
        and all(value is not None and value >= 0.0 for value in split_gains)
        and bootstrap_gain is not None
        and bootstrap_gain["lower_95"] > 0.0
    ):
        return {
            "verdict": "PROJECTOR GO",
            "reason": f"{best_field} clears pooled, hard-negative, cross-user, split, and query-bootstrap gates",
        }

    both_primary_flat = all(
        gain is not None and gain <= 0.01 for gain, _ in primary_gains.values()
    )
    both_hard_flat = all(
        _gain(hard, field) is not None and _gain(hard, field) <= 0.01
        for field in ("rho", "projected_norm")
    )
    contextual = summary["hard_negative_types"]["contextual_requirement"]
    override = summary["hard_negative_types"]["override_conflict"]
    held_out = summary["by_split"].get("held_out")
    difficult_groups_flat = all(
        _gain(block, field) is not None and _gain(block, field) <= 0.01
        for block in (contextual, override)
        for field in ("rho", "projected_norm")
    )
    held_out_flat = held_out is not None and all(
        _gain(held_out, field) is not None and _gain(held_out, field) <= 0.01
        for field in ("rho", "projected_norm")
    )
    bootstrap_rejects_material_gain = all(
        summary["query_bootstrap_95_ci"]["gain_over_raw_cosine"][field]["auroc"]
        is not None
        and summary["query_bootstrap_95_ci"]["gain_over_raw_cosine"][field]["auroc"][
            "upper_95"
        ]
        <= 0.01
        for field in ("rho", "projected_norm")
    )
    if (
        both_primary_flat
        and both_hard_flat
        and difficult_groups_flat
        and held_out_flat
        and bootstrap_rejects_material_gain
    ):
        return {
            "verdict": "PROJECTOR STOP",
            "reason": (
                "neither projection metric beats cosine overall, on hard/contextual/override negatives, "
                "or held-out queries; query-bootstrap upper bounds exclude a material AUROC gain"
            ),
        }
    return {
        "verdict": "PROJECTOR INCONCLUSIVE",
        "reason": "projection evidence is mixed, subgroup-inconsistent, or below the conservative margins",
    }


def _legacy_summarize_projector_records(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 2603,
) -> dict[str, Any]:
    if not records:
        return {
            "coverage": {"pair_count": 0, "query_count": 0, "user_count": 0},
            "decision": {
                "verdict": "PROJECTOR INCONCLUSIVE — EVALUATOR DATA INSUFFICIENT",
                "reason": "no labelled query-memory pairs",
            },
        }
    label_counts = {
        label.value: sum(value["label"] == label.value for value in records)
        for label in ProjectorLabel
    }
    distributions = {
        label.value: {
            field: _distribution(
                [float(value[field]) for value in records if value["label"] == label.value]
            )
            for field in SCORE_FIELDS
        }
        for label in ProjectorLabel
    }
    labels, scores = _binary_rows(records, PRIMARY_NEGATIVES)
    primary_scores = {
        field: {"auroc": _auroc(labels, value), "auprc": _auprc(labels, value)}
        for field, value in scores.items()
    }
    negative_subgroups: dict[str, Any] = {}
    for negative in PRIMARY_NEGATIVES:
        subgroup_labels, subgroup_scores = _binary_rows(records, [negative])
        negative_subgroups[negative] = {
            "negative_count": int(np.count_nonzero(subgroup_labels == 0)),
            "scores": {
                field: {
                    "auroc": _auroc(subgroup_labels, value),
                    "auprc": _auprc(subgroup_labels, value),
                }
                for field, value in subgroup_scores.items()
            },
        }
    users = {value.get("user_id") for value in records if value.get("user_id") is not None}
    categories = {
        value.get("current_category")
        for value in records
        if value.get("current_category") is not None
    }
    null_records = [value for value in records if value["label"] in PRIMARY_NEGATIVES]
    summary: dict[str, Any] = {
        "primary_label_policy": {
            "positive": PRIMARY_POSITIVE,
            "negative": list(PRIMARY_NEGATIVES),
            "excluded_diagnostic_group": "relevant_but_redundant",
        },
        "coverage": {
            "pair_count": len(records),
            "query_count": len({value["fixture_id"] for value in records}),
            "user_count": len(users),
            "category_count": len(categories),
            "users": sorted(str(value) for value in users),
            "categories": sorted(str(value) for value in categories),
        },
        "label_counts": label_counts,
        "label_distributions": distributions,
        "primary_binary": {
            "counts": {
                "positive": int(np.count_nonzero(labels == 1)),
                "negative": int(np.count_nonzero(labels == 0)),
            },
            "scores": primary_scores,
        },
        "negative_subgroups": negative_subgroups,
        "empirical_null": {
            field: _distribution([float(value[field]) for value in null_records])
            for field in SCORE_FIELDS
        },
        "query_bootstrap_95_ci": _query_bootstrap(
            records,
            negative_labels=PRIMARY_NEGATIVES,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }
    summary["decision"] = _decision(summary)
    return summary


def summarize_projector_records(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 2603,
) -> dict[str, Any]:
    if not records:
        return {
            "coverage": {"pair_count": 0, "query_count": 0, "user_count": 0},
            "decision": {"verdict": "PROJECTOR INCONCLUSIVE", "reason": "no labelled query-memory pairs"},
        }
    label_counts = {
        label.value: sum(value["label"] == label.value for value in records)
        for label in ProjectorLabel
    }
    distributions = {
        label.value: {
            field: _distribution(
                [float(value[field]) for value in records if value["label"] == label.value]
            )
            for field in DISTRIBUTION_FIELDS
        }
        for label in ProjectorLabel
    }
    primary = _binary_summary(records, PRIMARY_NEGATIVES)
    negative_subgroups = {
        negative: _binary_summary(records, [negative]) for negative in PRIMARY_NEGATIVES
    }
    users = {value.get("user_id") for value in records if value.get("user_id") is not None}
    categories = {
        value.get("current_category")
        for value in records
        if value.get("current_category") is not None
    }
    primary_null_records = [
        value
        for value in records
        if value["label"] in PRIMARY_NEGATIVES
        and value.get("memory_polarity", "positive") != MemoryPolarity.NEGATIVE.value
    ]
    hard_types = (
        "same_category",
        "contextual_requirement",
        "override_conflict",
        "nearby_non_portable",
        "easy_distractor",
    )
    hard_type_blocks: dict[str, Any] = {}
    hard_negative_counts: dict[str, int] = {}
    positives = [
        value
        for value in records
        if value["label"] == PRIMARY_POSITIVE
        and value.get("memory_polarity", "positive") != MemoryPolarity.NEGATIVE.value
    ]
    for hard_type in hard_types:
        negatives = [
            value
            for value in records
            if (
                value["label"] == ProjectorLabel.SAME_CATEGORY_HARD_NEGATIVE.value
                if hard_type == "same_category"
                else value.get("hard_negative_type") == hard_type
            )
            and value["label"] in PRIMARY_NEGATIVES
            and value.get("memory_polarity", "positive")
            != MemoryPolarity.NEGATIVE.value
        ]
        hard_negative_counts[hard_type] = len(negatives)
        block = _binary_summary(positives + negatives, PRIMARY_NEGATIVES)
        block["negative_distribution"] = {
            field: _distribution([float(value[field]) for value in negatives])
            for field in DISTRIBUTION_FIELDS
        }
        hard_type_blocks[hard_type] = block
    hard_records = positives + [
        value
        for value in records
        if value.get("hard_negative_type")
        in {
            "same_category",
            "contextual_requirement",
            "override_conflict",
            "nearby_non_portable",
        }
        and value["label"] in PRIMARY_NEGATIVES
        and value.get("memory_polarity", "positive") != MemoryPolarity.NEGATIVE.value
    ]
    grouping_fields = {
        "by_user": "user_id",
        "by_product_family": "product_family",
        "by_split": "split",
        "by_query_mode": "buying_or_browsing_label",
    }
    grouped_summaries: dict[str, Any] = {}
    for output_name, field_name in grouping_fields.items():
        groups = sorted(
            {value.get(field_name) for value in records if value.get(field_name) is not None},
            key=str,
        )
        grouped_summaries[output_name] = {
            str(group): _binary_summary(
                [value for value in records if value.get(field_name) == group],
                PRIMARY_NEGATIVES,
            )
            for group in groups
        }
    positive_users = {value.get("user_id") for value in positives}
    fixture_ids = {value["fixture_id"] for value in records}
    useful_fixture_ids = {value["fixture_id"] for value in positives}
    negative_polarity = [
        value
        for value in records
        if value.get("memory_polarity") == MemoryPolarity.NEGATIVE.value
    ]
    summary: dict[str, Any] = {
        "primary_label_policy": {
            "positive": PRIMARY_POSITIVE,
            "negative": list(PRIMARY_NEGATIVES),
            "excluded_diagnostic_group": ProjectorLabel.RELEVANT_BUT_REDUNDANT.value,
            "negative_polarity_excluded": True,
        },
        "coverage": {
            "pair_count": len(records),
            "query_count": len(fixture_ids),
            "user_count": len(users),
            "category_count": len(categories),
            "users": sorted(str(value) for value in users),
            "categories": sorted(str(value) for value in categories),
            "positive_user_count": len(positive_users),
            "positive_users": sorted(str(value) for value in positive_users),
            "no_useful_memory_query_count": len(fixture_ids - useful_fixture_ids),
            "no_useful_memory_fixture_ids": sorted(fixture_ids - useful_fixture_ids),
            "entangled_memory_query_count": len(
                {value["fixture_id"] for value in records if value.get("has_entangled_memory")}
            ),
            "query_mode_counts": {
                mode: len(
                    {
                        value["fixture_id"]
                        for value in records
                        if value.get("buying_or_browsing_label") == mode
                    }
                )
                for mode in sorted(VALID_QUERY_MODES)
            },
        },
        "label_counts": label_counts,
        "label_distributions": distributions,
        "primary_binary": primary,
        "hard_negative_binary": _binary_summary(hard_records, PRIMARY_NEGATIVES),
        "negative_subgroups": negative_subgroups,
        "hard_negative_counts": hard_negative_counts,
        "hard_negative_types": hard_type_blocks,
        "redundant_memories": {
            field: _distribution(
                [
                    float(value[field])
                    for value in records
                    if value["label"] == ProjectorLabel.RELEVANT_BUT_REDUNDANT.value
                ]
            )
            for field in DISTRIBUTION_FIELDS
        },
        "negative_polarity": {
            "count": len(negative_polarity),
            "distributions": {
                field: _distribution([float(value[field]) for value in negative_polarity])
                for field in DISTRIBUTION_FIELDS
            },
        },
        "empirical_null": {
            "all_primary_negatives": {
                field: _distribution([float(value[field]) for value in primary_null_records])
                for field in DISTRIBUTION_FIELDS
            },
            "hard_negatives": {
                field: _distribution(
                    [float(value[field]) for value in hard_records if value["label"] in PRIMARY_NEGATIVES]
                )
                for field in DISTRIBUTION_FIELDS
            },
            "easy_distractors": hard_type_blocks["easy_distractor"]["negative_distribution"],
        },
        "analytical_null": {
            "dimension": 3072,
            "rank": 16,
            "r_over_d_minus_1": 16.0 / 3071.0,
            "percent": 100.0 * 16.0 / 3071.0,
            "production_threshold": False,
        },
        "query_bootstrap_95_ci": _query_bootstrap(
            records,
            negative_labels=PRIMARY_NEGATIVES,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }
    summary.update(grouped_summaries)
    summary["decision"] = _decision(summary)
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_freeze(project_root: str | Path) -> dict[str, Any]:
    """Hash meaningful M0 and QLMP source/config, excluding caches/secrets."""

    root = Path(project_root)
    shopping = root / "nickolas" / "shopping_agent"
    qlmp = root / "nickolas" / "memory" / "qlmp"
    m0_paths = [
        shopping / "agent.py",
        shopping / "agent_openai.py",
        shopping / "embedding_backends.py",
        shopping / "run_m0.py",
        shopping / "configs" / "m0_openai.json",
        shopping / "tests" / "test_dense_vector_interface.py",
        shopping / "tests" / "test_state_routing.py",
    ]
    qlmp_paths = sorted(
        [path for path in qlmp.glob("*.py")]
        + [path for path in (qlmp / "tests").glob("test_*.py")]
        + [qlmp / "README.md"]
    )
    component_paths = [
        shopping / "longitudinal_eval" / "qlmp_component_eval.py",
        shopping / "longitudinal_eval" / "build_projector_fixture.py",
        shopping / "run_projector_isolation.py",
        shopping / "tests" / "test_m0_qlmp_contract.py",
    ]

    def render(paths: Sequence[Path]) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in paths
            if path.is_file()
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "m0": render(m0_paths),
        "qlmp_phase_1_2": render(qlmp_paths),
        "component_harness": render(component_paths),
    }


def build_run_manifest(
    scorer: Any,
    fixture_set: ProjectorFixtureSet,
    config: QLMPIntegrationConfig,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    fixtures = fixture_set.fixtures
    annotations = [
        annotation
        for fixture in fixtures
        for annotation in fixture.annotations_by_memory_id.values()
    ]
    label_counts = {
        label.value: sum(annotation.label is label for annotation in annotations)
        for label in ProjectorLabel
    }
    return {
        "run_type": "projector_isolation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_version": fixture_set.fixture_version,
        "fixture_sha256": fixture_set.source_sha256,
        "fixture_vector_snapshot_sha256": fixture_set.vector_snapshot_sha256,
        "embedding_space_id": fixture_set.embedding_space_id,
        "candidate_universe": config.candidate_universe.value,
        "local_k": config.local_k,
        "rank": config.projection.rank,
        "projection_epsilon": config.projection.epsilon,
        "fixture_count": len(fixtures),
        "query_count": len(fixtures),
        "memory_pair_count": len(annotations),
        "label_counts": label_counts,
        "negative_polarity_count": sum(
            item.polarity is MemoryPolarity.NEGATIVE
            for fixture in fixtures
            for item in fixture.candidate_memories.items
        ),
        "users": sorted(
            {value.snapshot.user_id for value in fixtures if value.snapshot.user_id}
        ),
        "sessions": sorted(
            {value.snapshot.session_id for value in fixtures if value.snapshot.session_id}
        ),
        "buying_or_browsing_labels": sorted(
            {value.buying_or_browsing_label for value in fixtures}
        ),
        "splits": sorted({value.split for value in fixtures}),
        "product_families": sorted({value.product_family for value in fixtures}),
        "catalogue_row_count": int(np.asarray(scorer.catalog_embeddings).shape[0]),
        "catalogue_fingerprint": getattr(scorer, "catalog_fingerprint", None),
        "product_text_fingerprint": getattr(scorer, "product_text_fingerprint", None),
        "embedding_cache": {
            "path": str(getattr(scorer, "embedding_cache_path", "")),
            "hashed": False,
            "reused_without_rebuild": True,
        },
        "projector_replay_external_calls": {"llm": 0, "openai": 0},
        "source_freeze": build_source_freeze(project_root),
        "b3_retrieval_steering_enabled": False,
    }


def write_projector_artifacts(
    output_root: str | Path,
    run_id: str,
    evaluation: ProjectorEvaluation,
    manifest: Mapping[str, Any],
) -> Path:
    """Write run-specific JSONL/CSV/summary/manifest without overwriting."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    pair_jsonl = run_dir / "projector_pairs.jsonl"
    with pair_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in evaluation.pair_records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    pair_csv = run_dir / "projector_pairs.csv"
    fieldnames = list(evaluation.pair_records[0]) if evaluation.pair_records else []
    with pair_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(evaluation.pair_records)
    query_jsonl = run_dir / "projector_queries.jsonl"
    with query_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in evaluation.query_diagnostics:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    query_csv = run_dir / "projector_queries.csv"
    query_fields = list(evaluation.query_diagnostics[0]) if evaluation.query_diagnostics else []
    with query_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=query_fields)
        if query_fields:
            writer.writeheader()
            writer.writerows(evaluation.query_diagnostics)
    (run_dir / "summary.json").write_text(
        json.dumps(evaluation.summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_dir


__all__ = [
    "FIXTURE_VERSION",
    "PRIMARY_NEGATIVES",
    "PRIMARY_POSITIVE",
    "ProjectorEvaluation",
    "ProjectorFixture",
    "ProjectorFixtureSet",
    "ProjectorLabel",
    "ProjectorMemoryAnnotation",
    "build_run_manifest",
    "build_source_freeze",
    "candidate_batch_from_records",
    "evaluate_projector_fixtures",
    "load_projector_fixture",
    "summarize_projector_records",
    "write_projector_artifacts",
]
