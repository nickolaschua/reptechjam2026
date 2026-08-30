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
PRIMARY_POSITIVE = "useful_additional_steering"
PRIMARY_NEGATIVES = (
    "irrelevant",
    "same_category_hard_negative",
    "cross_domain_distractor",
)
SCORE_FIELDS = ("raw_cosine", "rho", "projected_norm")


class ProjectorLabel(str, Enum):
    USEFUL_ADDITIONAL_STEERING = "useful_additional_steering"
    RELEVANT_BUT_REDUNDANT = "relevant_but_redundant"
    IRRELEVANT = "irrelevant"
    SAME_CATEGORY_HARD_NEGATIVE = "same_category_hard_negative"
    CROSS_DOMAIN_DISTRACTOR = "cross_domain_distractor"


@dataclass(frozen=True)
class ProjectorFixture:
    """Safe component input plus evaluator-private labels/metadata."""

    snapshot: DenseQuerySnapshot
    candidate_memories: CandidateMemoryBatch
    labels_by_memory_id: Mapping[str, ProjectorLabel | str]
    sequence_index: int | None = None
    turn_index: int | None = None
    scenario_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, DenseQuerySnapshot):
            raise ValueError("snapshot must be a DenseQuerySnapshot")
        if not isinstance(self.candidate_memories, CandidateMemoryBatch):
            raise ValueError("candidate_memories must be a CandidateMemoryBatch")
        if self.snapshot.embedding_space_id != self.candidate_memories.embedding_space_id:
            raise ValueError("query and memory embedding spaces differ")
        labels: dict[str, ProjectorLabel] = {}
        for memory_id, raw_label in self.labels_by_memory_id.items():
            try:
                labels[str(memory_id)] = ProjectorLabel(raw_label)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid projector label for {memory_id!r}") from exc
        memory_ids = {item.id for item in self.candidate_memories.items}
        if set(labels) != memory_ids:
            missing = sorted(memory_ids.difference(labels))
            extra = sorted(set(labels).difference(memory_ids))
            raise ValueError(
                f"labels must match candidate memory IDs; missing={missing}, extra={extra}"
            )
        for name in ("sequence_index", "turn_index"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be None or a non-negative integer")
        object.__setattr__(self, "labels_by_memory_id", labels)


@dataclass(frozen=True)
class ProjectorFixtureSet:
    fixture_version: str
    embedding_space_id: str
    candidate_universe: CandidateUniverse
    fixtures: tuple[ProjectorFixture, ...]
    source_sha256: str | None = None


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
    try:
        universe = CandidateUniverse(payload["candidate_universe"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fixture candidate_universe must be explicit") from exc
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise ValueError("fixture must contain a non-empty fixtures list")

    fixtures: list[ProjectorFixture] = []
    seen_fixture_ids: set[str] = set()
    for raw in raw_fixtures:
        if not isinstance(raw, Mapping):
            raise ValueError("every fixture entry must be an object")
        if "q_m0" not in raw:
            raise ValueError("every fixture must persist exact q_m0; embedding is disabled")
        fixture_id = str(raw.get("fixture_id", ""))
        if not fixture_id or fixture_id in seen_fixture_ids:
            raise ValueError("fixture_id values must be unique and non-empty")
        seen_fixture_ids.add(fixture_id)
        raw_memories = raw.get("memories")
        if not isinstance(raw_memories, list):
            raise ValueError(f"fixture {fixture_id!r} memories must be a list")
        if any("embedding" not in item for item in raw_memories if isinstance(item, Mapping)):
            raise ValueError("every memory must persist an embedding; embedding is disabled")
        memories = tuple(_memory_from_payload(item) for item in raw_memories)
        labels = {
            item.id: raw_value["label"]
            for item, raw_value in zip(memories, raw_memories)
        }
        snapshot = DenseQuerySnapshot(
            example_id=fixture_id,
            raw_user_message=str(raw.get("current_message", "")),
            effective_query_text=str(raw.get("effective_query_text", "")),
            query_embedding=np.asarray(raw["q_m0"], dtype=np.float32),
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
        fixtures.append(
            ProjectorFixture(
                snapshot=snapshot,
                candidate_memories=CandidateMemoryBatch(
                    items=memories,
                    embedding_space_id=embedding_space_id,
                ),
                labels_by_memory_id=labels,
                sequence_index=raw.get("sequence_index"),
                turn_index=raw.get("turn_index"),
                scenario_type=(
                    None if raw.get("scenario_type") is None
                    else str(raw["scenario_type"])
                ),
            )
        )
    return ProjectorFixtureSet(
        fixture_version=FIXTURE_VERSION,
        embedding_space_id=embedding_space_id,
        candidate_universe=universe,
        fixtures=tuple(fixtures),
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _pair_records_after_projection(
    fixture: ProjectorFixture,
    result: ProjectorIsolationResult,
) -> tuple[dict[str, Any], ...]:
    """The only function that joins private labels to computed projections."""

    labels = fixture.labels_by_memory_id
    snapshot = fixture.snapshot
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
                "scenario_type": fixture.scenario_type,
                "memory_id": value.memory_id,
                "label": labels[value.memory_id].value,
                "memory_text": value.memory_text,
                "query_scope": snapshot.query_scope,
                "current_category": snapshot.current_category,
                "memory_scope": value.memory_scope,
                "memory_source": value.memory_source,
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
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
    }


def _binary_rows(
    records: Sequence[Mapping[str, Any]], negative_labels: Sequence[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    chosen = [
        value
        for value in records
        if value["label"] == PRIMARY_POSITIVE or value["label"] in negative_labels
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
    for _ in range(samples):
        sampled_ids = rng.choice(fixture_ids, size=len(fixture_ids), replace=True)
        sampled = [value for fixture_id in sampled_ids for value in grouped[str(fixture_id)]]
        labels, scores = _binary_rows(sampled, negative_labels)
        for field in SCORE_FIELDS:
            auc = _auroc(labels, scores[field])
            ap = _auprc(labels, scores[field])
            if auc is not None:
                collected[field]["auroc"].append(auc)
            if ap is not None:
                collected[field]["auprc"].append(ap)
    result: dict[str, Any] = {
        "unit": "query",
        "samples_requested": samples,
        "seed": seed,
        "available": True,
        "scores": {},
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
    return result


def _decision(summary: Mapping[str, Any]) -> dict[str, str]:
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


def summarize_projector_records(
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
    }


def build_run_manifest(
    scorer: Any,
    fixture_set: ProjectorFixtureSet,
    config: QLMPIntegrationConfig,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    fixtures = fixture_set.fixtures
    return {
        "run_type": "projector_isolation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_version": fixture_set.fixture_version,
        "fixture_sha256": fixture_set.source_sha256,
        "embedding_space_id": fixture_set.embedding_space_id,
        "candidate_universe": config.candidate_universe.value,
        "local_k": config.local_k,
        "rank": config.projection.rank,
        "projection_epsilon": config.projection.epsilon,
        "fixture_count": len(fixtures),
        "users": sorted(
            {value.snapshot.user_id for value in fixtures if value.snapshot.user_id}
        ),
        "sessions": sorted(
            {value.snapshot.session_id for value in fixtures if value.snapshot.session_id}
        ),
        "scenario_types": sorted(
            {value.scenario_type for value in fixtures if value.scenario_type}
        ),
        "catalogue_row_count": int(np.asarray(scorer.catalog_embeddings).shape[0]),
        "catalogue_fingerprint": getattr(scorer, "catalog_fingerprint", None),
        "product_text_fingerprint": getattr(scorer, "product_text_fingerprint", None),
        "embedding_cache": {
            "path": str(getattr(scorer, "embedding_cache_path", "")),
            "hashed": False,
        },
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
    "build_run_manifest",
    "build_source_freeze",
    "candidate_batch_from_records",
    "evaluate_projector_fixtures",
    "load_projector_fixture",
    "summarize_projector_records",
    "write_projector_artifacts",
]
