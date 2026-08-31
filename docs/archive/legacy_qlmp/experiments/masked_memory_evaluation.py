"""Frozen dual-OpenAI-space evaluation for masked longitudinal memory steering.

The scientific path is deliberately separate from production agents.  It reuses
the exact large-space projector fixture, can explicitly freeze the corresponding
small-space vectors once, and then performs cache-only M0/M1/M2/M3 replay.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from nickolas.memory.qlmp import MemoryItem, MemoryPolarity, MemorySource
from ..agent import Agent, DenseQuerySnapshot
from ..embedding_backends import (
    CacheExpectation,
    OPENAI_EMBEDDING_SPACE_ID,
    OPENAI_SMALL_MODEL,
    OpenAIEmbeddingBackend,
    PRODUCT_TEXT_VERSION,
    cache_filename,
    fingerprint_file,
    fingerprint_texts,
    load_embedding_cache,
    make_embedding_space_id,
    save_embedding_cache,
)
from .masked_memory_steering import (
    DEFAULT_KEEP_RATIO,
    DEFAULT_LAMBDA_MEMORY,
    SteeringConfig,
    SteeringMethod,
    aggregate_user_memory,
    steer_query_with_diagnostics,
)


CURRENT_DIR = Path(__file__).resolve().parent
SHOPPING_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SHOPPING_DIR.parents[1]
DEFAULT_FIXTURE = CURRENT_DIR / "projector_fixture_v1.json"
DEFAULT_CATALOG = PROJECT_ROOT / "techjam-conversational-search" / "data" / "catalog.jsonl"
DEFAULT_OUTPUT = CURRENT_DIR / "results" / "masked_memory_steering"
SMALL_BACKEND_ID = "openai-text-embedding-3-small"
SMALL_METADATA = CURRENT_DIR / "masked_memory_small_fixture.json"
SMALL_VECTORS = CURRENT_DIR / "masked_memory_small_fixture.vectors.npz"
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_SAMPLES = 10_000


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _product_text(product: Mapping[str, Any]) -> str:
    title = product.get("title") or ""
    categories = ", ".join(product.get("categories") or [])
    features = "; ".join((product.get("features") or [])[:3])
    return f"Product: {title}. Categories: {categories}. Features: {features}.".strip()


def _catalog_inputs(path: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            ids.append(str(product["parent_asin"]))
            texts.append(_product_text(product))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("catalogue IDs must be non-empty and unique")
    return ids, texts


def _logical_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": str(fixture["fixture_id"]),
            "user_id": str(fixture["user_id"]),
            "session_id": str(fixture["session_id"]),
            "sequence_index": int(fixture["sequence_index"]),
            "target_product_id": str(fixture["target_product_id"]),
            "effective_query_text": str(fixture["effective_query_text"]),
            "memories": [
                {
                    "id": str(memory["id"]),
                    "text": str(memory["text"]),
                    "polarity": str(memory.get("polarity", "positive")),
                    "origin_sequence_index": int(memory["origin_sequence_index"]),
                }
                for memory in fixture["memories"]
            ],
        }
        for fixture in payload["fixtures"]
    ]


@dataclass(frozen=True)
class EmbeddingSpace:
    backend_id: str
    model_id: str
    embedding_space_id: str
    dimension: int
    normalized: bool = True


@dataclass(frozen=True)
class FrozenSession:
    fixture_id: str
    user_id: str
    session_id: str
    sequence_index: int
    scenario_type: str
    split: str
    target_asin: str
    query_text: str
    query: np.ndarray
    memories: tuple[MemoryItem, ...]
    memory_origin_indices: tuple[int, ...]
    space: EmbeddingSpace


@dataclass(frozen=True)
class FrozenBundle:
    space: EmbeddingSpace
    sessions: tuple[FrozenSession, ...]
    source_fixture_sha256: str
    vector_snapshot_sha256: str
    logical_manifest_sha256: str


def validate_vector_space(
    vector: np.ndarray,
    actual_space: EmbeddingSpace,
    expected_space: EmbeddingSpace,
    name: str,
) -> None:
    if actual_space.embedding_space_id != expected_space.embedding_space_id:
        raise ValueError(
            f"{name} embedding space mismatch: {actual_space.embedding_space_id} != "
            f"{expected_space.embedding_space_id}"
        )
    array = np.asarray(vector)
    if array.ndim != 1 or array.shape[0] != expected_space.dimension:
        raise ValueError(
            f"{name} dimension mismatch: expected {expected_space.dimension}, got {array.shape}"
        )


def validate_same_space(
    query: tuple[np.ndarray, EmbeddingSpace],
    memory: tuple[np.ndarray, EmbeddingSpace] | None,
    catalog_space: EmbeddingSpace,
) -> None:
    validate_vector_space(query[0], query[1], catalog_space, "query")
    if memory is not None:
        validate_vector_space(memory[0], memory[1], catalog_space, "memory")


def _load_vector_map(path: Path, expected_hash: str, dimension: int) -> dict[str, np.ndarray]:
    if _sha256_file(path) != expected_hash:
        raise ValueError(f"vector snapshot SHA-256 mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"keys", "vectors"}:
            raise ValueError("vector snapshot must contain only keys and vectors")
        keys = [str(value) for value in data["keys"].tolist()]
        vectors = np.asarray(data["vectors"], dtype=np.float64)
    if vectors.shape != (len(keys), dimension) or len(keys) != len(set(keys)):
        raise ValueError("vector snapshot shape/keys do not match metadata")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or not np.allclose(norms, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("all frozen vectors must be finite and L2-normalized")
    return {key: vectors[index] for index, key in enumerate(keys)}


def _memory_item(raw: Mapping[str, Any], vector: np.ndarray) -> MemoryItem:
    timestamp = raw.get("timestamp")
    return MemoryItem(
        id=str(raw["id"]),
        text=str(raw["text"]),
        embedding=vector,
        source=MemorySource(raw.get("source", "user")),
        polarity=MemoryPolarity(raw.get("polarity", "positive")),
        scope=None if raw.get("scope") is None else str(raw["scope"]),
        timestamp=None if timestamp is None else datetime.fromisoformat(str(timestamp)),
        confidence=float(raw.get("confidence", 1.0)),
    )


def _materialize_bundle(
    fixture_payload: Mapping[str, Any],
    vector_map: Mapping[str, np.ndarray],
    space: EmbeddingSpace,
    *,
    source_fixture_sha256: str,
    vector_snapshot_sha256: str,
) -> FrozenBundle:
    sessions: list[FrozenSession] = []
    for raw in fixture_payload["fixtures"]:
        if str(raw.get("buying_or_browsing_label")) != "Buying":
            continue
        query_key = str(raw["q_m0_key"])
        if query_key not in vector_map:
            raise ValueError(f"missing frozen query vector {query_key}")
        memories: list[MemoryItem] = []
        origins: list[int] = []
        for memory in raw["memories"]:
            key = str(memory["embedding_key"])
            if key not in vector_map:
                raise ValueError(f"missing frozen memory vector {key}")
            origin = int(memory["origin_sequence_index"])
            if origin >= int(raw["sequence_index"]):
                raise ValueError(f"current/future-session leakage in {raw['fixture_id']}")
            memories.append(_memory_item(memory, vector_map[key]))
            origins.append(origin)
        query = np.asarray(vector_map[query_key], dtype=np.float64)
        validate_vector_space(query, space, space, "query")
        sessions.append(
            FrozenSession(
                fixture_id=str(raw["fixture_id"]),
                user_id=str(raw["user_id"]),
                session_id=str(raw["session_id"]),
                sequence_index=int(raw["sequence_index"]),
                scenario_type="buying",
                split=str(raw["split"]),
                target_asin=str(raw["target_product_id"]),
                query_text=str(raw["effective_query_text"]),
                query=query,
                memories=tuple(memories),
                memory_origin_indices=tuple(origins),
                space=space,
            )
        )
    if not sessions:
        raise ValueError("no Buyer sessions in fixture")
    return FrozenBundle(
        space=space,
        sessions=tuple(sessions),
        source_fixture_sha256=source_fixture_sha256,
        vector_snapshot_sha256=vector_snapshot_sha256,
        logical_manifest_sha256=_sha256_json(_logical_manifest(fixture_payload)),
    )


def load_large_bundle(path: Path = DEFAULT_FIXTURE) -> FrozenBundle:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("embedding_space_id") != OPENAI_EMBEDDING_SPACE_ID:
        raise ValueError("large fixture embedding space is not canonical M0_OPENAI")
    snapshot = payload["vector_snapshot"]
    dimension = int(snapshot["dimension"])
    space = EmbeddingSpace(
        "openai-text-embedding-3-large",
        "text-embedding-3-large",
        OPENAI_EMBEDDING_SPACE_ID,
        dimension,
    )
    vector_path = path.parent / str(snapshot["path"])
    vector_map = _load_vector_map(vector_path, str(snapshot["sha256"]), dimension)
    return _materialize_bundle(
        payload,
        vector_map,
        space,
        source_fixture_sha256=hashlib.sha256(raw).hexdigest(),
        vector_snapshot_sha256=str(snapshot["sha256"]),
    )


def _small_expectation(
    metadata: Mapping[str, Any], catalog_ids: Sequence[str], catalog_path: Path
) -> CacheExpectation:
    return CacheExpectation(
        backend_id=str(metadata["backend_id"]),
        model_id=str(metadata["model_id"]),
        embedding_space_id=str(metadata["embedding_space_id"]),
        catalog_ids=catalog_ids,
        product_text_fingerprint=str(metadata["product_text_fingerprint"]),
        catalog_fingerprint=fingerprint_file(catalog_path),
        vector_dimension=int(metadata["dimension"]),
        normalized=True,
    )


def freeze_small_bundle(
    fixture_path: Path = DEFAULT_FIXTURE,
    catalog_path: Path = DEFAULT_CATALOG,
    metadata_path: Path = SMALL_METADATA,
    vectors_path: Path = SMALL_VECTORS,
) -> dict[str, Any]:
    """Make the only hosted calls in this experiment and freeze their outputs."""

    fixture_bytes = fixture_path.read_bytes()
    payload = json.loads(fixture_bytes.decode("utf-8"))
    logical = _logical_manifest(payload)
    catalog_ids, catalog_texts = _catalog_inputs(catalog_path)
    backend = OpenAIEmbeddingBackend(
        model_id=OPENAI_SMALL_MODEL,
        backend_id=SMALL_BACKEND_ID,
        vector_dimension=None,
    )

    catalog_vectors = backend.embed_catalog(catalog_texts)
    dimension = int(catalog_vectors.shape[1])
    space_id = make_embedding_space_id(
        SMALL_BACKEND_ID, OPENAI_SMALL_MODEL, dimension
    )
    if backend.embedding_space_id != space_id:
        raise ValueError("small backend output dimension/space metadata disagree")
    product_text_fingerprint = fingerprint_texts(catalog_texts)
    expectation = CacheExpectation(
        backend_id=SMALL_BACKEND_ID,
        model_id=OPENAI_SMALL_MODEL,
        embedding_space_id=space_id,
        catalog_ids=catalog_ids,
        product_text_fingerprint=product_text_fingerprint,
        catalog_fingerprint=fingerprint_file(catalog_path),
        vector_dimension=dimension,
        normalized=True,
    )
    catalog_cache_path = SHOPPING_DIR / "embedding_cache" / cache_filename(SMALL_BACKEND_ID)
    save_embedding_cache(catalog_cache_path, catalog_vectors, expectation)

    vector_inputs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fixture in payload["fixtures"]:
        query_entry = (str(fixture["q_m0_key"]), str(fixture["effective_query_text"]))
        if query_entry[0] not in seen:
            seen.add(query_entry[0])
            vector_inputs.append(query_entry)
        for memory in fixture["memories"]:
            entry = (str(memory["embedding_key"]), str(memory["text"]))
            if entry[0] not in seen:
                seen.add(entry[0])
                vector_inputs.append(entry)
    fixture_vectors = backend.embed_catalog([text for _, text in vector_inputs])
    if fixture_vectors.shape != (len(vector_inputs), dimension):
        raise ValueError("small fixture vector shape mismatch")
    temporary = vectors_path.with_name(vectors_path.name + ".tmp.npz")
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        temporary,
        keys=np.asarray([key for key, _ in vector_inputs], dtype=np.str_),
        vectors=np.asarray(fixture_vectors, dtype=np.float64),
    )
    temporary.replace(vectors_path)
    vector_hash = _sha256_file(vectors_path)
    metadata = {
        "schema_version": 1,
        "backend_id": SMALL_BACKEND_ID,
        "model_id": OPENAI_SMALL_MODEL,
        "embedding_space_id": space_id,
        "dimension": dimension,
        "normalized": True,
        "normalization_policy": "l2",
        "product_text_version": PRODUCT_TEXT_VERSION,
        "catalogue_fingerprint": fingerprint_file(catalog_path),
        "catalogue_row_count": len(catalog_ids),
        "catalogue_cache_path": str(catalog_cache_path.relative_to(SHOPPING_DIR)),
        "catalogue_cache_sha256": _sha256_file(catalog_cache_path),
        "product_text_fingerprint": product_text_fingerprint,
        "product_order_fingerprint": fingerprint_texts(catalog_ids),
        "source_fixture_path": fixture_path.name,
        "source_fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "source_fixture_fingerprint": _sha256_json(logical),
        "query_text_fingerprint": fingerprint_texts(
            [str(value["effective_query_text"]) for value in payload["fixtures"]]
        ),
        "memory_text_fingerprint": fingerprint_texts(
            [str(memory["text"]) for value in payload["fixtures"] for memory in value["memories"]]
        ),
        "vector_snapshot": {
            "path": vectors_path.name,
            "sha256": vector_hash,
            "vector_count": len(vector_inputs),
            "dimension": dimension,
            "storage_dtype": "float64",
        },
        "embedding_usage": backend.usage_snapshot(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return metadata


def load_small_bundle(
    fixture_path: Path = DEFAULT_FIXTURE,
    catalog_path: Path = DEFAULT_CATALOG,
    metadata_path: Path = SMALL_METADATA,
) -> FrozenBundle:
    fixture_bytes = fixture_path.read_bytes()
    payload = json.loads(fixture_bytes.decode("utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "backend_id": SMALL_BACKEND_ID,
        "model_id": OPENAI_SMALL_MODEL,
        "normalized": True,
        "product_text_version": PRODUCT_TEXT_VERSION,
        "catalogue_fingerprint": fingerprint_file(catalog_path),
        "source_fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "source_fixture_fingerprint": _sha256_json(_logical_manifest(payload)),
    }
    problems = [key for key, expected in required.items() if metadata.get(key) != expected]
    catalog_ids, catalog_texts = _catalog_inputs(catalog_path)
    fingerprints = {
        "catalogue_row_count": len(catalog_ids),
        "product_text_fingerprint": fingerprint_texts(catalog_texts),
        "product_order_fingerprint": fingerprint_texts(catalog_ids),
    }
    problems.extend(
        key for key, expected in fingerprints.items() if metadata.get(key) != expected
    )
    dimension = int(metadata.get("dimension", 0))
    expected_space = make_embedding_space_id(
        SMALL_BACKEND_ID, OPENAI_SMALL_MODEL, dimension
    ) if dimension > 0 else ""
    if metadata.get("embedding_space_id") != expected_space:
        problems.append("embedding_space_id")
    cache_path = SHOPPING_DIR / str(metadata.get("catalogue_cache_path", ""))
    if not cache_path.is_file() or _sha256_file(cache_path) != metadata.get("catalogue_cache_sha256"):
        problems.append("catalogue_cache_sha256")
    if problems:
        raise ValueError("small frozen cache failed validation: " + ", ".join(sorted(set(problems))))
    load_embedding_cache(cache_path, _small_expectation(metadata, catalog_ids, catalog_path))
    snapshot = metadata["vector_snapshot"]
    vectors_path = metadata_path.parent / str(snapshot["path"])
    vector_map = _load_vector_map(vectors_path, str(snapshot["sha256"]), dimension)
    space = EmbeddingSpace(SMALL_BACKEND_ID, OPENAI_SMALL_MODEL, expected_space, dimension)
    return _materialize_bundle(
        payload,
        vector_map,
        space,
        source_fixture_sha256=str(metadata["source_fixture_sha256"]),
        vector_snapshot_sha256=str(snapshot["sha256"]),
    )


def assert_identical_samples(large: FrozenBundle, small: FrozenBundle) -> None:
    large_ids = [session.session_id for session in large.sessions]
    small_ids = [session.session_id for session in small.sessions]
    if large_ids != small_ids:
        raise ValueError("large_session_ids != small_session_ids")
    if [s.target_asin for s in large.sessions] != [s.target_asin for s in small.sessions]:
        raise ValueError("large_target_asins != small_target_asins")
    for left, right in zip(large.sessions, small.sessions):
        left_ids = [item.id for item in left.memories]
        right_ids = [item.id for item in right.memories]
        if left_ids != right_ids:
            raise ValueError(f"memory IDs differ for {left.session_id}")
        if [item.text for item in left.memories] != [item.text for item in right.memories]:
            raise ValueError(f"memory texts differ for {left.session_id}")
        if left.memory_origin_indices != right.memory_origin_indices:
            raise ValueError(f"memory timestamps differ for {left.session_id}")


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["target_rank"]) for row in rows]
    return {
        "session_count": len(rows),
        "hr_at_10": float(np.mean([rank <= 10 for rank in ranks])),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])),
        "mean_target_rank": float(np.mean(ranks)),
        "median_target_rank": float(median(ranks)),
    }


def _pairwise(
    by_method: Mapping[str, Sequence[Mapping[str, Any]]], compared: str, reference: str
) -> dict[str, Any]:
    right = {str(row["session_id"]): row for row in by_method[reference]}
    changes = []
    rr_changes = []
    for row in by_method[compared]:
        base = right[str(row["session_id"])]
        changes.append(int(base["target_rank"]) - int(row["target_rank"]))
        rr_changes.append(float(row["reciprocal_rank"]) - float(base["reciprocal_rank"]))
    return {
        "compared": compared,
        "reference": reference,
        "sessions_improved": sum(value > 0 for value in changes),
        "sessions_unchanged": sum(value == 0 for value in changes),
        "sessions_regressed": sum(value < 0 for value in changes),
        "mean_rank_change_reference_minus_compared": float(np.mean(changes)),
        "mrr_change_compared_minus_reference": float(np.mean(rr_changes)),
    }


def _bootstrap_mrr_delta(
    m3: Sequence[Mapping[str, Any]], m0: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    differences = np.asarray(
        [float(a["reciprocal_rank"]) - float(b["reciprocal_rank"]) for a, b in zip(m3, m0)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = differences[rng.integers(0, len(differences), size=(BOOTSTRAP_SAMPLES, len(differences)))].mean(axis=1)
    return {
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "delta_mrr": float(differences.mean()),
        "percentile_95_ci": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "caution": "descriptive paired bootstrap; the fixture is small and previously inspected",
    }


def _evaluate_bundle(bundle: FrozenBundle, agent: Agent) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if agent.embedding_space_id != bundle.space.embedding_space_id:
        raise ValueError("scorer and fixture embedding spaces differ")
    if list(agent.catalog_ids) != list(dict.fromkeys(agent.catalog_ids)):
        raise ValueError("catalogue product IDs are not uniquely ordered")
    rows: list[dict[str, Any]] = []
    for session in bundle.sessions:
        if any(origin >= session.sequence_index for origin in session.memory_origin_indices):
            raise ValueError(f"temporal leakage in {session.session_id}")
        eligible = tuple(
            item for item in session.memories if item.polarity is not MemoryPolarity.NEGATIVE
        )
        memory = aggregate_user_memory(eligible)
        validate_same_space(
            (session.query, session.space),
            None if memory is None else (memory, session.space),
            bundle.space,
        )
        base_queries: list[np.ndarray] = []
        session_rows: list[dict[str, Any]] = []
        for method in SteeringMethod:
            base_queries.append(session.query)
            final_query, diagnostic = steer_query_with_diagnostics(
                session.query,
                memory,
                SteeringConfig(method, DEFAULT_KEEP_RATIO, DEFAULT_LAMBDA_MEMORY),
                memory_item_count=len(eligible),
                buyer_active=True,
            )
            validate_vector_space(final_query, session.space, bundle.space, "q_final")
            result = agent.dense_retrieve_vector(final_query, top_n=len(agent.catalog_ids))
            try:
                rank = result.product_ids.index(session.target_asin) + 1
            except ValueError as exc:
                raise ValueError(f"target {session.target_asin} absent from catalogue") from exc
            data = diagnostic.to_dict()
            session_rows.append(
                {
                    "user_id": session.user_id,
                    "session_id": session.session_id,
                    "fixture_id": session.fixture_id,
                    "sequence_index": session.sequence_index,
                    "split": session.split,
                    "scenario_type": session.scenario_type,
                    "embedding_model": bundle.space.model_id,
                    "embedding_space_id": bundle.space.embedding_space_id,
                    "embedding_dimension": bundle.space.dimension,
                    "method": method.value,
                    "target_asin": session.target_asin,
                    "target_rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "hit_at_10": rank <= 10,
                    "keep_ratio": DEFAULT_KEEP_RATIO,
                    "lambda_memory": DEFAULT_LAMBDA_MEMORY,
                    "number_of_historical_memories": len(eligible),
                    "historical_memory_ids": [item.id for item in eligible],
                    "query_norm": float(np.linalg.norm(session.query)),
                    "aggregate_memory_norm": 0.0 if memory is None else float(np.linalg.norm(memory)),
                    "cleaned_memory_norm": data["cleaned_memory_norm"],
                    "kept_dimensions": data["kept_dimensions"],
                    "kept_fraction": data["retained_fraction"],
                    "cosine_q_m": data["query_memory_cosine"],
                    "cosine_q_m_clean": data["query_cleaned_memory_cosine"],
                    "cosine_q_q_final": data["query_steered_cosine"],
                    "recommendations_top_10": list(result.product_ids[:10]),
                    "scores_top_10": [float(value) for value in result.scores[:10]],
                }
            )
        if any(query is not session.query for query in base_queries):
            raise AssertionError("variants do not share the same base query")
        baseline_rank = next(row["target_rank"] for row in session_rows if row["method"] == "M0")
        for row in session_rows:
            row["baseline_rank"] = baseline_rank
            row["rank_delta_versus_m0"] = baseline_rank - row["target_rank"]
            rows.append(row)
    by_method = {
        method.value: [row for row in rows if row["method"] == method.value]
        for method in SteeringMethod
    }
    comparisons = {}
    for compared, reference in (("M1", "M0"), ("M2", "M0"), ("M3", "M0"), ("M3", "M1"), ("M3", "M2")):
        comparisons[f"{compared}-{reference}"] = _pairwise(by_method, compared, reference)
    rank_map = {
        (row["session_id"], row["method"]): int(row["target_rank"])
        for row in rows
    }
    rescues = [
        session.session_id for session in bundle.sessions
        if rank_map[(session.session_id, "M1")] > rank_map[(session.session_id, "M0")]
        and rank_map[(session.session_id, "M3")] <= rank_map[(session.session_id, "M0")]
    ]
    destroyed = [
        session.session_id for session in bundle.sessions
        if rank_map[(session.session_id, "M1")] < rank_map[(session.session_id, "M0")]
        and rank_map[(session.session_id, "M3")] > rank_map[(session.session_id, "M1")]
    ]
    summary = {
        "embedding_model": bundle.space.model_id,
        "embedding_space_id": bundle.space.embedding_space_id,
        "metrics": {method: _metrics(values) for method, values in by_method.items()},
        "pairwise": comparisons,
        "masking_rescue_count": len(rescues),
        "masking_rescue_session_ids": rescues,
        "masking_destroyed_useful_memory_count": len(destroyed),
        "masking_destroyed_useful_memory_session_ids": destroyed,
        "paired_bootstrap_m3_minus_m0": _bootstrap_mrr_delta(by_method["M3"], by_method["M0"]),
    }
    return rows, summary


def _results_markdown(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    large = summary["text-embedding-3-large"]
    small = summary["text-embedding-3-small"]
    def metric_table(metric: str, digits: int = 6) -> str:
        lines = ["| Embedding | M0 | M1 | M2 | M3 |", "| --- | ---: | ---: | ---: | ---: |"]
        for label, value in (("text-embedding-3-large", large), ("text-embedding-3-small", small)):
            cells = [f"{value['metrics'][method][metric]:.{digits}f}" for method in ("M0", "M1", "M2", "M3")]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        return "\n".join(lines)
    def pair_lines(value: Mapping[str, Any]) -> str:
        lines = ["| Pair | Improved | Unchanged | Regressed | Mean rank change* | MRR change |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for key in ("M1-M0", "M2-M0", "M3-M0", "M3-M1", "M3-M2"):
            row = value["pairwise"][key]
            lines.append(
                f"| {key} | {row['sessions_improved']} | {row['sessions_unchanged']} | "
                f"{row['sessions_regressed']} | {row['mean_rank_change_reference_minus_compared']:.3f} | "
                f"{row['mrr_change_compared_minus_reference']:.6f} |"
            )
        return "\n".join(lines)
    large_uplift = large["metrics"]["M3"]["mrr"] - large["metrics"]["M0"]["mrr"]
    small_uplift = small["metrics"]["M3"]["mrr"] - small["metrics"]["M0"]["mrr"]
    if large_uplift > 0 and small_uplift > 0:
        interpretation = "M3 improved MRR in both frozen embedding spaces, which is directionally consistent with cross-space generalization."
    elif large_uplift <= 0 and small_uplift <= 0:
        interpretation = "M3 did not improve MRR over M0 in either frozen embedding space; this run does not support a retrieval benefit."
    else:
        interpretation = "M3 uplift changed sign across embedding spaces, so the mechanism is embedding-model dependent in this fixture."
    return f"""# Masked Memory Steering Results

# Experiment

Locked evaluator-private Buyer experiment on `{manifest['fixture_path']}` with `keep_ratio = 0.20`, `lambda_memory = 0.20`, equal-weight normalized eligible positive/neutral memory, the unchanged full-catalogue dot-product scorer, and exact target ranks over {manifest['catalogue_row_count']} products. All {manifest['buyer_session_count']} sessions are scored under M0/M1/M2/M3 without tuning.

The fixture uses the pre-existing curated projector memory set (109 session-memory pairs, 44 unique memory texts), not every memory committed in the 40-session source replay. Its fixture, baseline rankings, and projector results were previously inspected, so this is reproducible diagnostic evidence rather than clean held-out evidence.

# Embedding spaces

`text-embedding-3-large` and `text-embedding-3-small` are validated as independent spaces. Query, memory, and catalogue operands must share the exact embedding-space identifier and provider-returned dimension. Cross-model operations fail before scoring. Product text and row ordering are identical.

# Primary results

MRR:

{metric_table('mrr')}

HR@10:

{metric_table('hr_at_10')}

Mean target rank (lower is better):

{metric_table('mean_target_rank', 3)}

Median target rank (lower is better):

{metric_table('median_target_rank', 3)}

# Pairwise deltas

Positive mean rank change means the compared method moved the target upward relative to the reference.

## text-embedding-3-large

{pair_lines(large)}

## text-embedding-3-small

{pair_lines(small)}

# Session-level behaviour

- Large masking rescues: {large['masking_rescue_count']}; destroyed-useful-memory cases: {large['masking_destroyed_useful_memory_count']}.
- Small masking rescues: {small['masking_rescue_count']}; destroyed-useful-memory cases: {small['masking_destroyed_useful_memory_count']}.
- Session-level rows, exact ranks, diagnostics, and top-10 scores are in `session_results.jsonl`.

# Large vs small

Absolute M0 baseline MRR is {large['metrics']['M0']['mrr']:.6f} for large and {small['metrics']['M0']['mrr']:.6f} for small. This is retrieval-backend quality, not memory uplift.

M3 minus M0 MRR uplift is {large_uplift:+.6f} for large and {small_uplift:+.6f} for small. This within-space comparison is the relevant cross-model test of the memory mechanism.

# Sensitivity

Not run. With 18 previously inspected sessions, a keep-ratio sweep would be weak sensitivity evidence and is intentionally deferred; 0.20 remains the locked primary value.

# Interpretation

{interpretation} The paired bootstrap intervals are descriptive only; no significance claim is made for this small, previously inspected fixture.

# Limitations

- Dense coordinates are distributed and are not human-interpretable features.
- Coordinate-wise masking is a heuristic.
- Buyer-only evaluation with four users and a limited session sample.
- The frozen projector fixture contains a curated historical-memory subset rather than all committed prior memories.
- Equal-weight aggregate memory and no negative-memory steering.
- Fixed lambda and primary keep ratio.
- Results may depend on embedding model.
- This fixture and related projector/M0 artifacts were inspected during earlier method work; future tuning and evaluation partitions must remain separated.

# Conclusion

{interpretation}
"""


def run_evaluation(
    output_dir: Path = DEFAULT_OUTPUT,
    fixture_path: Path = DEFAULT_FIXTURE,
    catalog_path: Path = DEFAULT_CATALOG,
) -> Path:
    large = load_large_bundle(fixture_path)
    small = load_small_bundle(fixture_path, catalog_path)
    assert_identical_samples(large, small)
    catalog_ids, catalog_texts = _catalog_inputs(catalog_path)
    large_expectation = CacheExpectation(
        backend_id=large.space.backend_id,
        model_id=large.space.model_id,
        embedding_space_id=large.space.embedding_space_id,
        catalog_ids=catalog_ids,
        product_text_fingerprint=fingerprint_texts(catalog_texts),
        catalog_fingerprint=fingerprint_file(catalog_path),
        vector_dimension=large.space.dimension,
        normalized=True,
    )
    large_cache_path = SHOPPING_DIR / "embedding_cache" / cache_filename(large.space.backend_id)
    load_embedding_cache(large_cache_path, large_expectation)
    small_meta = json.loads(SMALL_METADATA.read_text(encoding="utf-8"))

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for bundle in (large, small):
        backend = OpenAIEmbeddingBackend(
            model_id=bundle.space.model_id,
            backend_id=bundle.space.backend_id,
            vector_dimension=bundle.space.dimension,
        )
        agent = Agent(
            catalog_path=catalog_path,
            embedding_backend=backend,
            allow_catalog_embedding=False,
        )
        try:
            if fingerprint_texts(agent.catalog_ids) != fingerprint_texts(catalog_ids):
                raise ValueError("product ordering changed while constructing scorer")
            rows, bundle_summary = _evaluate_bundle(bundle, agent)
            if backend.usage_snapshot()["request_count"] != 0:
                raise RuntimeError("cache-only evaluation attempted an embedding request")
            all_rows.extend(rows)
            summaries[bundle.space.model_id] = bundle_summary
        finally:
            agent.connection.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    session_manifest = [session.session_id for session in large.sessions]
    (output_dir / "session_manifest.json").write_text(
        json.dumps(
            {
                "ordered_session_ids": session_manifest,
                "ordered_target_asins": [session.target_asin for session in large.sessions],
                "ordered_memory_ids": {
                    session.session_id: [item.id for item in session.memories]
                    for session in large.sessions
                },
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "session_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "locked_masked_memory_steering_dual_openai",
        "fixture_path": str(fixture_path.relative_to(PROJECT_ROOT)),
        "fixture_sha256": large.source_fixture_sha256,
        "logical_memory_manifest_sha256": large.logical_manifest_sha256,
        "large_vector_snapshot_sha256": large.vector_snapshot_sha256,
        "small_metadata_sha256": _sha256_file(SMALL_METADATA),
        "small_vector_snapshot_sha256": small.vector_snapshot_sha256,
        "large_catalogue_cache_sha256": _sha256_file(large_cache_path),
        "small_catalogue_cache_sha256": str(small_meta["catalogue_cache_sha256"]),
        "catalogue_fingerprint": fingerprint_file(catalog_path),
        "product_text_fingerprint": fingerprint_texts(catalog_texts),
        "product_order_fingerprint": fingerprint_texts(catalog_ids),
        "catalogue_row_count": len(catalog_ids),
        "user_count": len({session.user_id for session in large.sessions}),
        "buyer_session_count": len(large.sessions),
        "ordered_session_ids": session_manifest,
        "keep_ratio": DEFAULT_KEEP_RATIO,
        "lambda_memory": DEFAULT_LAMBDA_MEMORY,
        "embedding_calls_during_evaluation": 0,
        "small_freeze_embedding_usage": small_meta.get("embedding_usage", {}),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "sensitivity_run": False,
        "previously_inspected_fixture": True,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results_path = CURRENT_DIR / "MASKED_MEMORY_STEERING_RESULTS.md"
    results_path.write_text(_results_markdown(manifest, summaries), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-small", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    if args.freeze_small:
        metadata = freeze_small_bundle(args.fixture, args.catalog)
        print(json.dumps({"small_freeze": metadata["embedding_usage"]}, indent=2))
    destination = run_evaluation(args.output, args.fixture, args.catalog)
    print(f"Artifacts: {destination}")


if __name__ == "__main__":
    main()

