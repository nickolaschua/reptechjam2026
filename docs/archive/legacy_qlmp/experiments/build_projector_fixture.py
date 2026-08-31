"""Freeze the curated Phase-3A component fixture from real longitudinal logs.

Audit mode is read-only and reports every missing vector before any hosted call.
Build mode reuses vectors from an existing destination, embeds only deduplicated
missing texts, and writes one self-contained offline-replay fixture atomically.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..agent import Agent, _state_to_retrieval_query
from ..embedding_backends import OPENAI_EMBEDDING_SPACE_ID
from .qlmp_component_eval import FIXTURE_VERSION, ProjectorLabel


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = CURRENT_DIR / "projector_fixture_spec.json"
DEFAULT_OUTPUT = CURRENT_DIR / "projector_fixture_v1.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_vectors(path: Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    if not path.is_file():
        return {}, {}
    payload = _load_json(path)
    stored: dict[str, list[float]] = {}
    snapshot = payload.get("vector_snapshot")
    if isinstance(snapshot, Mapping) and snapshot.get("path"):
        with np.load(path.parent / str(snapshot["path"]), allow_pickle=False) as data:
            keys = [str(value) for value in data["keys"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float64)
        stored = {key: vectors[index].tolist() for index, key in enumerate(keys)}
    queries = {
        str(value["effective_query_text"]): (
            value["q_m0"] if "q_m0" in value else stored[str(value["q_m0_key"])]
        )
        for value in payload.get("fixtures", [])
        if "q_m0" in value or str(value.get("q_m0_key", "")) in stored
    }
    memories = {
        str(value["text"]): (
            value["embedding"]
            if "embedding" in value
            else stored[str(value["embedding_key"])]
        )
        for fixture in payload.get("fixtures", [])
        for value in fixture.get("memories", [])
        if "embedding" in value or str(value.get("embedding_key", "")) in stored
    }
    return queries, memories


def _normalize_memory64(value: Any) -> list[float]:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3072,) or not np.all(np.isfinite(vector)) or norm <= 0.0:
        raise ValueError("memory vector must be finite, nonzero, and 3072-dimensional")
    return (vector / norm).tolist()


def _memory_vector_key(text: str) -> str:
    return "memory:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_curation(
    spec_path: str | Path,
    *,
    source_run_path: str | Path | None = None,
    query_limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec_file = Path(spec_path)
    spec = _load_json(spec_file)
    source = (
        Path(source_run_path)
        if source_run_path is not None
        else spec_file.parent / str(spec["source_run"])
    )
    run = _load_json(source)
    sessions = {
        (str(value["user_id"]), int(value["sequence_index"])): value
        for value in run["sessions"]
    }
    raw_queries = list(spec["queries"])
    if query_limit is not None:
        raw_queries = raw_queries[:query_limit]
    resolved: list[dict[str, Any]] = []
    seen_fixture_ids: set[str] = set()
    for raw_query in raw_queries:
        user_id = str(raw_query["user_id"])
        sequence_index = int(raw_query["sequence_index"])
        session = sessions.get((user_id, sequence_index))
        if session is None:
            raise ValueError(f"missing source session {user_id} S{sequence_index}")
        fixture_id = f"{user_id}_s{sequence_index}_final"
        if fixture_id in seen_fixture_ids:
            raise ValueError(f"duplicate fixture ID {fixture_id}")
        seen_fixture_ids.add(fixture_id)
        selected: list[tuple[int, int, dict[str, Any], Mapping[str, Any]]] = []
        for annotation in raw_query["memories"]:
            ProjectorLabel(annotation["label"])
            origin_index = int(annotation["origin_sequence_index"])
            if origin_index >= sequence_index:
                raise ValueError(f"future-memory leakage in {fixture_id}: S{origin_index}")
            origin = sessions.get((user_id, origin_index))
            if origin is None:
                raise ValueError(f"missing origin {user_id} S{origin_index}")
            matches = [
                (position, memory)
                for position, memory in enumerate(origin["committed_memory_items"])
                if str(memory["text"]) == str(annotation["text"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {annotation['text']!r} in {user_id} S{origin_index}; got {len(matches)}"
                )
            position, memory = matches[0]
            selected.append((origin_index, position, dict(memory), annotation))
        selected.sort(key=lambda value: (value[0], value[1], str(value[2]["id"])))
        if len({value[2]["id"] for value in selected}) != len(selected):
            raise ValueError(f"duplicate memory ID in {fixture_id}")
        messages = [str(value["shopper"]) for value in session["turns"]]
        final_state = session["final_fast_memory"]
        effective = _state_to_retrieval_query(final_state)
        if not effective:
            raise ValueError(f"empty effective query for {fixture_id}")
        resolved.append(
            {
                "fixture_id": fixture_id,
                "user_id": user_id,
                "session_id": str(session["session_id"]),
                "sequence_index": sequence_index,
                "turn_index": int(session["turns"][-1]["turn"]),
                "buying_or_browsing_label": "Buying",
                "split": str(raw_query["split"]),
                "raw_current_message": messages[-1],
                "current_conversation_messages": messages,
                "effective_query_text": effective,
                "query_scope": str(raw_query["product_family"]),
                "current_category": str(final_state.get("category", "")),
                "product_family": str(raw_query["product_family"]),
                "target_product_id": str(session["target_asin"]),
                "has_entangled_memory": bool(raw_query.get("has_entangled_memory", False)),
                "memories": [
                    {
                        **memory,
                        "source": "user",
                        "confidence": 1.0,
                        "origin_user_id": user_id,
                        "origin_session_id": str(sessions[(user_id, origin_index)]["session_id"]),
                        "origin_sequence_index": origin_index,
                        "label": str(annotation["label"]),
                        "label_reason": str(annotation["label_reason"]),
                        "hard_negative_type": str(annotation["hard_negative_type"]),
                    }
                    for origin_index, _, memory, annotation in selected
                ],
            }
        )
    return spec, resolved


def construction_audit(
    queries: list[Mapping[str, Any]], vector_cache: str | Path
) -> dict[str, Any]:
    existing_q, existing_memory = _existing_vectors(Path(vector_cache))
    query_texts = [str(value["effective_query_text"]) for value in queries]
    memory_texts = [
        str(memory["text"]) for value in queries for memory in value["memories"]
    ]
    unique_memory_texts = sorted(set(memory_texts))
    missing_q = [text for text in query_texts if text not in existing_q]
    missing_memory = [text for text in unique_memory_texts if text not in existing_memory]
    return {
        "query_count": len(query_texts),
        "memory_pair_count": len(memory_texts),
        "unique_memory_text_count": len(unique_memory_texts),
        "q_embeddings_reused": len(query_texts) - len(missing_q),
        "missing_q_embeddings": len(missing_q),
        "memory_embeddings_reused": len(unique_memory_texts) - len(missing_memory),
        "missing_unique_memory_embeddings": len(missing_memory),
        "deduplicated_texts_requiring_embedding": len(set(missing_q).union(missing_memory)),
        "estimated_request_count": len(missing_q) + (1 if missing_memory else 0),
        "label_counts": dict(
            sorted(
                Counter(
                    str(memory["label"])
                    for value in queries
                    for memory in value["memories"]
                ).items()
            )
        ),
        "negative_polarity_count": sum(
            memory.get("polarity") == "negative"
            for value in queries
            for memory in value["memories"]
        ),
        "missing_q_texts": missing_q,
        "missing_memory_texts": missing_memory,
    }


def build_fixture(
    spec_path: str | Path,
    destination: str | Path,
    *,
    source_run_path: str | Path | None = None,
    query_limit: int | None = None,
    vector_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    spec, queries = resolve_curation(
        spec_path, source_run_path=source_run_path, query_limit=query_limit
    )
    destination_path = Path(destination)
    vector_cache = Path(vector_cache_path) if vector_cache_path is not None else destination_path
    audit = construction_audit(queries, vector_cache)
    existing_q, existing_memory = _existing_vectors(vector_cache)
    prior_construction = (
        _load_json(vector_cache).get("construction", {})
        if vector_cache.is_file()
        else {}
    )
    agent = Agent(allow_catalog_embedding=False)
    try:
        q_vectors = dict(existing_q)
        for query in queries:
            text = str(query["effective_query_text"])
            if text not in q_vectors:
                q_vectors[text] = agent.embed_dense_query(text).tolist()
        missing_memory = audit["missing_memory_texts"]
        memory_vectors = dict(existing_memory)
        if missing_memory:
            embedded = agent.embedding_backend.embed_catalog(missing_memory)
            for text, vector in zip(missing_memory, embedded):
                memory_vectors[text] = np.asarray(vector, dtype=np.float32).tolist()
        usage = agent.embedding_backend.usage_snapshot()
    finally:
        connection = getattr(agent, "connection", None)
        if connection is not None:
            connection.close()
    fixtures = []
    for query in queries:
        fixture_id = str(query["fixture_id"])
        fixtures.append(
            {
                **{key: value for key, value in query.items() if key != "raw_current_message"},
                "current_message": query["raw_current_message"],
                "q_m0_key": f"query:{fixture_id}",
                "memories": [
                    {
                        **memory,
                        "embedding_space_id": OPENAI_EMBEDDING_SPACE_ID,
                        "embedding_key": _memory_vector_key(str(memory["text"])),
                    }
                    for memory in query["memories"]
                ],
            }
        )
    vector_items: list[tuple[str, np.ndarray]] = []
    for query in queries:
        vector_items.append(
            (
                f"query:{query['fixture_id']}",
                np.asarray(q_vectors[str(query["effective_query_text"])], dtype=np.float64),
            )
        )
    for text in sorted({str(memory["text"]) for query in queries for memory in query["memories"]}):
        vector_items.append(
            (
                _memory_vector_key(text),
                np.asarray(_normalize_memory64(memory_vectors[text]), dtype=np.float64),
            )
        )
    vector_path = destination_path.with_suffix(".vectors.npz")
    vector_temporary = vector_path.with_name(vector_path.name + ".tmp.npz")
    np.savez_compressed(
        vector_temporary,
        keys=np.asarray([key for key, _ in vector_items], dtype=np.str_),
        vectors=np.vstack([value for _, value in vector_items]),
    )
    vector_temporary.replace(vector_path)
    vector_sha256 = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    payload = {
        "fixture_version": FIXTURE_VERSION,
        "curation_spec_version": spec["spec_version"],
        "embedding_space_id": OPENAI_EMBEDDING_SPACE_ID,
        "candidate_universe": spec["candidate_universe"],
        "vector_snapshot": {
            "path": vector_path.name,
            "sha256": vector_sha256,
            "vector_count": len(vector_items),
            "dimension": 3072,
            "storage_dtype": "float64",
        },
        "construction": {
            **{key: value for key, value in audit.items() if not key.endswith("_texts")},
            "actual_embedding_usage": usage,
            "initial_embedding_usage": prior_construction.get(
                "initial_embedding_usage",
                prior_construction.get("actual_embedding_usage", usage),
            ),
        },
        "fixtures": fixtures,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(destination_path)
    return payload["construction"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--vector-cache", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    _, queries = resolve_curation(
        args.spec, source_run_path=args.source_run, query_limit=args.query_limit
    )
    if args.audit_only:
        result = construction_audit(queries, args.vector_cache or args.output)
    else:
        result = build_fixture(
            args.spec,
            args.output,
            source_run_path=args.source_run,
            query_limit=args.query_limit,
            vector_cache_path=args.vector_cache,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
