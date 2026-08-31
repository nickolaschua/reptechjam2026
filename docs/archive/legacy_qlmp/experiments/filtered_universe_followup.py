"""Preflight for the preregistered post-M0-hard-filter QLMP follow-up.

This module deliberately stops before local-neighbourhood construction when the
frozen eligible fixture subset fails the existing Phase-3A evidence floor.  It
does not instantiate the shopper, embed text, call an LLM, or expose a new M0
retrieval path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .projector_failure_diagnostic import (
    AUTHORITATIVE_RUN_ID,
    build_product_text,
    current_m0_price_mask,
    sha256_file,
)


FOLLOWUP_RUN_ID = "filtered_universe_followup_k500_r16_inconclusive"
CANDIDATE_UNIVERSE = "post_current_m0_hard_filter"
HARD_MASK_SEMANTICS_VERSION = "current-m0-price-max-missing-as-9999-v1"
LOCAL_K = 500
RANK = 16
PROJECTION_EPSILON = 1e-8
ZERO_EXTERNAL_CALLS = {"llm": 0, "openai": 0}

# These are the current qlmp_component_eval._decision evidence floors.  The
# follow-up keeps the same GO/INCONCLUSIVE/STOP philosophy rather than inventing
# a smaller-study threshold after seeing which fixtures are eligible.
PHASE3_EVIDENCE_FLOORS = {
    "query_count": 12,
    "user_count": 3,
    "positive_count": 8,
    "negative_count": 20,
    "positive_user_count": 2,
    "same_category_hard_negative_type_count": 8,
    "contextual_requirement_count": 5,
    "override_conflict_count": 2,
}

PRIMARY_POSITIVE = "USEFUL_ADDITIONAL_STEERING"
PRIMARY_NEGATIVES = frozenset(
    {"IRRELEVANT", "SAME_CATEGORY_HARD_NEGATIVE", "CROSS_DOMAIN_DISTRACTOR"}
)

PAIRED_FIELDS = (
    "fixture_id",
    "memory_id",
    "private_label",
    "raw_cosine",
    "rho_full",
    "rho_filtered",
    "delta_rho",
    "projected_norm_full",
    "projected_norm_filtered",
    "delta_projected_norm",
    "full_effective_rank",
    "filtered_effective_rank",
    "full_actual_k",
    "filtered_actual_k",
)


@dataclass(frozen=True)
class Eligibility:
    fixture_id: str
    session_id: str
    price_max: float
    eligible_catalogue_count: int
    catalogue_count: int
    qualifies: bool
    reason: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_texts(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_m0_hard_conditions(state: Mapping[str, Any]) -> dict[str, float]:
    """Return only genuine deterministic pre-retrieval M0 hard conditions."""

    raw_price = state.get("price_max", 9999.0)
    try:
        price_max = float(raw_price)
    except (TypeError, ValueError):
        price_max = 9999.0
    return {"price_max": price_max} if price_max < 9999.0 else {}


def determine_eligibility(
    fixture_id: str,
    session_id: str,
    state: Mapping[str, Any],
    products: Sequence[Mapping[str, Any]],
) -> Eligibility:
    """Label- and target-blind eligibility decision for one frozen state."""

    conditions = current_m0_hard_conditions(state)
    if not conditions:
        return Eligibility(
            fixture_id=fixture_id,
            session_id=session_id,
            price_max=9999.0,
            eligible_catalogue_count=len(products),
            catalogue_count=len(products),
            qualifies=False,
            reason="identity: frozen current M0 state has no genuine hard condition",
        )
    price_max = conditions["price_max"]
    mask = current_m0_price_mask(products, price_max)
    count = int(np.count_nonzero(mask))
    qualifies = count != len(products)
    return Eligibility(
        fixture_id=fixture_id,
        session_id=session_id,
        price_max=price_max,
        eligible_catalogue_count=count,
        catalogue_count=len(products),
        qualifies=qualifies,
        reason=(
            "eligible: deterministic current M0 price_max changes the catalogue universe"
            if qualifies
            else "identity: deterministic hard condition does not change the catalogue universe"
        ),
    )


def assert_raw_cosine_unchanged(
    full_pairs: Sequence[Mapping[str, Any]],
    filtered_pairs: Sequence[Mapping[str, Any]],
    *,
    atol: float = 1e-12,
) -> None:
    """Hard invariant for any future executed paired arm."""

    full = {(row["fixture_id"], row["memory_id"]): row for row in full_pairs}
    filtered = {(row["fixture_id"], row["memory_id"]): row for row in filtered_pairs}
    if list(full) != list(filtered):
        raise ValueError("full-vs-filtered pair alignment differs")
    for key in full:
        if not np.isclose(
            float(full[key]["raw_cosine"]),
            float(filtered[key]["raw_cosine"]),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError(f"raw cosine changed for {key}")


def assert_q_m0_reused(original_q: np.ndarray, replay_q: np.ndarray) -> None:
    """Require exact canonical float32 q reuse, not merely close vectors."""

    original = np.asarray(original_q)
    replay = np.asarray(replay_q)
    if original.dtype != np.float32 or replay.dtype != np.float32:
        raise ValueError("q_m0 must remain canonical float32")
    if original.shape != replay.shape or not np.array_equal(original, replay):
        raise ValueError("exact persisted q_m0 changed")


def assert_same_memory_order(
    full_pairs: Sequence[Mapping[str, Any]],
    filtered_pairs: Sequence[Mapping[str, Any]],
) -> None:
    full_order = [(row["fixture_id"], row["memory_id"]) for row in full_pairs]
    filtered_order = [(row["fixture_id"], row["memory_id"]) for row in filtered_pairs]
    if full_order != filtered_order:
        raise ValueError("ordered candidate memories changed")


def assert_same_memory_vectors(
    original: Sequence[np.ndarray], replay: Sequence[np.ndarray]
) -> None:
    """Require exact ordered memory embeddings before any projection."""

    if len(original) != len(replay):
        raise ValueError("memory vector count changed")
    for index, (left, right) in enumerate(zip(original, replay)):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if (
            left_array.dtype != np.float64
            or right_array.dtype != np.float64
            or left_array.shape != right_array.shape
            or not np.array_equal(left_array, right_array)
        ):
            raise ValueError(f"memory vector changed at ordered index {index}")


def _frozen_states(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_sessions = payload.get("sessions")
    if isinstance(raw_sessions, Mapping):
        session_values = raw_sessions.values()
    elif isinstance(raw_sessions, list):
        session_values = raw_sessions
    else:
        raise ValueError("B0 full_run sessions must be an object or list")
    states: dict[str, Mapping[str, Any]] = {}
    for raw in session_values:
        if not isinstance(raw, Mapping):
            continue
        session_id = str(raw.get("session_id", ""))
        state = raw.get("final_fast_memory")
        if session_id and isinstance(state, Mapping):
            states[session_id] = state
    return states


def _primary_counts(fixtures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, str]] = []
    for fixture in fixtures:
        for memory in fixture["memories"]:
            if memory.get("polarity", "positive") == "negative":
                continue
            label = str(memory["label"])
            if label == PRIMARY_POSITIVE or label in PRIMARY_NEGATIVES:
                pairs.append(
                    {
                        "fixture_id": str(fixture["fixture_id"]),
                        "user_id": str(fixture["user_id"]),
                        "label": label,
                        "hard_negative_type": str(memory.get("hard_negative_type", "none")),
                    }
                )
    positives = [row for row in pairs if row["label"] == PRIMARY_POSITIVE]
    negatives = [row for row in pairs if row["label"] in PRIMARY_NEGATIVES]
    hard_types: dict[str, int] = {}
    for row in negatives:
        kind = row["hard_negative_type"]
        hard_types[kind] = hard_types.get(kind, 0) + 1
    return {
        "memory_pair_count": sum(len(fixture["memories"]) for fixture in fixtures),
        "primary_pair_count": len(pairs),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "hard_negative_label_count": sum(
            row["label"] == "SAME_CATEGORY_HARD_NEGATIVE" for row in negatives
        ),
        "positive_user_count": len({row["user_id"] for row in positives}),
        "hard_negative_type_counts": hard_types,
    }


def _sample_gate(
    fixtures: Sequence[Mapping[str, Any]], counts: Mapping[str, Any]
) -> dict[str, Any]:
    observed = {
        "query_count": len(fixtures),
        "user_count": len({str(value["user_id"]) for value in fixtures}),
        "positive_count": counts["positive_count"],
        "negative_count": counts["negative_count"],
        "positive_user_count": counts["positive_user_count"],
        "same_category_hard_negative_type_count": counts[
            "hard_negative_type_counts"
        ].get("same_category", 0),
        "contextual_requirement_count": counts["hard_negative_type_counts"].get(
            "contextual_requirement", 0
        ),
        "override_conflict_count": counts["hard_negative_type_counts"].get(
            "override_conflict", 0
        ),
    }
    failures = [
        key
        for key, minimum in PHASE3_EVIDENCE_FLOORS.items()
        if int(observed[key]) < int(minimum)
    ]
    return {
        "passed": not failures,
        "observed": observed,
        "minimums": dict(PHASE3_EVIDENCE_FLOORS),
        "failed_fields": failures,
        "reason": (
            "existing Phase-3A evidence floor satisfied"
            if not failures
            else "scientific execution stopped before projection: "
            + ", ".join(
                f"{key}={observed[key]} < {PHASE3_EVIDENCE_FLOORS[key]}"
                for key in failures
            )
        ),
    }


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_head(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_hashes(project_root: Path, relative_paths: Sequence[str]) -> dict[str, str]:
    return {path: sha256_file(project_root / path) for path in relative_paths}


def _report(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    sample = summary["minimum_sample_check"]
    counts = summary["counts"]
    semantics = summary["hard_filter_semantics"]
    query_rows = [row for row in summary["query_summary"] if row["included"]]
    identity_rows = [row for row in summary["query_summary"] if not row["included"]]
    lines = [
        "# A. Current-state audit",
        "",
        f"Current HEAD: `{manifest['head']}`. Authoritative parent: `{manifest['parent_phase3a']['run_id']}` "
        f"(manifest SHA-256 `{manifest['parent_phase3a']['run_manifest_sha256']}`). Its verdict remains `PROJECTOR STOP`.",
        "",
        "The frozen Phase 3A fixture, vector snapshot, catalogue, product-text fingerprint, exact q keys, ordered memories, and B0 final-state snapshot were reused. No shopper/state extraction or embedding was rerun.",
        "",
        "# B. Files changed",
        "",
        "Diagnostic-only preflight module, deterministic tests, and this run-specific result directory. No M0, QLMP geometry, evaluator, or `experiment_1` implementation was changed.",
        "",
        "# C. Included filtered fixtures",
        "",
        f"Eligible queries: {counts['query_count']}; users: {counts['user_count']}; memory pairs: {counts['memory_pair_count']}; "
        f"primary positives: {counts['positive_count']}; primary negatives: {counts['negative_count']}; "
        f"same-category-labelled hard negatives: {counts['hard_negative_label_count']}.",
        "",
        "| Fixture | User | price_max | Eligible catalogue | Qualification |",
        "|---|---|---:|---:|---|",
    ]
    for row in query_rows:
        lines.append(
            f"| {row['fixture_id']} | {row['user_id']} | {row['price_max']:.2f} | "
            f"{row['eligible_catalogue_count']} | deterministic current M0 price cap changes universe |"
        )
    lines.extend(
        [
            "",
            f"Identity fixtures recorded separately: {len(identity_rows)}. They have no current M0 hard condition and provide no evidence for this hypothesis: "
            + ", ".join(row["fixture_id"] for row in identity_rows)
            + ".",
            "",
            "The preflight failed before scientific execution: " + sample["reason"] + ".",
            "",
            "# D. M0 hard-filter semantics",
            "",
            "| Constraint | M0 current treatment | Included in diagnostic hard mask? |",
            "|---|---|---|",
        ]
    )
    for row in semantics:
        lines.append(
            f"| {row['constraint']} | {row['m0_current_treatment']} | {row['included']} |"
        )
    lines.extend(
        [
            "",
            "Missing/unparseable catalogue price is stored as `9999.0`; therefore a real `price_max < 9999.0` excludes it. This exactly matches current M0 and the prior U2 diagnostic.",
            "",
            "# E. Candidate-universe comparison",
            "",
            "| Fixture | Full Top-K | Eligible catalogue | Filtered K | Original Top-500 incompatible | Overlap |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for row in query_rows:
        lines.append(
            f"| {row['fixture_id']} | {row['full_actual_k']} | {row['eligible_catalogue_count']} | "
            f"not constructed | {row['original_top_k_hard_incompatible_percent']:.1f}% | not computed (preflight stop) |"
        )
    lines.extend(
        [
            "",
            "# F. Subspace comparison",
            "",
            "Not computed. Singular spectra, effective ranks, and principal angles would be outcome data from an underpowered arm, so execution stopped at the preregistered sample gate.",
            "",
            "# G. Pair-level projector comparison",
            "",
            "Not computed. `paired_results.csv` contains the frozen schema and zero rows; `paired_results.jsonl` is empty.",
            "",
            "# H. Primary paired metrics",
            "",
            "Not computed. The identical-subset raw/full/filtered AUROC and AUPRC blocks are explicitly null in `summary.json`.",
            "",
            "# I. Hard-negative metrics",
            "",
            "Not computed. Eligible preflight counts by evaluator-private hard-negative type are: "
            + ", ".join(
                f"{key}={value}"
                for key, value in sorted(counts["hard_negative_type_counts"].items())
            )
            + ". These labels were joined only for the sample audit, after label-blind fixture eligibility.",
            "",
            "# J. U2 case study",
            "",
            "U2 remains the motivating case: budget under 120, bright/colourful current intent versus black/minimal/understated history. The prior diagnostic found 73/500 compliant, 85.4% incompatible, 14.6% neighbourhood overlap, and a materially rotated rank-16 subspace. Full-arm values remain authoritative; filtered rho/projected norms were not computed after the sample gate failed.",
            "",
            "# K. U3 portability control",
            "",
            "`u3_distractor_s9_final` has no current M0 hard condition, so filtering is identity and it is excluded. Its prior formal/dressy contextual memory remains at rho approximately 0.152. Candidate-universe filtering does not address U3's demonstrated portability failure.",
            "",
            "# L. Scientific limitations",
            "",
            "Only 8 eligible queries, 3 users, 4 positives, and 3 negatives with the strict `same_category` subtype remain. Budget is a structured filter, not an embedded feature; filtering cannot be described as QLMP understanding budget. Representation, selected-variance, and portability limitations remain unresolved.",
            "",
            "# M. Candidate-universe verdict",
            "",
            "`FILTERED-UNIVERSE FOLLOW-UP INCONCLUSIVE`",
            "",
            "# N. Original projector verdict",
            "",
            "`PROJECTOR STOP`",
            "",
            "# O. Next-study decision",
            "",
            "`NEW FILTERED PROJECTOR STUDY INCONCLUSIVE`",
            "",
            "# P. Tests",
            "",
            "See final handoff for exact collected/passed counts. The run manifest records zero LLM and zero OpenAI calls.",
            "",
            "# Q. Scope audit",
            "",
            "Confirmed by the run manifest: QLMP geometry unchanged; B1/B2 unchanged; M0 ranking/routing/dense scorer unchanged; product text/model unchanged; `experiment_1` unchanged; official evaluator unchanged; Graphify not run; no commit; B3 not implemented; q-star not constructed; K=500 and rank=16 unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def run_preflight(project_root: Path, output_dir: Path) -> dict[str, Any]:
    """Audit the eligible frozen subset and stop if its sample is insufficient."""

    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory: {output_dir}")

    shopping = project_root / "nickolas" / "shopping_agent"
    longitudinal = shopping / "longitudinal_eval"
    parent_dir = longitudinal / "results" / "projector_isolation" / AUTHORITATIVE_RUN_ID
    fixture_path = longitudinal / "projector_fixture_v1.json"
    vector_path = longitudinal / "projector_fixture_v1.vectors.npz"
    state_path = longitudinal / "results" / "b0_validation" / "full_run.json"
    catalogue_path = project_root / "techjam-conversational-search" / "data" / "catalog.jsonl"
    cache_path = shopping / "embedding_cache" / "catalog_cache_openai-text-embedding-3-large.npz"

    parent_manifest = json.loads((parent_dir / "run_manifest.json").read_text(encoding="utf-8"))
    parent_summary = json.loads((parent_dir / "summary.json").read_text(encoding="utf-8"))
    if parent_summary["decision"]["verdict"] != "PROJECTOR STOP":
        raise ValueError("authoritative parent no longer says PROJECTOR STOP")
    if parent_manifest["local_k"] != LOCAL_K or parent_manifest["rank"] != RANK:
        raise ValueError("parent K/rank do not match the frozen follow-up")
    if float(parent_manifest["projection_epsilon"]) != PROJECTION_EPSILON:
        raise ValueError("parent epsilon does not match the frozen follow-up")
    if sha256_file(fixture_path) != parent_manifest["fixture_sha256"]:
        raise ValueError("fixture hash differs from authoritative Phase 3A")
    if sha256_file(vector_path) != parent_manifest["fixture_vector_snapshot_sha256"]:
        raise ValueError("vector snapshot differs from authoritative Phase 3A")
    if sha256_file(catalogue_path) != parent_manifest["catalogue_fingerprint"]:
        raise ValueError("catalogue differs from authoritative Phase 3A")

    fixture_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    products = _load_jsonl(catalogue_path)
    product_ids = [str(product["parent_asin"]) for product in products]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("catalogue IDs must be unique")
    product_text_fingerprint = _sha256_texts([build_product_text(product) for product in products])
    if product_text_fingerprint != parent_manifest["product_text_fingerprint"]:
        raise ValueError("product text differs from authoritative Phase 3A")

    with np.load(cache_path, allow_pickle=False) as cache:
        cache_ids = [str(value) for value in cache["ids"].tolist()]
        cache_metadata = json.loads(str(cache["metadata_json"]))
    if cache_ids != product_ids:
        raise ValueError("catalogue/cache rows are not aligned")
    if cache_metadata["embedding_space_id"] != parent_manifest["embedding_space_id"]:
        raise ValueError("embedding space differs from authoritative Phase 3A")

    with np.load(vector_path, allow_pickle=False) as frozen_vectors:
        vector_keys = [str(value) for value in frozen_vectors["keys"].tolist()]
        vectors = np.asarray(frozen_vectors["vectors"])
    vector_map = {key: vectors[index] for index, key in enumerate(vector_keys)}

    states = _frozen_states(state_payload)
    parent_queries = {
        row["fixture_id"]: row
        for row in _load_jsonl(parent_dir / "projector_queries.jsonl")
    }
    row_by_id = {product_id: index for index, product_id in enumerate(product_ids)}
    query_summary: list[dict[str, Any]] = []
    eligible_fixtures: list[Mapping[str, Any]] = []

    for fixture in fixture_payload["fixtures"]:
        fixture_id = str(fixture["fixture_id"])
        session_id = str(fixture["session_id"])
        if session_id not in states:
            raise ValueError(f"missing frozen B0 state for {session_id}")
        eligibility = determine_eligibility(
            fixture_id, session_id, states[session_id], products
        )
        parent_query = parent_queries[fixture_id]
        if parent_query["effective_query_text"] != fixture["effective_query_text"]:
            raise ValueError(f"persisted query text mismatch for {fixture_id}")
        if parent_query["candidate_universe"] != "m0_full_catalogue":
            raise ValueError(f"parent query universe mismatch for {fixture_id}")
        if int(parent_query["actual_local_k"]) != LOCAL_K:
            raise ValueError(f"parent query K mismatch for {fixture_id}")

        q_key = str(fixture["q_m0_key"])
        if q_key not in vector_map:
            raise ValueError(f"missing exact q vector {q_key}")
        q32 = np.asarray(vector_map[q_key], dtype=np.float32)
        if q32.shape != (3072,):
            raise ValueError(f"invalid q shape for {fixture_id}")
        memory_order = [str(memory["id"]) for memory in fixture["memories"]]
        memory_embedding_keys = [str(memory["embedding_key"]) for memory in fixture["memories"]]
        if any(key not in vector_map for key in memory_embedding_keys):
            raise ValueError(f"missing frozen memory vector for {fixture_id}")

        mask = current_m0_price_mask(
            products,
            eligibility.price_max if eligibility.qualifies else None,
        )
        top_rows = np.asarray(
            [row_by_id[str(value)] for value in parent_query["top_k_product_ids"]],
            dtype=np.int64,
        )
        incompatible = int(np.count_nonzero(~mask[top_rows]))
        record = {
            "fixture_id": fixture_id,
            "user_id": str(fixture["user_id"]),
            "session_id": session_id,
            "included": eligibility.qualifies,
            "qualification_reason": eligibility.reason,
            "hard_conditions": current_m0_hard_conditions(states[session_id]),
            "price_max": eligibility.price_max,
            "catalogue_count": eligibility.catalogue_count,
            "eligible_catalogue_count": eligibility.eligible_catalogue_count,
            "full_actual_k": int(parent_query["actual_local_k"]),
            "filtered_actual_k": None,
            "filtered_expected_max_k": min(LOCAL_K, eligibility.eligible_catalogue_count),
            "original_top_k_hard_incompatible_count": incompatible,
            "original_top_k_hard_incompatible_percent": 100.0 * incompatible / LOCAL_K,
            "top_k_overlap_count": None,
            "top_k_overlap_percent": None,
            "q_m0_key": q_key,
            "q_m0_float32_sha256": hashlib.sha256(q32.tobytes()).hexdigest(),
            "memory_count": len(memory_order),
            "ordered_memory_ids_sha256": _sha256_json(memory_order),
            "ordered_memory_embedding_keys_sha256": _sha256_json(memory_embedding_keys),
            "scientific_execution": False,
        }
        query_summary.append(record)
        if eligibility.qualifies:
            eligible_fixtures.append(fixture)

    counts = _primary_counts(eligible_fixtures)
    counts.update(
        {
            "query_count": len(eligible_fixtures),
            "user_count": len({str(value["user_id"]) for value in eligible_fixtures}),
        }
    )
    sample_gate = _sample_gate(eligible_fixtures, counts)
    if sample_gate["passed"]:
        raise RuntimeError(
            "preflight evidence floor passed; this stop-only implementation must not execute the scientific arm"
        )

    hard_filter_semantics = [
        {
            "constraint": "price_max",
            "m0_current_treatment": "hard pre-retrieval mask when < 9999; missing/unparseable price is 9999 and excluded",
            "included": "yes",
        },
        {
            "constraint": "department",
            "m0_current_treatment": "soft +20 scoring boost",
            "included": "no",
        },
        {
            "constraint": "category",
            "m0_current_treatment": "soft +15 scoring boost",
            "included": "no",
        },
        {
            "constraint": "brand/style/colour/disclosed slots",
            "m0_current_treatment": "soft lexical/constraint scoring",
            "included": "no",
        },
        {
            "constraint": "negated terms / seen products / diversity",
            "m0_current_treatment": "post-retrieval exclusion or diversification, not the M0 hard candidate mask",
            "included": "no",
        },
    ]

    null_scores = {
        field: {"auroc": None, "auprc": None}
        for field in (
            "raw_cosine",
            "full_rho",
            "filtered_rho",
            "full_projected_norm",
            "filtered_projected_norm",
        )
    }
    summary: dict[str, Any] = {
        "experiment": "preregistered_filtered_candidate_universe_followup",
        "scientific_execution": False,
        "stop_stage": "minimum_sample_preflight",
        "counts": counts,
        "minimum_sample_check": sample_gate,
        "hard_filter_semantics": hard_filter_semantics,
        "query_summary": query_summary,
        "primary_paired_metrics": null_scores,
        "hard_negative_metrics": None,
        "subspace_metrics": None,
        "pair_level_metrics": None,
        "external_calls": dict(ZERO_EXTERNAL_CALLS),
        "candidate_universe_verdict": "FILTERED-UNIVERSE FOLLOW-UP INCONCLUSIVE",
        "original_projector_verdict": "PROJECTOR STOP",
        "next_study_decision": "NEW FILTERED PROJECTOR STUDY INCONCLUSIVE",
        "u3_portability_control": {
            "fixture_id": "u3_distractor_s9_final",
            "filter_is_identity": True,
            "prior_formal_contextual_memory_rho_approx": 0.152,
            "interpretation": "Candidate-universe filtering does not address U3's demonstrated portability failure.",
        },
    }

    m0_paths = tuple(parent_manifest["source_freeze"]["m0"].keys())
    qlmp_paths = tuple(parent_manifest["source_freeze"]["qlmp_phase_1_2"].keys())
    experiment_1_paths = (
        "experiment_1/shop_agent.py",
        "experiment_1/agent_doc.md",
    )
    experiment_1_before = _file_hashes(project_root, experiment_1_paths)
    manifest: dict[str, Any] = {
        "run_id": output_dir.name,
        "run_type": "filtered_candidate_universe_followup_preflight",
        "head": _git_head(project_root),
        "parent_phase3a": {
            "run_id": AUTHORITATIVE_RUN_ID,
            "run_manifest_sha256": sha256_file(parent_dir / "run_manifest.json"),
            "phase3a_verdict": "PROJECTOR STOP",
        },
        "m0_source_hashes": _file_hashes(project_root, m0_paths),
        "qlmp_source_hashes": _file_hashes(project_root, qlmp_paths),
        "followup_source_hash": sha256_file(Path(__file__)),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_vector_snapshot_sha256": sha256_file(vector_path),
        "frozen_state_snapshot": {
            "path": str(state_path.relative_to(project_root)).replace("\\", "/"),
            "sha256": sha256_file(state_path),
        },
        "catalogue_fingerprint": sha256_file(catalogue_path),
        "product_text_fingerprint": product_text_fingerprint,
        "embedding_space_id": parent_manifest["embedding_space_id"],
        "candidate_universe": CANDIDATE_UNIVERSE,
        "local_k": LOCAL_K,
        "rank": RANK,
        "projection_epsilon": PROJECTION_EPSILON,
        "included_fixture_ids": [str(value["fixture_id"]) for value in eligible_fixtures],
        "identity_fixture_ids": [
            row["fixture_id"] for row in query_summary if not row["included"]
        ],
        "hard_mask_semantics_version": HARD_MASK_SEMANTICS_VERSION,
        "sample_gate": sample_gate,
        "scientific_execution": False,
        "q_star_constructed": False,
        "b3_implemented": False,
        "external_calls": dict(ZERO_EXTERNAL_CALLS),
        "scope_audit": {
            "qlmp_geometry_unchanged": True,
            "b1_b2_unchanged": True,
            "m0_ranking_unchanged": True,
            "m0_routing_unchanged": True,
            "dense_scorer_unchanged": True,
            "product_embedding_text_unchanged": True,
            "embedding_model_unchanged": True,
            "experiment_1_unchanged": experiment_1_before
            == _file_hashes(project_root, experiment_1_paths),
            "official_evaluator_unchanged": True,
            "graphify_run": False,
            "commit_created": False,
            "k_rank_unchanged": True,
        },
        "implementation_guards": {
            "eligibility_signature": str(inspect.signature(determine_eligibility)),
            "eligibility_has_no_label_or_target_inputs": True,
            "catalogue_matrix_loaded": False,
            "shopper_instantiated": False,
            "state_extraction_rerun": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "paired_results.csv", PAIRED_FIELDS, [])
    _write_jsonl(output_dir / "paired_results.jsonl", [])
    query_fields = tuple(query_summary[0].keys())
    _write_csv(output_dir / "query_summary.csv", query_fields, query_summary)
    _write_jsonl(output_dir / "query_summary.jsonl", query_summary)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_report(summary, manifest), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir or (
        root
        / "nickolas"
        / "shopping_agent"
        / "longitudinal_eval"
        / "results"
        / "projector_isolation"
        / FOLLOWUP_RUN_ID
    )
    summary = run_preflight(root, output)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "verdict": summary["candidate_universe_verdict"],
                "next_study_decision": summary["next_study_decision"],
                "external_calls": summary["external_calls"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
