"""Locked cache-only evaluation for M4 relevant-memory retrieval steering."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from nickolas.memory.qlmp import MemoryPolarity
from ..agent import Agent
from ..embedding_backends import (
    CacheExpectation,
    OpenAIEmbeddingBackend,
    cache_filename,
    fingerprint_file,
    fingerprint_texts,
    load_embedding_cache,
)
from .masked_memory_evaluation import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_CATALOG,
    DEFAULT_FIXTURE,
    PROJECT_ROOT,
    SHOPPING_DIR,
    SMALL_METADATA,
    FrozenBundle,
    _bootstrap_mrr_delta,
    _catalog_inputs,
    _metrics,
    assert_identical_samples,
    load_large_bundle,
    load_small_bundle,
    validate_vector_space,
)
from .relevant_memory_retrieval import (
    M4_LAMBDA_MEMORY,
    M4_TOP_K,
    assert_logical_memory_parity,
    score_relevant_memory_query,
)


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CURRENT_DIR / "results" / "relevant_memory_retrieval"
FROZEN_RESULTS_DIR = CURRENT_DIR / "results" / "masked_memory_steering"
RESULTS_DOCUMENT = CURRENT_DIR / "RELEVANT_MEMORY_RETRIEVAL_RESULTS.md"

# These hashes freeze the already-completed M0/M1/M2/M3 artifacts.  M4 reads
# them for comparison and never invokes the earlier steering implementation.
FROZEN_ARTIFACT_SHA256 = {
    "session_results.jsonl": "f5bd42f45561114f6c07bddcd277c74f6be33d241780400d544b04d499df2811",
    "summary.json": "0f4623fc1fe498fb5a3b6232897b9cdffde0600af331941d555140de54cc7f58",
    "run_manifest.json": "2a11f16572276263b1dd0686a41223652254aaab26e231a7ba1de914995609c8",
    "session_manifest.json": "10913ec671809a8401ce0c4f903055dbab4700b9ce3eb4a4b03c2e5bdb40cd37",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen_baselines() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    for name, expected in FROZEN_ARTIFACT_SHA256.items():
        actual = _sha256_file(FROZEN_RESULTS_DIR / name)
        if actual != expected:
            raise ValueError(f"frozen M0/M1/M2/M3 artifact changed: {name}")
    rows = [
        json.loads(line)
        for line in (FROZEN_RESULTS_DIR / "session_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((FROZEN_RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((FROZEN_RESULTS_DIR / "run_manifest.json").read_text(encoding="utf-8"))
    if len(rows) != 18 * 2 * 4:
        raise ValueError("frozen baseline row count is not 144")
    return rows, summary, manifest


def _pairwise_m4(
    m4_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    reference: str,
) -> dict[str, Any]:
    baseline = {str(row["session_id"]): row for row in baseline_rows}
    changes = [
        int(baseline[str(row["session_id"])]["target_rank"]) - int(row["target_rank"])
        for row in m4_rows
    ]
    rr_changes = [
        float(row["reciprocal_rank"])
        - float(baseline[str(row["session_id"])]["reciprocal_rank"])
        for row in m4_rows
    ]
    return {
        "compared": "M4",
        "reference": reference,
        "sessions_improved": sum(value > 0 for value in changes),
        "sessions_unchanged": sum(value == 0 for value in changes),
        "sessions_regressed": sum(value < 0 for value in changes),
        "mean_rank_change_reference_minus_compared": float(np.mean(changes)),
        "median_rank_change_reference_minus_compared": float(median(changes)),
        "mrr_change_compared_minus_reference": float(np.mean(rr_changes)),
    }


def _baseline_map(
    frozen_rows: Sequence[Mapping[str, Any]], model_id: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result = {
        (str(row["session_id"]), str(row["method"])): row
        for row in frozen_rows
        if str(row["embedding_model"]) == model_id
    }
    if len(result) != 18 * 4:
        raise ValueError(f"frozen baseline coverage is incomplete for {model_id}")
    return result


def _evaluate_bundle_m4(
    bundle: FrozenBundle,
    agent: Agent,
    frozen_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if agent.embedding_space_id != bundle.space.embedding_space_id:
        raise ValueError("scorer and fixture embedding spaces differ")
    baseline = _baseline_map(frozen_rows, bundle.space.model_id)
    rows: list[dict[str, Any]] = []
    for session in bundle.sessions:
        if any(origin >= session.sequence_index for origin in session.memory_origin_indices):
            raise ValueError(f"temporal leakage in {session.session_id}")
        eligible_pairs = tuple(
            (item, origin)
            for item, origin in zip(session.memories, session.memory_origin_indices)
            if item.polarity is not MemoryPolarity.NEGATIVE
        )
        eligible = tuple(item for item, _ in eligible_pairs)
        origins = tuple(origin for _, origin in eligible_pairs)
        result, q_m4, diagnostics = score_relevant_memory_query(
            agent,
            session.query,
            eligible,
            origins,
            current_sequence_index=session.sequence_index,
            query_space_id=session.space.embedding_space_id,
            memory_space_id=bundle.space.embedding_space_id,
            top_n=len(agent.catalog_ids),
            buyer_active=True,
        )
        validate_vector_space(q_m4, session.space, bundle.space, "q_M4")
        try:
            m4_rank = result.product_ids.index(session.target_asin) + 1
        except ValueError as exc:
            raise ValueError(f"target {session.target_asin} absent from catalogue") from exc
        ranks = {
            method: int(baseline[(session.session_id, method)]["target_rank"])
            for method in ("M0", "M1", "M2", "M3")
        }
        data = diagnostics.to_dict()
        rows.append(
            {
                "user_id": session.user_id,
                "session_id": session.session_id,
                "fixture_id": session.fixture_id,
                "sequence_index": session.sequence_index,
                "split": session.split,
                "scenario_type": session.scenario_type,
                "target_asin": session.target_asin,
                "current_buyer_query": session.query_text,
                "embedding_model": bundle.space.model_id,
                "embedding_space_id": bundle.space.embedding_space_id,
                "embedding_dimension": bundle.space.dimension,
                "method": "M4",
                "k": M4_TOP_K,
                "lambda_memory": M4_LAMBDA_MEMORY,
                "number_of_eligible_memories": len(eligible),
                "eligible_memory_ids": list(data["eligible_memory_ids"]),
                "eligible_memory_texts": list(data["eligible_memory_texts"]),
                "eligible_memory_origin_indices": list(data["eligible_memory_origin_indices"]),
                "eligible_memory_similarity_scores": list(data["eligible_similarity_scores"]),
                "selected_top_k_memory_ids": list(data["selected_memory_ids"]),
                "selected_top_k_memory_texts": list(data["selected_memory_texts"]),
                "selected_top_k_memory_origin_indices": list(data["selected_memory_origin_indices"]),
                "selected_similarity_scores": list(data["selected_similarity_scores"]),
                "selected_memory_count": int(data["selected_memory_count"]),
                "aggregate_selected_memory_norm": float(data["aggregate_selected_memory_norm"]),
                "cosine_q_m_top": data["cosine_q_m_top"],
                "cosine_q_q_m4": float(data["cosine_q_q_m4"]),
                "m0_target_rank": ranks["M0"],
                "m1_target_rank": ranks["M1"],
                "m2_target_rank": ranks["M2"],
                "m3_target_rank": ranks["M3"],
                "m4_target_rank": m4_rank,
                "target_rank": m4_rank,
                "reciprocal_rank_m4": 1.0 / m4_rank,
                "reciprocal_rank": 1.0 / m4_rank,
                "hit_at_10_m4": m4_rank <= 10,
                "hit_at_10": m4_rank <= 10,
                "rank_delta_m4_vs_m0": ranks["M0"] - m4_rank,
                "rank_delta_m4_vs_m1": ranks["M1"] - m4_rank,
                "rank_delta_m4_vs_m3": ranks["M3"] - m4_rank,
                "recommendations_top_10": list(result.product_ids[:10]),
                "scores_top_10": [float(value) for value in result.scores[:10]],
            }
        )

    baseline_by_method = {
        method: [baseline[(session.session_id, method)] for session in bundle.sessions]
        for method in ("M0", "M1", "M2", "M3")
    }
    pairwise = {
        f"M4-{reference}": _pairwise_m4(rows, baseline_by_method[reference], reference)
        for reference in ("M0", "M1", "M3")
    }
    rank = {
        row["session_id"]: {
            "M0": row["m0_target_rank"],
            "M1": row["m1_target_rank"],
            "M3": row["m3_target_rank"],
            "M4": row["m4_target_rank"],
        }
        for row in rows
    }
    rescue = [sid for sid, value in rank.items() if value["M1"] > value["M0"] and value["M4"] <= value["M0"]]
    partial = [sid for sid, value in rank.items() if value["M1"] > value["M0"] and value["M4"] < value["M1"] and value["M4"] > value["M0"]]
    improvement = [sid for sid, value in rank.items() if value["M4"] < value["M0"]]
    destroyed = [sid for sid, value in rank.items() if value["M1"] < value["M0"] and value["M4"] > value["M1"]]
    beats_m3 = [sid for sid, value in rank.items() if value["M4"] < value["M3"]]
    summary = {
        "embedding_model": bundle.space.model_id,
        "embedding_space_id": bundle.space.embedding_space_id,
        "m4_metrics": _metrics(rows),
        "pairwise": pairwise,
        "relevant_memory_rescue_count": len(rescue),
        "relevant_memory_rescue_session_ids": rescue,
        "relevant_memory_partial_rescue_count": len(partial),
        "relevant_memory_partial_rescue_session_ids": partial,
        "relevant_memory_improvement_count": len(improvement),
        "relevant_memory_improvement_session_ids": improvement,
        "relevant_memory_destroyed_useful_memory_count": len(destroyed),
        "relevant_memory_destroyed_useful_memory_session_ids": destroyed,
        "m4_beats_coordinate_masking_count": len(beats_m3),
        "m4_beats_coordinate_masking_session_ids": beats_m3,
        "m4_minus_m3_mean_rank_change": pairwise["M4-M3"]["mean_rank_change_reference_minus_compared"],
        "paired_bootstrap_m4_minus_m0": _bootstrap_mrr_delta(rows, baseline_by_method["M0"]),
    }
    return rows, summary


def _audit(
    large: FrozenBundle,
    small: FrozenBundle,
    fixture_path: Path,
) -> dict[str, Any]:
    counts = [len(session.memories) for session in large.sessions]
    per_session = []
    for left, right in zip(large.sessions, small.sessions):
        per_session.append(
            {
                "user_id": left.user_id,
                "session_id": left.session_id,
                "current_sequence_index": left.sequence_index,
                "eligible_memory_count": len(left.memories),
                "eligible_memory_ids": [item.id for item in left.memories],
                "origin_sequence_indices": list(left.memory_origin_indices),
                "all_origins_strictly_prior": all(
                    origin < left.sequence_index for origin in left.memory_origin_indices
                ),
                "large_small_logical_candidate_parity": (
                    [item.id for item in left.memories] == [item.id for item in right.memories]
                    and [item.text for item in left.memories] == [item.text for item in right.memories]
                    and left.memory_origin_indices == right.memory_origin_indices
                ),
            }
        )
    return {
        "fixture_path": str(fixture_path.relative_to(PROJECT_ROOT)),
        "fixture_sha256": large.source_fixture_sha256,
        "buyer_session_count": len(large.sessions),
        "eligible_memory_count_distribution": {
            str(key): value for key, value in sorted(Counter(counts).items())
        },
        "eligible_memory_pair_count": sum(counts),
        "unique_memory_text_count": len({item.text for session in large.sessions for item in session.memories}),
        "unique_memory_id_count": len({item.id for session in large.sessions for item in session.memories}),
        "sessions_with_fewer_than_three": [
            session.session_id for session in large.sessions if len(session.memories) < M4_TOP_K
        ],
        "exact_individual_vectors_available_before_aggregation": True,
        "large_small_logical_memory_parity": all(
            row["large_small_logical_candidate_parity"] for row in per_session
        ),
        "all_memories_strictly_temporally_isolated": all(
            row["all_origins_strictly_prior"] for row in per_session
        ),
        "reembedding_required": False,
        "large_dimension": large.space.dimension,
        "small_dimension": small.space.dimension,
        "per_session": per_session,
    }


def _metric_table(summary: Mapping[str, Any], metric: str, digits: int = 6) -> str:
    lines = ["| Embedding | M0 | M1 | M3 | M4 |", "| --- | ---: | ---: | ---: | ---: |"]
    for model in ("text-embedding-3-large", "text-embedding-3-small"):
        frozen = summary[model]["frozen_metrics"]
        m4 = summary[model]["m4_metrics"]
        values = [frozen[method][metric] for method in ("M0", "M1", "M3")] + [m4[metric]]
        lines.append(f"| {model} | " + " | ".join(f"{value:.{digits}f}" for value in values) + " |")
    return "\n".join(lines)


def _results_markdown(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    pair_lines = [
        "| Pair | Improved | Same | Regressed | Mean Rank Δ | Median Rank Δ | ΔMRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_label, model in (("Large", "text-embedding-3-large"), ("Small", "text-embedding-3-small")):
        for reference in ("M0", "M1", "M3"):
            value = summary[model]["pairwise"][f"M4-{reference}"]
            pair_lines.append(
                f"| {model_label} M4-{reference} | {value['sessions_improved']} | "
                f"{value['sessions_unchanged']} | {value['sessions_regressed']} | "
                f"{value['mean_rank_change_reference_minus_compared']:.3f} | "
                f"{value['median_rank_change_reference_minus_compared']:.3f} | "
                f"{value['mrr_change_compared_minus_reference']:+.6f} |"
            )

    representative_ids = (
        "u1_stable_s5", "u2_override_s7", "u3_distractor_s9", "u4_negative_s2", "u4_negative_s9"
    )
    large_rows = {
        str(row["session_id"]): row
        for row in rows
        if row["embedding_model"] == "text-embedding-3-large"
    }
    examples = [
        "| Current Buyer Query | Top Selected Memories (large) | M0 Rank | M4 Rank |",
        "| --- | --- | ---: | ---: |",
    ]
    for session_id in representative_ids:
        row = large_rows[session_id]
        query = str(row["current_buyer_query"]).replace("|", "\\|")
        memories = "<br>".join(str(value).replace("|", "\\|") for value in row["selected_top_k_memory_texts"])
        examples.append(f"| {query} | {memories} | {row['m0_target_rank']} | {row['m4_target_rank']} |")

    full_lines = [
        "| Embedding | Method | MRR | HR@10 | Mean Rank | Median Rank |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model in ("text-embedding-3-large", "text-embedding-3-small"):
        for method in ("M0", "M1", "M2", "M3", "M4"):
            value = summary[model]["all_metrics"][method]
            full_lines.append(
                f"| {model} | {method} | {value['mrr']:.6f} | {value['hr_at_10']:.6f} | "
                f"{value['mean_target_rank']:.3f} | {value['median_target_rank']:.3f} |"
            )

    large = summary["text-embedding-3-large"]
    small = summary["text-embedding-3-small"]
    large_delta = large["pairwise"]["M4-M0"]["mrr_change_compared_minus_reference"]
    small_delta = small["pairwise"]["M4-M0"]["mrr_change_compared_minus_reference"]
    if large_delta > 0 and small_delta > 0:
        interpretation = "M4 beat M0 in both frozen spaces, providing preliminary fixture-specific evidence for memory-level relevance selection."
    elif large_delta <= 0 and small_delta <= 0:
        interpretation = (
            "M4 did not beat M0 in either frozen space. The hypothesis is not supported on this fixture, "
            "and no K or algorithm tuning was performed. The next scientific question is whether the stored "
            "MemoryItems themselves are predictive enough to improve recommendation ranking."
        )
    else:
        interpretation = (
            "M4's uplift versus M0 changed sign across embedding spaces, so the behavior did not generalize "
            "consistently on this fixture. No K or algorithm tuning was performed."
        )
    different = summary["cross_space_selected_top_k_differences"]
    return f"""# Relevant-Memory Retrieval Results

## Frozen experiment

This cache-only run reused the exact 18-session Buyer fixture and the unchanged frozen M0/M1/M2/M3 artifacts. M4 used `K = 3`, `lambda_memory = 0.20`, equal-weight selected-memory aggregation, no threshold, and the canonical full-catalogue scorer. No embeddings were generated.

Positive rank Δ means M4 moved the target upward (reference rank minus M4 rank).

## Primary metrics

MRR:

{_metric_table(summary, 'mrr')}

HR@10:

{_metric_table(summary, 'hr_at_10')}

Mean target rank (lower is better):

{_metric_table(summary, 'mean_target_rank', 3)}

Median target rank (lower is better):

{_metric_table(summary, 'median_target_rank', 3)}

The prior M0/M1/M2/M3 values above are copied unchanged from the hash-locked result artifact. The complete M0–M4 table remains in `summary.json`.

### Complete M0–M4 table

{chr(10).join(full_lines)}

## Pairwise results

{chr(10).join(pair_lines)}

## Primary questions

- **Q1 — M4 vs M0:** No. M4 MRR is lower in both spaces ({large_delta:+.6f} large; {small_delta:+.6f} small), with worse mean and median target rank.
- **Q2 — M4 vs M1:** No at aggregate level. M4 MRR is lower than M1 by {large['pairwise']['M4-M1']['mrr_change_compared_minus_reference']:+.6f} large and {small['pairwise']['M4-M1']['mrr_change_compared_minus_reference']:+.6f} small; top-three selection did not avoid the negative transfer.
- **Q3 — M4 vs M3:** No. M4 MRR is lower than M3 by {large['pairwise']['M4-M3']['mrr_change_compared_minus_reference']:+.6f} large and {small['pairwise']['M4-M3']['mrr_change_compared_minus_reference']:+.6f} small.
- **Q4 — cross-space behavior:** The beneficial hypothesis did not generalize. M4 minus M0 is negative in both spaces, while the selected top-three IDs differ in {len(different)} of 18 sessions.

## Negative-transfer diagnostics

- Large: relevant-memory rescues {large['relevant_memory_rescue_count']}; partial rescues {large['relevant_memory_partial_rescue_count']}; M4 improvements over M0 {large['relevant_memory_improvement_count']}; destroyed useful-memory cases {large['relevant_memory_destroyed_useful_memory_count']}; M4 beats M3 in {large['m4_beats_coordinate_masking_count']} sessions (mean rank Δ {large['m4_minus_m3_mean_rank_change']:+.3f}).
- Small: relevant-memory rescues {small['relevant_memory_rescue_count']}; partial rescues {small['relevant_memory_partial_rescue_count']}; M4 improvements over M0 {small['relevant_memory_improvement_count']}; destroyed useful-memory cases {small['relevant_memory_destroyed_useful_memory_count']}; M4 beats M3 in {small['m4_beats_coordinate_masking_count']} sessions (mean rank Δ {small['m4_minus_m3_mean_rank_change']:+.3f}).
- Large M4-over-M0 sessions: {', '.join(large['relevant_memory_improvement_session_ids']) or 'none'}.
- Small M4-over-M0 sessions: {', '.join(small['relevant_memory_improvement_session_ids']) or 'none'}.
- Large partial rescues: {', '.join(large['relevant_memory_partial_rescue_session_ids']) or 'none'}; small partial rescues: {', '.join(small['relevant_memory_partial_rescue_session_ids']) or 'none'}.

## Representative selections

These examples are qualitative diagnostics only. Plausible text is not evidence of correctness; target rank is the objective result.

{chr(10).join(examples)}

Across the complete session artifact, selections often reflect surface attribute overlap (for example cotton with cotton memories, or zipper with double zippers), but several omit central intent attributes or retrieve only broadly related style/material memories. This is a qualitative observation, not a correctness or causality label, and the aggregate ranking result remains negative.

The large and small models selected different top-three memory ID sets in {len(different)} of 18 sessions: {', '.join(different) if different else 'none'}.

## Paired bootstrap: M4 minus M0 MRR

- Large: ΔMRR {large['paired_bootstrap_m4_minus_m0']['delta_mrr']:+.6f}; descriptive 95% percentile CI {large['paired_bootstrap_m4_minus_m0']['percentile_95_ci']}.
- Small: ΔMRR {small['paired_bootstrap_m4_minus_m0']['delta_mrr']:+.6f}; descriptive 95% percentile CI {small['paired_bootstrap_m4_minus_m0']['percentile_95_ci']}.
- Seed `{BOOTSTRAP_SEED}`, samples `{BOOTSTRAP_SAMPLES}`. These intervals are descriptive only; the fixture is small and previously inspected, so no significance claim is made.

## Artifacts

- `results/relevant_memory_retrieval/session_results.jsonl`: all 36 model-session diagnostics, including every candidate score and selected text.
- `results/relevant_memory_retrieval/fixture_audit.json`: per-session eligibility, logical parity, and temporal-isolation audit.
- `results/relevant_memory_retrieval/summary.json`: frozen M0–M3 metrics plus M4 metrics, pairwise comparisons, rescue categories, and bootstrap intervals.
- `results/relevant_memory_retrieval/run_manifest.json`: hashes and locked run parameters.

## Interpretation

{interpretation}
"""


def run_evaluation(
    output_dir: Path = DEFAULT_OUTPUT,
    fixture_path: Path = DEFAULT_FIXTURE,
    catalog_path: Path = DEFAULT_CATALOG,
) -> Path:
    frozen_rows, frozen_summary, frozen_manifest = _load_frozen_baselines()
    large = load_large_bundle(fixture_path)
    small = load_small_bundle(fixture_path, catalog_path)
    assert_identical_samples(large, small)
    assert_logical_memory_parity(large.sessions, small.sessions)
    if large.source_fixture_sha256 != frozen_manifest["fixture_sha256"]:
        raise ValueError("M4 fixture differs from the frozen M0/M1/M2/M3 fixture")

    catalog_ids, catalog_texts = _catalog_inputs(catalog_path)
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    cache_hashes: dict[str, str] = {}
    for bundle in (large, small):
        expectation = CacheExpectation(
            backend_id=bundle.space.backend_id,
            model_id=bundle.space.model_id,
            embedding_space_id=bundle.space.embedding_space_id,
            catalog_ids=catalog_ids,
            product_text_fingerprint=fingerprint_texts(catalog_texts),
            catalog_fingerprint=fingerprint_file(catalog_path),
            vector_dimension=bundle.space.dimension,
            normalized=True,
        )
        cache_path = SHOPPING_DIR / "embedding_cache" / cache_filename(bundle.space.backend_id)
        load_embedding_cache(cache_path, expectation)
        cache_hashes[bundle.space.model_id] = _sha256_file(cache_path)
        backend = OpenAIEmbeddingBackend(
            model_id=bundle.space.model_id,
            backend_id=bundle.space.backend_id,
            vector_dimension=bundle.space.dimension,
        )
        agent = Agent(catalog_path=catalog_path, embedding_backend=backend, allow_catalog_embedding=False)
        try:
            if fingerprint_texts(agent.catalog_ids) != frozen_manifest["product_order_fingerprint"]:
                raise ValueError("canonical catalogue ordering changed")
            rows, model_summary = _evaluate_bundle_m4(bundle, agent, frozen_rows)
            if backend.usage_snapshot()["request_count"] != 0:
                raise RuntimeError("cache-only M4 evaluation attempted an embedding request")
            model_summary["frozen_metrics"] = frozen_summary[bundle.space.model_id]["metrics"]
            model_summary["all_metrics"] = {
                **frozen_summary[bundle.space.model_id]["metrics"],
                "M4": model_summary["m4_metrics"],
            }
            all_rows.extend(rows)
            summaries[bundle.space.model_id] = model_summary
        finally:
            agent.connection.close()

    large_selected = {
        row["session_id"]: row["selected_top_k_memory_ids"]
        for row in all_rows if row["embedding_model"] == "text-embedding-3-large"
    }
    small_selected = {
        row["session_id"]: row["selected_top_k_memory_ids"]
        for row in all_rows if row["embedding_model"] == "text-embedding-3-small"
    }
    differences = [session_id for session_id in large_selected if large_selected[session_id] != small_selected[session_id]]
    summaries["cross_space_selected_top_k_differences"] = differences
    summaries["cross_space_selected_top_k_difference_count"] = len(differences)

    audit = _audit(large, small, fixture_path)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "locked_relevant_memory_retrieval_dual_openai",
        "fixture_path": str(fixture_path.relative_to(PROJECT_ROOT)),
        "fixture_sha256": large.source_fixture_sha256,
        "logical_memory_manifest_sha256": large.logical_manifest_sha256,
        "large_vector_snapshot_sha256": large.vector_snapshot_sha256,
        "small_vector_snapshot_sha256": small.vector_snapshot_sha256,
        "catalogue_fingerprint": fingerprint_file(catalog_path),
        "product_text_fingerprint": fingerprint_texts(catalog_texts),
        "product_order_fingerprint": fingerprint_texts(catalog_ids),
        "catalogue_row_count": len(catalog_ids),
        "buyer_session_count": len(large.sessions),
        "ordered_session_ids": [session.session_id for session in large.sessions],
        "k": M4_TOP_K,
        "lambda_memory": M4_LAMBDA_MEMORY,
        "similarity_threshold": None,
        "selected_memory_weighting": "equal",
        "embedding_calls_during_evaluation": 0,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "previously_inspected_fixture": True,
        "frozen_m0_m3_artifact_sha256": FROZEN_ARTIFACT_SHA256,
        "catalogue_cache_sha256": cache_hashes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "session_results.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "fixture_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    RESULTS_DOCUMENT.write_text(_results_markdown(summaries, all_rows), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    destination = run_evaluation(args.output, args.fixture, args.catalog)
    print(f"Artifacts: {destination}")


if __name__ == "__main__":
    main()
