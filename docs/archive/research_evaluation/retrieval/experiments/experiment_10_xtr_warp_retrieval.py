from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import logging
import math
import shutil
import stat
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import MAX_TURNS, RRF_DEPTH, RRF_K, TOP_K
from .experiment_07_residual_failure_analysis import (
    FAILURE_TAGS,
    RetrievalInput,
    _comparison,
    _metrics_for_subset,
    _rank_distribution_rows,
    _rank_in_top,
    deterministic_rrf,
    failure_category_counts,
    fallback_reasons,
    load_frozen_split,
)
from .harness import Harness, experiment_logger, percentile_summary, sha256, write_csv, write_json


EXPERIMENT_NUMBER = 10
EXPERIMENT_SLUG = "xtr_warp_retrieval"
EXPERIMENT_DIRECTORY = f"experiment_{EXPERIMENT_NUMBER:02d}_{EXPERIMENT_SLUG}"

EXACT_METHOD = "exact_only"
BM25_METHOD = "experiment_07_exact_stateful_bm25_rrf"
WARP_METHOD = "exact_stateful_xtr_warp_rrf"
METHODS = (EXACT_METHOD, BM25_METHOD, WARP_METHOD)

EXPECTED_INPUT_ZIP_SHA256 = "511ad7b4231c7f4529590a6a8efa8ecd65d2179af96aa1c1636db2a4994fe538"
EXPECTED_SOURCE_REVISION = "cca97613e6f969ac89f259946b976f8c5a6f1399"
EXPECTED_MODEL_REVISION = "f40cd399e67dfc8ec974e922ad828610e3c83a36"
EXPECTED_MODEL_ID = "google/xtr-base-en"
EXPECTED_QUERY_COUNT = 600
EXPECTED_DOCUMENT_COUNT = 50_000
EXPECTED_RESULT_COUNT = 1_000

REQUIRED_COLAB_MEMBERS = {
    "README.txt",
    "SHA256SUMS",
    "experiment_10_config.json",
    "input_manifest.json",
    "manifest.json",
    "pid_to_asin.json",
    "provenance.json",
    "rankings/query_manifest.json",
    "rankings/rankings.jsonl",
    "rankings/rankings.tsv",
    "xtr_warp_compat.patch",
}

REQUIRED_ARTIFACTS = (
    "summary.md",
    "metrics.json",
    "rows.csv",
    "sessions.json",
    "baseline_reproduction.json",
    "colab_import_report.json",
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
    "weak_success_improvements.csv",
    "weak_success_improvements.json",
    "rank_distributions.csv",
    "rank_distributions.json",
    "rescue_comparison.png",
    "rank_distributions.png",
    "technical_score_comparison.png",
    "source_snapshot.py",
    "source_snapshot.json",
    "run.log",
)


@dataclass(frozen=True)
class FrozenXTRTurnRanking:
    retrieval_input: RetrievalInput
    fallback_activated: bool
    fallback_reasons: tuple[str, ...]
    all_phrases_exact_candidate_count: int
    highest_exact_match_count: int
    highest_exact_match_tier_count: int
    xtr_warp_top_10: tuple[int, ...]
    xtr_warp_top_10_scores: tuple[float, ...]
    method_top_10: Mapping[str, tuple[int, ...]]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_zip_member(archive: zipfile.ZipFile, name: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(name) as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _safe_archive_infos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = [item for item in archive.infolist() if not item.is_dir()]
    names = [item.filename for item in infos]
    if len(names) != len(set(names)):
        raise RuntimeError("Colab output contains duplicate member names")
    if len(infos) > 1_000:
        raise RuntimeError("Colab output contains unexpectedly many members")
    if sum(item.file_size for item in infos) > 2 * 1024**3:
        raise RuntimeError("Colab output exceeds the 2 GiB extraction safety limit")
    for item in infos:
        raw_name = item.filename
        raw_parts = raw_name.split("/")
        path = PurePosixPath(item.filename)
        mode = item.external_attr >> 16
        if (
            not raw_name
            or "\\" in raw_name
            or "\x00" in raw_name
            or path.is_absolute()
            or any(part in {"", ".", ".."} or ":" in part for part in raw_parts)
        ):
            raise RuntimeError(f"Unsafe Colab output path: {item.filename!r}")
        if stat.S_ISLNK(mode) or item.flag_bits & 0x1:
            raise RuntimeError(f"Links/encrypted members are not allowed: {item.filename!r}")
    return infos


def _validate_ranking_row(row: dict, expected_query_id: int) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    if row.get("query_id") != expected_query_id:
        raise RuntimeError(f"Ranking query ID mismatch at row {expected_query_id}")
    query = row.get("query")
    if not isinstance(query, str) or not query:
        raise RuntimeError(f"Ranking query {expected_query_id} is empty")
    expected_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    if row.get("query_sha256") != expected_hash:
        raise RuntimeError(f"Ranking query hash mismatch for query {expected_query_id}")
    results = row.get("results")
    if not isinstance(results, list) or len(results) != EXPECTED_RESULT_COUNT:
        raise RuntimeError(f"Ranking {expected_query_id} does not contain Top-{EXPECTED_RESULT_COUNT}")
    pids: list[int] = []
    scores: list[float] = []
    previous_key: tuple[float, int] | None = None
    for rank, pair in enumerate(results, 1):
        if not isinstance(pair, list) or len(pair) != 2:
            raise RuntimeError(f"Malformed result at query {expected_query_id}, rank {rank}")
        pid, score = pair
        if not isinstance(pid, int) or not 0 <= pid < EXPECTED_DOCUMENT_COUNT:
            raise RuntimeError(f"PID out of range at query {expected_query_id}, rank {rank}")
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise RuntimeError(f"Non-finite score at query {expected_query_id}, rank {rank}")
        key = (-float(score), pid)
        if previous_key is not None and key < previous_key:
            raise RuntimeError(f"Ranking order/tie-break violation at query {expected_query_id}, rank {rank}")
        previous_key = key
        pids.append(pid)
        scores.append(float(score))
    if len(set(pids)) != EXPECTED_RESULT_COUNT:
        raise RuntimeError(f"Ranking {expected_query_id} contains duplicate PIDs")
    return query, tuple(pids), tuple(scores)


def _validate_colab_archive(
    archive_path: Path,
    input_archive_path: Path,
    h: Harness,
) -> tuple[dict, dict[str, tuple[tuple[int, ...], tuple[float, ...]]], list[zipfile.ZipInfo]]:
    if not archive_path.is_file():
        raise RuntimeError(f"Colab output archive is missing: {archive_path}")
    if not input_archive_path.is_file():
        raise RuntimeError(f"Revision-3 Colab input archive is missing: {input_archive_path}")
    input_zip_sha = sha256(input_archive_path)
    if input_zip_sha != EXPECTED_INPUT_ZIP_SHA256:
        raise RuntimeError(f"Wrong Colab input ZIP revision: {input_zip_sha}")

    outer_sha = sha256(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        infos = _safe_archive_infos(archive)
        names = {item.filename for item in infos}
        missing_required = REQUIRED_COLAB_MEMBERS - names
        if missing_required or not any(name.startswith("warp_index/") for name in names):
            raise RuntimeError(f"Colab output is incomplete: missing={sorted(missing_required)}")

        manifest = json.loads(archive.read("manifest.json"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("archive_type") != "experiment_10_colab_output"
            or manifest.get("experiment") != EXPERIMENT_NUMBER
            or manifest.get("document_count") != EXPECTED_DOCUMENT_COUNT
            or manifest.get("query_count") != EXPECTED_QUERY_COUNT
        ):
            raise RuntimeError("Colab output manifest identity/count mismatch")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise RuntimeError("Colab output manifest has no file entries")
        declared = {entry["path"]: entry for entry in entries}
        if len(declared) != len(entries):
            raise RuntimeError("Colab output manifest contains duplicate paths")
        if names != set(declared) | {"manifest.json", "SHA256SUMS"}:
            raise RuntimeError("Colab output member set does not match manifest")

        sums: dict[str, str] = {}
        for line in archive.read("SHA256SUMS").decode("utf-8").splitlines():
            try:
                digest, name = line.split("  ", 1)
            except ValueError as exc:
                raise RuntimeError(f"Malformed SHA256SUMS entry: {line!r}") from exc
            if name in sums or len(digest) != 64:
                raise RuntimeError(f"Malformed SHA256SUMS entry: {line!r}")
            sums[name] = digest
        if set(sums) != set(declared) | {"manifest.json"}:
            raise RuntimeError("Colab output SHA256SUMS coverage mismatch")
        for name, expected_digest in sorted(sums.items()):
            actual_digest, actual_size = _sha256_zip_member(archive, name)
            if actual_digest != expected_digest:
                raise RuntimeError(f"Colab output checksum mismatch: {name}")
            if name in declared:
                entry = declared[name]
                if actual_digest != entry.get("sha256") or actual_size != int(entry.get("bytes", -1)):
                    raise RuntimeError(f"Colab output manifest size/hash mismatch: {name}")

        with zipfile.ZipFile(input_archive_path) as input_archive:
            local_input_manifest = input_archive.read("manifest.json")
        returned_input_manifest = archive.read("input_manifest.json")
        if returned_input_manifest != local_input_manifest:
            raise RuntimeError("Returned input manifest does not match the supplied revision-3 input")
        input_manifest = json.loads(returned_input_manifest)
        if input_manifest.get("bundle_revision") != 3 or any(input_manifest.get("privacy", {}).values()):
            raise RuntimeError("Returned input manifest is the wrong revision or violates the privacy contract")

        config_bytes = archive.read("experiment_10_config.json")
        config = json.loads(config_bytes)
        provenance = json.loads(archive.read("provenance.json"))
        hashes = provenance.get("hashes", {})
        if hashes.get("configuration_sha256") != _sha256_bytes(config_bytes):
            raise RuntimeError("Returned configuration hash mismatch")
        if hashes.get("input_manifest_sha256") != _sha256_bytes(returned_input_manifest):
            raise RuntimeError("Returned input manifest provenance mismatch")
        if hashes.get("compatibility_patch_sha256") != _sha256_bytes(archive.read("xtr_warp_compat.patch")):
            raise RuntimeError("Returned compatibility patch provenance mismatch")
        if config.get("hashes", {}).get("experiment_07_source_sha256") != sha256(
            h.repo / "nickolas" / "experiments" / "experiment_07_residual_failure_analysis.py"
        ):
            raise RuntimeError("Experiment 7 source changed since the Colab query bundle was frozen")
        if hashes.get("catalog_jsonl_sha256") != h.catalog_hash or hashes.get("public_set_jsonl_sha256") != h.public_hash:
            raise RuntimeError("Returned catalog/public-set hashes do not match the local harness")
        if provenance.get("source", {}).get("revision") != EXPECTED_SOURCE_REVISION:
            raise RuntimeError("Returned xtr-warp revision mismatch")
        if provenance.get("model") != {"model_id": EXPECTED_MODEL_ID, "revision": EXPECTED_MODEL_REVISION}:
            raise RuntimeError("Returned XTR model identity/revision mismatch")
        if provenance.get("training_performed") is not False:
            raise RuntimeError("Returned provenance unexpectedly reports model training")
        retrieval = provenance.get("retrieval", {})
        if retrieval.get("device") != "cpu" or retrieval.get("nprobe") != 32 or retrieval.get("top_k") != 1000:
            raise RuntimeError("Returned retrieval configuration mismatch")

        pid_map = json.loads(archive.read("pid_to_asin.json"))
        if (
            pid_map.get("schema_version") != 1
            or pid_map.get("pid_semantics") != "array_index"
            or pid_map.get("document_count") != EXPECTED_DOCUMENT_COUNT
            or pid_map.get("asins") != h.ids
        ):
            raise RuntimeError("Returned PID-to-ASIN mapping does not match the local catalog order")

        query_manifest = json.loads(archive.read("rankings/query_manifest.json"))
        if (
            query_manifest.get("schema_version") != 1
            or query_manifest.get("scope") != "full"
            or query_manifest.get("query_count") != EXPECTED_QUERY_COUNT
            or query_manifest.get("top_k") != EXPECTED_RESULT_COUNT
            or query_manifest.get("nprobe") != 32
            or query_manifest.get("tie_break") != "score_desc_then_pid_asc"
        ):
            raise RuntimeError("Returned query manifest mismatch")
        if query_manifest.get("rankings_jsonl_sha256") != _sha256_zip_member(archive, "rankings/rankings.jsonl")[0]:
            raise RuntimeError("Query manifest JSONL hash mismatch")
        if query_manifest.get("rankings_tsv_sha256") != _sha256_zip_member(archive, "rankings/rankings.tsv")[0]:
            raise RuntimeError("Query manifest TSV hash mismatch")

        rankings_by_query: dict[str, tuple[tuple[int, ...], tuple[float, ...]]] = {}
        rankings_by_id: dict[int, tuple[tuple[int, ...], tuple[float, ...]]] = {}
        queries_by_id: dict[int, str] = {}
        with archive.open("rankings/rankings.jsonl") as raw:
            for expected_query_id, line in enumerate(raw):
                row = json.loads(line)
                query, pids, scores = _validate_ranking_row(row, expected_query_id)
                if query in rankings_by_query:
                    raise RuntimeError(f"Duplicate frozen query text at query {expected_query_id}")
                rankings_by_query[query] = (pids, scores)
                rankings_by_id[expected_query_id] = (pids, scores)
                queries_by_id[expected_query_id] = query
        if len(rankings_by_query) != EXPECTED_QUERY_COUNT:
            raise RuntimeError(f"Expected {EXPECTED_QUERY_COUNT} frozen queries, got {len(rankings_by_query)}")

        query_entries = query_manifest.get("queries")
        if not isinstance(query_entries, list) or len(query_entries) != EXPECTED_QUERY_COUNT:
            raise RuntimeError("Query manifest entry count mismatch")
        for query_id, entry in enumerate(query_entries):
            query = queries_by_id[query_id]
            if (
                entry.get("query_id") != query_id
                or entry.get("query_sha256") != hashlib.sha256(query.encode("utf-8")).hexdigest()
                or entry.get("result_count") != EXPECTED_RESULT_COUNT
            ):
                raise RuntimeError(f"Query manifest entry mismatch at query {query_id}")

        tsv_count = 0
        with archive.open("rankings/rankings.tsv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""), delimiter="\t")
            if reader.fieldnames != ["query_id", "rank", "pid", "score"]:
                raise RuntimeError("Ranking TSV header mismatch")
            for row in reader:
                query_id = int(row["query_id"])
                rank = int(row["rank"])
                if query_id not in rankings_by_id or not 1 <= rank <= EXPECTED_RESULT_COUNT:
                    raise RuntimeError("Ranking TSV query/rank out of range")
                pids, scores = rankings_by_id[query_id]
                if int(row["pid"]) != pids[rank - 1] or float(row["score"]) != scores[rank - 1]:
                    raise RuntimeError(f"Ranking TSV/JSONL disagreement at query {query_id}, rank {rank}")
                tsv_count += 1
        if tsv_count != EXPECTED_QUERY_COUNT * EXPECTED_RESULT_COUNT:
            raise RuntimeError(f"Ranking TSV row count mismatch: {tsv_count}")

        report = {
            "archive": str(archive_path.relative_to(h.repo)),
            "archive_sha256": outer_sha,
            "archive_bytes": archive_path.stat().st_size,
            "member_count": len(infos),
            "uncompressed_bytes": sum(item.file_size for item in infos),
            "checksums_verified": len(sums),
            "input_zip": str(input_archive_path.relative_to(h.repo)),
            "input_zip_sha256": input_zip_sha,
            "input_manifest_sha256": _sha256_bytes(returned_input_manifest),
            "bundle_revision": input_manifest["bundle_revision"],
            "document_count": EXPECTED_DOCUMENT_COUNT,
            "query_count": len(rankings_by_query),
            "ranking_rows": tsv_count,
            "source_revision": EXPECTED_SOURCE_REVISION,
            "model_id": EXPECTED_MODEL_ID,
            "model_revision": EXPECTED_MODEL_REVISION,
            "nprobe": 32,
            "top_k": EXPECTED_RESULT_COUNT,
            "privacy_contract_passed": True,
            "pid_mapping_matches_local_catalog": True,
            "jsonl_tsv_identity_passed": True,
            "all_member_checksums_passed": True,
        }
        return report, rankings_by_query, infos


def _import_colab_archive(archive_path: Path, directory: Path, report: dict) -> Path:
    destination = directory / "imported"
    receipt_path = destination / "archive_receipt.json"
    if destination.exists():
        if not receipt_path.is_file():
            raise RuntimeError(f"Existing import has no receipt: {destination}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("archive_sha256") != report["archive_sha256"]:
            raise RuntimeError("Existing Experiment 10 import belongs to a different archive")
        required = destination / "rankings" / "rankings.jsonl"
        index = destination / "warp_index" / "techjam26-products-xtr-warp.nbits=4"
        if not required.is_file() or not index.is_dir():
            raise RuntimeError("Existing Experiment 10 import is incomplete")
        return destination

    temporary = directory / ".colab_import_in_progress"
    if temporary.exists():
        raise RuntimeError(f"Stale import staging directory requires inspection: {temporary}")
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = _safe_archive_infos(archive)
            for item in infos:
                target = temporary.joinpath(*PurePosixPath(item.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
        write_json(temporary / "archive_receipt.json", report)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _rank_input(
    h: Harness,
    retrieval_input: RetrievalInput,
    xtr_result: tuple[tuple[int, ...], tuple[float, ...]],
) -> FrozenXTRTurnRanking:
    exact_counts, _, _ = h.exact_scores(retrieval_input.phrases)
    exact_order, _ = h.exact_ranked(retrieval_input.phrases)
    highest = int(exact_counts.max()) if exact_counts.size else 0
    highest_tier_count = int(np.count_nonzero(exact_counts == highest)) if highest else 0
    all_exact_count = int(np.count_nonzero(exact_counts == len(retrieval_input.phrases)))
    reasons = fallback_reasons(retrieval_input.active_constraints, all_exact_count, highest_tier_count)
    bm25_order, _ = h.lexical.ranked(retrieval_input.query, "bm25", RRF_DEPTH)
    xtr_order = np.asarray(xtr_result[0], dtype=np.int64)

    if reasons:
        bm25_fused, _ = deterministic_rrf((exact_order[:RRF_DEPTH], bm25_order), h.ids)
        xtr_fused, _ = deterministic_rrf((exact_order[:RRF_DEPTH], xtr_order), h.ids)
    else:
        bm25_fused = exact_order
        xtr_fused = exact_order
    return FrozenXTRTurnRanking(
        retrieval_input=retrieval_input,
        fallback_activated=bool(reasons),
        fallback_reasons=reasons,
        all_phrases_exact_candidate_count=all_exact_count,
        highest_exact_match_count=highest,
        highest_exact_match_tier_count=highest_tier_count,
        xtr_warp_top_10=tuple(int(value) for value in xtr_order[:TOP_K]),
        xtr_warp_top_10_scores=tuple(float(value) for value in xtr_result[1][:TOP_K]),
        method_top_10={
            EXACT_METHOD: tuple(int(value) for value in exact_order[:TOP_K]),
            BM25_METHOD: tuple(int(value) for value in bm25_fused[:TOP_K]),
            WARP_METHOD: tuple(int(value) for value in xtr_fused[:TOP_K]),
        },
    )


def _freeze_rankings(
    h: Harness,
    rankings_by_query: Mapping[str, tuple[tuple[int, ...], tuple[float, ...]]],
    logger: logging.Logger,
) -> list[FrozenXTRTurnRanking]:
    inputs = [RetrievalInput(state.category, tuple(state.active_constraints)) for state in h.traces]
    unique_queries = {item.query for item in inputs}
    if unique_queries != set(rankings_by_query):
        raise RuntimeError(
            f"Frozen query set mismatch: missing={len(unique_queries-set(rankings_by_query))}, "
            f"unexpected={len(set(rankings_by_query)-unique_queries)}"
        )
    cache: dict[RetrievalInput, FrozenXTRTurnRanking] = {}
    frozen: list[FrozenXTRTurnRanking] = []
    for number, item in enumerate(inputs, 1):
        ranking = cache.get(item)
        if ranking is None:
            ranking = _rank_input(h, item, rankings_by_query[item.query])
            cache[item] = ranking
        frozen.append(ranking)
        if number % 250 == 0:
            logger.info("Frozen %d/%d exact/BM25/XTR-WARP turn rankings", number, len(inputs))
    if len(cache) != EXPECTED_QUERY_COUNT or len(frozen) != len(h.traces):
        raise RuntimeError("Experiment 10 ranking freeze is incomplete")
    for ranking in frozen:
        if not ranking.fallback_activated:
            exact = ranking.method_top_10[EXACT_METHOD]
            if ranking.method_top_10[BM25_METHOD] != exact or ranking.method_top_10[WARP_METHOD] != exact:
                raise RuntimeError("A conditional cascade changed exact ordering while fallback was inactive")
    return frozen


def _replay(h: Harness, frozen: Sequence[FrozenXTRTurnRanking], method: str) -> list[dict]:
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


def _canonical_session(row: dict) -> dict:
    return {
        "sample_id": row["sample_id"],
        "scenario_type": row.get("scenario_type", row.get("oracle_scenario_type")),
        "hit": row["hit"],
        "first_hit_turn": row["first_hit_turn"],
        "best_rank": row["best_rank"],
        "reciprocal_rank": row["reciprocal_rank"],
    }


def _assert_control_identity(
    h: Harness,
    frozen: Sequence[FrozenXTRTurnRanking],
    session_sets: Mapping[str, Sequence[dict]],
    exp7_directory: Path,
) -> dict:
    rows_path = exp7_directory / "rows.csv"
    sessions_path = exp7_directory / "sessions.json"
    metrics_path = exp7_directory / "metrics.json"
    if not rows_path.is_file() or not sessions_path.is_file() or not metrics_path.is_file():
        raise RuntimeError("Experiment 7 frozen control artifacts are incomplete")
    with rows_path.open(encoding="utf-8", newline="") as stream:
        expected_rows = list(csv.DictReader(stream))
    if len(expected_rows) != len(frozen):
        raise RuntimeError("Experiment 7 turn-row count mismatch")
    for number, (state, ranking, expected) in enumerate(zip(h.traces, frozen, expected_rows), 1):
        if expected["sample_id"] != state.sample_id or int(expected["turn"]) != state.turn:
            raise RuntimeError(f"Experiment 7 turn alignment mismatch at row {number}")
        if expected["agent_query"] != ranking.retrieval_input.query:
            raise RuntimeError(f"Experiment 7 query identity mismatch at row {number}")
        actual_exact = [h.ids[idx] for idx in ranking.method_top_10[EXACT_METHOD]]
        actual_bm25 = [h.ids[idx] for idx in ranking.method_top_10[BM25_METHOD]]
        if actual_exact != json.loads(expected["exact_only_top_10"]):
            raise RuntimeError(f"Exact control slate mismatch at row {number}")
        if actual_bm25 != json.loads(expected["exact_stateful_bm25_rrf_top_10"]):
            raise RuntimeError(f"Experiment 7 BM25 control slate mismatch at row {number}")
        if ranking.fallback_activated != (expected["fallback_activated"] == "True"):
            raise RuntimeError(f"Experiment 7 fallback decision mismatch at row {number}")

    expected_sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    for actual_method, expected_method in (
        (EXACT_METHOD, "exact_only"),
        (BM25_METHOD, "exact_stateful_bm25_rrf"),
    ):
        actual = [_canonical_session(row) for row in session_sets[actual_method]]
        expected = [_canonical_session(row) for row in expected_sessions[expected_method]]
        if actual != expected:
            raise RuntimeError(f"Experiment 7 {actual_method} session outcomes do not reproduce bit-for-bit")

    exp7_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    all_ids = {row["sample_id"] for row in session_sets[EXACT_METHOD]}
    if _metrics_for_subset(session_sets[EXACT_METHOD], all_ids) != exp7_metrics["method_metrics"]["exact_only"]["full"]:
        raise RuntimeError("Experiment 7 exact control full metrics mismatch")
    if _metrics_for_subset(session_sets[BM25_METHOD], all_ids) != exp7_metrics["method_metrics"]["exact_stateful_bm25_rrf"]["full"]:
        raise RuntimeError("Experiment 7 BM25 control full metrics mismatch")
    return {
        "passed": True,
        "turn_slates_checked_per_control": len(frozen),
        "session_outcomes_checked_per_control": len(h.samples),
        "exact_top_10_bit_for_bit": True,
        "bm25_top_10_bit_for_bit": True,
        "exact_sessions_bit_for_bit": True,
        "bm25_sessions_bit_for_bit": True,
        "exact_full_metrics_bit_for_bit": True,
        "bm25_full_metrics_bit_for_bit": True,
        "sources": {
            "rows": {"path": str(rows_path.relative_to(h.repo)), "sha256": sha256(rows_path)},
            "sessions": {"path": str(sessions_path.relative_to(h.repo)), "sha256": sha256(sessions_path)},
            "metrics": {"path": str(metrics_path.relative_to(h.repo)), "sha256": sha256(metrics_path)},
        },
    }


def _joined_rows(h: Harness, frozen: Sequence[FrozenXTRTurnRanking], calibration: set[str]) -> list[dict]:
    rows: list[dict] = []
    histories: dict[str, list[str]] = defaultdict(list)
    for state, ranking in zip(h.traces, frozen):
        histories[state.sample_id].append(state.message)
        target_idx = h.id_to_idx[state.target_asin]
        row = {
            "sample_id": state.sample_id,
            "split": "calibration" if state.sample_id in calibration else "evaluation",
            "oracle_scenario_type": state.scenario_type,
            "turn": state.turn,
            "evaluator_message": state.message,
            "message_history": list(histories[state.sample_id]),
            "retrieval_category": ranking.retrieval_input.category,
            "active_evidence": list(ranking.retrieval_input.active_constraints),
            "agent_query": ranking.retrieval_input.query,
            "system_ask_attribute": "other",
            "override_applied": state.override_applied,
            "fallback_activated": ranking.fallback_activated,
            "fallback_reasons": list(ranking.fallback_reasons),
            "exact_all_phrases_candidate_count": ranking.all_phrases_exact_candidate_count,
            "exact_highest_match_count": ranking.highest_exact_match_count,
            "exact_highest_tier_candidate_count": ranking.highest_exact_match_tier_count,
            "xtr_warp_component_top_10": [h.ids[idx] for idx in ranking.xtr_warp_top_10],
            "xtr_warp_component_top_10_scores": list(ranking.xtr_warp_top_10_scores),
            "oracle_target_asin": state.target_asin,
        }
        for method in METHODS:
            order = ranking.method_top_10[method]
            row[f"{method}_top_10"] = [h.ids[idx] for idx in order]
            row[f"diagnostic_{method}_target_rank"] = _rank_in_top(order, target_idx)
        rows.append(row)
    return rows


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


def _residual_turn_rows(
    h: Harness,
    frozen: Sequence[FrozenXTRTurnRanking],
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
        row = {
            "sample_id": state.sample_id,
            "split": "calibration" if state.sample_id in calibration else "evaluation",
            "oracle_scenario_type": state.scenario_type,
            "diagnostic_residual_type": residual_types[state.sample_id],
            "turn": state.turn,
            "message": state.message,
            "message_history": list(histories[state.sample_id]),
            "retrieval_category": ranking.retrieval_input.category,
            "active_evidence": list(ranking.retrieval_input.active_constraints),
            "agent_query": ranking.retrieval_input.query,
            "override_applied": state.override_applied,
            "fallback_activated": ranking.fallback_activated,
            "fallback_reasons": list(ranking.fallback_reasons),
            "oracle_target_asin": state.target_asin,
            "xtr_warp_component_top_10": [h.ids[idx] for idx in ranking.xtr_warp_top_10],
            "xtr_warp_component_top_10_scores": list(ranking.xtr_warp_top_10_scores),
        }
        for method in METHODS:
            order = ranking.method_top_10[method]
            row[f"{method}_top_10_candidates"] = [h.ids[idx] for idx in order]
            row[f"diagnostic_{method}_target_rank"] = _rank_in_top(order, target_idx)
        rows.append(row)
    return rows


def _decorate_residual_diagnostics(
    source_rows: Sequence[dict],
    by_method_by_id: Mapping[str, Mapping[str, dict]],
) -> list[dict]:
    output: list[dict] = []
    for source in source_rows:
        row = copy.deepcopy(source)
        sid = row["sample_id"]
        row["experiment_10_route_outcomes"] = {
            method: {
                "hit": by_method_by_id[method][sid]["hit"],
                "first_hit_turn": by_method_by_id[method][sid]["first_hit_turn"],
                "best_rank": by_method_by_id[method][sid]["best_rank"],
            }
            for method in METHODS
        }
        row["experiment_10_xtr_warp_rescue"] = (
            not by_method_by_id[EXACT_METHOD][sid]["hit"] and by_method_by_id[WARP_METHOD][sid]["hit"]
        )
        row["experiment_10_xtr_warp_regression"] = (
            by_method_by_id[EXACT_METHOD][sid]["hit"] and not by_method_by_id[WARP_METHOD][sid]["hit"]
        )
        output.append(row)
    return output


def _weak_improvements(
    h: Harness,
    frozen: Sequence[FrozenXTRTurnRanking],
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
            sid, turn = session["sample_id"], session["first_hit_turn"]
            target = h.id_to_idx[target_by_sid[sid]]
            candidate_rank = _rank_in_top(by_key[(sid, turn)].method_top_10[method], target)
            capped = candidate_rank if candidate_rank is not None else TOP_K + 1
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
                    "capped_rank_improvement": int(session["best_rank"] - capped),
                }
            )

        def summarize(selected: Sequence[dict]) -> dict:
            values = [item["capped_rank_improvement"] for item in selected]
            return {
                **percentile_summary(values),
                "improved": sum(value > 0 for value in values),
                "unchanged": sum(value == 0 for value in values),
                "regressed": sum(value < 0 for value in values),
                "rank_cap_for_misses": TOP_K + 1,
            }

        method_rows = [row for row in rows if row["method"] == method]
        summaries[method] = {}
        for split_name in ("full", "calibration", "evaluation"):
            split_rows = method_rows if split_name == "full" else [row for row in method_rows if row["split"] == split_name]
            summary = summarize(split_rows)
            summary["scenario_metrics"] = {
                scenario: summarize([row for row in split_rows if row["oracle_scenario_type"] == scenario])
                for scenario in sorted({row["oracle_scenario_type"] for row in split_rows})
            }
            summaries[method][split_name] = summary
    return rows, summaries


def _rescue_by_category(
    hard_failures: Sequence[dict],
    by_method_by_id: Mapping[str, Mapping[str, dict]],
    split_sets: Mapping[str, set[str]],
) -> list[dict]:
    rows: list[dict] = []
    for method in METHODS:
        for tag in FAILURE_TAGS:
            tagged = [row for row in hard_failures if tag in row.get("diagnostic_failure_tags", [])]
            for split, ids in split_sets.items():
                selected = [row for row in tagged if row["sample_id"] in ids]
                rescued = [row for row in selected if by_method_by_id[method][row["sample_id"]]["hit"]]
                rows.append(
                    {
                        "method": method,
                        "split": split,
                        "failure_category": tag,
                        "tagged_hard_failures": len(selected),
                        "rescued_hard_failures": len(rescued),
                        "rescue_percentage": round(len(rescued) / len(selected), 6) if selected else 0.0,
                        "non_exclusive": True,
                    }
                )
    return rows


def _write_charts(
    directory: Path,
    comparisons_to_exact: Mapping[str, Mapping[str, dict]],
    rank_distributions: Sequence[dict],
    method_metrics: Mapping[str, Mapping[str, dict]],
) -> None:
    labels = [BM25_METHOD, WARP_METHOD]
    rescues = [comparisons_to_exact[method]["evaluation"]["hard_failure_rescues"] for method in labels]
    regressions = [comparisons_to_exact[method]["evaluation"]["regressions"] for method in labels]
    display = ["Experiment 7\nBM25-RRF", "Experiment 10\nXTR-WARP-RRF"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - 0.2, rescues, 0.4, label="Hard-failure rescues", color="#2a9d8f")
    ax.bar(x + 0.2, regressions, 0.4, label="Regressions", color="#e76f51")
    ax.set_title("Held-out rescue/regression comparison vs exact-only")
    ax.set_ylabel("Sessions")
    ax.set_xticks(x, display)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "rescue_comparison.png", dpi=150)
    plt.close(fig)

    eval_rows = [row for row in rank_distributions if row["split"] == "evaluation"]
    rank_labels = [str(rank) for rank in range(1, TOP_K + 1)] + ["miss"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in METHODS:
        values = [
            next(row["proportion"] for row in eval_rows if row["method"] == method and row["rank"] == label)
            for label in rank_labels
        ]
        ax.plot(rank_labels, values, marker="o", linewidth=1.5, label=method)
    ax.set_title("Held-out first-hit rank distributions")
    ax.set_xlabel("Rank at conversion")
    ax.set_ylabel("Session proportion")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "rank_distributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    calibration_scores = [method_metrics[method]["calibration"]["technical_score"] for method in METHODS]
    evaluation_scores = [method_metrics[method]["evaluation"]["technical_score"] for method in METHODS]
    x = np.arange(len(METHODS))
    ax.bar(x - width / 2, calibration_scores, width, label="Calibration", color="#457b9d")
    ax.bar(x + width / 2, evaluation_scores, width, label="Held-out", color="#f4a261")
    ax.set_xticks(x, ["Exact", "Exp. 7 BM25", "Exp. 10 WARP"])
    ax.set_ylabel("TechnicalScore")
    ax.set_ylim(0.75, max(calibration_scores + evaluation_scores) + 0.02)
    ax.set_title("TechnicalScore comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "technical_score_comparison.png", dpi=150)
    plt.close(fig)


def _validate_local_artifacts(
    metrics: dict,
    rows: Sequence[dict],
    sessions: Mapping[str, Sequence[dict]],
    residual_turns: Sequence[dict],
    hard_failures: Sequence[dict],
    weak_successes: Sequence[dict],
) -> None:
    if metrics.get("experiment") != EXPERIMENT_NUMBER or set(metrics.get("method_metrics", {})) != set(METHODS):
        raise RuntimeError("Experiment 10 metrics schema is incomplete")
    if len(rows) != len(sessions[EXACT_METHOD]) * MAX_TURNS or len(rows) != 2_000:
        raise RuntimeError("Experiment 10 turn rows are incomplete")
    if any(len(sessions[method]) != 200 for method in METHODS):
        raise RuntimeError("Experiment 10 sessions are incomplete")
    if len(residual_turns) != (len(hard_failures) + len(weak_successes)) * MAX_TURNS:
        raise RuntimeError("Experiment 10 residual traces are incomplete")
    if not metrics["baseline_reproduction"]["passed"]:
        raise RuntimeError("Experiment 10 control reproduction did not pass")


def promotion_gates(warp_eval: Mapping[str, float], bm25_eval: Mapping[str, float], warp_vs_exact: Mapping[str, int]) -> dict[str, bool]:
    """Apply the preregistered held-out WARP promotion rule without tuning."""
    return {
        "technical_score_strictly_beats_experiment_07_bm25": (
            warp_eval["technical_score"] > bm25_eval["technical_score"]
        ),
        "rescues_at_least_one_exact_hard_failure": warp_vs_exact["hard_failure_rescues"] >= 1,
        "regressions_do_not_exceed_rescues_vs_exact": (
            warp_vs_exact["regressions"] <= warp_vs_exact["hard_failure_rescues"]
        ),
    }


def run_10(h: Harness, root_logger: logging.Logger) -> dict:
    directory = h.results_dir / EXPERIMENT_DIRECTORY
    logger = experiment_logger(root_logger, directory)
    started = time.perf_counter()
    logger.info("AGENT-REALISTIC XTR/WARP EVALUATION: validating returned Colab archive")

    archive_path = h.repo / "nickolas" / "colab" / "experiment_10_colab_output.zip"
    input_archive_path = h.repo / "nickolas" / "colab" / "experiment_10_colab_input.zip"
    import_report, rankings_by_query, _ = _validate_colab_archive(archive_path, input_archive_path, h)
    imported = _import_colab_archive(archive_path, directory, import_report)
    import_report["imported_directory"] = str(imported.relative_to(h.repo))
    import_report["reusable_index_directory"] = str(
        (imported / "warp_index" / "techjam26-products-xtr-warp.nbits=4").relative_to(h.repo)
    )
    write_json(directory / "colab_import_report.json", import_report)
    logger.info("Validated all returned checksums; imported %d members", import_report["member_count"])

    split_path = h.results_dir / "experiment_06_slate_width_counterfactuals" / "metrics.json"
    public_ids = [sample["sample_id"] for sample in h.samples]
    calibration, evaluation, split = load_frozen_split(split_path, public_ids)
    all_ids = set(public_ids)

    frozen = _freeze_rankings(h, rankings_by_query, logger)
    session_sets = {method: _replay(h, frozen, method) for method in METHODS}
    for method_rows in session_sets.values():
        for row in method_rows:
            row["split"] = "calibration" if row["sample_id"] in calibration else "evaluation"

    exp7_directory = h.results_dir / "experiment_07_residual_failure_analysis"
    baseline_reproduction = _assert_control_identity(h, frozen, session_sets, exp7_directory)
    write_json(directory / "baseline_reproduction.json", baseline_reproduction)
    logger.info("Exact and BM25 controls reproduced all turn slates and session outcomes bit-for-bit")

    exact_sessions = session_sets[EXACT_METHOD]
    bm25_sessions = session_sets[BM25_METHOD]
    warp_sessions = session_sets[WARP_METHOD]
    hard_sessions = [row for row in exact_sessions if not row["hit"]]
    weak_sessions = [row for row in exact_sessions if row["hit"] and 6 <= int(row["best_rank"]) <= 10]
    hard_ids = {row["sample_id"] for row in hard_sessions}
    weak_ids = {row["sample_id"] for row in weak_sessions}
    bm25_weak_ids = {row["sample_id"] for row in bm25_sessions if row["hit"] and 6 <= int(row["best_rank"]) <= 10}

    method_metrics = {
        method: {
            "full": _metrics_for_subset(rows, all_ids),
            "calibration": _metrics_for_subset(rows, calibration),
            "evaluation": _metrics_for_subset(rows, evaluation),
        }
        for method, rows in session_sets.items()
    }
    comparisons_to_exact = {
        method: {
            "full": _comparison(exact_sessions, rows, all_ids, weak_ids),
            "calibration": _comparison(exact_sessions, rows, calibration, weak_ids),
            "evaluation": _comparison(exact_sessions, rows, evaluation, weak_ids),
        }
        for method, rows in session_sets.items()
    }
    comparisons_to_bm25 = {
        method: {
            "full": _comparison(bm25_sessions, rows, all_ids, bm25_weak_ids),
            "calibration": _comparison(bm25_sessions, rows, calibration, bm25_weak_ids),
            "evaluation": _comparison(bm25_sessions, rows, evaluation, bm25_weak_ids),
        }
        for method, rows in session_sets.items()
    }

    warp_eval = method_metrics[WARP_METHOD]["evaluation"]
    bm25_eval = method_metrics[BM25_METHOD]["evaluation"]
    warp_vs_exact = comparisons_to_exact[WARP_METHOD]["evaluation"]
    gates = promotion_gates(warp_eval, bm25_eval, warp_vs_exact)
    recommend_warp = all(gates.values())
    recommendation = WARP_METHOD if recommend_warp else BM25_METHOD

    joined_rows = _joined_rows(h, frozen, calibration)
    serialized_sessions = _serialize_sessions(session_sets, calibration)
    by_method_by_id = {method: {row["sample_id"]: row for row in rows} for method, rows in session_sets.items()}
    residual_types = {**{sid: "hard_failure" for sid in hard_ids}, **{sid: "weak_success" for sid in weak_ids}}
    residual_turns = _residual_turn_rows(h, frozen, residual_types, calibration)

    exp7_hard_path = exp7_directory / "hard_failures.json"
    exp7_weak_path = exp7_directory / "weak_successes.json"
    exp7_hard = json.loads(exp7_hard_path.read_text(encoding="utf-8"))
    exp7_weak = json.loads(exp7_weak_path.read_text(encoding="utf-8"))
    if {row["sample_id"] for row in exp7_hard} != hard_ids or {row["sample_id"] for row in exp7_weak} != weak_ids:
        raise RuntimeError("Experiment 7 residual session identities changed")
    hard_failure_rows = _decorate_residual_diagnostics(exp7_hard, by_method_by_id)
    weak_success_rows = _decorate_residual_diagnostics(exp7_weak, by_method_by_id)
    category_counts = failure_category_counts([*hard_failure_rows, *weak_success_rows])
    split_sets = {"full": all_ids, "calibration": calibration, "evaluation": evaluation}
    rescue_by_category = _rescue_by_category(hard_failure_rows, by_method_by_id, split_sets)
    weak_rows, weak_summary = _weak_improvements(h, frozen, weak_sessions, calibration)
    rank_distributions = _rank_distribution_rows(session_sets, split_sets)
    comparison_rows = [
        {"baseline": EXACT_METHOD, "method": method, "split": split_name, **values}
        for method, split_values in comparisons_to_exact.items()
        for split_name, values in split_values.items()
    ] + [
        {"baseline": BM25_METHOD, "method": method, "split": split_name, **values}
        for method, split_values in comparisons_to_bm25.items()
        for split_name, values in split_values.items()
    ]

    fallback_turns = sum(ranking.fallback_activated for ranking in frozen)
    source_path = Path(__file__).resolve()
    source_hash = sha256(source_path)
    metrics = {
        "experiment": EXPERIMENT_NUMBER,
        "slug": EXPERIMENT_SLUG,
        "label": "AGENT-REALISTIC EVALUATION + ORACLE-AFTER-FREEZE DIAGNOSTIC",
        "retrieval_contract": {
            "ranker_input_fields": ["category", "active_constraints"],
            "excluded_from_colab_input_and_rankers": [
                "target_asin",
                "scenario_type",
                "sample_id",
                "oracle_card",
                "future_turns",
                "user_profile",
            ],
            "rankings_frozen_in_colab_before_oracle_join": True,
            "training_performed": False,
            "query_count": EXPECTED_QUERY_COUNT,
        },
        "cascade_config": {
            "conditional": True,
            "components": {
                BM25_METHOD: ["exact", "stateful_bm25"],
                WARP_METHOD: ["exact", "xtr_warp"],
            },
            "rrf_k": RRF_K,
            "depth": RRF_DEPTH,
            "weights": "equal",
            "tie_break": "ascending_parent_asin",
            "xtr_warp_retrieval": {"nprobe": 32, "top_k": 1000, "device": "cpu"},
            "fallback_activation_identical_to_experiment_07": True,
            "exact_order_preserved_when_inactive": True,
        },
        "split": split,
        "colab_import": import_report,
        "baseline_reproduction": baseline_reproduction,
        "residuals": {
            "hard_failures": len(hard_failure_rows),
            "weak_successes_rank_6_to_10": len(weak_success_rows),
            "residual_trace_turns": len(residual_turns),
            "fallback_turns": fallback_turns,
            "fallback_turn_percentage": round(fallback_turns / len(frozen), 6),
            "failure_category_counts_are_non_exclusive": True,
        },
        "method_metrics": method_metrics,
        "comparisons_to_exact": comparisons_to_exact,
        "comparisons_to_experiment_07_bm25": comparisons_to_bm25,
        "weak_success_same_turn_capped_rank_improvement": weak_summary,
        "failure_category_counts": category_counts,
        "selection": {
            "candidate": WARP_METHOD,
            "baseline": BM25_METHOD,
            "held_out_gates": gates,
            "production_recommendation": recommendation,
            "recommend_xtr_warp": recommend_warp,
            "strict_improvement_required": True,
            "held_out_results_did_not_tune_or_reselect_configuration": True,
            "starter_agent_modified": False,
        },
        "source": {"path": str(source_path.relative_to(h.repo)), "sha256": source_hash},
    }
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 3)

    _validate_local_artifacts(
        metrics,
        joined_rows,
        serialized_sessions,
        residual_turns,
        hard_failure_rows,
        weak_success_rows,
    )
    _write_charts(directory, comparisons_to_exact, rank_distributions, method_metrics)
    write_json(directory / "metrics.json", metrics)
    write_csv(directory / "rows.csv", joined_rows)
    write_json(directory / "sessions.json", serialized_sessions)
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
    write_csv(directory / "weak_success_improvements.csv", weak_rows)
    write_json(directory / "weak_success_improvements.json", weak_rows)
    write_csv(directory / "rank_distributions.csv", rank_distributions)
    write_json(directory / "rank_distributions.json", rank_distributions)
    (directory / "source_snapshot.py").write_bytes(source_path.read_bytes())
    snapshot_hash = sha256(directory / "source_snapshot.py")
    if snapshot_hash != source_hash:
        raise RuntimeError("Experiment 10 source snapshot hash mismatch")
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

    bm25_vs_exact = comparisons_to_exact[BM25_METHOD]["evaluation"]
    warp_vs_bm25 = comparisons_to_bm25[WARP_METHOD]["evaluation"]
    decision = (
        f"recommend `{WARP_METHOD}`"
        if recommend_warp
        else f"retain `{BM25_METHOD}`"
    )
    summary = f"""# Experiment 10 — XTR/WARP retrieval

> **AGENT-REALISTIC RANKING, ORACLE-AFTER-FREEZE DIAGNOSTICS.** The 600 XTR/WARP queries contained only category plus active disclosed dialogue evidence. Targets, scenarios, sample IDs, hidden cards, future turns, and profiles were absent from Colab and joined locally only after Top-1000 rankings were frozen.

The returned archive passed all member checksums and provenance checks. It contains a reusable converted 4-bit WARP index built from 50,000 products with pinned `{EXPECTED_MODEL_ID}` and `nprobe=32` CPU retrieval. Exact-only and Experiment 7 BM25-RRF reproduced all **2,000 Top-10 turn slates** and all **200 session outcomes each** bit-for-bit.

On the frozen 140-session held-out set, exact-only scored **{method_metrics[EXACT_METHOD]['evaluation']['technical_score']:.6f}**, Experiment 7 BM25-RRF scored **{bm25_eval['technical_score']:.6f}**, and XTR/WARP-RRF scored **{warp_eval['technical_score']:.6f}**. Relative to exact-only, BM25 rescued **{bm25_vs_exact['hard_failure_rescues']}** hard failures with **{bm25_vs_exact['regressions']}** regressions; XTR/WARP rescued **{warp_vs_exact['hard_failure_rescues']}** with **{warp_vs_exact['regressions']}** regressions. Relative to BM25, XTR/WARP rescued **{warp_vs_bm25['hard_failure_rescues']}** failures and introduced **{warp_vs_bm25['regressions']}** regressions.

The preregistered decision is to **{decision}**. XTR/WARP is promoted only when it strictly beats BM25 held-out TechnicalScore and passes the existing exact-baseline rescue/regression gates. No starter-agent file was modified.
"""
    (directory / "summary.md").write_text(summary, encoding="utf-8")

    missing = [name for name in REQUIRED_ARTIFACTS if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"Experiment 10 artifact set is incomplete: {missing}")
    logger.info(
        "Completed %s in %.2fs; WARP held-out score=%.6f BM25=%.6f recommendation=%s",
        EXPERIMENT_DIRECTORY,
        metrics["elapsed_seconds"],
        warp_eval["technical_score"],
        bm25_eval["technical_score"],
        recommendation,
    )
    return metrics
