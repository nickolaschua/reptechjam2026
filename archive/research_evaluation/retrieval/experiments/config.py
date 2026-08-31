from __future__ import annotations

from pathlib import Path

SEED = 20260826
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_MAX_SEQ_LENGTH = 128
MAX_TURNS = 10
TOP_K = 10
RRF_K = 60
RRF_DEPTH = 1000
BM25_RELATIVE_CUTOFF = 0.50
DENSE_ABSOLUTE_CUTOFF = 0.25
DENSE_RELATIVE_CUTOFF = 0.80

EXPERIMENTS = (
    (1, "constraint_uniqueness"),
    (2, "target_rank_curves"),
    (3, "field_signal"),
    (4, "constraint_classification"),
    (5, "candidate_set_shrinkage"),
    (6, "slate_width_counterfactuals"),
    (7, "residual_failure_analysis"),
    (8, "intent_routed_dense_browsing"),
    (9, "adaptive_hybrid_architecture"),
    (10, "xtr_warp_retrieval"),
    (11, "clean_fts5_candidate"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_catalog() -> Path:
    return repo_root() / "techjam-conversational-search" / "data" / "catalog.jsonl"


def default_public_set() -> Path:
    return repo_root() / "techjam-conversational-search" / "data" / "public_set.jsonl"


def default_results() -> Path:
    return repo_root() / "nickolas" / "results"
