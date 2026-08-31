"""Command-line entry point for the frozen 40-probe M0/M3 evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .agent import Agent
from .longitudinal_eval.evaluator_v2 import (
    run_v2, validate_fixture_v2, verify_bundle, write_bundle,
)
from .memory_store import InMemoryVectorMemoryStore


HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "longitudinal_eval" / "users_40_v2.json"
DEFAULT_CATALOG = HERE.parents[1] / "techjam-conversational-search" / "data" / "catalog.jsonl"
DEFAULT_OUTPUT = HERE / "longitudinal_eval" / "results" / "v2"


def deterministic_comparison(
    first_rows: Sequence[Mapping[str, Any]], first_arrays: Mapping[str, np.ndarray],
    second_rows: Sequence[Mapping[str, Any]], second_arrays: Mapping[str, np.ndarray],
    *, tolerance: float = 1e-6,
) -> dict[str, Any]:
    scalar_fields = ("record_type", "session_id", "m0_relevant_rank", "m3_relevant_rank",
                     "m0_target_rank", "m3_target_rank", "eligible_count", "gate_passed")
    scalar_equal = len(first_rows) == len(second_rows) and all(
        all(left.get(field) == right.get(field) for field in scalar_fields)
        for left, right in zip(first_rows, second_rows)
    )
    keys_equal = set(first_arrays) == set(second_arrays)
    arrays_close = keys_equal and all(
        np.array_equal(first_arrays[key], second_arrays[key])
        if first_arrays[key].dtype.kind in "bOUS" else
        np.allclose(first_arrays[key], second_arrays[key], rtol=tolerance, atol=tolerance)
        for key in first_arrays
    )
    return {"passed": bool(scalar_equal and arrays_close), "scalar_ranks_equal": scalar_equal,
            "array_keys_equal": keys_equal, "arrays_within_tolerance": arrays_close,
            "float_tolerance": tolerance}


def _agent(catalog: Path) -> Agent:
    return Agent(catalog_path=catalog, allow_catalog_embedding=False,
                 memory_store=InMemoryVectorMemoryStore())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--deterministic-rerun", action="store_true")
    parser.add_argument("--complete-tests-passed", action="store_true",
                        help="Set only after the complete repository test suite passes.")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    validation = validate_fixture_v2(fixture)
    if not validation["valid"]:
        raise SystemExit("fixture validation failed: " + "; ".join(validation["errors"]))
    agent = _agent(args.catalog)
    try:
        rows, arrays = run_v2(agent, fixture)
        cache_path = agent.embedding_cache_path
        space = agent.embedding_space_id
    finally:
        agent.connection.close()
    rerun = {"passed": False, "reason": "not requested"}
    if args.deterministic_rerun:
        second = _agent(args.catalog)
        try:
            rows2, arrays2 = run_v2(second, fixture)
        finally:
            second.connection.close()
        rerun = deterministic_comparison(rows, arrays, rows2, arrays2)
    checks = {"fixture_validation": True, "paired_invariants": True,
              "artifact_hashes": True, "offline_reconstruction": False,
              "deterministic_rerun": bool(rerun["passed"]),
              "complete_test_suite": bool(args.complete_tests_passed)}
    sources = [Path(__file__), HERE / "agent.py", HERE / "memory_store.py",
               HERE / "vector_memory.py", HERE / "longitudinal_eval" / "evaluator_v2.py"]
    write_bundle(args.output, rows=rows, arrays=arrays, fixture_path=args.fixture,
                 catalog_path=args.catalog, embedding_cache_path=cache_path,
                 embedding_space_id=space, source_paths=sources, verification=checks)
    offline = verify_bundle(args.output)
    checks["offline_reconstruction"] = bool(offline["valid"])
    write_bundle(args.output, rows=rows, arrays=arrays, fixture_path=args.fixture,
                 catalog_path=args.catalog, embedding_cache_path=cache_path,
                 embedding_space_id=space, source_paths=sources, verification=checks)
    final_verification = verify_bundle(args.output)
    print(json.dumps({"fixture": validation, "rerun": rerun,
                      "offline_replay": final_verification,
                      "bundle": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["deterministic_comparison", "main"]
