"""CLI for Long-Term Memory P1 portability component evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:  # direct-script compatibility
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_FIXTURE = CURRENT_DIR / "longitudinal_eval" / "projector_fixture_v1.json"
DEFAULT_OUTPUT_ROOT = CURRENT_DIR / "longitudinal_eval" / "results" / "portability"
DEFAULT_ENV_FILE = CURRENT_DIR / ".env"
DRY_FIXTURES = (
    "u1_stable_s8_final",
    "u2_override_s9_final",
    "u3_distractor_s9_final",
    "u3_distractor_s3_final",
)


def _load_env_value(path: Path, name: str) -> str | None:
    direct = os.environ.get(name)
    if direct and direct.strip():
        return direct.strip()
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"").strip("'")
    return None


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a target-isolated, query-conditioned portability judge over the "
            "existing exact-q Phase 3A fixture. No q_star or retrieval fusion is built."
        )
    )
    parser.add_argument("--run-mode", choices=("dry", "full"), required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--deployable-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    try:
        from .longitudinal_eval.portability_eval import (
            ANNOTATION_VERSION,
            DEFAULT_TEMPERATURE,
            OUTPUT_SCHEMA,
            PROMPT_SHA256,
            PROMPT_VERSION,
            OpenAIPortabilityJudge,
            estimated_prompt_tokens,
            evaluate_with_judge,
            fixture_coverage,
            load_portability_fixture,
            write_artifacts,
        )
    except ImportError:  # pragma: no cover - direct script compatibility
        from longitudinal_eval.portability_eval import (
            ANNOTATION_VERSION,
            DEFAULT_TEMPERATURE,
            OUTPUT_SCHEMA,
            PROMPT_SHA256,
            PROMPT_VERSION,
            OpenAIPortabilityJudge,
            estimated_prompt_tokens,
            evaluate_with_judge,
            fixture_coverage,
            load_portability_fixture,
            write_artifacts,
        )

    fixture = load_portability_fixture(args.fixture)
    query_ids = DRY_FIXTURES if args.run_mode == "dry" else None
    coverage = fixture_coverage(fixture, query_ids)
    preflight = estimated_prompt_tokens(
        fixture, query_ids=query_ids, deployable_k=args.deployable_k
    )
    print("Preflight: " + json.dumps({**coverage, **preflight}, sort_keys=True))
    api_key = _load_env_value(args.env_file, "OPENAI_API_KEY")
    if not api_key or api_key.startswith("your_") or "placeholder" in api_key.casefold():
        raise SystemExit("OPENAI_API_KEY is unavailable; no provider fallback is permitted")

    judge = OpenAIPortabilityJudge(
        api_key=api_key,
        model=args.model,
        temperature=DEFAULT_TEMPERATURE,
        max_retries=1,
    )
    evaluation = evaluate_with_judge(
        fixture,
        judge,
        query_ids=query_ids,
        deployable_k=args.deployable_k,
        bootstrap_samples=(0 if args.run_mode == "dry" else args.bootstrap_samples),
    )
    run_id = args.run_id or (
        f"{args.run_mode}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir = args.output_root / run_id
    module_path = CURRENT_DIR / "longitudinal_eval" / "portability_eval.py"
    runner_path = Path(__file__).resolve()
    status = _git("status", "--short")
    manifest = {
        "experiment": "query-conditioned-memory-portability-isolation-p1",
        "run_mode": args.run_mode,
        "scientific_interpretation": args.run_mode == "full",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "provider": "openai",
        "model": args.model,
        "temperature": DEFAULT_TEMPERATURE,
        "provider_fallback": False,
        "model_fallback": False,
        "max_retries_same_provider_model": 1,
        "one_call_per_query_per_evaluation_mode": True,
        "evaluation_modes": ["judge_capability_all", "deployable_cosine_top10"],
        "deployable_candidate_k": args.deployable_k,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "few_shot_examples": [],
        "output_schema": OUTPUT_SCHEMA,
        "runtime_output_cardinality": "results.minItems = results.maxItems = candidate_count",
        "annotation_version": ANNOTATION_VERSION,
        "source_fixture": str(args.fixture.resolve()),
        "source_fixture_sha256": fixture.source_fixture_sha256,
        "source_vector_sha256": fixture.source_vector_sha256,
        "implementation_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "query_ids": [query.fixture_id for query in fixture.queries if query_ids is None or query.fixture_id in set(query_ids)],
        "preflight": preflight,
        "coverage": coverage,
        "actual_usage": evaluation["usage"],
        "failure_records": evaluation["failures"],
        "frozen_invariants": {
            "embeddings_reused": True,
            "q_m0_reused": True,
            "q_star_constructed": False,
            "b3_implemented": False,
            "projector_modified": False,
            "m0_modified": False,
            "graphify_run": False,
        },
    }
    destination = write_artifacts(
        output_dir,
        fixture=fixture,
        evaluation=evaluation,
        manifest=manifest,
        query_ids=query_ids,
        scientific=args.run_mode == "full",
    )
    print(f"Artifacts: {destination}")
    if args.run_mode == "dry":
        print("DRY RUN COMPLETE — NO SCIENTIFIC CLAIM")
    else:
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        print(summary["decision"]["verdict"])
        print(summary["decision"]["reason"])


if __name__ == "__main__":
    main()
