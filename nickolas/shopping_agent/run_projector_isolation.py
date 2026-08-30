"""Cache-only CLI for the real-catalogue QLMP projector-isolation study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    from .agent import Agent
    from .longitudinal_eval.qlmp_component_eval import (
        build_run_manifest,
        evaluate_projector_fixtures,
        load_projector_fixture,
        write_projector_artifacts,
    )
    from .qlmp_integration import (
        CandidateUniverse,
        MemoryMode,
        QLMPIntegrationConfig,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from agent import Agent
    from longitudinal_eval.qlmp_component_eval import (
        build_run_manifest,
        evaluate_projector_fixtures,
        load_projector_fixture,
        write_projector_artifacts,
    )
    from qlmp_integration import CandidateUniverse, MemoryMode, QLMPIntegrationConfig

from nickolas.memory.qlmp import ProjectionConfig


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
DEFAULT_OUTPUT = CURRENT_DIR / "longitudinal_eval" / "results" / "projector_isolation"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run QLMP projector isolation from persisted exact-q/memory vectors. "
            "This command never embeds missing inputs or enables B3 retrieval."
        )
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id")
    parser.add_argument("--local-k", type=int, default=500)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--candidate-universe",
        required=True,
        choices=[value.value for value in CandidateUniverse],
        help="Scientific artifacts must name the neighbourhood universe explicitly.",
    )
    args = parser.parse_args()

    fixture_set = load_projector_fixture(args.fixture)
    universe = CandidateUniverse(args.candidate_universe)
    if fixture_set.candidate_universe is not universe:
        raise SystemExit(
            "fixture and --candidate-universe differ; refusing an ambiguous experiment"
        )
    config = QLMPIntegrationConfig(
        memory_mode=MemoryMode.PROJECTION,
        embedding_space_id=fixture_set.embedding_space_id,
        embedding_dimension=3072,
        projection=ProjectionConfig(rank=args.rank),
        local_k=args.local_k,
        candidate_universe=universe,
    )

    # A missing/rejected cache is a hard stop.  allow_catalog_embedding=False
    # guarantees this scientific command cannot launch a paid catalogue build.
    agent = Agent(allow_catalog_embedding=False)
    try:
        evaluation = evaluate_projector_fixtures(
            agent,
            fixture_set,
            config=config,
            bootstrap_samples=args.bootstrap_samples,
        )
        manifest = build_run_manifest(
            agent,
            fixture_set,
            config,
            project_root=PROJECT_ROOT,
        )
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = write_projector_artifacts(
            args.output_root,
            run_id,
            evaluation,
            manifest,
        )
        print(f"Artifacts: {run_dir}")
        print(evaluation.summary["decision"]["verdict"])
        print(evaluation.summary["decision"]["reason"])
    finally:
        for owner in (agent, getattr(agent, "baseline_agent", None)):
            connection = getattr(owner, "connection", None)
            if connection is not None:
                connection.close()


if __name__ == "__main__":
    main()
