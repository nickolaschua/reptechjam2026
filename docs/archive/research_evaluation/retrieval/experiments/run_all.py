from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nickolas.experiments.config import EXPERIMENTS, SEED, default_catalog, default_public_set, default_results, repo_root
    from nickolas.experiments.experiment_07_residual_failure_analysis import run_07
    from nickolas.experiments.experiment_08_intent_routed_dense_browsing import run_08
    from nickolas.experiments.experiment_09_adaptive_hybrid_architecture import run_09
    from nickolas.experiments.experiment_10_xtr_warp_retrieval import run_10
    from nickolas.experiments.experiment_11_clean_fts5_candidate import run_11
    from nickolas.experiments.experiments import RUNNERS as LEGACY_RUNNERS
    from nickolas.experiments.harness import Harness, make_logger, manifest_base, write_json
else:
    from .config import EXPERIMENTS, SEED, default_catalog, default_public_set, default_results, repo_root
    from .experiment_07_residual_failure_analysis import run_07
    from .experiment_08_intent_routed_dense_browsing import run_08
    from .experiment_09_adaptive_hybrid_architecture import run_09
    from .experiment_10_xtr_warp_retrieval import run_10
    from .experiment_11_clean_fts5_candidate import run_11
    from .experiments import RUNNERS as LEGACY_RUNNERS
    from .harness import Harness, make_logger, manifest_base, write_json

RUNNERS = {**LEGACY_RUNNERS, 7: run_07, 8: run_08, 9: run_09, 10: run_10, 11: run_11}


def verify_baseline(repo: Path, catalog: Path, public_set: Path, results: Path, logger: logging.Logger) -> dict:
    kit = repo / "techjam-conversational-search-participant-kit"
    output = results / "baseline_verification.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(kit)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable, str(kit / "evaluator" / "local_evaluator.py"),
        "--catalog", str(catalog), "--dataset", str(public_set), "--output", str(output),
    ]
    logger.info("Verifying unmodified starter baseline")
    subprocess.run(command, cwd=kit, env=env, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    actual = payload["recommended_technical_score"]
    if actual != 0.10671:
        raise RuntimeError(f"Starter score mismatch: expected 0.10671, got {actual}")
    logger.info("Baseline verified at 0.10671")
    return {"expected": 0.10671, "actual": actual, "passed": True, "command": subprocess.list2cmdline(command)}


def render_indexes(results: Path, status: dict, manifest: dict) -> None:
    links = []
    chart_names = {
        1: "candidate_uniqueness.png", 2: "target_rank_curves.png", 3: "field_coverage.png",
        4: "classification_distribution.png", 5: "shrinkage_curves.png", 6: "slate_comparison.png",
        7: "rescue_comparison.png",
        8: "route_comparison.png", 9: "ablation_comparison.png", 10: "technical_score_comparison.png",
        11: "agent_comparison.png",
    }
    for number, slug in EXPERIMENTS:
        directory = f"experiment_{number:02d}_{slug}"
        state = status.get(str(number), {}).get("status", "not_run")
        extras = f", [chart]({directory}/{chart_names[number]})"
        if number == 2:
            extras += f", [early-termination sessions]({directory}/early_termination_sessions.json)"
        if number == 7:
            extras += f", [residual turns]({directory}/residual_turns.json), [hard failures]({directory}/hard_failures.json), [weak successes]({directory}/weak_successes.json)"
        if number == 8:
            extras += f", [route diagnostics]({directory}/route_diagnostics.json), [paraphrase stress test]({directory}/paraphrase_results.json)"
        if number == 9:
            extras += f", [state diagnostics]({directory}/state_diagnostics.json), [ablations]({directory}/ablation_comparisons.json)"
        if number == 10:
            extras += f", [Colab import report]({directory}/colab_import_report.json), [control reproduction]({directory}/baseline_reproduction.json), [rescue comparison]({directory}/rescue_comparisons.json)"
        if number == 11:
            extras += f", [comparisons]({directory}/comparisons.json), [candidate snapshot]({directory}/candidate_agent_snapshot.py), [latency]({directory}/latency.json)"
        links.append(f"- Experiment {number}: [{slug}]({directory}/summary.md) — `{state}` ([metrics]({directory}/metrics.json), [raw rows]({directory}/rows.csv), [log]({directory}/run.log){extras})")
    readme = "# Eleven-experiment evaluation results\n\n" + "\n".join(links) + "\n\n"
    readme += "- [Combined summary](combined_summary.md)\n- [Methodology](../experiments/methodology.md)\n- [Manifest](manifest.json)\n- [Baseline verification](baseline_verification.json)\n- [Runner log](run_all.log)\n- [Cache documentation](cache/README.md)\n"
    readme += "\n## Interactive Experiment 7 CLI\n\nOpen existing traces with `python -m nickolas.experiments.experiment_07_cli`. "
    readme += "Add `--run` to rerun Experiment 7 with live logs before opening the viewer.\n"
    readme += "Run `python -m nickolas.experiments.experiment_07_cli --demo` for a presentation-ready single-case rescue replay.\n"
    readme += "\n## Experiment 10 CLI\n\nInspect the WARP comparison with `python -m nickolas.experiments.experiment_10_cli --summary`; use `--demo` for a held-out case replay or `--run` to validate/import and rerun Experiment 10.\n"
    (results / "README.md").write_text(readme, encoding="utf-8")

    completed = [(n, slug, status[str(n)].get("metrics", {})) for n, slug in EXPERIMENTS if status.get(str(n), {}).get("status") == "completed"]
    best = None
    if any(n == 2 for n, _, _ in completed):
        exp2 = next(m for n, _, m in completed if n == 2)
        best = exp2.get("best_method")
    combined = "# Combined experiment summary\n\n"
    combined += "> Oracle diagnostics (1, 3, 4, 5) inspect reconstructed hidden labels only for dataset analysis. Agent-realistic claims (2, 6, 7, 8, 9, 10, 11) use only information disclosed at each turn. Experiment 11 is retrospective because its public set was inspected before evaluation.\n\n"
    combined += "| Experiment | Mode | Central result |\n|---|---|---|\n"
    by_number = {n: m for n, _, m in completed}
    if 1 in by_number:
        agg = by_number[1]["aggregates"]
        exact = agg["exact_phrase_candidates"]["by_position"].get("constraint_position=4", {})
        combined += f"| 1. Constraint uniqueness | Oracle | Four constraints leave median {exact.get('median', 'n/a')} exact-phrase candidates; uniqueness {exact.get('uniqueness_rate', 0):.1%}. |\n"
    if 2 in by_number:
        m = by_number[2]
        selected = m["early_termination"][m["best_method"]]
        combined += f"| 2. Target-rank curves | Agent-realistic | {m['best_method']} leads: score {selected['technical_score']:.6f}, Hit@10 {selected['hit_rate_at_10']:.1%}, MRR {selected['mrr']:.3f}. |\n"
    if 3 in by_number:
        fields = by_number[3]["field_metrics"]
        top = max(fields, key=lambda key: fields[key]["exact_match_frequency"])
        combined += f"| 3. Field signal | Oracle | {top} has the highest exact coverage ({fields[top]['exact_match_frequency']:.1%}); {by_number[3]['multi_field_constraints']} constraints overlap fields. |\n"
    if 4 in by_number:
        m = by_number[4]
        combined += f"| 4. Classification | Oracle | `other` covers all {m['other_revealed']} constraints; {m['multi_rule_precedence_resolutions']} strings trigger classifier precedence. |\n"
    if 5 in by_number:
        final = by_number[5]["aggregates"]["exact_phrase_candidates"]["by_position"].get("constraint_count=4", {})
        combined += f"| 5. Candidate shrinkage | Oracle | Exact candidates shrink to median {final.get('median', 'n/a')} at four constraints; soft sets remain deliberately broader. |\n"
    if 6 in by_number:
        m = by_number[6]
        top10 = m["fixed_widths"]["10"]["evaluation"]["technical_score"]
        adaptive = m["adaptive"]["evaluation"]["technical_score"]
        combined += f"| 6. Slate widths | Agent-realistic | Held-out Top-10 scores {top10:.6f} vs {adaptive:.6f} adaptive; full width preserves more recall. |\n"
    if 7 in by_number:
        m = by_number[7]
        selected = m["selection"]["selected_on_calibration"]
        recommendation = m["selection"]["production_recommendation"]
        selected_eval = m["method_metrics"][selected]["evaluation"]["technical_score"]
        exact_eval = m["method_metrics"]["exact_only"]["evaluation"]["technical_score"]
        rescues = m["comparisons_to_exact"][selected]["evaluation"]["hard_failure_rescues"]
        regressions = m["comparisons_to_exact"][selected]["evaluation"]["regressions"]
        combined += f"| 7. Residual failures | Agent-realistic + oracle-after-freeze | Calibration selected {selected}; held-out score {selected_eval:.6f} vs exact {exact_eval:.6f}, with {rescues} rescues/{regressions} regressions. Recommend {recommendation}. |\n"
    if 8 in by_number:
        m = by_number[8]
        control = m["method_metrics"]["exp7_same_parser_control"]["evaluation"]
        treatment = m["method_metrics"]["intent_routed_dense_browsing"]["evaluation"]
        combined += f"| 8. Intent-routed dense browsing | Agent-realistic | Held-out routed score {treatment['technical_score']:.6f} vs Experiment 7 {control['technical_score']:.6f}; routing accuracy {m['routing']['accuracy']:.1%}. |\n"
    if 9 in by_number:
        m = by_number[9]
        selection = m["selection"]
        combined += f"| 9. Adaptive hybrid | Agent-realistic | Calibration selected {selection['selected_on_calibration']}; held-out score {selection['held_out_metrics']['technical_score']:.6f}; promote={selection['recommend_promotion']}. |\n"
    if 10 in by_number:
        m = by_number[10]
        exact = m["method_metrics"]["exact_only"]["evaluation"]["technical_score"]
        bm25 = m["method_metrics"]["experiment_07_exact_stateful_bm25_rrf"]["evaluation"]["technical_score"]
        warp = m["method_metrics"]["exact_stateful_xtr_warp_rrf"]["evaluation"]["technical_score"]
        combined += f"| 10. XTR/WARP retrieval | Agent-realistic + oracle-after-freeze | Held-out scores: exact {exact:.6f}, Experiment 7 BM25 {bm25:.6f}, WARP {warp:.6f}; recommend {m['selection']['production_recommendation']}. |\n"
    if 11 in by_number:
        m = by_number[11]
        selected = m["selection"]["selected_on_calibration"]
        full = m["method_metrics"][selected]["full"]["technical_score"]
        evaluation = m["method_metrics"][selected]["evaluation"]["technical_score"]
        combined += f"| 11. Clean FTS5 candidate | Retrospective agent evaluation | Selected {selected}; full score {full:.6f}, evaluation-partition score {evaluation:.6f}. Private validation required. |\n"
    combined += "\n"
    for number, slug, metrics in completed:
        combined += f"- [{number}. {slug.replace('_', ' ').title()}](experiment_{number:02d}_{slug}/summary.md) ({metrics.get('elapsed_seconds', 'n/a')} seconds)\n"
    if 11 in by_number:
        selection = by_number[11]["selection"]
        selected = selection["selected_on_calibration"]
        if selection["all_diagnostic_gates_passed"]:
            combined += f"\n## Recommendation\n\nPrioritize **{selected}** for private or newly generated validation. It passed every retrospective diagnostic gate, but Experiment 11 does not authorize production promotion because the public data were already inspected. The starter agent remains unchanged.\n"
        else:
            combined += "\n## Recommendation\n\nRetain the current submission agent. The clean FTS5 candidate did not pass all retrospective diagnostic gates, and no starter-agent file was modified.\n"
    elif 10 in by_number:
        selection = by_number[10]["selection"]
        if selection["recommend_xtr_warp"]:
            combined += f"\n## Recommendation\n\nPromote **{selection['production_recommendation']}** in a separate production change. It strictly beat Experiment 7 BM25 on held-out TechnicalScore and passed the exact-baseline rescue/regression gates. Experiment 10 did not modify the submission agent automatically.\n"
        else:
            combined += "\n## Recommendation\n\nRetain the **Experiment 7 exact-first + conditional BM25 submission agent**. XTR/WARP did not pass every preregistered held-out promotion gate. Experiment 10 did not modify the submission agent.\n"
    elif 9 in by_number:
        selection = by_number[9]["selection"]
        if selection["recommend_promotion"]:
            combined += f"\n## Recommendation\n\nPromote **{selection['selected_on_calibration']}** in a separate production change; it passed all held-out TechnicalScore, MRR, rescue, and regression gates. Experiments 8 and 9 did not modify the submission agent automatically.\n"
        else:
            combined += "\n## Recommendation\n\nRetain the **Experiment 7 exact-first + conditional BM25 submission agent**. The Experiment 8 routed treatment regressed held-out performance, and the Experiment 9 calibration winner was the unchanged structured-state identity, which rescued no held-out hard failure. Neither experiment modified the submission agent.\n"
    elif 7 in by_number:
        selected = by_number[7]["selection"]["selected_on_calibration"]
        recommendation = by_number[7]["selection"]["production_recommendation"]
        if recommendation == "exact_only":
            combined += f"\n## Recommendation\n\nRetain **exact-only** in production. Calibration selected **{selected}**, but it did not pass all preregistered held-out safety gates. Experiment 7 did not modify the starter agent.\n"
        else:
            combined += f"\n## Recommendation\n\nAdopt the calibrated conditional cascade **{recommendation}** in a separate production change. It passed the held-out score, rescue, and regression gates. Experiment 7 itself did not modify the starter agent.\n"
    elif best:
        combined += f"\n## Recommendation\n\nBuild an **exact-phrase-first, stateful retrieval cascade**. Use normalized exact phrase matches as the high-precision first stage, fall back to stateful BM25 when exact evidence is absent or ambiguous, and use field-aware BM25 + MiniLM RRF as a semantic expansion layer rather than the sole ranker. Preserve explicit accumulated state and remove overridden preferences. Keep a Top-10 slate by default: the calibrated adaptive policy lost held-out score. This architecture follows the agent-realistic winner (**{best}**) while respecting the oracle finding that some generated/noisy constraints should not become mandatory filters.\n"
    (results / "combined_summary.md").write_text(combined, encoding="utf-8")
    manifest["experiments"] = status
    write_json(results / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all eleven TechJam diagnostic experiments sequentially")
    parser.add_argument("--catalog", type=Path, default=default_catalog())
    parser.add_argument("--public-set", type=Path, default=default_public_set())
    parser.add_argument("--results", type=Path, default=default_results())
    parser.add_argument("--only", type=int, nargs="*", choices=range(1, 12))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", help="Attempt later experiments after a failure")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = repo_root()
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    logger = make_logger(results)
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    status: dict[str, dict] = {}
    if args.only and (results / "manifest.json").exists():
        try:
            status.update(json.loads((results / "manifest.json").read_text(encoding="utf-8")).get("experiments", {}))
        except (OSError, ValueError):
            pass
    manifest: dict = {"seed": SEED, "commands": [command]}
    try:
        if not args.skip_baseline:
            manifest["baseline_verification"] = verify_baseline(repo, args.catalog.resolve(), args.public_set.resolve(), results, logger)
        harness = Harness(repo, args.catalog.resolve(), args.public_set.resolve(), results, logger)
        manifest = {**manifest_base(harness, command), **manifest}
    except Exception:
        logger.exception("Harness initialization failed")
        write_json(results / "manifest.json", {**manifest, "fatal_error": traceback.format_exc()})
        return 1

    selected = set(args.only or range(1, 12))
    exit_code = 0
    for number, slug in EXPERIMENTS:
        if number not in selected:
            status.setdefault(str(number), {"status": "skipped"})
            continue
        started = time.perf_counter()
        try:
            metrics = RUNNERS[number](harness, logger)
            stale_failure = results / f"experiment_{number:02d}_{slug}" / "failure.txt"
            if stale_failure.exists():
                stale_failure.unlink()
            status[str(number)] = {"status": "completed", "elapsed_seconds": round(time.perf_counter() - started, 3), "metrics": metrics}
        except Exception as exc:
            exit_code = 1
            directory = results / f"experiment_{number:02d}_{slug}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            status[str(number)] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(time.perf_counter() - started, 3)}
            logger.exception("Experiment %d failed; prior results remain intact", number)
            render_indexes(results, status, manifest)
            if not args.continue_on_error:
                break
        render_indexes(results, status, manifest)
    render_indexes(results, status, manifest)
    logger.info("Run complete with exit code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
