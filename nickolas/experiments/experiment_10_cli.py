"""Local viewer and runner for Experiment 10's frozen XTR/WARP evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import default_results, repo_root
from .experiment_10_xtr_warp_retrieval import BM25_METHOD, EXACT_METHOD, WARP_METHOD


METHODS = (EXACT_METHOD, BM25_METHOD, WARP_METHOD)


def _parse_cell(value: str) -> Any:
    if value in {"True", "False"}:
        return value == "True"
    if value and value[0] in "[{\"":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


class Viewer:
    def __init__(self, results: Path) -> None:
        self.directory = results / "experiment_10_xtr_warp_retrieval"
        required = ("metrics.json", "sessions.json", "rows.csv")
        missing = [name for name in required if not (self.directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Experiment 10 artifacts are missing ({', '.join(missing)}). Run with --run first."
            )
        self.metrics = json.loads((self.directory / "metrics.json").read_text(encoding="utf-8"))
        self.sessions = json.loads((self.directory / "sessions.json").read_text(encoding="utf-8"))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with (self.directory / "rows.csv").open(encoding="utf-8", newline="") as stream:
            for raw in csv.DictReader(stream):
                row = {key: _parse_cell(value) for key, value in raw.items()}
                grouped[str(row["sample_id"])].append(row)
        self.rows = {sample_id: sorted(rows, key=lambda row: row["turn"]) for sample_id, rows in grouped.items()}
        self.by_method = {
            method: {str(row["sample_id"]): row for row in values}
            for method, values in self.sessions.items()
        }

    @staticmethod
    def _outcome(row: dict[str, Any]) -> str:
        return f"T{row['first_hit_turn']}/R{row['best_rank']}" if row["hit"] else "MISS"

    def summary(self) -> None:
        print("EXPERIMENT 10 · XTR/WARP RETRIEVAL")
        print("Held-out method comparison (140 frozen evaluation sessions)")
        print(f"{'method':42} {'score':>10} {'Hit@10':>9} {'MRR':>9} {'MTTC':>8}")
        print("-" * 82)
        for method in METHODS:
            row = self.metrics["method_metrics"][method]["evaluation"]
            print(
                f"{method:42} {row['technical_score']:10.6f} "
                f"{row['hit_rate_at_10']:9.3%} {row['mrr']:9.6f} {row['mttc']:8.3f}"
            )
        selection = self.metrics["selection"]
        print(f"\nRecommendation: {selection['production_recommendation']}")
        for gate, passed in selection["held_out_gates"].items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {gate}")
        reproduction = self.metrics["baseline_reproduction"]
        print(
            "Control reproduction: "
            f"{'PASS' if reproduction['passed'] else 'FAIL'} "
            f"({reproduction['turn_slates_checked_per_control']} turn slates/control, "
            f"{reproduction['session_outcomes_checked_per_control']} sessions/control)"
        )

    def list_cases(self, kind: str = "different", limit: int = 30) -> None:
        results: list[tuple[str, str, str, str]] = []
        for sample_id in sorted(self.rows):
            first = self.rows[sample_id][0]
            if first["split"] != "evaluation":
                continue
            exact = self.by_method[EXACT_METHOD][sample_id]
            bm25 = self.by_method[BM25_METHOD][sample_id]
            warp = self.by_method[WARP_METHOD][sample_id]
            tags: list[str] = []
            if not bm25["hit"] and warp["hit"]:
                tags.append("warp_rescue_vs_bm25")
            if bm25["hit"] and not warp["hit"]:
                tags.append("warp_regression_vs_bm25")
            if not exact["hit"] and warp["hit"]:
                tags.append("warp_rescue_vs_exact")
            if exact["hit"] and not warp["hit"]:
                tags.append("warp_regression_vs_exact")
            if not tags and self._outcome(bm25) != self._outcome(warp):
                tags.append("rank_or_turn_change")
            if kind != "all" and kind not in tags and not (kind == "different" and tags):
                continue
            results.append((sample_id, ",".join(tags) or "same", self._outcome(bm25), self._outcome(warp)))
        print(f"{'sample':12} {'case':34} {'BM25':>9} {'WARP':>9}")
        print("-" * 70)
        for sample_id, tags, bm25, warp in results[:limit]:
            print(f"{sample_id:12} {tags:34} {bm25:>9} {warp:>9}")
        if len(results) > limit:
            print(f"... {len(results) - limit} more")

    def demo_id(self) -> str:
        priorities = (
            lambda b, w: not b["hit"] and w["hit"],
            lambda b, w: b["hit"] and not w["hit"],
            lambda b, w: self._outcome(b) != self._outcome(w),
        )
        evaluation_ids = [sid for sid, rows in self.rows.items() if rows[0]["split"] == "evaluation"]
        for predicate in priorities:
            for sample_id in sorted(evaluation_ids):
                if predicate(self.by_method[BM25_METHOD][sample_id], self.by_method[WARP_METHOD][sample_id]):
                    return sample_id
        return sorted(evaluation_ids)[0]

    def show(self, sample_id: str) -> None:
        if sample_id not in self.rows:
            raise KeyError(f"Unknown sample ID: {sample_id}")
        rows = self.rows[sample_id]
        print(f"SESSION {sample_id} · {rows[0]['split']} · oracle scenario={rows[0]['oracle_scenario_type']}")
        for method in METHODS:
            print(f"  {method}: {self._outcome(self.by_method[method][sample_id])}")
        print()
        for row in rows:
            print(f"Turn {row['turn']}: {row['evaluator_message']}")
            print(f"  agent-visible query: {row['agent_query']}")
            print(f"  fallback={row['fallback_activated']} reasons={row['fallback_reasons']}")
            for method in METHODS:
                rank = row.get(f"diagnostic_{method}_target_rank")
                slate = row.get(f"{method}_top_10") or []
                print(f"  {method}: target_rank={rank or 'miss'} top3={slate[:3]}")
            if row["override_applied"] and any(
                self.by_method[method][sample_id].get("first_hit_turn") == row["turn"] for method in METHODS
            ):
                print("  ^ at least one method converts here")
            print()


def _run(results: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "nickolas.experiments.run_all",
        "--only",
        "10",
        "--skip-baseline",
        "--results",
        str(results),
    ]
    return subprocess.run(command, cwd=repo_root(), check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or rerun Experiment 10")
    parser.add_argument("--results", type=Path, default=default_results())
    parser.add_argument("--run", action="store_true", help="Validate/import the Colab output and rerun Experiment 10")
    parser.add_argument("--summary", action="store_true", help="Print the held-out comparison and promotion gates")
    parser.add_argument("--demo", action="store_true", help="Replay a representative held-out BM25/WARP difference")
    parser.add_argument("--session", help="Replay a particular public sample ID")
    parser.add_argument("--list", choices=("all", "different", "warp_rescue_vs_bm25", "warp_regression_vs_bm25"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = args.results.resolve()
    if args.run and _run(results):
        return 1
    viewer = Viewer(results)
    if args.session:
        viewer.show(args.session)
    elif args.demo:
        viewer.show(viewer.demo_id())
    elif args.list:
        viewer.list_cases(args.list)
    else:
        viewer.summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
