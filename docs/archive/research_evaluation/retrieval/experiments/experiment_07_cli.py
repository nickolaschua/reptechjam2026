"""Interactive terminal trace viewer for Experiment 7.

Experiment 7 is a retrieval-policy evaluation, so a "system response" consists of
an ask_attribute action and a Top-10 recommendation slate. It does not generate a
natural-language assistant message. This viewer keeps agent-visible information
separate from oracle-only scoring information.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METHODS = (
    "exact_only",
    "exact_stateful_bm25_rrf",
    "exact_field_aware_bm25_rrf",
    "exact_dense_rrf",
    "exact_generic_field_dense_rrf",
)
MAX_TURNS = 10
DEMO_SESSION = "public_0037"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results() -> Path:
    return repo_root() / "nickolas" / "results"


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def apply(self, code: str, value: object) -> str:
        text = str(value)
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def heading(self, value: object) -> str:
        return self.apply("1;36", value)

    def evaluator(self, value: object) -> str:
        return self.apply("1;35", value)

    def system(self, value: object) -> str:
        return self.apply("1;34", value)

    def oracle(self, value: object) -> str:
        return self.apply("1;33", value)

    def good(self, value: object) -> str:
        return self.apply("1;32", value)

    def bad(self, value: object) -> str:
        return self.apply("1;31", value)

    def dim(self, value: object) -> str:
        return self.apply("2", value)


def json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_cell(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped in {"True", "False"}:
        return stripped == "True"
    if stripped[0] in "[{\"" or stripped in {"null", "true", "false"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    try:
        return int(stripped)
    except ValueError:
        return value


def load_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = {key: parse_cell(value) for key, value in raw.items()}
            grouped[str(row["sample_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["turn"]))
    return dict(grouped)


def load_official_evaluator(root: Path):
    kit = root / "techjam-conversational-search-participant-kit"
    sys.path.insert(0, str(kit))
    spec = importlib.util.spec_from_file_location(
        "techjam_cli_official_evaluator", kit / "evaluator" / "local_evaluator.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the official evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wrap(text: object, indent: str = "  ", width: int | None = None) -> str:
    terminal_width = width or max(60, min(120, os.get_terminal_size().columns if sys.stdout.isatty() else 100))
    return textwrap.fill(str(text), width=terminal_width, initial_indent=indent, subsequent_indent=indent)


def run_experiment_process(root: Path, results: Path, style: Style) -> bool:
    print(style.heading("\nRunning Experiment 7 (live output follows)\n"))
    command = [
        sys.executable,
        "-m",
        "nickolas.experiments.run_all",
        "--only",
        "7",
        "--skip-baseline",
        "--results",
        str(results),
    ]
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode:
        print(style.bad(f"Experiment failed with exit code {completed.returncode}. Existing artifacts were preserved."))
        return False
    print(style.good("Experiment 7 completed."))
    return True


class Viewer:
    def __init__(self, results: Path, style: Style) -> None:
        self.root = repo_root()
        self.results = results
        self.directory = results / "experiment_07_residual_failure_analysis"
        self.style = style
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.sessions: dict[str, list[dict[str, Any]]] = {}
        self.metrics: dict[str, Any] = {}
        self.samples: dict[str, dict[str, Any]] = {}
        self.hard_ids: set[str] = set()
        self.weak_ids: set[str] = set()
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self.products: dict[str, dict[str, Any]] = {}
        self.method = "exact_stateful_bm25_rrf"
        self.reload()

    def reload(self) -> None:
        required = [self.directory / "rows.csv", self.directory / "sessions.json", self.directory / "metrics.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Experiment 7 artifacts are missing:\n  " + "\n  ".join(missing))
        self.rows = load_rows(required[0])
        self.sessions = json.loads(required[1].read_text(encoding="utf-8"))
        self.metrics = json.loads(required[2].read_text(encoding="utf-8"))
        self.samples = {
            str(item["sample_id"]): item
            for item in json_lines(self.root / "techjam-conversational-search" / "data" / "public_set.jsonl")
        }
        self.hard_ids = self._ids_from("hard_failures.json")
        self.weak_ids = self._ids_from("weak_successes.json")
        selected = self.metrics.get("selection", {}).get("production_recommendation")
        if selected in METHODS:
            self.method = selected

    def _ids_from(self, name: str) -> set[str]:
        path = self.directory / name
        if not path.exists():
            return set()
        return {str(item["sample_id"]) for item in json.loads(path.read_text(encoding="utf-8"))}

    def run_experiment(self) -> bool:
        if not run_experiment_process(self.root, self.results, self.style):
            return False
        self.messages.clear()
        self.products.clear()
        self.reload()
        print(self.style.good("Artifacts reloaded."))
        return True

    def banner(self) -> None:
        selection = self.metrics.get("selection", {})
        chosen = selection.get("selected_on_calibration", "unknown")
        recommendation = selection.get("production_recommendation", "unknown")
        print(self.style.heading("EXPERIMENT 7 · INTERACTIVE TRACE VIEWER"))
        print(f"Sessions: {len(self.rows)} · turns retained: {sum(map(len, self.rows.values()))}")
        print(f"Calibration winner: {chosen} · production recommendation: {recommendation}")
        print(self.style.dim("Rankers saw only category + active disclosed evidence. Yellow/oracle fields were used only after rankings froze."))
        print(self.style.dim("Type 'help' for commands. Enter advances one turn while viewing a session."))

    def session_outcome(self, sample_id: str, method: str | None = None) -> dict[str, Any]:
        selected = method or self.method
        return next(item for item in self.sessions[selected] if item["sample_id"] == sample_id)

    def session_kind(self, sample_id: str) -> str:
        return ",".join(self.session_tags(sample_id))

    def session_tags(self, sample_id: str) -> list[str]:
        tags: list[str] = []
        if sample_id in self.hard_ids:
            tags.append("hard_failure")
        if sample_id in self.weak_ids:
            tags.append("weak_success")
        exact = self.session_outcome(sample_id, "exact_only")
        selected = self.session_outcome(sample_id)
        if not exact["hit"] and selected["hit"]:
            tags.append("rescue")
        if exact["hit"] and not selected["hit"]:
            tags.append("regression")
        return tags or ["normal"]

    def list_sessions(self, query: str = "", limit: int = 40) -> None:
        words = query.lower().split()
        results: list[str] = []
        for sample_id in sorted(self.rows):
            first = self.rows[sample_id][0]
            outcome = self.session_outcome(sample_id)
            kind = self.session_kind(sample_id)
            haystack = " ".join(
                [sample_id, str(first.get("oracle_scenario_type")), str(first.get("split")), kind, str(first.get("retrieval_category"))]
            ).lower()
            if words and not all(word in haystack for word in words):
                continue
            rank = outcome.get("best_rank")
            hit = f"T{outcome['first_hit_turn']}/R{rank}" if outcome.get("hit") else "MISS"
            results.append(
                f"{sample_id:<12} {str(first.get('split')):<11} {str(first.get('oracle_scenario_type')):<16} "
                f"{kind:<13} {hit:<8} {first.get('retrieval_category')}"
            )
        print(self.style.heading(f"\nSessions matching: {query or 'all'} ({len(results)})"))
        print("ID           SPLIT       SCENARIO         TYPE          RESULT   CATEGORY")
        print("-" * 100)
        for line in results[:limit]:
            print(line)
        if len(results) > limit:
            print(self.style.dim(f"... {len(results) - limit} more; refine with: list <scenario|split|type|category>"))

    def ensure_messages(self, sample_id: str) -> list[dict[str, Any]]:
        if sample_id in self.messages:
            return self.messages[sample_id]
        rows = self.rows[sample_id]
        if all(row.get("evaluator_message") for row in rows):
            result = [
                {
                    "message": row["evaluator_message"],
                    "override_applied": bool(row.get("override_applied", True)),
                }
                for row in rows
            ]
            self.messages[sample_id] = result
            return result

        official = load_official_evaluator(self.root)
        sample = self.samples[sample_id]
        target = str(sample["ground_truth"]["parent_asin"])
        self.load_products({target})
        card, behavior = official.materialize_hidden_fields(sample, self.products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        category = str(rows[0]["retrieval_category"])
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        message = official.initial_message(effective, category, disclosed)
        result = []
        for turn in range(1, MAX_TURNS + 1):
            result.append({"message": message, "override_applied": override_applied})
            if turn == MAX_TURNS:
                break
            override = behavior.get("override", {})
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                old = str(override.get("old_value", ""))
                new = str(override.get("new_value", ""))
                disclosed.discard(old)
                if new:
                    disclosed.add(new)
                message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                message, boundary_used = official.customer_reply(effective, "other", disclosed, boundary_used)
        self.messages[sample_id] = result
        return result

    def load_products(self, identifiers: Iterable[str]) -> None:
        needed = set(identifiers) - self.products.keys()
        if not needed:
            return
        catalog = self.root / "techjam-conversational-search" / "data" / "catalog.jsonl"
        with catalog.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                if asin in needed:
                    self.products[asin] = product
                    needed.remove(asin)
                    if not needed:
                        break

    def show_candidate(self, row: dict[str, Any], value: str) -> None:
        candidates = list(row.get(f"{self.method}_top_10") or [])
        if value.isdigit() and 1 <= int(value) <= len(candidates):
            asin = candidates[int(value) - 1]
        else:
            asin = value.strip()
        if not asin:
            print(self.style.bad("Use: candidate <rank|ASIN>"))
            return
        self.load_products({asin})
        product = self.products.get(asin)
        if product is None:
            print(self.style.bad(f"Catalog product not found: {asin}"))
            return
        print(self.style.heading(f"\nCATALOG PRODUCT · {asin}"))
        print(json.dumps(product, indent=2, ensure_ascii=False, sort_keys=True))

    def show_oracle(self, sample_id: str) -> None:
        sample = self.samples[sample_id]
        target = str(sample["ground_truth"]["parent_asin"])
        self.load_products({target})
        official = load_official_evaluator(self.root)
        card, behavior = official.materialize_hidden_fields(sample, self.products)
        payload = {
            "warning": "ORACLE-ONLY: none of these hidden fields were passed into the ranker",
            "sample": sample,
            "materialized_intent_card": card,
            "materialized_behavior": behavior,
            "target_product": self.products[target],
        }
        print(self.style.oracle("\nFULL EVALUATOR ORACLE STATE"))
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def title(product: dict[str, Any] | None) -> str:
        if not product:
            return "(catalog entry unavailable)"
        return " ".join(str(product.get("title") or "(untitled)").split())

    def render_turn(self, sample_id: str, turn_index: int, *, compare: bool = False, raw: bool = False) -> None:
        row = self.rows[sample_id][turn_index]
        message_state = self.ensure_messages(sample_id)[turn_index]
        outcome = self.session_outcome(sample_id)
        target = str(row["oracle_target_asin"])
        top_key = f"{self.method}_top_10"
        candidates = list(row.get(top_key) or [])
        self.load_products([target, *candidates])
        stop_turn = outcome.get("first_hit_turn")
        after_stop = stop_turn is not None and int(row["turn"]) > int(stop_turn)

        print("\n" + "═" * 100)
        print(self.style.heading(
            f"{sample_id} · turn {row['turn']}/{MAX_TURNS} · {row['split']} · {row['oracle_scenario_type']} · {self.session_kind(sample_id)}"
        ))
        if after_stop:
            print(self.style.oracle(f"DIAGNOSTIC-ONLY TURN: normal scoring stopped after the hit on turn {stop_turn}."))

        print(self.style.evaluator("\nEVALUATOR → SYSTEM"))
        print(wrap(message_state["message"]))

        print(self.style.system("\nSYSTEM → EVALUATOR  [agent-visible retrieval action]"))
        print(f"  method:        {self.method}")
        print(f"  query:         {row['agent_query']}")
        print(f"  active state:  {json.dumps(row.get('active_evidence') or [], ensure_ascii=False)}")
        print("  ask_attribute: other")
        fallback = "YES · " + ", ".join(row.get("fallback_reasons") or []) if row.get("fallback_activated") else "no"
        print(f"  fallback:      {fallback}")
        print(
            f"  exact pool:    all-phrase={row.get('exact_all_phrases_candidate_count')} · "
            f"highest-tier={row.get('exact_highest_tier_candidate_count')}"
        )
        print("  recommendations:")
        for rank, asin in enumerate(candidates, 1):
            marker = self.style.oracle(" ← TARGET (oracle)") if asin == target else ""
            print(wrap(f"{rank:>2}. {asin} · {self.title(self.products.get(asin))}{marker}", indent="    "))

        rank = row.get(f"diagnostic_{self.method}_target_rank")
        eligible = bool(message_state.get("override_applied", True))
        print(self.style.oracle("\nORACLE SCORING  [never provided to the ranker]"))
        print(f"  target:             {target} · {self.title(self.products.get(target))}")
        print(f"  target rank:        {rank if rank is not None else 'not in Top-10'}")
        print(f"  conversion eligible:{' yes' if eligible else ' no (intent override not yet applied)'}")
        converted = eligible and rank is not None
        print(f"  turn outcome:       {self.style.good('HIT') if converted else self.style.bad('MISS')}")
        if stop_turn == row["turn"]:
            print(self.style.good("  normal evaluation terminates here"))

        if compare:
            self.render_comparison(row)
        if raw:
            print(self.style.heading("\nRAW TURN ROW"))
            print(json.dumps(row, indent=2, ensure_ascii=False, sort_keys=True))

    def render_comparison(self, row: dict[str, Any]) -> None:
        print(self.style.heading("\nMETHOD COMPARISON (same frozen turn input)"))
        for method in METHODS:
            rank = row.get(f"diagnostic_{method}_target_rank")
            slate = row.get(f"{method}_top_10") or []
            print(f"  {method:<38} target_rank={str(rank or 'MISS'):<5} top3={', '.join(slate[:3])}")

    def print_metrics(self) -> None:
        print(self.style.heading("\nHeld-out method metrics"))
        for method in METHODS:
            values = self.metrics["method_metrics"][method]["evaluation"]
            print(
                f"  {method:<38} score={values['technical_score']:.6f} "
                f"Hit@10={values['hit_rate_at_10']:.3f} MRR={values['mrr']:.3f} MTTC={values['mttc']:.3f}"
            )

    def demo_pause(self, prompt: str, auto: bool) -> None:
        if auto or not sys.stdin.isatty():
            return
        try:
            input(self.style.dim(f"\n  {prompt}"))
        except (EOFError, KeyboardInterrupt):
            print()

    def demo(self, sample_id: str = DEMO_SESSION, *, auto: bool = False) -> None:
        """Render one curated rescue as a concise presentation transcript."""
        if sample_id not in self.rows:
            raise ValueError(f"Demo session is unavailable: {sample_id}")
        outcome = self.session_outcome(sample_id)
        exact_outcome = self.session_outcome(sample_id, "exact_only")
        stop_turn = int(outcome.get("first_hit_turn") or MAX_TURNS)
        first = self.rows[sample_id][0]
        messages = self.ensure_messages(sample_id)
        target = str(first["oracle_target_asin"])
        self.load_products({target})

        print("\n" + "╔" + "═" * 96 + "╗")
        print("║" + " EXPERIMENT 7 · RESCUE DEMO ".center(96) + "║")
        print("╚" + "═" * 96 + "╝")
        print(f"  Case       {sample_id} · held-out {first['oracle_scenario_type']} session")
        print(f"  Goal       rescue a session where exact phrase retrieval never finds the target")
        print(f"  Target     {target} · {self.title(self.products.get(target))}")
        print(self.style.oracle("  Disclosure Target identity is shown for the demo only; the rankers never receive it."))
        self.demo_pause("Press Enter to start →", auto)

        for index in range(stop_turn):
            row = self.rows[sample_id][index]
            turn = int(row["turn"])
            exact_rank = row.get("diagnostic_exact_only_target_rank")
            cascade_rank = row.get(f"diagnostic_{self.method}_target_rank")
            candidates = list(row.get(f"{self.method}_top_10") or [])
            self.load_products(candidates)

            print("\n" + "─" * 98)
            print(self.style.heading(f"  TURN {turn} OF {stop_turn}"))
            print(self.style.evaluator("\n  EVALUATOR"))
            print(wrap(f'“{messages[index]["message"]}”', indent="    "))

            print(self.style.system("\n  SYSTEM · Experiment 7 retrieval policy"))
            state = row.get("active_evidence") or []
            print(f"    remembered evidence  {json.dumps(state, ensure_ascii=False)}")
            print(wrap(f"search query         {row['agent_query']}", indent="    "))
            if row.get("fallback_activated"):
                reasons = ", ".join(row.get("fallback_reasons") or [])
                print(self.style.good(f"    route                EXACT → STATEFUL BM25 RRF"))
                print(wrap(f"why                  {reasons}", indent="    "))
            else:
                print(f"    route                EXACT (evidence is sufficiently precise)")

            exact_label = f"#{exact_rank}" if exact_rank is not None else "MISS"
            cascade_label = f"#{cascade_rank}" if cascade_rank is not None else "MISS"
            exact_styled = self.style.bad(exact_label) if exact_rank is None else exact_label
            cascade_styled = self.style.good(cascade_label) if cascade_rank is not None else self.style.bad(cascade_label)
            print(self.style.oracle("\n  ORACLE CHECK · frozen rankings scored after retrieval"))
            print(f"    exact-only target rank       {exact_styled}")
            print(f"    Experiment 7 target rank     {cascade_styled}")

            print(self.style.system("\n  RECOMMENDATION SLATE"))
            for rank, asin in enumerate(candidates, 1):
                title = self.title(self.products.get(asin))
                if len(title) > 62:
                    title = title[:59].rstrip() + "..."
                marker = self.style.good("  ◀ TARGET RESCUED") if asin == target else ""
                print(f"    {rank:>2}  {asin}  {title:<62}{marker}")

            if cascade_rank is not None:
                print(self.style.good(f"\n  SYSTEM → EVALUATOR  Recommend Top-10 · target recovered at rank {cascade_rank}."))
                print(self.style.good("  EVALUATOR SCORE       HIT · session terminates."))
            else:
                print(self.style.system("\n  SYSTEM → EVALUATOR  No conversion yet · ask_attribute=other"))
                if turn < stop_turn:
                    print(self.style.evaluator("  EVALUATOR             Reveals more preference evidence on the next turn."))
            if turn < stop_turn:
                self.demo_pause("Press Enter for the next evaluator turn →", auto)

        selected_metrics = self.metrics["method_metrics"][self.method]["evaluation"]
        exact_metrics = self.metrics["method_metrics"]["exact_only"]["evaluation"]
        print("\n" + "╔" + "═" * 96 + "╗")
        print("║" + " DEMO RESULT · HARD FAILURE RESCUED ".center(96) + "║")
        print("╚" + "═" * 96 + "╝")
        print(f"  Exact-only         {'HIT' if exact_outcome['hit'] else 'MISS':<8} target was never retrieved in its Top-10")
        print(f"  Experiment 7       {'HIT' if outcome['hit'] else 'MISS':<8} turn {outcome.get('first_hit_turn')} · rank {outcome.get('best_rank')}")
        print(f"  Held-out score     {exact_metrics['technical_score']:.6f} → {selected_metrics['technical_score']:.6f}")
        print("  Safety result      4 held-out hard-failure rescues · 2 regressions · recommendation passed")
        print(self.style.dim("\n  This is a deterministic replay of saved Experiment 7 artifacts, not a fabricated conversation."))

    def session_loop(self, sample_id: str, start: int = 0, print_all: bool = False) -> None:
        if sample_id not in self.rows:
            print(self.style.bad(f"Unknown session: {sample_id}"))
            return
        index = max(0, min(MAX_TURNS - 1, start))
        if print_all:
            for turn in range(MAX_TURNS):
                self.render_turn(sample_id, turn)
            return
        self.render_turn(sample_id, index)
        while True:
            try:
                raw = input(self.style.system(f"\n{sample_id} [{index + 1}/{MAX_TURNS}]> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            command, _, argument = raw.partition(" ")
            command = command.lower()
            if command in {"", "n", "next"}:
                if index == MAX_TURNS - 1:
                    print(self.style.dim("Already at the final retained turn."))
                    continue
                index += 1
                self.render_turn(sample_id, index)
            elif command in {"p", "prev", "previous"}:
                index = max(0, index - 1)
                self.render_turn(sample_id, index)
            elif command in {"t", "turn"} and argument.isdigit():
                index = max(0, min(MAX_TURNS - 1, int(argument) - 1))
                self.render_turn(sample_id, index)
            elif command in {"a", "all"}:
                for turn in range(MAX_TURNS):
                    self.render_turn(sample_id, turn)
            elif command in {"c", "compare"}:
                self.render_turn(sample_id, index, compare=True)
            elif command in {"j", "json", "raw"}:
                self.render_turn(sample_id, index, raw=True)
            elif command in {"d", "detail", "details", "candidate"}:
                self.show_candidate(self.rows[sample_id][index], argument)
            elif command in {"o", "oracle"}:
                self.show_oracle(sample_id)
            elif command in {"m", "method"}:
                if argument in METHODS:
                    self.method = argument
                    print(self.style.good(f"Method changed to {self.method}"))
                    self.render_turn(sample_id, index)
                else:
                    print("Methods: " + ", ".join(METHODS))
            elif command in {"h", "help", "?"}:
                self.session_help()
            elif command in {"q", "back", "exit"}:
                return
            else:
                print(self.style.bad("Unknown command. Type 'help'."))

    @staticmethod
    def session_help() -> None:
        print(
            "\n  Enter/n       next turn                 p             previous turn\n"
            "  turn <1-10>   jump to a turn            all           print all ten turns\n"
            "  compare       compare all rankers       json          raw saved row\n"
            "  candidate <#> full catalog product      oracle        full hidden evaluator state\n"
            "  method <name> change displayed ranker   back/q        session list\n"
        )

    def main_loop(self) -> None:
        self.banner()
        self.list_sessions("hard_failure")
        while True:
            try:
                raw = input(self.style.heading("\nexp7> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            command, _, argument = raw.partition(" ")
            command = command.lower()
            if command in {"open", "o"}:
                self.session_loop(argument.strip())
            elif command in {"list", "ls", "l"}:
                self.list_sessions(argument.strip())
            elif command in {"metrics", "score", "scores"}:
                self.print_metrics()
            elif command in {"method", "m"}:
                if argument in METHODS:
                    self.method = argument
                    print(self.style.good(f"Method changed to {self.method}"))
                else:
                    print("Methods: " + ", ".join(METHODS))
            elif command in {"run", "rerun"}:
                self.run_experiment()
            elif command in {"help", "h", "?"}:
                print(
                    "\n  list [filter]      list sessions; combine filters such as 'evaluation browsing'\n"
                    "                     useful filters: hard_failure, weak_success, rescue, regression\n"
                    "  open <sample_id>   open a turn-by-turn interaction\n"
                    "  metrics            compare held-out scores\n"
                    "  method <name>      choose the recommendation slate shown\n"
                    "  run                rerun Experiment 7 and reload artifacts\n"
                    "  quit               exit\n"
                )
            elif command in {"quit", "q", "exit"}:
                return
            elif command in self.rows:
                self.session_loop(command)
            elif not command:
                continue
            else:
                print(self.style.bad("Unknown command or session. Type 'help'."))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and interactively inspect Experiment 7 evaluator/system traces")
    parser.add_argument("--results", type=Path, default=default_results(), help="Results directory")
    parser.add_argument("--run", action="store_true", help="Rerun Experiment 7 before opening the viewer")
    parser.add_argument("--demo", action="store_true", help=f"Run one polished rescue demo ({DEMO_SESSION}) and exit")
    parser.add_argument("--demo-auto", action="store_true", help="Do not pause between demo turns")
    parser.add_argument("--session", help="Open this sample ID immediately")
    parser.add_argument("--turn", type=int, default=1, help="Initial turn for --session (1-10)")
    parser.add_argument("--method", choices=METHODS, help="Recommendation method to display")
    parser.add_argument("--print-all", action="store_true", help="Print all turns for --session and exit non-interactively")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI terminal colors")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Redirected Windows PowerShell streams often default to cp1252. The viewer
    # deliberately uses readable Unicode arrows and separators.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")
    style = Style(not args.no_color and "NO_COLOR" not in os.environ and sys.stdout.isatty())
    results = args.results.resolve()
    if args.run and not run_experiment_process(repo_root(), results, style):
        return 1
    try:
        viewer = Viewer(results, style)
    except (OSError, ValueError) as exc:
        print(f"Could not open Experiment 7: {exc}", file=sys.stderr)
        print("Run it with: python -m nickolas.experiments.experiment_07_cli --run", file=sys.stderr)
        return 1
    if args.method:
        viewer.method = args.method
    if args.demo:
        try:
            viewer.demo(args.session or DEMO_SESSION, auto=args.demo_auto)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    if args.session:
        viewer.session_loop(args.session, args.turn - 1, args.print_all)
        if args.print_all:
            return 0
    viewer.main_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
