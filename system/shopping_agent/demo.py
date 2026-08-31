"""Canonical interactive and scripted TechJam shopping-agent demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .agent import Agent
from .config import DEMO_TOP_K, MEMORY_STORE_PATH
from .memory_store import JsonFileVectorMemoryStore
from .vector_memory import BuyerMode


SCENARIO_PATH = Path(__file__).with_name("demo_scenarios.json")
EXPLORATORY_CUES = (
    "just browsing",
    "browsing for now",
    "still exploring",
    "open to options",
    "looking around",
    "comparing options",
    "not ready to buy",
    "haven't decided",
    "have not decided",
)


def classify_buyer_mode(message: str) -> BuyerMode:
    """Provide a conservative caller fallback when live intent is unavailable."""

    normalized = " ".join(str(message).lower().split())
    if any(cue in normalized for cue in EXPLORATORY_CUES):
        return BuyerMode.BROWSING
    return BuyerMode.BUYING


@dataclass
class ActiveDemoSession:
    user_id: str
    session_id: str
    sequence_index: int
    turn: int = 0
    mode_override: BuyerMode | None = None


class DemoApplication:
    """Small lifecycle wrapper around the real Agent and persistent store."""

    def __init__(
        self,
        *,
        memory_path: str | Path = MEMORY_STORE_PATH,
        top_k: int = DEMO_TOP_K,
        agent: Agent | None = None,
        store: JsonFileVectorMemoryStore | None = None,
    ) -> None:
        self.store = store or JsonFileVectorMemoryStore(memory_path)
        self.agent = agent or Agent(memory_store=self.store, allow_catalog_embedding=False)
        self.top_k = int(top_k)
        self.active: ActiveDemoSession | None = None

    def start_session(
        self, user_id: str, *, mode: str | BuyerMode | None = None
    ) -> ActiveDemoSession:
        if self.active is not None:
            raise RuntimeError("finish the active session before starting another")
        user = str(user_id).strip()
        if not user:
            raise ValueError("user_id must be non-empty")
        sequence = self.store.next_sequence_index(user)
        session_id = f"demo-{user}-{sequence}-{uuid4().hex[:8]}"
        resolved_mode = None if mode is None else BuyerMode(mode)
        self.agent.reset(
            session_id, {}, user_id=user, sequence_index=sequence
        )
        self.active = ActiveDemoSession(user, session_id, sequence, mode_override=resolved_mode)
        return self.active

    def send(self, message: str, *, mode: str | BuyerMode | None = None) -> dict[str, Any]:
        if self.active is None:
            raise RuntimeError("start_session must be called before send")
        text = str(message).strip()
        if not text:
            raise ValueError("message must be non-empty")
        selected = (
            BuyerMode(mode)
            if mode is not None
            else self.active.mode_override or classify_buyer_mode(text)
        )
        next_turn = self.active.turn + 1
        response = self.agent.respond(
            self.active.session_id,
            text,
            next_turn,
            self.top_k,
            buyer_mode=selected,
            debug=True,
        )
        self.active.turn = next_turn
        return response

    def finish_session(self) -> dict[str, Any] | None:
        if self.active is None:
            return None
        session_id = self.active.session_id
        self.agent.end_session(session_id)
        trace = self.agent.get_memory_debug(session_id, consume=True)
        self.active = None
        return trace

    def close(self) -> None:
        try:
            if self.active is not None:
                self.finish_session()
        finally:
            self.agent.close()
            self.active = None

    def inspect_user(self, user_id: str) -> dict[str, Any]:
        return self.store.describe_user(user_id)

    def reset_user(self, user_id: str) -> None:
        if self.active is not None and self.active.user_id == user_id:
            raise RuntimeError("finish the active session before resetting its user")
        self.store.clear_user(user_id)

    def reset_all(self) -> None:
        if self.active is not None:
            raise RuntimeError("finish the active session before resetting all memory")
        self.store.clear()


def _products(response: dict[str, Any]) -> list[dict[str, Any]]:
    return response["debug"]["memory_trace"]["returned"]


def _print_response(response: dict[str, Any], *, show_debug: bool) -> None:
    print(f"\nAgent: {response['message']}")
    for index, product in enumerate(_products(response), 1):
        print(f"  {index}. {product['title']} [{product['parent_asin']}]")
    if show_debug:
        trace = response["debug"]["memory_trace"]
        concise = {
            key: trace[key]
            for key in (
                "user_id", "buyer_mode", "intent_mode", "intent_source",
                "current_intent", "useful_slots",
                "prior_ltm_exists", "memory_version", "memory_update_count",
                "gate_cosine", "gate_threshold", "gate_passed", "a", "b",
                "catalog_rows_scored", "eligible_count", "fts_or_threshold",
                "keyword_route_threshold",
                "ltm_updated_after_turn", "memory_update_text",
            )
        }
        concise["top_recommended_products"] = [
            {"parent_asin": item["parent_asin"], "title": item["title"]}
            for item in trace["returned"]
        ]
        print("Trace:")
        print(json.dumps(concise, indent=2, sort_keys=True))


def _print_commit(trace: dict[str, Any] | None, *, show_debug: bool) -> None:
    if trace is None:
        return
    status = "updated" if trace["ltm_updated_after_turn"] else "unchanged"
    print(f"Session committed; LTM {status}.")
    if show_debug:
        print(json.dumps({
            "user_id": trace["user_id"],
            "memory_version": trace["memory_version"],
            "canonical_update_text": trace["preference_text"],
            "ltm_updated_after_turn": trace["ltm_updated_after_turn"],
            "post_update_memory": trace["post_update_memory"],
        }, indent=2, sort_keys=True))


def run_scripted(app: DemoApplication, *, show_debug: bool = True) -> None:
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    for user_id in scenario["users"]:
        app.reset_user(user_id)
    print(scenario["description"])
    for item in scenario["sessions"]:
        print(f"\n=== {item['label']} ({item['user_id']}) ===")
        app.start_session(item["user_id"], mode=item["buyer_mode"])
        for message in item["messages"]:
            print(f"\nShopper: {message}")
            _print_response(app.send(message), show_debug=show_debug)
        _print_commit(app.finish_session(), show_debug=show_debug)


def run_interactive(app: DemoApplication, user_id: str, *, show_debug: bool) -> None:
    current_user = user_id
    mode_override: BuyerMode | None = None
    app.start_session(current_user)
    print("System TechJam demo. Commands: /help, /new, /user ID, /mode MODE,")
    print("/memory, /reset-user, /reset-all, /debug, /exit")
    while True:
        try:
            value = input(f"\n{current_user}> ").strip()
        except (EOFError, KeyboardInterrupt):
            value = "/exit"
        if not value:
            continue
        command, _, argument = value.partition(" ")
        if command == "/exit":
            _print_commit(app.finish_session(), show_debug=show_debug)
            return
        if command == "/help":
            print("Messages continue short-term state. /new commits LTM and starts a new session.")
            continue
        if command == "/debug":
            show_debug = not show_debug
            print(f"Debug trace {'on' if show_debug else 'off'}.")
            continue
        if command == "/memory":
            print(json.dumps(app.inspect_user(current_user), indent=2, sort_keys=True))
            continue
        if command == "/mode":
            mode_override = BuyerMode(argument.strip().lower())
            app.active.mode_override = mode_override
            print(f"Mode fallback set to {mode_override.value}; live intent may override.")
            continue
        if command in {"/new", "/user"}:
            _print_commit(app.finish_session(), show_debug=show_debug)
            if command == "/user":
                current_user = argument.strip()
                if not current_user:
                    raise ValueError("/user requires an ID")
            app.start_session(current_user, mode=mode_override)
            continue
        if command == "/reset-user":
            app.finish_session()
            app.reset_user(current_user)
            app.start_session(current_user, mode=mode_override)
            print(f"Reset memory for {current_user}.")
            continue
        if command == "/reset-all":
            app.finish_session()
            app.reset_all()
            app.start_session(current_user, mode=mode_override)
            print("Reset all demo memory.")
            continue
        _print_response(app.send(value), show_debug=show_debug)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="demo-user", help="initial interactive user ID")
    parser.add_argument("--debug", action="store_true", help="show concise structured traces")
    parser.add_argument("--scripted", action="store_true", help="run the deterministic two-user presentation")
    parser.add_argument("--memory-file", type=Path, default=MEMORY_STORE_PATH)
    parser.add_argument("--top-k", type=int, default=DEMO_TOP_K)
    parser.add_argument("--inspect", metavar="USER_ID", help="inspect memory without loading the agent")
    parser.add_argument("--reset-user", metavar="USER_ID", help="reset one user without loading the agent")
    parser.add_argument("--reset-all", action="store_true", help="reset all demo memory without loading the agent")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.inspect or args.reset_user or args.reset_all:
        store = JsonFileVectorMemoryStore(args.memory_file)
        if args.reset_user:
            store.clear_user(args.reset_user)
            print(f"Reset memory for {args.reset_user}.")
        elif args.reset_all:
            store.clear()
            print("Reset all demo memory.")
        else:
            print(json.dumps(store.describe_user(args.inspect), indent=2, sort_keys=True))
        return
    app = DemoApplication(memory_path=args.memory_file, top_k=args.top_k)
    try:
        if args.scripted:
            run_scripted(app, show_debug=True)
        else:
            run_interactive(app, args.user, show_debug=args.debug)
    finally:
        app.close()


if __name__ == "__main__":
    main()
