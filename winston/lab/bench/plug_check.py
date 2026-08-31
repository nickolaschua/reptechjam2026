"""Plug the bolt-on into the DEPLOYED agent (starter/agent.py) and measure.

BoltOnAgent overrides only _update_state: BoltOnParser.parse first (None on any
template -> the stock regex path, byte for byte), else the ParsedTurn fills the
same legacy {category, constraints} that _legacy_recommendations already reads.
Negatives are never appended (they would become positive terms). Retrieval and
question policy untouched.

  python3 plug_check.py --public 60     # evaluator sessions: stock vs plugged must be identical
  python3 plug_check.py --bench         # 1,685 messy cases, cached parses, no LLM
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
REPO = WINSTON.parent
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON, LAB, BENCH, WINSTON / "experiments", KIT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from starter.agent import Agent            # noqa: E402
from bolt_on import BoltOnParser           # noqa: E402


class BoltOnAgent(Agent):
    def __init__(self, catalog_path, parser: BoltOnParser) -> None:
        super().__init__(catalog_path)
        self._parser = parser

    def _update_state(self, state, user_message: str, turn: int) -> None:
        update = self._parser.parse(user_message, turn)
        if update is None:                              # template -> stock path
            return super()._update_state(state, user_message, turn)
        if update.category:
            state["category"] = update.category
        for c in (*update.hard_constraints, *update.soft_preferences):
            self._append_constraint(state, c.value)


def rank_of(resp: dict, asin: str) -> int | None:
    ids = [r["parent_asin"] for r in resp["recommendations"]]
    return ids.index(asin) + 1 if asin in ids else None


def run_bench() -> None:
    cases = [json.loads(l) for l in (BENCH / "cases.jsonl").open() if l.strip()]
    parses = {}
    for l in (BENCH / "parses.jsonl").open():
        if l.strip():
            r = json.loads(l)
            parses[r["case_id"]] = r["parse"]           # last row wins
    by_msg = {c["utterance"]: parses[c["case_id"]] for c in cases if c["case_id"] in parses}
    stock = Agent(KIT / "data" / "catalog.jsonl")
    plugged = BoltOnAgent(KIT / "data" / "catalog.jsonl",
                          BoltOnParser(parse_fn=lambda m: by_msg[m], resolver=False))
    ranks = {"stock": [], "plugged": []}
    for i, c in enumerate(cases):
        for name, agent in (("stock", stock), ("plugged", plugged)):
            sid = f"{name}{i}"
            agent.reset(sid, {})
            ranks[name].append(rank_of(agent.respond(sid, c["utterance"], 1, 50), c["asin"]))
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(cases)}", flush=True)
    def hit(rs): return sum(1 for r in rs if r and r <= 10) / len(rs)
    def mrr(rs): return sum(1 / r for r in rs if r and r <= 10) / len(rs)
    print(f"\nDEPLOYED agent on {len(cases)} messy cases (turn 1):")
    for name in ("stock", "plugged"):
        print(f"  {name:8} hit@10 {hit(ranks[name]):.3f}  mrr {mrr(ranks[name]):.3f}")
    both = list(zip(ranks["stock"], ranks["plugged"]))
    w = sum(1 for s, p in both if (p or 999) < (s or 999)); l = sum(1 for s, p in both if (p or 999) > (s or 999))
    print(f"  paired: {w} better / {l} worse / {len(both) - w - l} tie")


def run_public(n: int) -> None:
    from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
    catalog_ids, categories, products = catalog_index(KIT / "data" / "catalog.jsonl")
    samples = load_jsonl(KIT / "data" / "public_set.jsonl")[:n]
    def qwen_should_never_run(msg):
        raise AssertionError(f"parser invoked on evaluator text: {msg!r}")
    out = {}
    for name, agent in (("stock", Agent(KIT / "data" / "catalog.jsonl")),
                        ("plugged", BoltOnAgent(KIT / "data" / "catalog.jsonl",
                                                BoltOnParser(parse_fn=qwen_should_never_run, resolver=False)))):
        result = evaluate(agent, samples, catalog_ids, categories, products)
        out[name] = result.get("metrics") or result.get("summary") or {k: v for k, v in result.items() if not isinstance(v, list)}
        print(f"  {name:8} {out[name]}")
    print("  identical" if out["stock"] == out["plugged"] else "  DIFFER - bolt-on touched a template session")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", type=int)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    if a.public:
        run_public(a.public)
    if a.bench:
        run_bench()
