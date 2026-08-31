"""Yang Xu's hybrid retriever with OUR reader in front of it.

Two overrides on his archived shop_agent.Agent, retrieval untouched:
  respond()               is_template() decides his simulator route instead of the
                          "i'm looking for " prefix, which sent 271/400 human messages
                          to his no-LLM exact matcher (hit@10 0.000 there).
  _update_state_via_llm() his free-JSON LLM state tracker is replaced by our cached,
                          cleaned parse, written through his own _set_constraint().
So his FTS5 -> BGE-fallback -> post-scoring runs on state produced by our parser.
No LLM call is made at all (parses are cached), so the run is deterministic.

    python3 shop_agent_plug.py [N]     # first N cases, default 400 -> results_shop_agent_plug.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
REPO = WINSTON.parent
for p in (BENCH, LAB, WINSTON, WINSTON / "experiments", REPO / "docs" / "archive" / "legacy_hybrid_agent",
          REPO / "techjam-conversational-search"):
    sys.path.insert(0, str(p))
for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(k, None)

import shop_agent                                   # noqa: E402  (archived, unmodified)
from bolt_on import clean_parse, is_template        # noqa: E402
from nlp_parse import tier_of                       # noqa: E402

for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(k, None)                         # his .env loader ran at import
shop_agent.HAS_OPENAI = False
shop_agent.HAS_GEMINI = False

OUT = BENCH / "results_shop_agent_plug.jsonl"
DEPT = {"womens": "women", "mens": "men", "girls": "girls", "boys": "boys",
        "unisex-child": "kids", "baby-girls": "toddler", "baby-boys": "toddler"}
_NEG_PREFIX = re.compile(r"^(?:not|no|non|without|never)\s+", re.I)


class PluggedShopAgent(shop_agent.Agent):
    def __init__(self, catalog_path, parses_by_message: dict[str, dict]) -> None:
        super().__init__(catalog_path)
        self._parses = parses_by_message

    def _call_llm(self, prompt, system_prompt="", session_id=None, response_json=False):
        return "Here are the top matches."          # dialogue only; state comes from our parse

    def respond(self, session_id, user_message, turn, top_k):
        if is_template(user_message):
            return super().respond(session_id, user_message, turn, top_k)   # his simulator route
        self._simulator_sessions[session_id] = False
        return self._respond_custom(session_id, user_message, turn, top_k)

    def _update_state_via_llm(self, session_id, user_message, turn=None):
        state = self._sessions[session_id]
        parse = clean_parse(self._parses[user_message], user_message)
        src = "initial_preference" if (turn or 1) <= 1 else "clarification"
        if parse.get("category_phrase"):
            state["category"] = parse["category_phrase"]
        for s in parse.get("slots", []):
            if tier_of(s) == "decline":
                continue
            if s.get("negated"):
                state["negated_terms"].add(_NEG_PREFIX.sub("", s["value"]).strip().lower())
                continue
            self._set_constraint(state, s["attribute"], s["value"], turn or 1, src, user_message)
        if parse.get("price_max"):
            state["price_max"] = float(parse["price_max"])
        dept = DEPT.get(parse.get("department") or "")
        if dept:
            state["target_department"] = dept
        self._rebuild_active_terms(state)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    cases = [json.loads(l) for l in (BENCH / "cases.jsonl").open() if l.strip()][:limit]
    parses = {}
    for l in (BENCH / "parses.jsonl").open():
        if l.strip():
            r = json.loads(l); parses[r["case_id"]] = r["parse"]        # last row wins
    by_msg = {c["utterance"]: parses[c["case_id"]] for c in cases}
    agent = PluggedShopAgent(REPO / "techjam-conversational-search" / "data" / "catalog.jsonl", by_msg)
    done = {json.loads(l)["case_id"] for l in OUT.open()} if OUT.exists() else set()
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"cases {len(cases)} | done {len(done)} | this run {len(todo)}", flush=True)
    t0 = time.time()
    with OUT.open("a") as fh:
        for i, c in enumerate(todo, 1):
            sid = f"plug_{c['case_id']}"
            agent.reset(sid, {})
            try:
                resp = agent.respond(sid, c["utterance"], 1, 50)
                ids = [r["parent_asin"] if isinstance(r, dict) else r for r in resp.get("recommendations", [])]
                rank = ids.index(c["asin"]) + 1 if c["asin"] in ids else None
                row = {"case_id": c["case_id"], "asin": c["asin"], "rank": rank,
                       "route": "template" if agent._simulator_sessions.get(sid) else "parsed",
                       "vector_fallback": bool((agent._sessions.get(sid, {}).get("debug_info") or {}).get("vector_fallback"))}
            except Exception as exc:                # noqa: BLE001
                row = {"case_id": c["case_id"], "asin": c["asin"], "rank": None, "error": f"{type(exc).__name__}: {exc}"[:200]}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el/i:.1f}s/case  eta {(len(todo)-i)*el/i/60:.0f} min", flush=True)
    report()


def report() -> None:
    def hit(rs): return sum(1 for r in rs if r and r <= 10) / len(rs) if rs else 0
    def mrr(rs): return sum(1 / r for r in rs if r and r <= 10) / len(rs) if rs else 0
    rows = [json.loads(l) for l in OUT.open() if l.strip()]
    ids = {r["case_id"] for r in rows}
    legacy = {json.loads(l)["case_id"]: json.loads(l) for l in (BENCH / "results_shop_agent.jsonl").open()}
    plug = {json.loads(l)["case_id"]: json.loads(l) for l in (BENCH / "results_plug.jsonl").open()}
    common = [r for r in rows if r["case_id"] in legacy and r["case_id"] in plug]
    print(f"\n{len(rows)} cases | errors {sum(1 for r in rows if r.get('error'))} | routes {dict((k, sum(1 for r in rows if r.get('route')==k)) for k in ('template','parsed'))} | BGE fallback used on {sum(1 for r in rows if r.get('vector_fallback'))}")
    print(f"{'system (same ' + str(len(common)) + ' cases)':40} {'hit@10':>7} {'mrr':>7}")
    print(f"{'Yang Xu legacy, as shipped':40} {hit([legacy[r['case_id']]['shop_agent_rank'] for r in common]):7.3f} {mrr([legacy[r['case_id']]['shop_agent_rank'] for r in common]):7.3f}")
    print(f"{'starter + bolt-on':40} {hit([plug[r['case_id']]['plugged_rank'] for r in common]):7.3f} {mrr([plug[r['case_id']]['plugged_rank'] for r in common]):7.3f}")
    print(f"{'Yang Xu retriever + our reader':40} {hit([r['rank'] for r in common]):7.3f} {mrr([r['rank'] for r in common]):7.3f}")
    w = sum(1 for r in common if (r["rank"] or 999) < (plug[r["case_id"]]["plugged_rank"] or 999))
    l = sum(1 for r in common if (r["rank"] or 999) > (plug[r["case_id"]]["plugged_rank"] or 999))
    print(f"his retriever+our reader vs starter+bolt-on, paired: {w} better / {l} worse / {len(common)-w-l} tie")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
