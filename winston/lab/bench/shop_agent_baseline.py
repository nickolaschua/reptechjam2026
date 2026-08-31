"""Yang Xu's hybrid agent (FTS5 + BGE embedder fallback + LLM state tracker) on the
messy benchmark - the existing-infra baseline the bolt-on must beat.

Local-only: no API keys are read; the state tracker (LLM Call 1) runs on Ollama
llama3.1 as the agent's own cascade dictates. The dialogue call (LLM Call 2,
response_json=False) is stubbed IN MEMORY ONLY - it shapes prose, not ranking -
so each case costs one LLM call instead of two. shop_agent.py is not modified.

    python3 shop_agent_baseline.py [N]       # first N cases (stratified prefix), resumable
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent.parent.parent
for p in (REPO / "experiment_1", REPO / "docs" / "archive" / "legacy_hybrid_agent",
          REPO / "techjam-conversational-search"):
    sys.path.insert(0, str(p))
for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(k, None)                     # local-only, by construction

OUT = BENCH / "results_shop_agent.jsonl"


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    import shop_agent
    agent = shop_agent.Agent(REPO / "techjam-conversational-search" / "data" / "catalog.jsonl")

    real_call = agent._call_llm
    def rank_only(prompt, system_prompt="", session_id=None, response_json=False):
        if not response_json:                   # LLM Call 2: dialogue only, no effect on ranks
            return "Here are the top matches."
        return real_call(prompt, system_prompt, session_id=session_id, response_json=True)
    agent._call_llm = rank_only

    # His Ollama branch hard-codes requests.post(..., timeout=3). An 8B JSON state
    # update takes 5-15 s locally, so every call timed out into the static fallback
    # and the agent ran without its state tracker (measured: 0.043, identical to the
    # regex agent). Override the timeout IN MEMORY for the local Ollama URL only.
    import requests
    _post = requests.post
    def patient_post(url, *a, **kw):
        if "localhost:11434" in str(url) and kw.get("timeout", 0) < 180:
            kw["timeout"] = 180
        return _post(url, *a, **kw)
    requests.post = patient_post

    cases = [json.loads(l) for l in (BENCH / "cases.jsonl").open() if l.strip()][:limit]
    done = {json.loads(l)["case_id"] for l in OUT.open()} if OUT.exists() else set()
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"cases {len(cases)} | done {len(done)} | this run {len(todo)}", flush=True)
    t0 = time.time()
    with OUT.open("a") as fh:
        for i, c in enumerate(todo, 1):
            sid = f"sa_{c['case_id']}"
            agent.reset(sid, {})
            try:
                resp = agent.respond(sid, c["utterance"], 1, 50)
                ids = [r["parent_asin"] if isinstance(r, dict) else r for r in resp.get("recommendations", [])]
                rank = ids.index(c["asin"]) + 1 if c["asin"] in ids else None
                model_used = (agent._sessions.get(sid, {}).get("debug_info") or {}).get("model")
                row = {"case_id": c["case_id"], "asin": c["asin"], "shop_agent_rank": rank,
                       "model_used": model_used}           # proves the state tracker actually ran
            except Exception as exc:            # noqa: BLE001 - a crash is a data point, not a stop
                row = {"case_id": c["case_id"], "asin": c["asin"], "shop_agent_rank": None,
                       "error": f"{type(exc).__name__}: {exc}"[:200]}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el/i:.1f}s/case  eta {(len(todo)-i)*el/i/60:.0f} min", flush=True)
    rows = [json.loads(l) for l in OUT.open() if l.strip()]
    ranks = [r["shop_agent_rank"] for r in rows]
    hit = sum(1 for r in ranks if r and r <= 10) / len(ranks)
    mrr = sum(1 / r for r in ranks if r and r <= 10) / len(ranks)
    from collections import Counter
    print(f"\nshop_agent on {len(rows)} messy cases: hit@10 {hit:.3f}  mrr {mrr:.3f}  errors {sum(1 for r in rows if r.get('error'))}")
    print("model that served the state update:", dict(Counter(r.get("model_used") for r in rows)))


if __name__ == "__main__":
    main()
