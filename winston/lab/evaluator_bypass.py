"""Gate 2, forced: run the evaluator's OWN turn-1 messages through the fall-through
pipeline as if a human had typed them, and compare with the keyword layer as shipped.

  template   exp11 respond(turn=1) - the keyword layer, its regex + its ranker
  lex        exp11's ranker fed by the parse (deterministic template parse, or 7B with --llm N)
  dense_raw  BGE on the raw message (DENSE_MODEL selects base / fine-tuned)
  fuse       RRF(lex, dense, dense-in-resolved-buckets) weighted by specificity, contradictions sunk

    DENSE_DEVICE=cpu python3 evaluator_bypass.py                 # deterministic parse, 200 sessions
    DENSE_DEVICE=cpu python3 evaluator_bypass.py --llm 60        # 7B parse on the first 60
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
WINSTON = LAB.parent
REPO = WINSTON.parent
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON, WINSTON / "experiments", LAB, LAB / "bench", KIT, REPO / "nickolas" / "experiments"):
    sys.path.insert(0, str(p))
from bolt_on import clean_parse, contradictions, n_hard  # noqa: E402
from dense_rank import _TAG, QUERY_PREFIX, catalog_index, load_model  # noqa: E402
from evaluator.local_evaluator import (catalog_index as ev_index, classify_constraint, coarse_category,  # noqa: E402
                                       initial_message, materialize_hidden_fields)
from score import parsed_state  # noqa: E402

K, RRF_K = 150, 60


def template_parse(message: str, sample: dict, category: str) -> dict:
    """What a regex would extract - the parse the keyword layer effectively has."""
    slots = []
    if sample["scenario_type"] == "buying":
        c = str(sample["intent_card"]["hard_constraints"][0])
        slots.append({"attribute": classify_constraint(c), "value": c, "declined": False, "negated": False})
    elif sample["scenario_type"] == "intent_override":
        c = str(sample["behavior"]["override"]["old_value"])
        slots.append({"attribute": classify_constraint(c), "value": c, "declined": False, "negated": False})
    return {"category_phrase": category, "department": None, "slots": slots, "price_max": None,
            "price_min": None, "quality_prior": "none",
            "exploring": sample["scenario_type"] in ("browsing", "boundary"), "specificity": None}


def rrf(*lists, weights=None):
    s = defaultdict(float)
    for w, lst in zip(weights or [1.0] * len(lists), lists):
        for i, a in enumerate(lst, 1):
            s[a] += w / (RRF_K + i)
    return sorted(s, key=lambda a: -s[a])


def rank_of(lst, asin): return lst.index(asin) + 1 if asin in lst else None
def hit(rs): return sum(1 for r in rs if r is not None and r <= 10) / len(rs)
def mrr(rs): return sum(1 / r for r in rs if r and r <= 10) / len(rs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0, help="parse the first N with qwen instead of the regex")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    args = ap.parse_args()

    from experiment_11_candidate_agent import CleanFTSAgent
    from common import get_index
    from pipeline import content_profiles, resolve, slot_terms
    _, cats, products = ev_index(KIT / "data" / "catalog.jsonl")
    samples = [json.loads(l) for l in (KIT / "data" / "public_set.jsonl").open() if l.strip()]
    if args.llm:
        samples = samples[:args.llm]
        from nlp_parse import parse_with_ollama
    agent = CleanFTSAgent(KIT / "data" / "catalog.jsonl", pagination_mode="none")
    ix = get_index(); profiles = content_profiles(ix)
    model = load_model(); emb, ids = catalog_index(model); ids = np.array(ids)

    def lexical(category, constraints, sid):
        agent.reset(sid, {}); st = agent.sessions[sid]
        st["category"], st["constraints"] = category, constraints
        return list(agent._rank(st, K))

    def dense(text):
        q = model.encode(QUERY_PREFIX + text, convert_to_numpy=True, normalize_embeddings=True)
        return list(ids[np.argsort(emb @ q)[::-1][:K]])

    rows = []
    for s in samples:
        s = dict(s)
        card, behavior = materialize_hidden_fields(s, products)
        s["intent_card"], s["behavior"] = card, behavior
        asin = s["ground_truth"]["parent_asin"]
        category = coarse_category(cats[asin])
        msg = initial_message(s, category, set())
        agent.reset(s["sample_id"], {})
        resp = agent.respond(s["sample_id"], msg, 1, 50)
        t_rank = rank_of([r["parent_asin"] for r in resp["recommendations"]], asin)
        parse = parse_with_ollama(msg, args.model) if args.llm else template_parse(msg, s, category)
        cp = clean_parse(parse, msg)
        lex = lexical(*parsed_state(cp), s["sample_id"] + "p")
        dr = dense(msg)
        top, conf = resolve([cp.get("category_phrase") or "", *slot_terms(cp)], ix, profiles, top_n=3)
        in_cat = {a for b in top for a in ix.buckets[b]}
        wl, wd = (0.5, 1.5) if (n_hard(cp) == 0 and conf < 0.2) else ((1.5, 1.0) if n_hard(cp) else (1.0, 1.0))
        fused = rrf(lex, dr, [a for a in dr if a in in_cat], weights=[wl, wd, conf])
        keep, sink = [], []
        for a in fused:
            (sink if contradictions(cp, ix.products[a], ix.text[a].lower()) else keep).append(a)
        rows.append({"sample_id": s["sample_id"], "scenario": s["scenario_type"], "message": msg,
                     "category_phrase": cp.get("category_phrase"), "n_slots": len(cp["slots"]),
                     "template": t_rank, "lex": rank_of(lex, asin), "dense_raw": rank_of(dr, asin),
                     "fuse": rank_of(keep + sink, asin)})
    tag = f"{_TAG}{'_llm' + str(args.llm) if args.llm else ''}"
    (LAB / "bench" / f"results_evaluator_bypass{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    systems = ("template", "lex", "dense_raw", "fuse")
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"]].append(r)
    print(f"\nevaluator turn-1 text, {len(rows)} sessions, embedder={'fine-tuned' if _TAG else 'base'}, parse={'qwen' if args.llm else 'regex'}")
    print(f"{'scenario':16s} n    " + "   ".join(f"{s:>11s}" for s in systems) + "   (HitRate@10 / MRR)")
    for g, rs in [("all", rows), *sorted(by.items())]:
        print(f"{g:16s} {len(rs):<4d} " + "   ".join(f"{hit([r[s] for r in rs]):.2f} / {mrr([r[s] for r in rs]):.2f}" for s in systems))
    b = sum(1 for r in rows if (r["fuse"] or 999) < (r["template"] or 999))
    w = sum(1 for r in rows if (r["fuse"] or 999) > (r["template"] or 999))
    print(f"fuse vs template per session: better {b} / worse {w} / same {len(rows) - b - w}")
    if args.llm:
        print("qwen category_phrase == evaluator category:",
              sum(1 for r in rows if (r["category_phrase"] or "").lower() == coarse_category(cats[[s for s in samples if s['sample_id'] == r['sample_id']][0]['ground_truth']['parent_asin']]).lower()), "/", len(rows))


if __name__ == "__main__":
    main()
