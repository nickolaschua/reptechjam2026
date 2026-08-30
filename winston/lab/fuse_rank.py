"""Where does the parse belong in a multi-route pipeline? RRF fusion of exp11's
lexical route (raw text vs parse-fed state) with the dense route (raw text).

  lex_raw       utterance -> exp11 state.category -> its _rank            (top-K ids)
  lex_parsed    parsed_state(parse) -> exp11 state -> its _rank           (top-K ids)
  dense_raw     utterance -> BGE (cached index)                           (top-K ids)
  fuse_raw      RRF(lex_raw, dense_raw)          <- the pipeline without the parse
  fuse_parsed   RRF(lex_parsed, dense_raw)       <- the pipeline with the parse
  category      resolver top-3 buckets (phrase + soft slots), members by popularity,
                weighted by resolver confidence   <- the brief's third route
  fuse3         RRF(lex_parsed, dense_raw, confidence * category)

    DENSE_DEVICE=cpu python3 fuse_rank.py
"""
from __future__ import annotations

import json
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
from bolt_on import clean_parse, contradictions  # noqa: E402
from dense_rank import _TAG, QUERY_PREFIX, catalog_index, load_model  # noqa: E402
from score import parsed_state  # noqa: E402

K, RRF_K = 150, 60


def rrf(*lists, weights=None) -> list[str]:
    s: dict[str, float] = defaultdict(float)
    for w, lst in zip(weights or [1.0] * len(lists), lists):
        for i, a in enumerate(lst, 1):
            s[a] += w / (RRF_K + i)
    return sorted(s, key=lambda a: -s[a])


def rank_of(lst: list[str], asin: str) -> int | None:
    return lst.index(asin) + 1 if asin in lst else None


def hit(rs): return sum(1 for r in rs if r is not None and r <= 10) / len(rs)
def mrr(rs): return sum(1 / r for r in rs if r and r <= 10) / len(rs)   # evaluator-style: credit only inside the top-10


def main() -> None:
    from experiment_11_candidate_agent import CleanFTSAgent
    cases = {r["case_id"]: r for r in map(json.loads, (LAB / "bench" / "cases.jsonl").open())}
    parses = {r["case_id"]: r["parse"] for r in map(json.loads, (LAB / "bench" / "parses.jsonl").open())}
    todo = [cases[k] for k in parses if k in cases]
    agent = CleanFTSAgent(KIT / "data" / "catalog.jsonl", pagination_mode="none")
    from common import get_index
    from pipeline import content_profiles, resolve, slot_terms
    ix = get_index()
    profiles = content_profiles(ix)
    model = load_model()
    emb, ids = catalog_index(model)
    ids = np.array(ids)

    def lexical(category: str, constraints: list[str], sid: str) -> list[str]:
        agent.reset(sid, {})
        st = agent.sessions[sid]
        st["category"], st["constraints"] = category, constraints
        return list(agent._rank(st, K))

    def dense(text: str) -> list[str]:
        q = model.encode(QUERY_PREFIX + text, convert_to_numpy=True, normalize_embeddings=True)
        return list(ids[np.argsort(emb @ q)[::-1][:K]])

    def category(parse: dict) -> tuple[list[str], float]:
        terms = [parse.get("category_phrase") or "", *slot_terms(parse)]
        top, conf = resolve(terms, ix, profiles, top_n=3)
        members = [a for b in top for a in sorted(ix.buckets[b], key=lambda a: -ix.popularity.get(a, 0.0))]
        return members[:K], conf

    systems = ("lex_raw", "lex_parsed", "dense_raw", "category", "fuse_raw", "fuse_parsed", "fuse3", "fuse3b", "fuse_filtered")

    def sink_contradictions(ranked: list[str], parse: dict) -> list[str]:
        keep, sink = [], []
        for a in ranked:
            (sink if contradictions(parse, ix.products[a], ix.text[a].lower()) else keep).append(a)
        return keep + sink
    rows = []
    for c in todo:
        lr = lexical(c["utterance"], [], c["case_id"] + "r")
        lp = lexical(*parsed_state(parses[c["case_id"]]), c["case_id"] + "p")
        dr = dense(c["utterance"])
        cat, conf = category(parses[c["case_id"]])
        in_cat = set(cat)
        dense_in_cat = [a for a in dr if a in in_cat]     # dense order, restricted to the resolved buckets
        lists = {"lex_raw": lr, "lex_parsed": lp, "dense_raw": dr, "category": cat,
                 "fuse_raw": rrf(lr, dr), "fuse_parsed": rrf(lp, dr),
                 "fuse3": rrf(lp, dr, cat, weights=[1.0, 1.0, conf]),
                 "fuse3b": rrf(lp, dr, dense_in_cat, weights=[1.0, 1.0, conf])}
        lists["fuse_filtered"] = sink_contradictions(lists["fuse3b"],
                                                     clean_parse(parses[c["case_id"]], c["utterance"]))
        rows.append({"case_id": c["case_id"], "style": c["style"], "modifiers": c["modifiers"],
                     "resolver_confidence": round(conf, 3),
                     **{k: rank_of(v, c["asin"]) for k, v in lists.items()}})
    (LAB / "bench" / f"results_fuse{_TAG}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def table(groups: dict, title: str) -> None:
        print(f"\n{title:16s} n   " + "  ".join(f"{s:>11s}" for s in systems) + "   (HitRate@10 / MRR)")
        for g, rs in groups.items():
            print(f"{g:16s} {len(rs):<3d} " + "  ".join(
                f"{hit([r[s] for r in rs]):.2f} / {mrr([r[s] for r in rs]):.2f}" for s in systems))
    table({"all": rows}, "overall")
    by = defaultdict(list)
    for r in rows: by[r["style"]].append(r)
    table(dict(sorted(by.items())), "by style")
    neg = defaultdict(list)
    for r in rows: neg[f"negation={'negation' in r['modifiers']}"].append(r)
    table(dict(sorted(neg.items())), "by negation")
    prods = {r["asin"]: r for r in map(json.loads, (LAB / "bench" / "products.jsonl").open())}
    cov = defaultdict(list)
    for r in rows:
        pr = prods.get(cases[r["case_id"]]["asin"], {})
        cov[f"price_present={bool(pr.get('price_present'))}"].append(r)
        cov[f"for_other={'for_other' in r['modifiers']}"].append(r)
    table(dict(sorted(cov.items())), "by covariate")
    for a, b in (("fuse_raw", "fuse_parsed"), ("fuse_parsed", "fuse3"), ("fuse_parsed", "fuse3b"),
                 ("fuse3b", "fuse_filtered")):
        better = sum(1 for r in rows if (r[b] or 999) < (r[a] or 999))
        worse = sum(1 for r in rows if (r[b] or 999) > (r[a] or 999))
        print(f"{b} vs {a} per case: better {better} / worse {worse} / same {len(rows) - better - worse}")


if __name__ == "__main__":
    main()
