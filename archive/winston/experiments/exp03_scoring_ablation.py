"""EXP03 - Which scoring components actually earn their place?

Ablates the popularity prior, IDF lexical matching, the patience policy, field
weighting, and the previously-unused dataset signals (preference_tags, price
presence, feature count). Every row is a full run of the official evaluator.
"""
from __future__ import annotations

import collections
import math
import statistics

from baseline_agent import BaselineAgent
from common import TOKEN_RE, get_index, intent_card, write_result
from evaluator.local_evaluator import evaluate


def run(ix, samples, **kwargs) -> dict:
    result = evaluate(BaselineAgent(**kwargs), samples, ix.ids, ix.categories, ix.products)
    return {
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "efficiency": result["efficiency"],
        "technical_score": result["recommended_technical_score"],
        "by_scenario": {k: {"hit_rate_at_10": v["hit_rate_at_10"], "mrr": v["mrr"],
                            "mttc": v["mttc"]} for k, v in result["scenario_metrics"].items()},
        "rank_histogram": dict(sorted(collections.Counter(
            s["best_rank"] for s in result["sessions"] if s["best_rank"]).items())),
    }


def main() -> dict:
    ix = get_index()
    samples = ix.samples()
    out: dict = {"published_baseline_weak_bm25": {"hit_rate_at_10": 0.125, "mrr": 0.068034,
                                                  "mttc": 9.81, "technical_score": 0.10671}}

    out["ablations"] = {
        "popularity_only_no_nlp": run(ix, samples, use_lexical=False, patience=False),
        "lexical_only_no_prior": run(ix, samples, use_popularity=False, patience=False),
        "lexical_plus_prior": run(ix, samples, patience=False),
        "lexical_plus_prior_plus_patience": run(ix, samples),
        "prior_weight_3x": run(ix, samples, popularity_weight=3.0, patience=False),
        "no_partial_credit": run(ix, samples, partial_credit=0.0),
    }

    # Previously unused dataset signals.
    out["unused_signals"] = {
        "preference_tags_w0.5": run(ix, samples, tag_weight=0.5),
        "preference_tags_w1.0": run(ix, samples, tag_weight=1.0),
    }

    # Are those signals predictive in isolation, even if they do not help the score?
    docs_text = ix.text
    tags = collections.Counter(t for s in samples for t in s["user_profile"].get("preference_tags", []))
    lift: dict = {}
    for tag, n in tags.most_common():
        word = tag.split()[0]
        base = sum(1 for t in docs_text.values() if word in t) / len(docs_text)
        group = [s for s in samples if tag in s["user_profile"].get("preference_tags", [])]
        hit = sum(1 for s in group if word in docs_text[s["ground_truth"]["parent_asin"]]) / len(group)
        lift[tag] = {"n_sessions": n, "catalog_rate": round(base, 4),
                     "target_rate": round(hit, 4), "lift": round(hit / base, 2) if base else None}
    out["preference_tag_lift"] = lift

    targets = [s["ground_truth"]["parent_asin"] for s in samples]
    def compare(fn):
        cat = [fn(p) for p in ix.products.values() if fn(p) is not None]
        tgt = [fn(ix.products[a]) for a in targets if fn(ix.products[a]) is not None]
        ordered = sorted(cat)
        pct = 100 * sum(1 for x in ordered if x < statistics.median(tgt)) / len(ordered)
        return {"catalog_median": round(statistics.median(cat), 2),
                "target_median": round(statistics.median(tgt), 2),
                "target_percentile": round(pct), "target_coverage": len(tgt)}
    out["target_vs_catalog"] = {
        "rating_number": compare(lambda p: p.get("rating_number")),
        "average_rating": compare(lambda p: p.get("average_rating")),
        "price": compare(lambda p: p.get("price") if isinstance(p.get("price"), (int, float)) else None),
    }

    # Field weighting, scored offline against the full card (no evaluator loop).
    cases = []
    for s in samples:
        t = s["ground_truth"]["parent_asin"]
        card = intent_card(ix.products[t])
        cases.append((t, ix.buckets[ix.bucket_of[t]],
                      card["hard_constraints"] + card["soft_preferences"]))

    def field_weighted(weights: dict[str, float], w_pop: float = 1.0) -> dict:
        rr = 0.0
        rank1 = 0
        for target, pool, cons in cases:
            prepared = [(c.lower(), [t for t in TOKEN_RE.findall(c.lower()) if t in ix.idf])
                        for c in cons]
            scored = []
            for a in pool:
                total = w_pop * ix.popularity[a]
                for lowered, terms in prepared:
                    if not terms:
                        continue
                    weight = sum(ix.idf[t] for t in terms)
                    best = 0.0
                    for field, fw in weights.items():
                        if not fw:
                            continue
                        if lowered in ix.fields[a][field]:
                            best = max(best, fw * weight)
                        else:
                            overlap = sum(ix.idf[t] for t in terms
                                          if t in set(TOKEN_RE.findall(ix.fields[a][field])))
                            if overlap:
                                best = max(best, fw * 0.3 * overlap)
                    total += best
                scored.append((total, a))
            scored.sort(reverse=True)
            rank = [a for _, a in scored].index(target) + 1
            rr += 1 / rank
            rank1 += rank == 1
        return {"mrr_offline": round(rr / len(cases), 4), "rank1_rate": round(rank1 / len(cases), 4)}

    flat = {f: 1.0 for f in ("title", "categories", "store", "features", "description", "details")}
    out["field_weighting"] = {
        "flat": field_weighted(flat),
        "title_3x": field_weighted({**flat, "title": 3.0}),
        "fts5_weights": field_weighted({"title": 6, "categories": 4, "store": 1.5,
                                        "features": 2.5, "description": 1.0, "details": 2.5}),
        "title3_feat2_det2": field_weighted({**flat, "title": 3.0, "features": 2.0, "details": 2.0}),
        "drop_description": field_weighted({**flat, "title": 3.0, "features": 2.0,
                                            "details": 2.0, "description": 0.0}),
    }
    return out


if __name__ == "__main__":
    print("EXP03 scoring ablation (runs the evaluator ~8 times, takes a few minutes)")
    write_result("exp03_scoring_ablation", main())
