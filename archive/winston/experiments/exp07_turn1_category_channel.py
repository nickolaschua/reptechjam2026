"""EXP07 - The turn-1 message hands you the target's bucket string verbatim.

`local_evaluator.initial_message` builds turn 1 as
    "I'm looking for {coarse_category(target)}. ..."
so the category route does not need to *infer* anything on turn 1 - it needs to
parse a carrier phrase and do a dict lookup. This measures how much of the work
that free channel actually does, and what is left over for an encoder or an LLM.

Answers the whiteboard question "Category - cut @ 2 buckets?": yes, but the cut
is the last two path segments (what the evaluator does), not L1+L2 from the root.
"""
from __future__ import annotations

import collections
import json
import re
import statistics

from common import KIT, get_index, write_result
from evaluator.local_evaluator import (
    coarse_category, intent_card, initial_message, classify_constraint,
)

CARRIER = re.compile(r"^i'm looking for (.+?)(?:,\s*but i'm still exploring)?\.?\s*(?:a key requirement is:\s*(.*?)\.?)?$",
                     re.I | re.DOTALL)


def parse_turn1(message: str) -> tuple[str | None, str | None]:
    """Split the simulator's turn-1 grammar into (category_phrase, constraint)."""
    head, _, tail = message.partition(". ")
    m = re.match(r"i'm looking for (.+?)(?:, but i'm still exploring)?\.?$", head.strip(), re.I)
    if not m:
        return None, None
    category = m.group(1).strip().rstrip(",")
    constraint = tail.strip() or None
    if constraint and constraint.lower().startswith("a key requirement is:"):
        constraint = constraint.split(":", 1)[1].strip().rstrip(".")
    return category, constraint


def main() -> dict:
    ix = get_index()
    samples = [json.loads(l) for l in (KIT / "data" / "public_set.jsonl").open()]
    buckets = {coarse_category(ix.categories[a]).lower(): None for a in ix.products}

    exact = lexical_top1 = lexical_top3 = 0
    target_in_bucket = 0
    pool_sizes, ranks = [], []
    constraint_kinds: collections.Counter = collections.Counter()
    misses = []

    for s in samples:
        target = s["ground_truth"]["parent_asin"]
        true_bucket = coarse_category(ix.categories[target])
        card = intent_card(ix.products[target])
        behavior = {"scenario_type": s["scenario_type"]}
        if s["scenario_type"] == "intent_override":
            behavior["override"] = {
                "turn": 3,
                "old_value": (card["soft_preferences"] or ["x"])[-1],
                "new_value": card["hard_constraints"][0],
                "message": "",
            }
        msg = initial_message({**s, "intent_card": card, "behavior": behavior},
                              true_bucket, set())

        phrase, constraint = parse_turn1(msg)
        if constraint:
            constraint_kinds[classify_constraint(constraint)] += 1

        # channel A: parse the carrier phrase, look the string up directly
        if phrase is not None and phrase.lower() in buckets:
            exact += 1
            pool_sizes.append(len(ix.buckets[phrase.lower()]))
            if target in ix.buckets[phrase.lower()]:
                target_in_bucket += 1
        else:
            misses.append({"sample_id": s["sample_id"], "phrase": phrase, "true_bucket": true_bucket})

        # channel B: what IDF resolution alone would have done with the same text
        ranked = ix.resolve_bucket(phrase or msg, top_n=len(ix.buckets))
        key = true_bucket.lower()
        rank = ranked.index(key) + 1 if key in ranked else None
        if rank:
            ranks.append(rank)
            lexical_top1 += rank == 1
            lexical_top3 += rank <= 3

    n = len(samples)
    return {
        "note": "turn-1 messages reconstructed with the evaluator's own initial_message()",
        "sessions": n,
        "carrier_phrase_parsed": n - sum(1 for m in misses if m["phrase"] is None),
        "exact_bucket_lookup": {
            "hit_rate": round(exact / n, 4),
            "target_inside_that_bucket": round(target_in_bucket / n, 4),
            "median_pool_size": statistics.median(pool_sizes) if pool_sizes else None,
            "max_pool_size": max(pool_sizes) if pool_sizes else None,
        },
        "idf_resolution_on_same_text": {
            "top1": round(lexical_top1 / n, 4),
            "top3": round(lexical_top3 / n, 4),
            "median_rank": statistics.median(ranks) if ranks else None,
        },
        "constraint_kinds": dict(constraint_kinds.most_common()),
        "misses": misses[:20],
        "catalog_granularity": {
            "full_leaf_paths": 1628,
            "evaluator_last2_buckets": len(ix.buckets),
            "leaf_labels_only": 800,
        },
        "verdict": (
            "Turn 1 contains the target's coarse_category string verbatim, so the "
            "category route is a regex plus a dict lookup - not an inference problem. "
            "Spend the LLM/encoder budget on turns 2+ and on the private set's "
            "paraphrase risk (EXP06), not on turn-1 category resolution."),
    }


if __name__ == "__main__":
    print("EXP07 turn-1 category channel")
    write_result("exp07_turn1_category_channel", main())
