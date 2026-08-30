"""EXP02 - How much does the public simulator leak?

Answers: whether ask_attribute="other" dominates every named attribute, how many
turns to extract the full intent card, whether intent_card() can be inverted to
identify the target outright, and how discriminative the leaked constraints are.
"""
from __future__ import annotations

import collections
import random
import statistics

from common import N_CATALOG, coarse_category, get_index, intent_card, write_result
from evaluator.local_evaluator import (
    behavior_for, classify_constraint, customer_reply, initial_message,
)

BUCKET_CYCLE = ["material", "feature", "color", "style", "size",
                "use_case", "budget", "brand", "category"]


def turns_to_full_card(ix, sample: dict, policy) -> int | None:
    target = sample["ground_truth"]["parent_asin"]
    card = intent_card(ix.products[target])
    seed = f"{sample['sample_id']}\0{sample['scenario_type']}"
    behavior = behavior_for(sample["scenario_type"], card, random.Random(seed))
    effective = {**sample, "intent_card": card, "behavior": behavior}
    full = set(card["hard_constraints"] + card["soft_preferences"])
    disclosed: set[str] = set()
    boundary_used = False
    initial_message(effective, coarse_category(ix.categories.get(target, [])), disclosed)
    for turn in range(2, 11):
        _, boundary_used = customer_reply(effective, policy(turn), disclosed, boundary_used)
        if disclosed >= full:
            return turn
    return None


def main() -> dict:
    ix = get_index()
    samples = ix.samples()
    out: dict = {}

    policies = {
        "always_other": lambda t: "other",
        "cycle_named_attributes": lambda t: BUCKET_CYCLE[(t - 2) % len(BUCKET_CYCLE)],
    }
    out["turns_to_full_card"] = {}
    for name, policy in policies.items():
        turns = [turns_to_full_card(ix, s, policy) for s in samples]
        done = [t for t in turns if t is not None]
        out["turns_to_full_card"][name] = {
            "median_turn": statistics.median(done) if done else None,
            "max_turn": max(done) if done else None,
            "never_extracted": sum(1 for t in turns if t is None),
            "distribution": dict(sorted(collections.Counter(
                t if t is not None else "never" for t in turns).items(), key=lambda x: str(x[0]))),
        }

    # intent_card() is a pure function of catalog metadata -> invertible.
    card_of = {
        a: frozenset(x.lower() for x in
                     (lambda c: c["hard_constraints"] + c["soft_preferences"])(intent_card(p)))
        for a, p in ix.products.items()
    }
    reverse: dict[frozenset, list[str]] = collections.defaultdict(list)
    for a, fs in card_of.items():
        reverse[fs].append(a)
    pools = []
    inside = unique = within_ten = 0
    for s in samples:
        t = s["ground_truth"]["parent_asin"]
        pool = reverse[card_of[t]]
        pools.append(len(pool))
        inside += t in pool
        unique += len(pool) == 1
        within_ten += len(pool) <= 10
    ordered = sorted(pools)
    out["card_reconstruction"] = {
        "target_inside_own_pool": f"{inside}/{len(samples)}",
        "pool_size_median": statistics.median(pools),
        "pool_size_p90": ordered[int(0.9 * len(ordered))],
        "pool_size_max": max(pools),
        "uniquely_identified": unique,
        "within_top_10": within_ten,
        "note": "private set ships precomputed intent_card; this inversion may not transfer",
    }

    # How discriminative is each leaked constraint?
    docs = list(ix.text.values())
    freqs = []
    for s in samples:
        t = s["ground_truth"]["parent_asin"]
        card = intent_card(ix.products[t])
        for v in card["hard_constraints"] + card["soft_preferences"]:
            freqs.append(sum(1 for d in docs if v.lower() in d))
    ordered = sorted(freqs)
    out["constraint_document_frequency"] = {
        "n_constraints": len(freqs),
        "median_df": statistics.median(freqs),
        "unique_df_1": sum(1 for f in freqs if f == 1),
        "near_unique_df_le_5": sum(1 for f in freqs if f <= 5),
        "boilerplate_df_gt_1000": sum(1 for f in freqs if f > 1000),
        "max_df": max(freqs),
    }

    buckets = collections.Counter()
    for s in samples:
        card = intent_card(ix.products[s["ground_truth"]["parent_asin"]])
        for v in card["hard_constraints"] + card["soft_preferences"]:
            buckets[classify_constraint(v)] += 1
    out["constraint_attribute_routing"] = dict(buckets.most_common())
    return out


if __name__ == "__main__":
    print("EXP02 simulator leakage")
    write_result("exp02_simulator_leakage", main())
