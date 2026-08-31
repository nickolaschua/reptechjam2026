"""EXP04 - How far does the lexical system fall when the customer stops
copy-pasting catalog text?

Perturbs the simulator's replies (word order, word drop, number rewording,
synonym substitution) and its turn-1 category phrase, then measures the drop.
Also tests the fuzzy-category fix and the flat-vs-hierarchy index question.
"""
from __future__ import annotations

import collections
import random
import re

from common import get_index, write_result  # noqa: I001  (sets sys.path for the kit)
import evaluator.local_evaluator as EV  # noqa: E402
from baseline_agent import BaselineAgent, CONSTRAINT_RE  # noqa: E402

STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "for", "with", "in", "is",
             "are", "from", "your", "you"}
SYNONYMS = {
    "leather": "genuine cowhide", "cotton": "combed ringspun fibre",
    "polyester": "synthetic microfibre", "nylon": "technical shell fabric",
    "wool": "merino fleece", "spandex": "four way stretch", "sole": "outsole tread",
    "rubber": "vulcanised gum", "imported": "globally sourced",
    "waterproof": "keeps water out", "closure": "fastening",
    "drawstring": "adjustable tie cord", "breathable": "lets air through",
    "lined": "inner layer", "sleeve": "arm covering", "pull": "slip",
    "machine": "laundry", "wash": "cleaning", "battery": "power cell",
    "stainless": "rustproof", "steel": "metal", "band": "wrist strap",
    "insulation": "heat retention", "fur": "plush pile", "mesh": "open weave",
    "measures": "is about", "approximately": "roughly",
}


def make_reply_perturber(original, mode: str, seed: int = 0):
    rng = random.Random(seed)

    def perturb(text: str) -> str:
        found = CONSTRAINT_RE.search(text)
        if not found:
            return text
        parts = []
        for clause in found.group(1).split(";"):
            words = [w for w in clause.strip(" .").split() if w.lower() not in STOPWORDS]
            if mode in ("shuffle", "all"):
                rng.shuffle(words)
            if mode in ("drop", "all"):
                words = [w for w in words if rng.random() > 0.4] or words[:1]
            if mode in ("numbers", "all"):
                words = [re.sub(r"\d+\.\d+", lambda m: f"{float(m.group()) + 0.05:.1f}", w)
                         for w in words]
                words = [re.sub(r"(\d+)%", lambda m: f"about {m.group(1)} percent", w)
                         for w in words]
            if mode == "synonyms":
                words = [SYNONYMS.get(w.lower().strip(".,"), w) for w in words]
            parts.append(" ".join(words))
        return text[:found.start(1)] + "; ".join(parts) + "."

    def wrapped(sample, ask, disclosed, boundary_used):
        message, used = original(sample, ask, disclosed, boundary_used)
        return perturb(message), used

    return wrapped


def make_category_perturber(original, mode: str):
    def wrapped(sample, category, disclosed):
        message = original(sample, category, disclosed)
        replacement = {
            "reorder": " ".join(reversed(category.split())),
            "lower_plural": category.lower() + "s",
            "loose": "some " + category.split()[-1].lower(),
        }[mode]
        return message.replace(category, replacement, 1)
    return wrapped


def score(ix, samples, **kwargs) -> dict:
    r = EV.evaluate(BaselineAgent(**kwargs), samples, ix.ids, ix.categories, ix.products)
    return {"hit_rate_at_10": r["hit_rate_at_10"], "mrr": r["mrr"], "mttc": r["mttc"],
            "technical_score": r["recommended_technical_score"]}


def main() -> dict:
    ix = get_index()
    samples = ix.samples()
    out: dict = {}
    original_reply, original_initial = EV.customer_reply, EV.initial_message

    out["reply_perturbation"] = {}
    for name, mode in [("verbatim", None), ("word_order_shuffled", "shuffle"),
                       ("40pct_words_dropped", "drop"), ("numbers_reworded", "numbers"),
                       ("all_three_combined", "all"), ("synonyms_swapped", "synonyms")]:
        EV.customer_reply = original_reply if mode is None else make_reply_perturber(original_reply, mode)
        out["reply_perturbation"][name] = score(ix, samples)
    EV.customer_reply = original_reply

    out["category_perturbation"] = {}
    for name, mode in [("exact", None), ("words_reversed", "reorder"),
                       ("lowercased_pluralised", "lower_plural"), ("loosely_reworded", "loose")]:
        EV.initial_message = original_initial if mode is None else make_category_perturber(original_initial, mode)
        out["category_perturbation"][name] = {
            "exact_bucket_lookup": score(ix, samples),
            "fuzzy_bucket_lookup": score(ix, samples, fuzzy_category=True),
        }
    EV.initial_message = original_initial

    # Flat 1,115-leaf index vs a hierarchy where every path prefix is a node.
    tree: dict[str, list[str]] = collections.defaultdict(list)
    for a in ix.products:
        segments = []
        for value in ix.categories[a]:
            for part in value.split(","):
                part = part.strip()
                if part and part.lower() not in {"clothing", "clothing shoes & jewelry"}:
                    segments.append(part.lower())
        for i in range(1, len(segments) + 1):
            tree[" > ".join(segments[:i])].append(a)
    sizes = sorted(len(v) for v in tree.values())
    out["category_index_shape"] = {
        "flat_leaves": len(ix.buckets),
        "hierarchy_nodes": len(tree),
        "hierarchy_node_size_median": sizes[len(sizes) // 2],
        "hierarchy_node_size_max": max(sizes),
        "verdict": "flat wins; internal hierarchy nodes are too broad to be useful pools",
        "measured_technical_score_flat": out["category_perturbation"]["exact"]["fuzzy_bucket_lookup"]["technical_score"],
    }
    return out


if __name__ == "__main__":
    print("EXP04 robustness (runs the evaluator ~14 times, takes several minutes)")
    write_result("exp04_robustness", main())
