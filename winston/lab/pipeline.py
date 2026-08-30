"""Parse -> Resolve -> Fuse. The five stages, with the plug points marked.

Stages 1 and 2 run on the standard library plus winston/experiments/common.py.
Stages 3-5 need nickolas' harness (numpy, scipy, scikit-learn, sentence-transformers,
torch) and are imported lazily, so you can work on the front half without a 2GB
download. See ARCHITECTURE.md for what each stage is for and who owns it.

    python3 pipeline.py            # stage 2 experiment: does slot evidence help?
"""
from __future__ import annotations

import collections
import json
import pickle
import statistics
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
WINSTON = LAB.parent
sys.path.insert(0, str(WINSTON / "experiments"))
sys.path.insert(0, str(WINSTON))

from common import TOKEN_RE, get_index  # noqa: E402


# ---------------------------------------------------------------- stage 1: PARSE
# Owner: winston/nlp_parse.py. Already built and measured (slot F1 0.333).
# Contract: message -> dict with category_phrase, department, slots[], price_*,
# quality_prior, exploring. Nothing downstream reads the raw message again.
#
# The template regex from exp07 stays as the fast path: when the message matches
# the simulator's grammar the bucket is handed over verbatim (200/200), so the
# LLM only fires on input the regex cannot parse. Free accuracy, no tokens.


def slot_terms(parse: dict, stances: tuple[str, ...] = ("soft",)) -> list[str]:
    """Slot values worth using as category evidence.

    Probe case 02 is the whole argument for this function: the parser emitted
    category_phrase="shoe" but also five use_case slots - lifting weights,
    running, racket sports, all purpose usage, sports. "Cross-training" IS that
    conjunction. The evidence was captured correctly and then thrown away
    because only category_phrase reached the resolver.
    """
    from nlp_parse import clean_slots, tier_of

    return [s["value"] for s in clean_slots(parse)
            if not s.get("declined") and not s.get("negated") and tier_of(s) in stances]


# -------------------------------------------------------------- stage 2: RESOLVE
# Owner: this file. THE priority - strata A and C fail here, not in retrieval.

_PROFILE_CACHE = LAB / ".cache"


def content_profiles(ix) -> dict[str, dict[str, float]]:
    """For each bucket, how often each title token appears inside it.

    Matching a query against the bucket LABEL fails whenever the label is
    generic ("women shoes") or a euphemism ("accessories breast petals").
    Matching against what the products inside actually say fixes those.
    exp06 measured label-only top3 0.392 vs combined 0.490.
    """
    _PROFILE_CACHE.mkdir(exist_ok=True)
    cached = _PROFILE_CACHE / "content_profiles.pkl"
    if cached.exists():
        with cached.open("rb") as fh:
            return pickle.load(fh)

    def stem(token: str) -> str:
        return token[:-1] if len(token) > 3 and token.endswith("s") else token

    profiles: dict[str, dict[str, float]] = {}
    for key, members in ix.buckets.items():
        counts: collections.Counter = collections.Counter()
        for asin in members:
            counts.update(set(TOKEN_RE.findall(ix.fields[asin]["title"])))
        n = len(members)
        profile: dict[str, float] = {}
        for token, count in counts.items():
            if count >= max(2, 0.02 * n):
                key_t = stem(token)
                profile[key_t] = max(profile.get(key_t, 0.0), count / n)
        profiles[key] = profile
    with cached.open("wb") as fh:
        pickle.dump(profiles, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return profiles


def resolve(query_terms: list[str], ix, profiles: dict, top_n: int = 3,
            w_label: float = 1.0, w_content: float = 1.0) -> tuple[list[str], float]:
    """Rank all 1,115 buckets against the query. Returns (ranked, confidence).

    confidence is the relative margin between the top two buckets. It is the
    signal stage 5 uses to decide whether to trust the category cut or widen
    the pool and ask a question - a flat top-2 means the resolver is guessing.
    """
    def stem(token: str) -> str:
        return token[:-1] if len(token) > 3 and token.endswith("s") else token

    q = [stem(t) for t in TOKEN_RE.findall(" ".join(query_terms).lower())]
    if not q:
        return [], 0.0

    def idf(token: str) -> float:
        return ix.idf.get(token, ix.idf.get(token + "s", 4.0))

    label_tokens = {k: {stem(t) for t in ix.bucket_tokens[k]} for k in ix.buckets}
    scored = []
    for key in ix.buckets:
        # the denominator penalises buckets whose label carries a lot of words
        # the query never mentioned, so "women shoes" does not beat
        # "athletic fitness & cross-training" just by being short and common
        label = sum(idf(t) for t in q if t in label_tokens[key]) / (
            1 + 0.15 * len(label_tokens[key] - set(q)))
        content = sum(profiles.get(key, {}).get(t, 0.0) * idf(t) for t in q)
        scored.append((w_label * label + w_content * content, key))
    scored.sort(reverse=True)

    top = scored[0][0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    confidence = 0.0 if top <= 0 else (top - runner_up) / top
    return [k for _, k in scored[:top_n]], confidence


# ------------------------------------------------- stages 3-5: RETRIEVE/AGREE/RANK
# Owner: open. These need nickolas' harness; import it lazily so stages 1-2 stay
# dependency-free. Everything below is wiring, not new retrieval code.
#
#   from nickolas.experiments.harness import Harness, replay_policy
#   sessions, metrics = replay_policy(harness, ranker)
#
# `ranker(state) -> (indices, scores)` is the entire contract. Write one function
# and the harness handles session replay, metrics, and the frozen split.


def agreement(lexical_ids, dense_ids, depth: int = 50) -> float:
    """Overlap between the two routes' top-N. The disagreement signal.

    exp08 measured that lexical scores higher but degrades 0.25-0.28 under
    paraphrase while dense degrades 0.06-0.12. Neither dominates, so the useful
    quantity is not "which route" but "do they agree" - low overlap means the
    query is vague or paraphrased, which is exactly when to widen and ask.
    """
    a, b = set(lexical_ids[:depth]), set(dense_ids[:depth])
    return len(a & b) / depth if depth else 0.0


def make_ranker(harness, parse_fn, ix, profiles):
    """Plug into replay_policy. Stub - stage 3/4/5 owner fills this in."""
    raise NotImplementedError(
        "Stage 3-5. Contract: ranker(state) -> (indices, scores).\n"
        "  1. parse   = parse_fn(state.message)\n"
        "  2. buckets, conf = resolve(...)     restrict or boost by bucket\n"
        "  3. lex = harness.lexical.ranked(q); dns = harness.dense.ranked(q)\n"
        "  4. agr = agreement(lex[0], dns[0])\n"
        "  5. fuse by RRF, weight by agr and conf, add popularity prior\n"
        "See ARCHITECTURE.md section 5.")


# ------------------------------------------------------------------- stage 2 test

def evaluate_resolver() -> dict:
    """Does adding slot evidence recover buckets that category_phrase alone misses?

    This is the experiment the lab exists to settle, and it needs no new
    dependencies. Ground truth: each probe case names a real ASIN, so its true
    bucket is known by construction.
    """
    gold_path = WINSTON / "probe_gold.json"
    pred_path = WINSTON / "preds-qwen2.5-7b-instruct.json"
    if not (gold_path.exists() and pred_path.exists()):
        raise SystemExit("run `python3 ../nlp_parse.py --model <m>` first")

    gold = {c["case"]: c for c in json.loads(gold_path.read_text())}
    preds = {p["case"]: p["pred"] for p in json.loads(pred_path.read_text())}
    ix = get_index()
    profiles = content_profiles(ix)

    arms = {
        "A category_phrase only        ": lambda p: [p["category_phrase"]],
        "B phrase + soft slots         ": lambda p: [p["category_phrase"], *slot_terms(p)],
        "C phrase + soft + hard slots  ": lambda p: [p["category_phrase"],
                                                    *slot_terms(p, ("soft", "hard"))],
        "D slots only (no phrase)      ": lambda p: slot_terms(p, ("soft", "hard")),
    }

    results: dict[str, dict] = {}
    for name, build in arms.items():
        ranks, top1, top3, pools = [], 0, 0, []
        for case, c in gold.items():
            if case not in preds:
                continue
            true_bucket = ix.bucket_of[c["asin"]]
            ranked, _ = resolve(build(preds[case]), ix, profiles, top_n=len(ix.buckets))
            rank = ranked.index(true_bucket) + 1 if true_bucket in ranked else None
            if rank:
                ranks.append(rank)
                top1 += rank == 1
                top3 += rank <= 3
            pools.append(len({a for b in ranked[:3] for a in ix.buckets[b]}))
        n = len(pools)
        results[name.strip()] = {
            "top1": round(top1 / n, 3), "top3": round(top3 / n, 3),
            "median_rank": statistics.median(ranks) if ranks else None,
            "median_pool_at_top3": statistics.median(pools),
        }
        print(f"  {name} top1={top1/n:.3f}  top3={top3/n:.3f}  "
              f"median_rank={statistics.median(ranks) if ranks else '-':>5}  "
              f"pool@3={statistics.median(pools):.0f}")
    return results


def main() -> None:
    print("Stage 2 - category resolution from parser output")
    print("  30 probe cases, true bucket known from the target ASIN\n")
    results = evaluate_resolver()

    out = LAB / "resolver_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n  -> {out.name}")

    best = max(results, key=lambda k: (results[k]["top3"], results[k]["top1"]))
    base = results["A category_phrase only"]
    print(f"\n  best arm: {best}")
    print(f"  top3 {base['top3']:.3f} -> {results[best]['top3']:.3f}")


if __name__ == "__main__":
    main()
