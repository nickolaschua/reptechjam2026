"""Spec section 6: where does the target rank, for each of three systems?

  template   exp11 as-is. Its turn-1 parser is regex-only, so a messy message
             leaves the query empty. Expected near zero; documents the limitation.
             Also records ask_attribute and question_hit: did the first question
             target an attribute the shopper's hidden intent card actually has?
  lexical    exp11's own FTS5 + reranker with the raw utterance as the query.
             The FAIR baseline - same retrieval, no template dependence.
  resolver   nlp_parse -> pipeline.resolve. Rank of the TRUE BUCKET among 1,115.
             Needs Ollama (qwen); parses are cached to parses.jsonl.

    python3 score.py --skip-resolver     # fast: the two lexical systems only
    python3 score.py                     # everything, resumable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
REPO = WINSTON.parents[2]
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON / "experiments", WINSTON, LAB, BENCH, KIT, REPO / "nickolas" / "experiments",
          REPO / "docs" / "archive" / "research_evaluation" / "retrieval" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CASES_PATH = BENCH / "cases.jsonl"
RESULTS_PATH = BENCH / "results.jsonl"
PARSES_PATH = BENCH / "parses.jsonl"
PARSER_MODEL = "qwen2.5:7b-instruct"
TOP_K = 50


def rank_of(ranked: list[str], target: str) -> int | None:
    try:
        return ranked.index(target) + 1
    except ValueError:
        return None


def resolver_query(parse: dict) -> list[str]:
    """category_phrase + soft slots - the arm that won in pipeline.py (B)."""
    from nlp_parse import clean_slots, tier_of
    terms = [parse.get("category_phrase") or ""]
    terms += [s["value"] for s in clean_slots(parse) if tier_of(s) == "soft" and not s.get("negated")]
    return [t for t in terms if t]


def parsed_state(parse: dict) -> tuple[str, list[str]]:
    """Map a parse onto exp11's own {category, constraints} state. The parse replaces
    the template regex; retrieval is exp11's untouched _rank. Negated, declined and
    junk slots are dropped; price is not a term. The parsed department is NOT a
    term unless the parser kept it, which clean_department() only allows when the
    utterance names a gender, recipient or age group."""
    from nlp_parse import clean_slots, tier_of
    # exp11 tokenises "women's" to "women"; the catalog's categories say Women/Men/Girls/Boys/Baby.
    # Safe only because clean_department() has vetoed departments the user never stated.
    dept = _DEPT_TERM.get(parse.get("department") or "")
    category = " ".join(t for t in (dept, parse.get("category_phrase")) if t)
    constraints = [s["value"] for s in clean_slots(parse)
                   if tier_of(s) != "decline" and not s.get("negated")]
    return category or "clothing item", constraints


_DEPT_TERM = {"womens": "women", "mens": "men", "girls": "girls", "boys": "boys",
              "baby-girls": "baby", "baby-boys": "baby"}


def parsed_rank(agent, sid: str, parse: dict, asin: str) -> int | None:
    agent.reset(sid, {})
    state = agent.sessions[sid]
    state["category"], state["constraints"] = parsed_state(parse)
    return rank_of(agent._rank(state, TOP_K), asin)


def specificity_counts(parse: dict) -> tuple[int, int]:
    """(n_hard, n_soft): the parse-side specificity axis. Declined slots count as neither."""
    from nlp_parse import clean_slots, tier_of
    tiers = [tier_of(s) for s in clean_slots(parse)]
    return tiers.count("hard"), tiers.count("soft")


def card_hard_said(utterance: str, product: dict) -> int:
    """How many of the intent card's hard constraints the shopper actually voiced.
    A free specificity label grounded in the utterance, not the style prompt."""
    # ponytail: content-word overlap, no stemming; good enough to stratify, not to grade
    from evaluator.local_evaluator import intent_card
    from prompts import content_words
    said = set(content_words(utterance))
    return sum(1 for c in intent_card(product)["hard_constraints"] if set(content_words(str(c))) & said)


def template_rank(agent, sid: str, utterance: str, asin: str) -> tuple[int | None, str | None]:
    """(rank, ask_attribute) from exp11 as shipped."""
    agent.reset(sid, {})
    resp = agent.respond(sid, utterance, 1, TOP_K)
    return rank_of([r["parent_asin"] for r in resp["recommendations"]], asin), resp.get("ask_attribute")


def question_hit(ask_attribute: str | None, product: dict) -> bool:
    """Would this question have elicited a real disclosure from the simulator?"""
    from evaluator.local_evaluator import intent_card, classify_constraint
    if not ask_attribute:
        return False
    return ask_attribute in {classify_constraint(str(c)) for c in intent_card(product).get("hard_constraints", [])}


def lexical_rank(agent, sid: str, utterance: str, asin: str) -> int | None:
    """Bypass the template regex: hand the utterance to exp11's own query path."""
    agent.reset(sid, {})
    state = agent.sessions[sid]
    state["category"] = utterance
    return rank_of(agent._rank(state, TOP_K), asin)


def load_parses() -> dict[str, dict]:
    if not PARSES_PATH.exists():
        return {}
    return {json.loads(l)["case_id"]: json.loads(l)["parse"] for l in PARSES_PATH.open() if l.strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-resolver", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cases", type=Path, default=CASES_PATH)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    ap.add_argument("--reparse-ids", type=Path, help="file of case_ids to parse afresh (ignores "
                    "their cached parse and prior result row); readers keep the LAST row per case")
    args = ap.parse_args()

    from common import get_index
    from experiment_11_candidate_agent import CleanFTSAgent
    ix = get_index()
    agent = CleanFTSAgent(KIT / "data" / "catalog.jsonl", pagination_mode="none")
    print(f"exp11 index built in {agent.index_build_seconds}s")

    profiles = None
    resolve = None
    parse_fn = None
    if not args.skip_resolver:
        from pipeline import content_profiles, resolve as _resolve
        from nlp_parse import parse_with_ollama, clean_department
        profiles = content_profiles(ix)
        resolve = _resolve
        parse_fn = parse_with_ollama

    cases = [json.loads(l) for l in args.cases.open() if l.strip()][:args.limit]
    done = {json.loads(l)["case_id"] for l in args.out.open()} if args.out.exists() else set()
    parses = load_parses()
    if args.reparse_ids:
        wanted = {l.strip() for l in args.reparse_ids.open() if l.strip()}
        todo = [c for c in cases if c["case_id"] in wanted]
        for c in todo:
            parses.pop(c["case_id"], None)
    else:
        todo = [c for c in cases if c["case_id"] not in done]
    print(f"cases {len(cases)} | scored {len(done)} | this run {len(todo)}")

    t0 = time.time()
    import contextlib
    with args.out.open("a") as out, (contextlib.nullcontext() if args.skip_resolver else PARSES_PATH.open("a")) as pf:
        for i, c in enumerate(todo, 1):
            sid = c["case_id"]
            product = ix.products[c["asin"]]
            t_rank, asked = template_rank(agent, sid, c["utterance"], c["asin"])
            row = {"case_id": sid, "asin": c["asin"],
                   "template_rank": t_rank,
                   "ask_attribute": asked,
                   "question_hit": question_hit(asked, product),
                   "lexical_rank": lexical_rank(agent, sid, c["utterance"], c["asin"]),
                   "card_hard_said": card_hard_said(c["utterance"], product)}
            if not args.skip_resolver:
                parse = parses.get(sid)
                if parse is None:
                    parse = parse_fn(c["utterance"], PARSER_MODEL)
                    pf.write(json.dumps({"case_id": sid, "parse": parse}) + "\n")
                    pf.flush()
                parse = clean_department(parse, c["utterance"])   # cached parses predate the gate
                ranked, conf = resolve(resolver_query(parse), ix, profiles, top_n=len(ix.buckets))
                row["bucket_rank"] = rank_of(ranked, ix.bucket_of[c["asin"]])
                row["resolver_confidence"] = round(conf, 3)
                row["category_phrase"] = parse.get("category_phrase")
                row["n_hard"], row["n_soft"] = specificity_counts(parse)
                row["n_negated"] = sum(1 for s in parse.get("slots", []) if s.get("negated"))
                row["price_stated"] = parse.get("price_max") is not None or parse.get("price_min") is not None
                row["specificity"] = parse.get("specificity")
                row["parsed_rank"] = parsed_rank(agent, sid + "p", parse, c["asin"])
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.1f}s/case", flush=True)


if __name__ == "__main__":
    main()
