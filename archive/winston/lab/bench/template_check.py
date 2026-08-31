"""Does the bolt-on path hold up on the simulator's own turn-1 messages?

The bolt-on never fires on a recognised template (bolt_on.is_template). This asks
what happens if it did - and what happens when it must: the organizer may add
natural-language paraphrasing, and then the regex misses. Three forms per public
target, each through three systems:

  browsing    "I'm looking for {cat}, but I'm still exploring."
  buying      "I'm looking for {cat}. A key requirement is: {c}."
  paraphrase  "hey, after {cat} - {c} is a must for me"       <- regex misses by design

  template    exp11's regex path (as shipped)
  lexical     the raw message through exp11's own FTS5 (no regex)
  parsed      qwen parse -> parsed_state -> exp11's own _rank  (the bolt-on)

Pass condition: on the two exact forms parsed == template rank case for case; on
the paraphrase parsed recovers what template loses.

    python3 template_check.py [N]        # N public targets, default 30 -> template_check.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
REPO = WINSTON.parents[2]
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON, LAB, BENCH, WINSTON / "experiments", KIT, REPO / "nickolas" / "experiments",
          REPO / "docs" / "archive" / "research_evaluation" / "retrieval" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from score import template_rank, lexical_rank, parsed_rank, PARSER_MODEL   # noqa: E402
from nlp_parse import parse_with_ollama                                    # noqa: E402
from experiment_11_candidate_agent import CleanFTSAgent                    # noqa: E402
from evaluator.local_evaluator import catalog_index, coarse_category, intent_card  # noqa: E402

OUT = BENCH / "template_check.json"
FORMS = ("browsing", "buying", "paraphrase")


def forms_for(cat: str, constraint: str) -> dict[str, str]:
    return {"browsing": f"I'm looking for {cat}, but I'm still exploring.",
            "buying": f"I'm looking for {cat}. A key requirement is: {constraint}.",
            "paraphrase": f"hey, after {cat} - {constraint} is a must for me"}


def hit(rs): return sum(1 for r in rs if r and r <= 10) / len(rs) if rs else 0.0
def mrr(rs): return sum(1 / r for r in rs if r and r <= 10) / len(rs) if rs else 0.0


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    agent = CleanFTSAgent(KIT / "data" / "catalog.jsonl", pagination_mode="none")
    _, cats, products = catalog_index(KIT / "data" / "catalog.jsonl")
    sessions = [json.loads(l) for l in (KIT / "data" / "public_set.jsonl").open()][:n]
    rows, t0 = [], time.time()
    for i, s in enumerate(sessions):
        asin = s["ground_truth"]["parent_asin"]
        cat = coarse_category(cats[asin])
        constraint = intent_card(products[asin])["hard_constraints"][0]
        for form, msg in forms_for(cat, constraint).items():
            parse = parse_with_ollama(msg, PARSER_MODEL)
            rows.append({"i": i, "asin": asin, "form": form, "msg": msg, "cat": cat,
                         "template": template_rank(agent, f"t{i}{form}", msg, asin)[0],
                         "lexical": lexical_rank(agent, f"l{i}{form}", msg, asin),
                         "parsed": parsed_rank(agent, f"p{i}{form}", parse, asin),
                         "phrase": parse.get("category_phrase")})
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{n}  {(time.time() - t0) / (i + 1):.0f}s/target", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))
    report(rows)


def report(rows: list[dict]) -> None:
    n = len({r["i"] for r in rows})
    print(f"\n{n} public targets, turn 1 only:")
    print(f"{'form':11} {'template':>18} {'raw lexical':>18} {'parsed (bolt-on)':>18}")
    for form in FORMS:
        rs = [r for r in rows if r["form"] == form]
        print(f"{form:11} " + " ".join(f"hit {hit([r[k] for r in rs]):.2f} mrr {mrr([r[k] for r in rs]):.2f}".rjust(18)
                                       for k in ("template", "lexical", "parsed")))
    exact = [r for r in rows if r["form"] != "paraphrase"]
    same = sum(1 for r in exact if r["template"] == r["parsed"])
    phrase_ok = sum(1 for r in exact if (r["phrase"] or "").lower().strip() == r["cat"].lower())
    print(f"\nexact forms: parsed rank == regex rank in {same}/{len(exact)}; "
          f"category_phrase == template category in {phrase_ok}/{len(exact)}")
    for r in exact:
        if r["template"] != r["parsed"]:
            print(f"  differs: {r['form']:8} template={r['template']} parsed={r['parsed']} phrase={r['phrase']!r} | {r['msg'][:70]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.loads(OUT.read_text()))
    else:
        main()
