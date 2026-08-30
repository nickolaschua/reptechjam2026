"""Aggregate probe-set score for every cached parser run, raw and through
bolt_on.clean_parse. Ollama-free: reads winston/preds-*.json.

    python3 compare_parsers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
WINSTON = LAB.parent
sys.path.insert(0, str(WINSTON)); sys.path.insert(0, str(LAB))
from nlp_parse import score  # noqa: E402
from bolt_on import clean_parse, intent_of, message_type_of  # noqa: E402


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda k: round(sum(float(r[k]) for r in rows) / n, 3)  # noqa: E731
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["stratum"], []).append(r["slot_f1"])
    return {"n": n, "f1": mean("slot_f1"), "P": mean("slot_precision"), "R": mean("slot_recall"),
            "dept": mean("department_ok"), "cat": mean("category_overlap"), "price": mean("price_ok"),
            "spurious": sum(r["spurious_slots"] for r in rows),
            "strata": {k: round(sum(v) / len(v), 2) for k, v in sorted(by.items())}}


def main() -> None:
    gold = {c["case"]: c for c in json.loads((WINSTON / "probe_gold.json").read_text())}
    for path in sorted(WINSTON.glob("preds-*.json")):
        preds = {p["case"]: p["pred"] for p in json.loads(path.read_text())}
        print(f"\n{path.name}  ({len(preds)} cached)")
        for label, fix in (("raw          ", lambda p, u: p), ("+ clean_parse", clean_parse)):
            rows = []
            for k, c in gold.items():
                if k not in preds:
                    continue
                pred = fix(preds[k], c["utterance"])
                rows.append({**score(pred, c["gold"], c["discard_spans"], c["utterance"]),
                             "stratum": c["stratum"]})
            s = summarize(rows)
            print(f"  {label}  f1={s['f1']}  P={s['P']} R={s['R']}  dept={s['dept']}  cat={s['cat']}  "
                  f"price={s['price']}  spurious={s['spurious']}  strata={s['strata']}")
        intents = [intent_of(clean_parse(preds[k], gold[k]["utterance"]), gold[k]["utterance"]) for k in gold if k in preds]
        types = [message_type_of(clean_parse(preds[k], gold[k]["utterance"]), gold[k]["utterance"]) for k in gold if k in preds]
        from collections import Counter
        print(f"  derived intent: {dict(Counter(intents))}   message_type: {dict(Counter(types))}")


if __name__ == "__main__":
    main()
