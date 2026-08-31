"""Teacher-student pairs for the LoRA parser.

  teacher   bench/parses.jsonl  (qwen2.5:7b, grammar-constrained) x bench/cases.jsonl
  gold      probe_gold.json     (30 hand-written parses)          -> always in TRAIN? no:
            probes are the hand-gold TEST set; they never train.

Split by product asin so no test product is seen in training. Prompt is the
parser's own PROMPT so the student sees exactly what the teacher saw.

    python3 build_sft.py            -> sft_train.jsonl, sft_dev.jsonl, sft_probe.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
WINSTON = LAB.parent
sys.path.insert(0, str(WINSTON))
from nlp_parse import PROMPT, SCHEMA  # noqa: E402

KEYS = list(SCHEMA["properties"])          # stable key order = easier target


def canon(parse: dict) -> str:
    slots = [{"attribute": s["attribute"], "value": s["value"], "declined": bool(s.get("declined")),
              "negated": bool(s.get("negated"))} for s in parse.get("slots", [])]
    out = {k: parse.get(k) for k in KEYS}
    out["slots"] = slots
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False)


def pair(utterance: str, parse: dict, **meta) -> dict:
    return {"prompt": PROMPT.format(utterance=utterance), "completion": canon(parse), **meta}


def main() -> None:
    cases = {r["case_id"]: r for r in map(json.loads, (LAB / "bench" / "cases.jsonl").open())}
    parses = [json.loads(l) for l in (LAB / "bench" / "parses.jsonl").open() if l.strip()]
    rows = [pair(cases[p["case_id"]]["utterance"], p["parse"], asin=cases[p["case_id"]]["asin"],
                 style=cases[p["case_id"]]["style"], case_id=p["case_id"])
            for p in parses if p["case_id"] in cases]
    asins = sorted({r["asin"] for r in rows})
    random.Random(20260830).shuffle(asins)
    dev_asins = set(asins[: max(1, len(asins) // 5)])
    train = [r for r in rows if r["asin"] not in dev_asins]
    dev = [r for r in rows if r["asin"] in dev_asins]
    probes = [pair(c["utterance"], c["gold"], case=c["case"], stratum=c["stratum"], asin=c["asin"])
              for c in json.loads((WINSTON / "probe_gold.json").read_text())]
    for name, data in (("sft_train", train), ("sft_dev", dev), ("sft_probe", probes)):
        (HERE / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in data) + "\n")
    print(f"train {len(train)} (teacher)  dev {len(dev)} (teacher, held-out products)  probe {len(probes)} (hand gold)")


if __name__ == "__main__":
    main()
