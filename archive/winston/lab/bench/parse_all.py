"""Parse every bench case once, cached to parses.jsonl. Resumable, append-only.

score.py also caches parses, but it builds the FTS5 agent and runs retrieval per
case - far more work than labelling needs. This is the parse and nothing else.

    python3 parse_all.py            # all remaining cases
    python3 parse_all.py --limit 50
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
for p in (WINSTON / "experiments", WINSTON, LAB, BENCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bolt_on import PARSER_MODEL  # noqa: E402
from nlp_parse import parse_with_ollama  # noqa: E402

CASES = BENCH / "cases.jsonl"
PARSES = BENCH / "parses.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=PARSER_MODEL)
    args = ap.parse_args()

    cases = [json.loads(l) for l in CASES.open() if l.strip()]
    done = {json.loads(l)["case_id"] for l in PARSES.open() if l.strip()} if PARSES.exists() else set()
    todo = [c for c in cases if c["case_id"] not in done][:args.limit]
    print(f"cases {len(cases)} | parsed {len(done)} | this run {len(todo)}", flush=True)

    t0 = time.time()
    with PARSES.open("a") as fh:
        for i, c in enumerate(todo, 1):
            parse = parse_with_ollama(c["utterance"], args.model)
            fh.write(json.dumps({"case_id": c["case_id"], "parse": parse}) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  {el / i:.1f}s/case  eta {(len(todo) - i) * el / i / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
