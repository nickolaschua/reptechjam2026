"""Spec section 4: turn sampled products into messy utterances.

    python3 generate.py --dry-run --limit 6    # offline smoke test, canned text
    python3 generate.py                        # full run, resumable

Resumable: cases already present in cases.jsonl are skipped, so a crash or an
Ollama restart costs nothing. Ground truth is the asin; nothing else is gold.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LAB = BENCH.parent
WINSTON = LAB.parent
REPO = WINSTON.parents[2]
KIT = REPO / "techjam-conversational-search"
for p in (WINSTON / "experiments", WINSTON, LAB, BENCH, KIT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prompts import (STYLES, GATED, INTENT_LABEL, MODIFIERS, MODIFIER_ONLY_FOR,  # noqa: E402
                     content_words, build_system_prompt, relation_for)

SEED = 20260829
# mistral:7b dropped: every output was "Hey there! I'm on the hunt for..." in 6-8 sentences,
# a fixed template of its own. Never the parser's qwen2.5.
GENERATORS = ("llama3.1:8b", "gemma2:9b")
MODIFIER_P = {"negation": 0.2, "for_other": 0.2, "vague_budget": 0.2, "format_noise": 0.3}
OVERLAP_LIMIT = 0.5
USER_TURN = "Start the conversation by telling the assistant what you are looking for."
PRODUCTS_PATH = BENCH / "products.jsonl"
CASES_PATH = BENCH / "cases.jsonl"
MANIFEST_PATH = BENCH / "manifest.json"


def overlap(utterance: str, product: dict) -> float:
    """Share of the utterance's content words that appear in title+features."""
    u = content_words(utterance)
    if not u:
        return 0.0
    listing = set(content_words(" ".join([str(product.get("title") or ""),
                                          *map(str, product.get("features") or [])])))
    return round(sum(1 for w in u if w in listing) / len(u), 3)


def plan_cases(products: list[dict], seed: int = SEED) -> list[dict]:
    """Deterministic case plan: which product x style x modifiers x generator.

    The LLM output is not reproducible (temperature 0.7); the PLAN is, so a
    partial run can be resumed and two runs can be compared case-for-case.
    """
    rng = random.Random(seed)
    plan: list[dict] = []
    gi = 0
    for prod in sorted(products, key=lambda r: r["asin"]):
        styles = [st for st in STYLES if st not in GATED or prod.get(GATED[st])]
        for style in styles:
            mods = [m for m in MODIFIERS
                    if MODIFIER_ONLY_FOR.get(m, style) == style and rng.random() < MODIFIER_P[m]]
            relation = relation_for(prod.get("department"), rng) if "for_other" in mods else None
            plan.append({
                "asin": prod["asin"],
                "style": style,
                "intent_label": INTENT_LABEL[style],
                "modifiers": mods,
                "relation": relation,
                "code": prod.get("model_code") if style == "exact" else None,
                "anchor": prod.get("compat_anchor") if style == "compatibility" else None,
                "generator": GENERATORS[gi % len(GENERATORS)],
            })
            gi += 1
    # ponytail: shuffled (seeded) so any --limit prefix is a stratified sample across
    # products, styles and generators - a 1h partial run can feed a first report.
    rng.shuffle(plan)
    for i, case in enumerate(plan, 1):
        case["case_id"] = f"c{i:04d}"
    return plan


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = "http://" + OLLAMA_HOST


def ollama_chat(model: str, system: str, user: str, temperature: float = 0.7,
                timeout: int = 180, host: str = OLLAMA_HOST) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 160},
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


def _canned(case: dict, product: dict) -> str:
    """Offline stand-in so the whole pipeline can be exercised without Ollama."""
    return f"[dry-run {case['style']}] looking for something like {str(product.get('title') or '')[:30].lower()}"


def run(products: list[dict], plan: list[dict], ix, samples_profile: dict, *, dry_run: bool,
        limit: int | None, out_path: Path = CASES_PATH) -> dict:
    from evaluator.local_evaluator import intent_card

    done = set()
    if out_path.exists():
        done = {json.loads(l)["case_id"] for l in out_path.open() if l.strip()}
    todo = [c for c in plan if c["case_id"] not in done][:limit]
    # ponytail: group by generator so Ollama swaps a 5 GB model once per run, not
    # once per call - on a 16 GB machine that swap was the whole cost. The shuffled
    # plan still gives a balanced prefix within each generator's block.
    todo.sort(key=lambda c: c["generator"])
    print(f"plan {len(plan)} | done {len(done)} | this run {len(todo)}")

    started = time.time()
    counts: dict[str, int] = {}
    skipped = 0
    with out_path.open("a") as fh:
        for i, case in enumerate(todo, 1):
            product = ix.products[case["asin"]]
            card = intent_card(product)
            system = build_system_prompt(product, card, samples_profile, case["style"],
                                         case["modifiers"], code=case["code"],
                                         relation=case["relation"], anchor=case["anchor"])
            if dry_run:
                text = _canned(case, product)
            else:
                # One slow Ollama moment (another consumer, a model swap) must not
                # kill a multi-hour run: retry once, then skip the case and move on.
                # The resumable loop picks skipped cases up on the next run.
                try:
                    text = _generate(case, product, system)
                except Exception as exc:                # noqa: BLE001 - logged, not masked
                    print(f"  {case['case_id']} SKIPPED {type(exc).__name__}: {exc}", flush=True)
                    skipped += 1
                    continue
            ov = overlap(text, product)
            row = {**case, "utterance": text, "overlap": ov, "overlap_flag": ov > OVERLAP_LIMIT}
            row.pop("code", None)
            row.pop("relation", None)
            row.pop("anchor", None)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            counts[case["style"]] = counts.get(case["style"], 0) + 1
            if i % 10 == 0 or i == len(todo):
                el = time.time() - started
                print(f"  {i}/{len(todo)}  {el / i:.1f}s/case  eta {(len(todo) - i) * el / i / 60:.0f} min", flush=True)
    return {"generated_this_run": len(todo) - skipped, "skipped": skipped,
            "seconds": round(time.time() - started, 1), "by_style": counts}


def _generate(case: dict, product: dict, system: str) -> str:
    last: Exception | None = None
    for _ in range(2):
        try:
            text = ollama_chat(case["generator"], system, USER_TURN)
            if overlap(text, product) > OVERLAP_LIMIT:
                text = ollama_chat(case["generator"], system, USER_TURN)
            return text
        except Exception as exc:                        # noqa: BLE001 - retried once
            last = exc
    raise RuntimeError(f"{case['generator']}: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="canned utterances, no Ollama")
    ap.add_argument("--limit", type=int, help="stop after N new cases")
    ap.add_argument("--out", type=Path, default=CASES_PATH)
    args = ap.parse_args()

    products = [json.loads(l) for l in PRODUCTS_PATH.open()]
    # ponytail: read the 269 sampled products straight from the catalog instead of
    # get_index() - that holds all 50k products (~1 GB resident) and this process
    # shares a 16 GB machine with the models it is calling.
    from types import SimpleNamespace
    wanted = {r["asin"] for r in products}
    catalog = {}
    with (KIT / "data" / "catalog.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row["parent_asin"] in wanted:
                    catalog[row["parent_asin"]] = row
    ix = SimpleNamespace(products=catalog)
    # profile text is only flavour for the shopper; keep it constant and neutral
    profile = {"purchase_frequency": "a few prior purchases", "preference_tags": ["fit", "comfort"],
               "summary": "Prior purchases emphasize fit and comfort."}
    plan = plan_cases(products)
    stats = run(products, plan, ix, profile, dry_run=args.dry_run, limit=args.limit, out_path=args.out)

    rows = [json.loads(l) for l in args.out.open() if l.strip()]
    manifest = {
        "seed": SEED, "generators": GENERATORS, "modifier_p": MODIFIER_P, "overlap_limit": OVERLAP_LIMIT,
        "planned": len(plan), "written": len(rows), "dry_run": args.dry_run,
        "by_style": {s: sum(1 for r in rows if r["style"] == s) for s in STYLES},
        "by_generator": {g: sum(1 for r in rows if r["generator"] == g) for g in GENERATORS},
        "by_modifier": {m: sum(1 for r in rows if m in r["modifiers"]) for m in MODIFIERS},
        "by_intent": {lab: sum(1 for r in rows if r["intent_label"] == lab) for lab in ("buying", "browsing")},
        "overlap_flagged": sum(1 for r in rows if r["overlap_flag"]),
        "last_run": stats,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["by_style"]), "->", MANIFEST_PATH.name)


if __name__ == "__main__":
    main()
