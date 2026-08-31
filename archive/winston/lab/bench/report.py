"""Spec section 6 reporting: every cell carries a bootstrap 95% CI.

    python3 report.py            # -> report.md
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

BENCH = Path(__file__).resolve().parent
CASES_PATH = BENCH / "cases.jsonl"
PRODUCTS_PATH = BENCH / "products.jsonl"
RESULTS_PATH = BENCH / "results.jsonl"
REPORT_PATH = BENCH / "report.md"
MIN_N = 20
COVARIATES = ("descriptiveness", "title_richness", "jargon", "bucket_size", "popularity")
FLAGS = ("silent_on_material", "has_near_duplicate", "has_model_code", "compat_eligible", "promo_bucket", "price_present")


def hit10(ranks: list[int | None]) -> float:
    return sum(1 for r in ranks if r is not None and r <= 10) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    # evaluator scores only the first 10 -> rank 11+ is a miss, same as None
    return sum(1.0 / r for r in ranks if r and r <= 10) / len(ranks) if ranks else 0.0


def bootstrap_ci(values: list, metric: Callable, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    stats = sorted(metric(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return round(stats[int(0.025 * n_boot)], 3), round(stats[int(0.975 * n_boot)], 3)


def quartile_label(value, population: list) -> str:
    if value is None:
        return "n/a"
    pop = sorted(v for v in population if v is not None)
    if not pop:
        return "n/a"
    cuts = [pop[len(pop) * q // 4] for q in (1, 2, 3)]
    return "Q" + str(1 + sum(value >= c for c in cuts))


def cell(ranks: list, metric: Callable) -> str:
    if not ranks:
        return "—"
    lo, hi = bootstrap_ci(ranks, metric)
    flag = " *" if len(ranks) < MIN_N else ""
    return f"{metric(ranks):.3f} [{lo:.3f}, {hi:.3f}] n={len(ranks)}{flag}"


def table(rows: list[dict], key: Callable[[dict], str], rank_field: str, title: str) -> list[str]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r.get(rank_field))
    out = [f"### {title} — `{rank_field}`", "", "| slice | HitRate@10 | MRR |", "|---|---|---|"]
    for g in sorted(groups):
        out.append(f"| {g} | {cell(groups[g], hit10)} | {cell(groups[g], mrr)} |")
    return out + [""]


def specificity_slices(rows: list[dict]) -> list[tuple[str, Callable]]:
    """Slices for the two intent axes, only where score.py recorded them."""
    out: list[tuple[str, Callable]] = []
    if any("card_hard_said" in r for r in rows):
        out.append(("By card hard-constraints voiced", lambda r: f"card_hard_said={min(r.get('card_hard_said', 0), 2)}{'+' if r.get('card_hard_said', 0) >= 2 else ''}"))
    if any("n_hard" in r for r in rows):
        out.append(("By parsed hard slots", lambda r: f"n_hard={min(r.get('n_hard', 0), 2)}{'+' if r.get('n_hard', 0) >= 2 else ''}"))
        out.append(("By resolver confidence", lambda r: f"conf {'>=0.2' if (r.get('resolver_confidence') or 0) >= 0.2 else '<0.2'}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=CASES_PATH)
    ap.add_argument("--products", type=Path, default=PRODUCTS_PATH)
    ap.add_argument("--results", type=Path, default=RESULTS_PATH)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args()

    cases = {r["case_id"]: r for r in map(json.loads, (l for l in args.cases.open() if l.strip()))}
    products = {r["asin"]: r for r in map(json.loads, (l for l in args.products.open() if l.strip()))}
    # last row per case wins: a resumed or duplicated pass may append a case twice
    results = list({r["case_id"]: r for r in (json.loads(l) for l in args.results.open() if l.strip())}.values())
    rows = [{**cases[r["case_id"]], **products[r["asin"]], **r} for r in results if r["case_id"] in cases]
    if not rows:
        raise SystemExit("no scored rows")

    rank_fields = [f for f in ("template_rank", "lexical_rank", "parsed_rank", "bucket_rank") if any(f in r for r in rows)]
    md = [f"# Messy benchmark report", "", f"{len(rows)} scored cases. `*` = n < {MIN_N}.", ""]
    if any("question_hit" in r for r in rows):
        md += ["## First-question quality (`question_hit`)", "",
               "| slice | hit rate |", "|---|---|"]
        for key, lab in (("all", lambda r: "all"), ("style", lambda r: r["style"]),
                         ("intent", lambda r: r["intent_label"])):
            groups: dict[str, list] = defaultdict(list)
            for r in rows:
                groups[lab(r)].append(1 if r.get("question_hit") else None)   # rank 1 = hit, None = miss
            for g in sorted(groups):
                md.append(f"| {key}={g} | {cell(groups[g], hit10)} |")
        md.append("")
    for rf in rank_fields:
        md += [f"## {rf}", ""]
        md += table(rows, lambda r: "all", rf, "Overall")
        md += table(rows, lambda r: r["style"], rf, "By style")
        md += table(rows, lambda r: r["intent_label"], rf, "By intent label (style prior)")
        for title, key in specificity_slices(rows):
            md += table(rows, key, rf, title)
        md += table(rows, lambda r: r["generator"], rf, "By generator")
        for m in ("negation", "for_other", "vague_budget", "format_noise"):
            md += table(rows, lambda r, m=m: f"{m}={m in r['modifiers']}", rf, f"By modifier {m}")
        md += table(rows, lambda r: f"overlap {quartile_label(r['overlap'], [x['overlap'] for x in rows])}", rf, "By listing overlap quartile")
        for cv in COVARIATES:
            pop = [x.get(cv) for x in rows]
            md += table(rows, lambda r, cv=cv, pop=pop: f"{cv} {quartile_label(r.get(cv), pop)}", rf, f"By {cv} quartile")
        for fl in FLAGS:
            md += table(rows, lambda r, fl=fl: f"{fl}={bool(r.get(fl))}", rf, f"By {fl}")
    args.out.write_text("\n".join(md))
    print(f"{len(rows)} rows -> {args.out.name}")


if __name__ == "__main__":
    main()
