"""Spec section 3.1: ~260 products, with the failure and gated-style pools over-sampled.

    python3 sample.py            # -> products.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
from covariates import load_all  # noqa: E402

SEED = 20260829
PRODUCTS_PATH = BENCH / "products.jsonl"


def sample_products(cov: dict[str, dict], seed: int = SEED, base: int = 120, per_pool: int = 30) -> list[dict]:
    rng = random.Random(seed)
    rows = [r for r in cov.values() if r.get("title_ok", True)]
    pops = sorted(r["popularity"] for r in rows)
    q1 = pops[len(pops) // 4]

    pools = {
        "silent_on_material": [r for r in rows if r["silent_on_material"]],
        "has_near_duplicate": [r for r in rows if r["has_near_duplicate"]],
        "low_popularity": [r for r in rows if r["popularity"] < q1],
        "has_model_code": [r for r in rows if r["has_model_code"]],
        "compat_eligible": [r for r in rows if r["compat_eligible"]],
    }
    chosen: dict[str, dict] = {}

    def take(pool: list[dict], n: int, tag: str) -> None:
        # sort first so rng.sample is deterministic regardless of dict order
        for r in rng.sample(sorted(pool, key=lambda x: x["asin"]), min(n, len(pool))):
            entry = chosen.setdefault(r["asin"], {**r, "pools": []})
            entry["pools"].append(tag)

    take(rows, base, "base")
    for tag, pool in pools.items():
        take(pool, per_pool, tag)
    return sorted(chosen.values(), key=lambda r: r["asin"])


def main() -> None:
    from common import get_index
    ix = get_index()
    cov = load_all(ix)
    for a, r in cov.items():
        r["title_ok"] = bool(str(ix.products[a].get("title") or "").strip())
    out = sample_products(cov)
    with PRODUCTS_PATH.open("w") as fh:
        for r in out:
            r.pop("title_ok", None)
            fh.write(json.dumps(r) + "\n")
    print(f"{len(out)} products -> {PRODUCTS_PATH.name}")
    tally: dict[str, int] = {}
    for r in out:
        for p in r["pools"]:
            tally[p] = tally.get(p, 0) + 1
    print("  pools:", tally)


if __name__ == "__main__":
    sys.path.insert(0, str(BENCH.parent.parent / "experiments"))
    main()
