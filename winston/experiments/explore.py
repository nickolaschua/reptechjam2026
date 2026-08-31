"""EXPLORE - scratch probes over the catalog. Add a @probe, rerun, read stdout.

    python3 explore.py                  # every probe, in definition order
    python3 explore.py ratings targets  # just these
    python3 explore.py --list           # names + one-liners
    python3 explore.py --test           # self-check the helpers

Unlike expNN_*.py this writes nothing to results/ - it is a scratchpad for
questions you ask once. Promote a probe to an expNN file when the answer
starts mattering to the agent.
"""
from __future__ import annotations

import bisect
import functools
import json
import statistics as st
import sys

from common import CATALOG, PUBLIC_SET

PROBES: dict[str, callable] = {}


def probe(fn):
    PROBES[fn.__name__] = fn
    return fn


# ---------------------------------------------------------------- data

@functools.cache
def rows() -> list[dict]:
    return [json.loads(line) for line in CATALOG.open()]


@functools.cache
def samples() -> list[dict]:
    return [json.loads(line) for line in PUBLIC_SET.open()]


@functools.cache
def targets() -> list[dict]:
    """The 200 public-set ground-truth products, as catalog rows."""
    wanted = {s["ground_truth"]["parent_asin"] for s in samples()}
    return [r for r in rows() if r["parent_asin"] in wanted]


# ---------------------------------------------------------------- helpers

def pctile_of(sorted_values: list, v) -> float:
    """Percentile rank of v within sorted_values, 0-100."""
    return 100.0 * bisect.bisect_left(sorted_values, v) / len(sorted_values)


def quantiles(values, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> dict[int, float]:
    s = sorted(values)
    return {p: s[min(int(p / 100 * len(s)), len(s) - 1)] for p in ps}


def table(rows_, headers) -> None:
    cells = [[str(c) for c in r] for r in ([headers] + list(rows_))]
    width = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for i, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, width)))
        if i == 0:
            print("  " + "  ".join("-" * w for w in width))


# ---------------------------------------------------------------- probes

@probe
def ratings() -> None:
    """average_rating: coverage, granularity, and whether it is shrunk."""
    ar = [r["average_rating"] for r in rows()]
    grid = sorted({v for v in ar})
    off_grid = [v for v in grid if round(v, 1) != v]
    print(f"rows={len(ar)}  null={sum(v is None for v in ar)}  distinct={len(grid)}")
    print(f"range={min(ar)}..{max(ar)}  mean={st.fmean(ar):.3f}  median={st.median(ar)}")
    print(f"values off the 0.1 grid: {len(off_grid)} -> raw rounded mean, not shrunk")

    counts = sorted(((ar.count(v), v) for v in grid), reverse=True)[:8]
    table([(v, c, f"{100 * c / len(ar):.2f}%") for c, v in counts],
          ["value", "rows", "share"])

    tiny = sum(1 for r in rows() if r["average_rating"] == 5.0 and r["rating_number"] < 10)
    perfect = sum(1 for v in ar if v == 5.0)
    print(f"\n5.0 rows: {perfect} ({100 * perfect / len(ar):.1f}% of catalog); "
          f"{tiny} of them ({100 * tiny / perfect:.0f}%) have <10 ratings")


@probe
def popularity() -> None:
    """rating_number distribution, and how it correlates with average_rating."""
    rn = [r["rating_number"] for r in rows()]
    q = quantiles(rn)
    print("rating_number  " + "  ".join(f"p{p}={v}" for p, v in q.items()))
    print(f"mean={st.fmean(rn):.1f}  exactly-one-rating={rn.count(1)}")

    print("\nmean average_rating by rating_number bucket:")
    buckets = [(1, 5), (5, 20), (20, 100), (100, 500), (500, 5_000), (5_000, 10**9)]
    table([(f"{lo}-{hi - 1}", len(g), f"{st.fmean(g):.3f}")
           for lo, hi in buckets
           if (g := [r["average_rating"] for r in rows() if lo <= r["rating_number"] < hi])],
          ["ratings", "rows", "mean avg"])


@probe
def target_skew() -> None:
    """Do public-set targets look like the catalog? (No. They are the head.)"""
    srn = sorted(r["rating_number"] for r in rows())
    sar = sorted(r["average_rating"] for r in rows())

    for name, group in (("catalog", rows()), ("targets", targets())):
        rn = [r["rating_number"] for r in group]
        ar = [r["average_rating"] for r in group]
        q = quantiles(rn, (25, 50, 75))
        print(f"{name:8} n={len(group):5d}  rating_number p25={q[25]:<7} p50={q[50]:<7} "
              f"p75={q[75]:<7} mean avg={st.fmean(ar):.3f}")

    tp = sorted(pctile_of(srn, r["rating_number"]) for r in targets())
    ta = sorted(pctile_of(sar, r["average_rating"]) for r in targets())
    q = lambda s, p: s[int(p / 100 * len(s))]  # noqa: E731
    print("\nwhere targets sit as a catalog percentile:")
    table([("rating_number", f"{q(tp, 10):.1f}", f"{q(tp, 50):.1f}", f"{q(tp, 90):.1f}"),
           ("average_rating", f"{q(ta, 10):.1f}", f"{q(ta, 50):.1f}", f"{q(ta, 90):.1f}")],
          ["field", "p10", "p50", "p90"])
    print(f"\ntargets in catalog top 5% by popularity: {sum(x >= 95 for x in tp)}/{len(tp)}")
    print(f"targets below catalog median popularity:  {sum(x < 50 for x in tp)}/{len(tp)}")


@probe
def shrinkage(m: int = 50) -> None:
    """Top-10 by raw average_rating vs a Bayesian-shrunk score."""
    prior = st.fmean(r["average_rating"] for r in rows())
    score = lambda r: (r["rating_number"] * r["average_rating"] + m * prior) / (r["rating_number"] + m)  # noqa: E731

    for label, key in (("raw average_rating", lambda r: (-r["average_rating"], -r["rating_number"])),
                       (f"shrunk (prior={prior:.3f}, m={m})", lambda r: -score(r))):
        print(f"top-10 by {label}:")
        table([(f"{score(r):.3f}", r["average_rating"], r["rating_number"], r["title"][:52])
               for r in sorted(rows(), key=key)[:10]],
              ["shrunk", "raw", "n", "title"])
        print()


# ---------------------------------------------------------------- run

def selftest() -> None:
    s = [1, 2, 2, 5, 9]
    assert pctile_of(s, 1) == 0.0 and pctile_of(s, 9) == 80.0
    assert pctile_of(s, 2) == 20.0, "ties rank at their first occurrence"
    assert pctile_of(s, 3) == 60.0, "absent values rank at the insertion point"
    assert quantiles(range(100), (0, 50, 99)) == {0: 0, 50: 50, 99: 99}
    assert quantiles([7], (99,)) == {99: 7}, "no index overrun on tiny inputs"
    print("ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--test" in args:
        selftest()
    elif "--list" in args:
        for name, fn in PROBES.items():
            print(f"  {name:14} {(fn.__doc__ or '').splitlines()[0]}")
    else:
        for name in args or PROBES:
            print(f"\n=== {name} " + "=" * (60 - len(name)))
            print(f"{(PROBES[name].__doc__ or '').splitlines()[0]}\n")
            PROBES[name]()
