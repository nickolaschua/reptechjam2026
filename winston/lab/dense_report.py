"""Slice bench/results_dense.jsonl: raw vs parsed vs clean query, by style and by
modifier (negation / for_other / vague_budget), with better/worse counts.

    python3 dense_report.py
"""
import json
from collections import defaultdict
from pathlib import Path

B = Path(__file__).resolve().parent / "bench"
rows = [json.loads(l) for l in (B / "results_dense.jsonl").open() if l.strip()]
cases = {r["case_id"]: r for r in map(json.loads, (B / "cases.jsonl").open())}
for r in rows:
    r["modifiers"] = cases[r["case_id"]]["modifiers"]

def hit(rs): return sum(1 for r in rs if r is not None and r <= 10) / len(rs)
def mrr(rs): return sum(1 / r for r in rs if r and r <= 10) / len(rs)   # evaluator-style: credit only inside the top-10
def bw(rs, a, b):
    better = sum(1 for r in rs if (r[b] or 999) < (r[a] or 999))
    worse = sum(1 for r in rs if (r[b] or 999) > (r[a] or 999))
    return f"{better}/{worse}/{len(rs) - better - worse}"

def table(groups: dict, title: str) -> None:
    print(f"\n{title:16s} n    raw H@10 MRR   | parsed H@10 MRR  | clean H@10 MRR  | parsed b/w/= | clean b/w/=")
    for g, rs in groups.items():
        cols = []
        for k in ("raw", "parsed", "clean"):
            v = [r[f"dense_{k}_rank"] for r in rs]
            cols.append(f"{hit(v):.2f} {mrr(v):.2f}")
        print(f"{g:16s} {len(rs):<4d} {cols[0]:14s}| {cols[1]:15s}| {cols[2]:15s}| "
              f"{bw(rs, 'dense_raw_rank', 'dense_parsed_rank'):12s} | {bw(rs, 'dense_raw_rank', 'dense_clean_rank')}")

table({"all": rows}, "overall")
by = defaultdict(list)
for r in rows: by[r["style"]].append(r)
table(dict(sorted(by.items())), "by style")
mods = defaultdict(list)
for r in rows:
    for m in ("negation", "for_other", "vague_budget"):
        mods[f"{m}={m in r['modifiers']}"].append(r)
table(dict(sorted(mods.items())), "by modifier")
