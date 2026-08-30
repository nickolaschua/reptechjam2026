"""Does the parse help the DENSE branch? Same BGE index Yang Xu's agent builds
(experiment_1/shop_agent.py: same catalog text, same query prefix, cosine), three
queries per benchmark case:

  raw      the utterance as typed            <- what shop_agent embeds today
  parsed   bench/score.parsed_state(parse)   <- category + non-negated slot values
  clean    same, after bolt_on.clean_parse   <- junk/dept rules applied

Rank of the target asin in the top-150. Needs torch + sentence-transformers;
the catalog index is encoded once and cached (~50k x 768 float32).

    python3 dense_rank.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
WINSTON = LAB.parent
KIT = WINSTON.parent / "techjam-conversational-search"
for p in (WINSTON, LAB, LAB / "bench", KIT):
    sys.path.insert(0, str(p))
from bolt_on import clean_parse  # noqa: E402
from score import parsed_state  # noqa: E402

import os
import re
# DENSE_MODEL=/path/to/finetuned overrides the embedder; the catalog index is cached per model
MODEL = os.environ.get("DENSE_MODEL", "BAAI/bge-base-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_TAG = "" if MODEL == "BAAI/bge-base-en-v1.5" else "_" + re.sub(r"[^A-Za-z0-9]+", "-", Path(MODEL).name)
CACHE = LAB / ".cache" / f"bge_catalog{_TAG}.npz"
TOP = 150


def catalog_text(p: dict) -> str:   # verbatim from shop_agent._build_vector_index
    cats = ", ".join(p.get("categories") or [])
    feats = "; ".join((p.get("features") or [])[:3])
    return f"Product: {p.get('title') or ''}. Categories: {cats}. Features: {feats}.".strip()


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    # ponytail: DENSE_DEVICE=cpu keeps the one-time 50k encode off the GPU that
    # Ollama's generators are using; the per-query encodes are trivial either way
    import os
    device = os.environ.get("DENSE_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    m = SentenceTransformer(MODEL, device=device)
    m.max_seq_length = 256
    return m


def catalog_index(model) -> tuple[np.ndarray, list[str]]:
    if CACHE.exists():
        d = np.load(CACHE)
        return d["emb"], list(d["ids"])
    ids, texts = [], []
    for line in (KIT / "data" / "catalog.jsonl").open(encoding="utf-8"):
        p = json.loads(line)
        ids.append(str(p["parent_asin"])); texts.append(catalog_text(p))
    t0 = time.time()
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, emb=emb, ids=np.array(ids))
    print(f"encoded {len(ids)} products in {time.time() - t0:.0f}s -> {CACHE.name}")
    return emb, ids


def hit(rs): return sum(1 for r in rs if r is not None and r <= 10) / len(rs)
def mrr(rs): return sum(1 / r for r in rs if r and r <= 10) / len(rs)   # evaluator-style: credit only inside the top-10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cases = {r["case_id"]: r for r in map(json.loads, (LAB / "bench" / "cases.jsonl").open())}
    parses = {r["case_id"]: r["parse"] for r in map(json.loads, (LAB / "bench" / "parses.jsonl").open())}
    todo = [cases[k] for k in parses if k in cases][:args.limit]
    print(f"{len(todo)} cases with a parse")

    model = load_model()
    emb, ids = catalog_index(model)
    pos = {a: i for i, a in enumerate(ids)}

    def rank(query: str, asin: str) -> int | None:
        q = model.encode(QUERY_PREFIX + query, convert_to_numpy=True, normalize_embeddings=True)
        top = np.argsort(emb @ q)[::-1][:TOP]
        hits = np.flatnonzero(top == pos[asin])
        return int(hits[0]) + 1 if hits.size else None

    rows = []
    for c in todo:
        p = parses[c["case_id"]]
        cat, cons = parsed_state(p)
        ccat, ccons = parsed_state(clean_parse(p, c["utterance"]))
        queries = {"raw": c["utterance"], "parsed": " ".join([cat, *cons]), "clean": " ".join([ccat, *ccons]),
                   "augmented": c["utterance"] + " " + cat}     # keep the sentence, add the category
        rows.append({"case_id": c["case_id"], "asin": c["asin"], "style": c["style"],
                     **{f"dense_{k}_rank": rank(v, c["asin"]) for k, v in queries.items()}})
    (LAB / "bench" / f"results_dense{_TAG}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print(f"\n{'query':8s} HitRate@10   MRR   in-top150")
    for k in ("raw", "parsed", "clean", "augmented"):
        rs = [r[f"dense_{k}_rank"] for r in rows]
        print(f"{k:8s}   {hit(rs):.3f}     {mrr(rs):.3f}    {sum(1 for x in rs if x)}/{len(rs)}")
    by = defaultdict(list)
    for r in rows:
        by[r["style"]].append(r)
    print(f"\n{'style':14s} n   raw hit  parsed hit  clean hit   parsed better/worse vs raw")
    for st, rs in sorted(by.items()):
        raw = [r["dense_raw_rank"] for r in rs]; pa = [r["dense_parsed_rank"] for r in rs]; cl = [r["dense_clean_rank"] for r in rs]
        b = sum(1 for a, c in zip(raw, pa) if (c or 999) < (a or 999)); w = sum(1 for a, c in zip(raw, pa) if (c or 999) > (a or 999))
        print(f"{st:14s} {len(rs):<3d}  {hit(raw):.2f}     {hit(pa):.2f}        {hit(cl):.2f}        {b} / {w}")


if __name__ == "__main__":
    main()
