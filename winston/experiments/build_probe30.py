"""Build probe30.json - the machine-readable form of probe_set.md.

Joins each hand-written case (utterance, expected parse, ceiling) to the full
catalog record for its target, plus the derived statistics an experiment needs
(bucket size, popularity rank, IDF profile, field coverage). probe_set.md stays
the thing a human edits; this file is what code reads.

    python3 experiments/build_probe30.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from common import ROOT, TOKEN_RE, coarse_category, get_index, _flatten_values  # noqa: E402

PROBE_MD = Path(__file__).resolve().parents[1] / "probe_set.md"
OUT = Path(__file__).resolve().parents[1] / "probe30.json"

CASE_RE = re.compile(
    r"## (?P<num>\d+)\. `(?P<asin>\w+)` — stratum (?P<stratum>.*?)\n"
    r".*?```yaml\n(?P<yaml>.*?)```",
    re.S,
)


def parse_yaml_block(block: str) -> dict:
    """The blocks carry inline comments; PyYAML handles those natively."""
    try:
        return yaml.safe_load(block) or {}
    except yaml.YAMLError as exc:  # keep the raw text rather than losing the case
        return {"_yaml_error": str(exc)}


def main() -> None:
    ix = get_index()
    text = PROBE_MD.read_text(encoding="utf-8")
    cases = list(CASE_RE.finditer(text))
    if len(cases) != 30:
        raise SystemExit(f"expected 30 cases in probe_set.md, found {len(cases)}")

    records = []
    for m in cases:
        asin = m.group("asin")
        product = ix.products[asin]
        parsed = parse_yaml_block(m.group("yaml"))
        expected = parsed.get("expected_parse") or {}

        bucket_key = ix.bucket_of[asin]
        members = ix.buckets[bucket_key]
        by_pop = sorted(members, key=lambda a: -ix.popularity[a])

        features = _flatten_values(product.get("features"))
        description = _flatten_values(product.get("description"))
        details = product.get("details") if isinstance(product.get("details"), dict) else {}

        utterance = parsed.get("utterance") or ""
        utt_tokens = [t for t in TOKEN_RE.findall(utterance.lower()) if t in ix.idf]
        # Which of the user's words actually appear in the target, and how rare are they?
        landing = sorted(
            ({"token": t, "in_target": t in ix.tokens[asin],
              "catalog_share": round(ix.doc_freq[t] / 50000, 5),
              "idf": round(ix.idf[t], 2)}
             for t in dict.fromkeys(utt_tokens)),
            key=lambda r: -r["idf"],
        )

        records.append({
            "case": int(m.group("num")),
            "stratum": m.group("stratum").strip(),
            "parent_asin": asin,
            "probe": {
                "utterance": utterance,
                "expected_parse": expected,
                "ceiling": expected.get("ceiling"),
                "notes": expected.get("notes"),
            },
            "product": {
                "title": product.get("title"),
                "categories": product.get("categories"),
                "features": features,
                "description": description,
                "details": details,
                "store": product.get("store"),
                "price": product.get("price"),
                "average_rating": product.get("average_rating"),
                "rating_number": product.get("rating_number"),
            },
            "derived": {
                "coarse_bucket": coarse_category(ix.categories[asin]),
                "bucket_size": len(members),
                "rank_by_popularity_in_bucket": by_pop.index(asin) + 1,
                "department": details.get("Department"),
                "has_price": product.get("price") is not None,
                "n_features": len(features),
                "n_description": len(description),
                "title_tokens": len(set(TOKEN_RE.findall(str(product.get("title") or "").lower()))),
                "max_feature_idf": round(max(
                    (ix.idf[t] for f in features for t in TOKEN_RE.findall(f.lower())
                     if t in ix.idf and len(t) >= 5), default=0.0), 2),
                "utterance_terms": landing,
                "utterance_terms_landing_in_target": sum(1 for r in landing if r["in_target"]),
                "utterance_terms_total": len(landing),
            },
        })

    records.sort(key=lambda r: r["case"])
    strata = collections.Counter(r["stratum"][0] for r in records)
    payload = {
        "source": "winston/probe_set.md",
        "n_cases": len(records),
        "summary": {
            "strata": dict(sorted(strata.items())),
            "with_price_constraint": sum(
                1 for r in records if "price" in (r["probe"]["expected_parse"].get("hard") or {})),
            "with_quality_constraint": sum(
                1 for r in records if r["probe"]["expected_parse"].get("quality")),
            "with_negation": sum(
                1 for r in records if r["probe"]["expected_parse"].get("negated")),
            "department_null": sum(
                1 for r in records
                if (r["probe"]["expected_parse"].get("hard") or {}).get("department", "MISSING") is None),
            "ceilings": dict(collections.Counter(
                str(r["probe"]["ceiling"]) for r in records).most_common()),
            "targets_with_price": sum(1 for r in records if r["derived"]["has_price"]),
            "targets_without_description": sum(
                1 for r in records if r["derived"]["n_description"] == 0),
        },
        "cases": records,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  -> {OUT.relative_to(ROOT)}  ({OUT.stat().st_size:,} bytes, {len(records)} cases)")


if __name__ == "__main__":
    main()
