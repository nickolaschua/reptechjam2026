"""EXP01 - What is actually in the 50k catalog, and what can be filtered on?

Answers: field coverage, the details schema, attribute selectivity, whether
attribute values are mutually exclusive, and whether a contradiction set can be
computed (the test that decides HARD vs SOFT).
"""
from __future__ import annotations

import collections
import re
import statistics

from common import N_CATALOG, TOKEN_RE, coarse_category, get_index, write_result

MATERIALS = ["leather", "cotton", "polyester", "nylon", "wool", "silk", "denim", "suede"]
COLOURS = ["black", "white", "blue", "red", "pink", "green", "brown", "gray", "purple", "yellow"]
DEPARTMENTS = [r"\bwomen", r"\bmen(?!tion)", r"\bgirls", r"\bboys"]
ANTONYMS = {
    "comfortable": ["uncomfortable", "stiff", "rigid"],
    "durable": ["fragile", "flimsy", "delicate"],
    "lightweight": ["heavyweight", "heavy duty"],
    "casual": ["formal", "black tie"],
    "summer": ["winter", "thermal", "insulated"],
    "breathable": ["waterproof", "sealed", "non-breathable"],
}


def non_empty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, dict)):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return True


def main() -> dict:
    ix = get_index()
    docs = list(ix.text.values())
    out: dict = {}

    out["field_coverage"] = {
        field: round(sum(1 for p in ix.products.values() if non_empty(p.get(field))) / N_CATALOG, 4)
        for field in ("title", "categories", "store", "details", "features",
                      "description", "price", "average_rating", "rating_number")
    }

    keys: collections.Counter = collections.Counter()
    values: dict[str, set] = collections.defaultdict(set)
    for p in ix.products.values():
        details = p.get("details")
        if isinstance(details, dict):
            for k, v in details.items():
                keys[k] += 1
                values[k].add(str(v)[:60])
    out["details_schema"] = {
        k: {"coverage": round(c / N_CATALOG, 4), "distinct_values": len(values[k])}
        for k, c in keys.most_common(16)
    }

    # Department is our best facet and it is not normalised.
    dept: collections.Counter = collections.Counter()
    for p in ix.products.values():
        d = p.get("details")
        if isinstance(d, dict) and d.get("Department"):
            dept[str(d["Department"])] += 1
    variants: dict[str, list] = collections.defaultdict(list)
    for k, c in dept.items():
        variants[k.lower().replace("-", " ")].append([k, c])
    out["department_casing_variants"] = {
        norm: sorted(v, key=lambda x: -x[1])
        for norm, v in variants.items() if len(v) > 1
    }

    def selectivity(pattern: str) -> float:
        rx = re.compile(pattern)
        return round(sum(1 for d in docs if rx.search(d)) / N_CATALOG, 4)

    out["selectivity"] = {
        group: {p: selectivity(p) for p in pats}
        for group, pats in {
            "material": MATERIALS, "colour": COLOURS, "department": DEPARTMENTS,
            "use_case": ["running", "hiking", "wedding", r"\bgym\b", "office", "travel"],
            "vibe": ["summer", "winter", "casual", "elegant", "vintage", "cozy"],
            "quality_claim": ["comfort", "breathab", "durab", "soft", "lightweight", "premium"],
        }.items()
    }

    # Exclusivity: if products routinely mention several values, you cannot exclude.
    def exclusivity(vals: list[str]) -> dict:
        rx = {v: re.compile(v) for v in vals}
        counts: collections.Counter = collections.Counter()
        for d in docs:
            counts[sum(1 for v in vals if rx[v].search(d))] += 1
        total = sum(counts.values())
        many = sum(c for k, c in counts.items() if k >= 2) / total
        return {"none": round(counts[0] / total, 4), "exactly_one": round(counts[1] / total, 4),
                "two_or_more": round(many, 4), "exclusivity_holds": bool(many < 0.15)}

    out["exclusivity"] = {name: exclusivity(v) for name, v in
                          (("material", MATERIALS), ("colour", COLOURS), ("department", DEPARTMENTS))}

    # Three-valued test: MATCHES / CONTRADICTS / SILENT for each material.
    rx = {m: re.compile(m) for m in MATERIALS}
    three: dict = {}
    for m in MATERIALS:
        match = contra = silent = 0
        for d in docs:
            if rx[m].search(d):
                match += 1
            elif any(rx[o].search(d) for o in MATERIALS if o != m):
                contra += 1
            else:
                silent += 1
        three[m] = {"matches": round(match / N_CATALOG, 4),
                    "contradicts": round(contra / N_CATALOG, 4),
                    "silent": round(silent / N_CATALOG, 4)}
    out["three_valued_material"] = three

    out["contradiction_detectable"] = {}
    for term, ants in ANTONYMS.items():
        pos = sum(1 for d in docs if term in d) / N_CATALOG
        neg = sum(1 for d in docs if any(a in d for a in ants)) / N_CATALOG
        both = sum(1 for d in docs if term in d and any(a in d for a in ants)) / N_CATALOG
        out["contradiction_detectable"][term] = {
            "says_term": round(pos, 4), "says_antonym": round(neg, 4),
            "says_both": round(both, 4), "usable_as_filter": bool(both < 0.01 and neg > 0.02)}

    # A literal text filter deletes true matches whose text uses another word.
    misses: dict = {}
    total_tp = total_missed = 0
    visible = {a: " ".join([ix.fields[a]["title"], ix.fields[a]["features"],
                            ix.fields[a]["description"]]) for a in ix.products}
    for m in MATERIALS:
        group = [a for a, p in ix.products.items()
                 if isinstance(p.get("details"), dict)
                 and m in str(p["details"].get("Material", "")).lower()]
        if len(group) < 5:
            continue
        hit = sum(1 for a in group if m in visible[a])
        misses[m] = {"true_positives": len(group), "word_present": hit,
                     "deleted_by_literal_filter": round((len(group) - hit) / len(group), 4)}
        total_tp += len(group)
        total_missed += len(group) - hit
    misses["OVERALL"] = {"true_positives": total_tp,
                         "deleted_by_literal_filter": round(total_missed / max(total_tp, 1), 4)}
    out["literal_filter_false_negatives"] = misses

    sizes = sorted(len(v) for v in ix.buckets.values())
    out["category_structure"] = {
        "distinct_leaves": len({(p.get("categories") or ["?"])[-1] for p in ix.products.values()}),
        "distinct_coarse_buckets": len(ix.buckets),
        "bucket_size_median": statistics.median(sizes),
        "bucket_size_p90": sizes[int(0.9 * len(sizes))],
        "bucket_size_max": max(sizes),
        "singleton_buckets": sum(1 for s in sizes if s == 1),
    }
    return out


if __name__ == "__main__":
    print("EXP01 catalog profile")
    write_result("exp01_catalog_profile", main())
