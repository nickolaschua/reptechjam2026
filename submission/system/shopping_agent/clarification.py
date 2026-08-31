"""Entropy-based clarification selector."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

import numpy as np

try:
    from .catalogue import Catalogue
except ImportError:  # Direct-module compatibility for the frozen dense interface tests.
    from catalogue import Catalogue


COLOR_VOCAB = {
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "gold", "silver",
}
MATERIAL_VOCAB = {
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "canvas", "denim", "rubber", "synthetic",
}
ATTRIBUTE_ORDER = (
    "material", "color", "size", "style", "brand", "budget", "use_case",
    "gender", "closure", "pattern", "waterproof", "rating", "reviews",
)
BUYING_ATTRIBUTE_ORDER = (
    "material", "brand", "color", "size", "style", "use_case", "budget",
)
BROWSING_ATTRIBUTE_ORDER = (
    "use_case", "style", "brand", "material", "color", "size", "budget",
)


def _priority_order(intent_mode: str) -> tuple[str, ...]:
    return BUYING_ATTRIBUTE_ORDER if str(intent_mode).lower() == "buying" else BROWSING_ATTRIBUTE_ORDER


def _attribute_values(catalogue: Catalogue, row: int, attribute: str) -> list[str]:
    asin = catalogue.ids[row]
    meta = catalogue.metadata[asin]
    bag = meta["searchable_bag"]
    if attribute == "brand":
        value = str(meta.get("brand") or "").strip().lower()
        return [value] if value not in {"", "unknown", "unspecified"} else []
    if attribute == "use_case":
        value = str(catalogue.departments[row])
        return [value] if value not in {"", "unspecified"} else []
    if attribute == "budget":
        price = float(catalogue.prices[row])
        if not np.isfinite(price) or not 0.0 < price <= 9000.0:
            return []
        return ["budget" if price < 20.0 else "mid" if price < 50.0 else "high" if price < 100.0 else "luxury"]
    if attribute == "color":
        found = sorted(value for value in COLOR_VOCAB if value in bag)
        return found[:1]
    if attribute == "material":
        found = sorted(value for value in MATERIAL_VOCAB if value in bag)
        return found[:1]
    if attribute == "size":
        details = meta.get("details", {})
        value = details.get("Size") or details.get("size")
        if not value:
            for feature in meta.get("features", []):
                match = re.search(r"\b(size\s+)?(s|m|l|xl|xxl|\d+(?:\.\d+)?)\b", feature.lower())
                if match:
                    value = match.group(2)
                    break
        normalized = str(value or "").strip().lower()
        return [normalized] if normalized not in {"", "unknown", "unspecified"} else []
    if attribute == "style":
        categories = meta.get("categories", [])
        if len(categories) > 3:
            value = str(categories[-1]).lower()
            return [value] if value not in {"", "unknown", "unspecified"} else []
        return []
    if attribute == "gender":
        department = str(catalogue.departments[row])
        mapping = {
            "men": ["men"], "women": ["women"], "girls": ["girls"], "boys": ["boys"],
            "unisex-adult": ["men", "women"], "unisex-kids": ["boys", "girls"],
            "baby-boys": ["boys", "toddler"], "baby-girls": ["girls", "toddler"],
            "baby": ["toddler", "boys", "girls"], "multi-demographic": ["men", "women"],
        }
        return mapping.get(department, [])
    if attribute == "closure":
        details = meta.get("details", {})
        value = details.get("Closure Type") or details.get("closure") or "unknown"
        if value == "unknown":
            feature_text = " ".join(meta.get("features", [])).lower()
            for term in ["drawstring", "zipper", "button", "elastic", "pull on", "lace up", "hook and eye"]:
                if term in feature_text:
                    value = term
                    break
        normalized = str(value).strip().lower()
        return [normalized] if normalized not in {"", "unknown", "unspecified"} else []
    if attribute == "pattern":
        details = meta.get("details", {})
        value = details.get("Pattern") or details.get("pattern") or "unknown"
        if value == "unknown":
            feature_text = " ".join(meta.get("features", [])).lower()
            for term in ["striped", "solid", "floral", "graphic", "plaid", "printed", "leopard", "camo"]:
                if term in feature_text:
                    value = term
                    break
        normalized = str(value).strip().lower()
        return [normalized] if normalized not in {"", "unknown", "unspecified"} else []
    if attribute == "waterproof":
        feature_text = " ".join(meta.get("features", [])).lower()
        return ["waterproof" if "waterproof" in feature_text or "water-resistant" in feature_text else "regular"]
    if attribute == "rating":
        rating = float(catalogue.avg_ratings[row])
        if rating <= 0.0:
            return []
        return ["excellent" if rating >= 4.5 else "good" if rating >= 4.0 else "average"]
    if attribute == "reviews":
        reviews = int(catalogue.rating_numbers[row])
        if reviews <= 0:
            return []
        return ["very popular" if reviews >= 1000 else "popular" if reviews >= 100 else "niche"]
    return []


def select_best_attributes(
    catalogue: Catalogue,
    candidate_ids: Iterable[str],
    remaining_attributes: set[str],
    *,
    top_n: int = 2,
    intent_mode: str = "browsing",
) -> list[str]:
    """Select attributes using entropy, gain ratio, coverage, and usefulness."""

    priority = _priority_order(intent_mode)
    if not remaining_attributes:
        return ["other"] * top_n
    rows = [catalogue.row_by_asin[asin] for asin in list(candidate_ids)[:100] if asin in catalogue.row_by_asin]
    if not rows:
        ordered = [attr for attr in priority if attr in remaining_attributes][:top_n]
        return ordered + ["other"] * (top_n - len(ordered))

    scores: list[tuple[float, int, int, str]] = []
    priority_rank = {attr: rank for rank, attr in enumerate(priority)}
    for attr in ATTRIBUTE_ORDER:
        if attr not in remaining_attributes:
            continue
        values: list[str] = []
        populated_count = 0
        for row in rows:
            found = _attribute_values(catalogue, row, attr)
            if found:
                values.extend(found)
                populated_count += 1
        if not values:
            scores.append((0.0, priority_rank.get(attr, len(priority)), ATTRIBUTE_ORDER.index(attr), attr))
            continue
        counts = Counter(values)
        value_total = len(values)
        entropy_candidates = np.log2(len(rows)) if rows else 0.0
        conditional_entropy = 0.0
        split_info = 0.0
        for count in counts.values():
            probability = count / value_total
            if probability > 0:
                split_info -= probability * np.log2(probability)
                conditional_entropy += probability * np.log2(count)
        gain = entropy_candidates - conditional_entropy
        gain_ratio = gain / (split_info + 1e-9)
        adjusted_gain = gain_ratio * (populated_count / len(rows))
        if adjusted_gain <= 0.05:
            adjusted_gain = 0.0
        scores.append((float(adjusted_gain), priority_rank.get(attr, len(priority)), ATTRIBUTE_ORDER.index(attr), attr))
    scores.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    result = [item[3] for item in scores[:top_n]]
    return result + ["other"] * (top_n - len(result))


def select_fixed_priority_attributes(
    catalogue: Catalogue,
    candidate_ids: Iterable[str],
    remaining_attributes: set[str],
    *,
    top_n: int = 2,
    intent_mode: str = "browsing",
) -> list[str]:
    """Select the first still-unasked attributes in the established intent order.

    ``catalogue`` and ``candidate_ids`` are intentionally accepted even though the
    control policy does not inspect them.  Keeping the same callable contract as
    :func:`select_best_attributes` makes clarification policy the only changed
    factor in an experiment.
    """

    del catalogue, candidate_ids
    ordered = [
        attribute
        for attribute in _priority_order(intent_mode)
        if attribute in remaining_attributes
    ][:top_n]
    return ordered + ["other"] * (top_n - len(ordered))


__all__ = [
    "ATTRIBUTE_ORDER", "BROWSING_ATTRIBUTE_ORDER", "BUYING_ATTRIBUTE_ORDER",
    "select_best_attributes", "select_fixed_priority_attributes",
]
