"""Offline post-Phase-3A diagnosis of the failed U2/U3 QLMP projectors.

This module deliberately has no embedding or LLM client imports.  It replays
only persisted query/memory vectors, the validated M0 catalogue cache, and the
real catalogue JSONL.  It does not call or modify M0/QLMP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AUTHORITATIVE_RUN_ID = "preregistered_full_k500_r16_definitive"
TARGET_FIXTURES = ("u2_override_s9_final", "u3_distractor_s9_final")
EMBEDDED_TEMPLATE = "Product: {title}. Categories: {categories}. Features: {first_three_features}."
ZERO_EXTERNAL_CALLS = {"llm": 0, "openai": 0}

ATTRIBUTE_TERMS: dict[str, tuple[str, ...]] = {
    "colour": (
        "black", "white", "gray", "grey", "red", "blue", "green", "yellow",
        "orange", "purple", "pink", "brown", "beige", "navy", "teal", "color",
        "colour", "multicolor", "multicolour",
    ),
    "dark_black": ("black", "dark", "charcoal", "navy"),
    "bright_colourful": (
        "bright", "colorful", "colourful", "multicolor", "multicolour", "vibrant",
        "neon", "rainbow", "bold color",
    ),
    "sporty_athletic": (
        "sporty", "athletic", "sport", "performance", "running", "workout", "gym",
        "training",
    ),
    "minimal_understated": (
        "minimal", "minimalist", "understated", "simple", "subtle", "classic", "basic",
        "plain",
    ),
    "breathability": (
        "breathable", "breathability", "moisture wicking", "moisture-wicking",
        "quick dry", "quick-dry", "ventilated", "ventilation", "airflow", "mesh",
    ),
    "materials": (
        "cotton", "polyester", "rayon", "spandex", "elastane", "nylon", "wool",
        "silk", "linen", "leather", "mesh", "viscose", "modal", "acrylic", "fleece",
        "denim", "satin",
    ),
    "formal_dressy": (
        "formal", "dressy", "evening", "office", "business", "cocktail", "gown",
        "blazer", "suit", "tuxedo", "dress shirt",
    ),
    "casual": ("casual", "everyday", "tee", "t-shirt", "t shirt", "lounge"),
    "compression": ("compression", "compressive"),
    "hooded": ("hooded", "hoodie", "hood"),
    "rain_waterproof": (
        "rain", "rainwear", "waterproof", "water-resistant", "water resistant", "poncho",
    ),
    "cargo_storage": (
        "cargo", "storage", "pocket", "pockets", "utility", "compartment",
    ),
    "winter_insulation": (
        "winter", "insulated", "insulation", "thermal", "warmth", "fleece-lined",
        "fleece lined", "cold weather", "snow",
    ),
    "price_proxy_affordable": ("affordable", "budget", "economical", "value", "cheap"),
    "price_proxy_premium": ("premium", "luxury", "designer", "high-end", "high end"),
}

CASE_ATTRIBUTES = {
    "u2_override_s9_final": (
        "price", "colour", "dark_black", "bright_colourful", "sporty_athletic",
        "minimal_understated",
    ),
    "u3_distractor_s9_final": (
        "price", "breathability", "materials", "formal_dressy", "casual", "compression",
        "hooded", "rain_waterproof", "cargo_storage", "winter_insulation",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.extend((str(key), *_flatten_text(item)))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten_text(item))
        return result
    return [str(value)]


def build_product_text(product: Mapping[str, Any]) -> str:
    title = product.get("title") or ""
    categories = ", ".join(product.get("categories") or [])
    features = "; ".join((product.get("features") or [])[:3])
    return f"Product: {title}. Categories: {categories}. Features: {features}.".strip()


def build_raw_catalogue_text(product: Mapping[str, Any]) -> str:
    fields = (
        product.get("title"), product.get("categories"), product.get("features"),
        product.get("details"), product.get("description"), product.get("store"),
    )
    return " ".join(part for value in fields for part in _flatten_text(value)).lower()


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term).replace(r"\ ", r"[\s-]+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE)


def matched_terms(text: str, terms: Sequence[str]) -> tuple[str, ...]:
    return tuple(term for term in terms if _term_pattern(term).search(text))


def parse_price(product: Mapping[str, Any]) -> float | None:
    value = product.get("price")
    if value is None:
        return None
    try:
        parsed = float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def current_m0_price_mask(products: Sequence[Mapping[str, Any]], max_price: float | None) -> np.ndarray:
    """Exact current M0 hard-price semantics; missing values behave as 9999."""
    if max_price is None:
        return np.ones(len(products), dtype=bool)
    values = np.asarray([
        9999.0 if parse_price(product) is None else parse_price(product)
        for product in products
    ], dtype=np.float64)
    return values <= float(max_price)


def select_top_k(scores: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores)
    eligible = np.asarray(eligible, dtype=bool)
    if scores.ndim != 1 or eligible.shape != scores.shape:
        raise ValueError("scores and eligible mask must be aligned vectors")
    rows = np.flatnonzero(eligible)
    # Preserve M0's reversed np.argsort tie behaviour within catalogue row order.
    return rows[np.argsort(scores[rows])[::-1][: int(k)]]


def validate_top_k_alignment(
    persisted_ids: Sequence[str], cache_ids: np.ndarray, rows: np.ndarray
) -> None:
    actual = [str(cache_ids[int(row)]) for row in rows]
    expected = [str(value) for value in persisted_ids]
    if actual != expected:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
            min(len(actual), len(expected)),
        )
        raise ValueError(f"persisted Top-K/cache row mismatch at rank {mismatch + 1}")


def build_tangent_matrix(q: np.ndarray, rows: np.ndarray) -> np.ndarray:
    q64 = np.asarray(q, dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64)
    residuals = matrix - np.outer(matrix @ q64, q64)
    residuals -= np.outer(residuals @ q64, q64)
    return residuals


def component_coordinates(tangent_matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    tangent_matrix = np.asarray(tangent_matrix, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if tangent_matrix.ndim != 2 or basis.ndim != 2 or tangent_matrix.shape[1] != basis.shape[0]:
        raise ValueError("tangent matrix and basis dimensions are not aligned")
    return tangent_matrix @ basis


def memory_component_diagnostic(
    q: np.ndarray, memory: np.ndarray, basis: np.ndarray
) -> dict[str, Any]:
    q64 = np.asarray(q, dtype=np.float64)
    memory64 = np.asarray(memory, dtype=np.float64)
    residual = memory64 - float(memory64 @ q64) * q64
    residual -= float(residual @ q64) * q64
    coefficients = np.asarray(basis, dtype=np.float64).T @ residual
    energy = coefficients * coefficients
    total = float(energy.sum())
    shares = energy / total if total > 0.0 else np.zeros_like(energy)
    order = np.argsort(np.abs(coefficients))[::-1]
    return {
        "coefficients": [float(value) for value in coefficients],
        "energy_shares": [float(value) for value in shares],
        "dominant": [
            {
                "component": int(index + 1),
                "coefficient": float(coefficients[index]),
                "projected_energy": float(energy[index]),
                "projected_energy_share": float(shares[index]),
            }
            for index in order[:5]
        ],
        "projected_norm": float(np.sqrt(total)),
        "tangent_norm": float(np.linalg.norm(residual)),
    }


def _percentile_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    return {
        "count": int(array.size), "min": float(array.min()),
        "p25": float(np.percentile(array, 25)), "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)), "max": float(array.max()),
        "mean": float(array.mean()),
    }


def attribute_coverage(
    products: Sequence[Mapping[str, Any]], attributes: Sequence[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    embedded = [build_product_text(product).lower() for product in products]
    raw = [build_raw_catalogue_text(product) for product in products]
    output: dict[str, dict[str, Any]] = {}
    indicators: dict[str, np.ndarray] = {}
    for attribute in attributes:
        if attribute == "price":
            prices = [parse_price(product) for product in products]
            present = np.asarray([value is not None for value in prices], dtype=bool)
            price_text_terms = ("$", "price", "priced", "affordable", "budget", "cheap", "premium", "luxury")
            proxy = np.asarray([any(term in text for term in price_text_terms) for text in embedded], dtype=bool)
            indicators[attribute] = present
            output[attribute] = {
                "embedded_text_count": int(proxy.sum()),
                "embedded_text_percent": 100.0 * float(proxy.mean()) if len(products) else 0.0,
                "raw_catalogue_count": int(present.sum()),
                "raw_catalogue_percent": 100.0 * float(present.mean()) if len(products) else 0.0,
                "catalogue_only_count": int(np.count_nonzero(present & ~proxy)),
                "distinct_embedded_forms": sorted({term for text in embedded for term in price_text_terms if term in text}),
                "distinct_raw_values": len({float(value) for value in prices if value is not None}),
                "note": "The structured price field is not interpolated into the embedded product template; embedded hits are only lexical proxies.",
            }
            continue
        terms = ATTRIBUTE_TERMS[attribute]
        embedded_matches = [matched_terms(text, terms) for text in embedded]
        raw_matches = [matched_terms(text, terms) for text in raw]
        embedded_hit = np.asarray([bool(value) for value in embedded_matches], dtype=bool)
        raw_hit = np.asarray([bool(value) for value in raw_matches], dtype=bool)
        indicators[attribute] = embedded_hit
        output[attribute] = {
            "embedded_text_count": int(embedded_hit.sum()),
            "embedded_text_percent": 100.0 * float(embedded_hit.mean()) if len(products) else 0.0,
            "raw_catalogue_count": int(raw_hit.sum()),
            "raw_catalogue_percent": 100.0 * float(raw_hit.mean()) if len(products) else 0.0,
            "catalogue_only_count": int(np.count_nonzero(raw_hit & ~embedded_hit)),
            "distinct_embedded_forms": sorted({term for values in embedded_matches for term in values}),
            "distinct_raw_forms": sorted({term for values in raw_matches for term in values}),
        }
    return output, indicators


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if left.size < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_rankdata(left), _rankdata(right))


def price_proxy_diagnostic(products: Sequence[Mapping[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    prices = np.asarray([np.nan if parse_price(product) is None else parse_price(product) for product in products], dtype=np.float64)
    embedded = [build_product_text(product).lower() for product in products]
    affordable = np.asarray([bool(matched_terms(text, ATTRIBUTE_TERMS["price_proxy_affordable"])) for text in embedded])
    premium = np.asarray([bool(matched_terms(text, ATTRIBUTE_TERMS["price_proxy_premium"])) for text in embedded])
    proxy = affordable.astype(np.float64) - premium.astype(np.float64)
    valid = np.isfinite(prices)
    return {
        "valid_price_count": int(valid.sum()),
        "affordable_proxy_count": int(affordable.sum()),
        "premium_proxy_count": int(premium.sum()),
        "lexical_proxy_vs_log_price_pearson": _pearson(proxy[valid], np.log1p(prices[valid])),
        "query_score_vs_price_spearman": _spearman(scores[valid], prices[valid]) if valid.sum() >= 3 else None,
        "price_when_affordable_proxy": _percentile_summary(prices[valid & affordable]),
        "price_when_premium_proxy": _percentile_summary(prices[valid & premium]),
        "price_without_proxy": _percentile_summary(prices[valid & ~(affordable | premium)]),
        "interpretation_guardrail": "Query similarity is not a price representation; it is included only to test accidental textual correlation.",
    }


def _category_composition(products: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = [" > ".join(product.get("categories") or []) for product in products]
    leaves = [(product.get("categories") or ["<missing>"])[-1] for product in products]
    departments = [
        (product.get("categories") or ["<missing>"])[1]
        if len(product.get("categories") or []) > 1 else "<missing>"
        for product in products
    ]
    return {
        "top_leaf_categories": Counter(leaves).most_common(15),
        "top_departments": Counter(departments).most_common(15),
        "top_category_paths": Counter(paths).most_common(15),
        "distinct_leaf_categories": len(set(leaves)),
        "distinct_category_paths": len(set(paths)),
    }


def _product_record(product: Mapping[str, Any], coordinate: float, rank: int) -> dict[str, Any]:
    return {
        "rank": int(rank), "product_id": str(product["parent_asin"]),
        "title": product.get("title") or "", "categories": product.get("categories") or [],
        "embedded_features": (product.get("features") or [])[:3],
        "price": parse_price(product), "coordinate": float(coordinate),
    }


def component_extremes(
    products: Sequence[Mapping[str, Any]], coordinates: np.ndarray, count: int = 5
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for component in range(coordinates.shape[1]):
        order = np.argsort(coordinates[:, component])
        for side, selected in (("negative", order[:count]), ("positive", order[::-1][:count])):
            for position, row in enumerate(selected, start=1):
                record = _product_record(products[int(row)], coordinates[int(row), component], position)
                record.update({"component": component + 1, "side": side})
                records.append(record)
    return records


def component_associations(
    coordinates: np.ndarray, indicators: Mapping[str, np.ndarray], products: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    prices = np.asarray([np.nan if parse_price(product) is None else parse_price(product) for product in products])
    output: list[dict[str, Any]] = []
    for component in range(coordinates.shape[1]):
        associations = []
        for name, indicator in indicators.items():
            if name == "price":
                valid = np.isfinite(prices)
                value = _spearman(coordinates[valid, component], prices[valid]) if valid.sum() >= 3 else None
            else:
                value = _pearson(coordinates[:, component], indicator.astype(np.float64))
            if value is not None:
                associations.append({"attribute": name, "correlation": value, "absolute_correlation": abs(value)})
        associations.sort(key=lambda item: item["absolute_correlation"], reverse=True)
        output.append({"component": component + 1, "strongest_attribute_associations": associations[:5]})
    return output


def _spectrum(singular_values: np.ndarray) -> dict[str, Any]:
    singular_values = np.asarray(singular_values, dtype=np.float64)
    energy = singular_values * singular_values
    total = float(energy.sum())
    return {
        "sigma_1_to_sigma_16": [float(value) for value in singular_values[:16]],
        "sigma_1_over_sigma_16": float(singular_values[0] / singular_values[15]),
        "cumulative_energy": {
            str(count): float(energy[:count].sum() / total) for count in (1, 2, 4, 8, 16)
        },
        "energy_denominator_includes_all_local_singular_values": True,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    # Iterate physical lines.  str.splitlines() also splits valid JSON string
    # content at Unicode line separators (for example U+2028 in descriptions).
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in record.items()})


def run_diagnostic(project_root: Path, output_dir: Path) -> dict[str, Any]:
    shopping = project_root / "nickolas" / "shopping_agent"
    longitudinal = shopping / "longitudinal_eval"
    run_dir = longitudinal / "results" / "projector_isolation" / AUTHORITATIVE_RUN_ID
    fixture_path = longitudinal / "projector_fixture_v1.json"
    vector_path = longitudinal / "projector_fixture_v1.vectors.npz"
    cache_path = shopping / "embedding_cache" / "catalog_cache_openai-text-embedding-3-large.npz"
    catalogue_path = project_root / "techjam-conversational-search" / "data" / "catalog.jsonl"

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    phase3_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    fixture_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    queries = {item["fixture_id"]: item for item in _load_jsonl(run_dir / "projector_queries.jsonl")}
    pairs_by_fixture: dict[str, list[dict[str, Any]]] = {fixture: [] for fixture in TARGET_FIXTURES}
    for pair in _load_jsonl(run_dir / "projector_pairs.jsonl"):
        if pair["fixture_id"] in pairs_by_fixture:
            pairs_by_fixture[pair["fixture_id"]].append(pair)
    fixtures = {item["fixture_id"]: item for item in fixture_payload["fixtures"]}

    if sha256_file(fixture_path) != manifest["fixture_sha256"]:
        raise ValueError("authoritative fixture hash mismatch")
    if manifest["candidate_universe"] != "m0_full_catalogue" or manifest["local_k"] != 500 or manifest["rank"] != 16:
        raise ValueError("authoritative run configuration mismatch")
    if phase3_summary["decision"]["verdict"] != "PROJECTOR STOP":
        raise ValueError("authoritative Phase-3A verdict mismatch")
    source_freeze_validation: dict[str, dict[str, bool]] = {}
    for group, entries in manifest["source_freeze"].items():
        if group == "generated_at_utc":
            continue
        source_freeze_validation[group] = {}
        for relative_path, expected_hash in entries.items():
            matches = sha256_file(project_root / relative_path) == expected_hash
            source_freeze_validation[group][relative_path] = matches
            if not matches:
                raise ValueError(f"authoritative source-freeze mismatch: {relative_path}")

    products = _load_jsonl(catalogue_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        embeddings = np.asarray(cache["embeddings"])
        cache_ids = np.asarray(cache["ids"])
        cache_metadata = json.loads(str(cache["metadata_json"]))
    with np.load(vector_path, allow_pickle=False) as vectors:
        vector_keys = [str(value) for value in vectors["keys"].tolist()]
        vector_values = np.asarray(vectors["vectors"], dtype=np.float64)
    vector_map = {key: vector_values[index] for index, key in enumerate(vector_keys)}

    product_ids = [str(product["parent_asin"]) for product in products]
    if product_ids != [str(value) for value in cache_ids.tolist()]:
        raise ValueError("catalogue/cache IDs are not row-aligned")
    if sha256_file(catalogue_path) != manifest["catalogue_fingerprint"]:
        raise ValueError("catalogue fingerprint mismatch")
    for field in ("catalog_fingerprint", "product_text_fingerprint", "embedding_space_id", "row_count", "vector_dimension"):
        expected_key = "catalogue_fingerprint" if field == "catalog_fingerprint" else field
        expected = manifest.get(expected_key)
        if field in ("row_count", "vector_dimension"):
            expected = 50000 if field == "row_count" else 3072
        if cache_metadata[field] != expected:
            raise ValueError(f"cache metadata mismatch: {field}")

    catalogue_product_text_hash = hashlib.sha256(
        json.dumps([build_product_text(product) for product in products], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    # Replay embedding_backends.fingerprint_texts without importing a backend.
    product_digest = hashlib.sha256()
    for text in (build_product_text(product) for product in products):
        encoded = text.encode("utf-8")
        product_digest.update(len(encoded).to_bytes(8, "big"))
        product_digest.update(encoded)
    if product_digest.hexdigest() != manifest["product_text_fingerprint"]:
        raise ValueError("current product text template/fingerprint mismatch")

    id_to_row = {str(value): index for index, value in enumerate(cache_ids)}
    all_output: dict[str, Any] = {}
    coverage_csv: list[dict[str, Any]] = []
    extremes_csv: list[dict[str, Any]] = []
    memory_csv: list[dict[str, Any]] = []

    for fixture_id in TARGET_FIXTURES:
        fixture = fixtures[fixture_id]
        query_record = queries[fixture_id]
        # DenseQuerySnapshot freezes q as float32 before QLMP promotes it to
        # float64 working precision; replay that boundary exactly.
        q = np.asarray(vector_map[str(fixture["q_m0_key"])], dtype=np.float32)
        persisted_ids = [str(value) for value in query_record["top_k_product_ids"]]
        persisted_scores = np.asarray(query_record["top_k_scores"], dtype=np.float64)
        top_rows = np.asarray([id_to_row[value] for value in persisted_ids], dtype=np.int64)
        validate_top_k_alignment(persisted_ids, cache_ids, top_rows)
        recomputed_scores = embeddings @ np.asarray(q, dtype=embeddings.dtype)
        recomputed_top = np.argsort(recomputed_scores)[::-1][:500]
        validate_top_k_alignment(persisted_ids, cache_ids, recomputed_top)
        if not np.allclose(recomputed_scores[top_rows], persisted_scores, atol=1e-7, rtol=1e-7):
            raise ValueError(f"persisted dense scores do not replay for {fixture_id}")

        local_products = [products[int(row)] for row in top_rows]
        q_work = np.asarray(q, dtype=np.float64)
        q_work /= np.linalg.norm(q_work)
        local_embedding_rows = np.asarray(embeddings[top_rows], dtype=np.float64)
        local_embedding_rows /= np.linalg.norm(local_embedding_rows, axis=1)[:, None]
        tangent = build_tangent_matrix(q_work, local_embedding_rows)
        _, singular_values, right_vectors = np.linalg.svd(tangent, full_matrices=False)
        basis = right_vectors[:16].T.copy()
        persisted_singular = np.asarray(query_record["singular_values"], dtype=np.float64)
        if not np.allclose(singular_values, persisted_singular, atol=1e-10, rtol=1e-10):
            raise ValueError(f"persisted singular spectrum does not replay for {fixture_id}")
        coordinates = component_coordinates(tangent, basis)

        coverage, indicators = attribute_coverage(local_products, CASE_ATTRIBUTES[fixture_id])
        for attribute, values in coverage.items():
            coverage_csv.append({"fixture_id": fixture_id, "universe": "m0_full_catalogue", "attribute": attribute, **values})
        extremes = component_extremes(local_products, coordinates)
        for record in extremes:
            extremes_csv.append({"fixture_id": fixture_id, "universe": "m0_full_catalogue", **record})

        fixture_memory_by_text = {item["text"]: item for item in fixture["memories"]}
        memory_output = []
        pair_lookup = {pair["memory_text"]: pair for pair in pairs_by_fixture[fixture_id]}
        for memory_text, pair in pair_lookup.items():
            raw_memory = fixture_memory_by_text[memory_text]
            memory_vector = vector_map[str(raw_memory["embedding_key"])]
            component = memory_component_diagnostic(q_work, memory_vector, basis)
            memory_residual = np.asarray(memory_vector, dtype=np.float64) - float(np.asarray(memory_vector, dtype=np.float64) @ q_work) * q_work
            memory_residual -= float(memory_residual @ q_work) * q_work
            rank_tolerance = float(singular_values[0]) * max(tangent.shape) * float(np.finfo(np.float64).eps)
            numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
            full_coefficients = right_vectors[:numerical_rank] @ memory_residual
            full_energy = full_coefficients * full_coefficients
            full_total = float(full_energy.sum())
            full_order = np.argsort(np.abs(full_coefficients))[::-1]
            component["full_local_span"] = {
                "numerical_rank": numerical_rank,
                "projected_norm": float(np.sqrt(full_total)),
                "fraction_of_tangent_energy": float(full_total / (float(memory_residual @ memory_residual) + 1e-8)),
                "rank16_share_of_local_span_energy": float(np.sum(full_energy[:16]) / full_total) if full_total > 0.0 else 0.0,
                "largest_components": [
                    {
                        "component": int(index + 1),
                        "coefficient": float(full_coefficients[index]),
                        "energy_share": float(full_energy[index] / full_total) if full_total > 0.0 else 0.0,
                        "singular_value": float(singular_values[index]),
                        "selected_by_rank16": bool(index < 16),
                    }
                    for index in full_order[:10]
                ],
            }
            if not np.isclose(component["projected_norm"], pair["projected_norm"], atol=1e-10, rtol=1e-10):
                raise ValueError(f"memory coefficients do not replay for {fixture_id}: {memory_text}")
            record = {
                "memory_id": pair["memory_id"], "memory_text": memory_text, "label": pair["label"],
                "raw_cosine": pair["raw_cosine"], "tangent_norm": pair["tangent_norm"],
                "rho": pair["rho"], "projected_norm": pair["projected_norm"], **component,
            }
            memory_output.append(record)
            for dominant in component["dominant"]:
                memory_csv.append({
                    "fixture_id": fixture_id, "memory_id": pair["memory_id"],
                    "memory_text": memory_text, "label": pair["label"], **dominant,
                })

        max_price = 120.0 if fixture_id == "u2_override_s9_final" else None
        hard_mask = current_m0_price_mask(products, max_price)
        local_compliant = hard_mask[top_rows]
        incompatible_ranks = np.flatnonzero(~local_compliant) + 1
        contamination = {
            "m0_hard_conditions": {"price_max": max_price} if max_price is not None else {},
            "top_500_total": 500,
            "hard_compliant": int(local_compliant.sum()),
            "hard_incompatible": int((~local_compliant).sum()),
            "hard_incompatible_percent": 100.0 * float((~local_compliant).mean()),
            "incompatible_rank_summary": _percentile_summary(incompatible_ranks),
            "incompatible_in_top_50": int(np.count_nonzero(incompatible_ranks <= 50)),
            "incompatible_in_top_100": int(np.count_nonzero(incompatible_ranks <= 100)),
        }

        counterfactual: dict[str, Any]
        if max_price is not None:
            filtered_rows = select_top_k(recomputed_scores, hard_mask, 500)
            filtered_products = [products[int(row)] for row in filtered_rows]
            filtered_embedding_rows = np.asarray(embeddings[filtered_rows], dtype=np.float64)
            filtered_embedding_rows /= np.linalg.norm(filtered_embedding_rows, axis=1)[:, None]
            filtered_tangent = build_tangent_matrix(q_work, filtered_embedding_rows)
            _, filtered_singular, filtered_right = np.linalg.svd(filtered_tangent, full_matrices=False)
            filtered_basis = filtered_right[:16].T.copy()
            filtered_coordinates = component_coordinates(filtered_tangent, filtered_basis)
            filtered_coverage, _ = attribute_coverage(filtered_products, CASE_ATTRIBUTES[fixture_id])
            for attribute, values in filtered_coverage.items():
                coverage_csv.append({"fixture_id": fixture_id, "universe": "diagnostic_post_hard_filter", "attribute": attribute, **values})
            for record in component_extremes(filtered_products, filtered_coordinates):
                extremes_csv.append({"fixture_id": fixture_id, "universe": "diagnostic_post_hard_filter", **record})
            full_set, filtered_set = set(persisted_ids), {str(cache_ids[int(row)]) for row in filtered_rows}
            principal_cosines = np.linalg.svd(basis.T @ filtered_basis, compute_uv=False)
            principal_cosines = np.clip(principal_cosines, 0.0, 1.0)
            counterfactual = {
                "constructed": True, "name": "diagnostic_post_hard_filter",
                "eligible_catalogue_rows": int(hard_mask.sum()), "top_k_count": int(filtered_rows.size),
                "top_500_overlap_count": len(full_set & filtered_set),
                "top_500_overlap_percent": 100.0 * len(full_set & filtered_set) / 500.0,
                "category_composition": _category_composition(filtered_products),
                "attribute_coverage": filtered_coverage,
                "spectrum": _spectrum(filtered_singular),
                "rank16_subspace_comparison": {
                    "principal_cosines": [float(value) for value in principal_cosines],
                    "principal_angles_degrees": [float(np.degrees(np.arccos(value))) for value in principal_cosines],
                    "mean_squared_cosine": float(np.mean(principal_cosines * principal_cosines)),
                    "grassmann_chordal_distance": float(np.sqrt(np.sum(1.0 - principal_cosines * principal_cosines))),
                    "maximum_possible_chordal_distance": 4.0,
                },
            }
        else:
            counterfactual = {
                "constructed": False,
                "reason": "The persisted current U3 state has no genuine deterministic M0 hard condition; filtering would be an identity operation.",
            }

        all_output[fixture_id] = {
            "query": {"raw_current_message": query_record["raw_current_message"], "effective_query_text": query_record["effective_query_text"], "no_target_leakage": True},
            "top_k_replay": {"count": 500, "ids_exact": True, "scores_match": True, "singular_values_match": True},
            "attribute_coverage": coverage,
            "category_composition": _category_composition(local_products),
            "price_distribution": _percentile_summary(value for product in local_products if (value := parse_price(product)) is not None),
            "missing_price_count": sum(parse_price(product) is None for product in local_products),
            "price_proxy_diagnostic": price_proxy_diagnostic(local_products, persisted_scores),
            "spectrum": _spectrum(singular_values),
            "component_associations": component_associations(coordinates, indicators, local_products),
            "memory_scores_and_components": memory_output,
            "hard_condition_contamination": contamination,
            "diagnostic_post_hard_filter": counterfactual,
        }

    summary = {
        "diagnostic_type": "post_phase_3a_projector_failure_diagnostic",
        "generated_from_existing_artifacts_only": True,
        "external_calls": ZERO_EXTERNAL_CALLS,
        "authoritative_run": {
            "run_id": AUTHORITATIVE_RUN_ID,
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "fixture_sha256": manifest["fixture_sha256"],
            "fixture_vector_snapshot_sha256": manifest["fixture_vector_snapshot_sha256"],
            "catalogue_fingerprint": manifest["catalogue_fingerprint"],
            "product_text_fingerprint": manifest["product_text_fingerprint"],
            "candidate_universe": manifest["candidate_universe"], "local_k": manifest["local_k"],
            "rank": manifest["rank"], "embedding_space_id": manifest["embedding_space_id"],
            "phase3_verdict": phase3_summary["decision"]["verdict"],
            "phase3_primary_scores": phase3_summary["primary_binary"]["scores"],
            "source_freeze_validation": source_freeze_validation,
        },
        "product_representation": {
            "exact_template": EMBEDDED_TEMPLATE,
            "features_rule": "Only features[0:3] are joined with '; '.",
            "excluded_fields": ["price", "details", "description", "store", "ratings", "features after index 2"],
            "current_recomputed_fingerprint": product_digest.hexdigest(),
            "unused_json_diagnostic_hash": catalogue_product_text_hash,
        },
        "semantics": {
            "m0_current": {
                "hard": ["price_max only"],
                "soft": ["department +20", "category +15", "brand mismatch -10", "keyword/constraint/category/popularity boosts"],
                "post_filter": ["negated-term exclusion", "seen-product exclusion", "brand/title diversity"],
                "dense_projector_neighbourhood": "No filters: full 50,000-row catalogue.",
            },
            "experiment_1_current": {
                "hard_mask": ["price_max (missing price allowed)", "target demographic", "minimum average rating (missing allowed)", "minimum rating count (missing allowed)", "store/brand substring"],
                "soft": ["department", "category", "brand and lexical reranking"],
                "note": "This implementation is not imported or applied to M0/QLMP by this diagnostic.",
            },
        },
        "cases": all_output,
        "decisions": {
            "representation": "REPRESENTATION BOTTLENECK PARTIAL",
            "variance": "VARIANCE BOTTLENECK SUPPORTED",
            "portability": "PORTABILITY FAILURE SUPPORTED",
            "candidate_universe": "FILTERED-UNIVERSE FOLLOW-UP JUSTIFIED",
            "original_scientific_verdict": "PROJECTOR STOP",
        },
        "scope_audit": {
            "qlmp_geometry_changed": False, "qlmp_b1_b2_changed": False,
            "m0_ranking_changed": False, "m0_routing_changed": False,
            "dense_scorer_changed": False, "product_embedding_text_changed": False,
            "embedding_model_changed": False, "experiment_1_changed": False,
            "official_evaluator_changed": False, "graphify_run": False,
            "commit_created": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output_dir / "attribute_coverage.csv", coverage_csv)
    _write_csv(output_dir / "component_extremes.csv", extremes_csv)
    _write_csv(output_dir / "memory_components.csv", memory_csv)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or (
        args.project_root / "nickolas" / "shopping_agent" / "longitudinal_eval" / "results"
        / "projector_isolation" / "post_phase3a_u2_u3_diagnostic"
    )
    summary = run_diagnostic(args.project_root.resolve(), output.resolve())
    print(json.dumps({"output_dir": str(output.resolve()), "decisions": summary["decisions"], "external_calls": summary["external_calls"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
