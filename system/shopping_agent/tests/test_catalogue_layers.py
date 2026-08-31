from __future__ import annotations

import json

import numpy as np

from system.shopping_agent.catalogue import (
    BROWSING_FTS_OR_THRESHOLD, BROWSING_KEYWORD_ROUTE_THRESHOLD,
    BUYING_FTS_OR_THRESHOLD, BUYING_KEYWORD_ROUTE_THRESHOLD,
    Catalogue, FTS_AND_LIMIT, FTS_BM25_WEIGHTS, FTS_OR_THRESHOLD,
    KEYWORD_ROUTE_THRESHOLD, VECTOR_FALLBACK_LIMIT, allowed_departments,
    standardize_department,
)
from system.shopping_agent.clarification import select_best_attributes


def _product(index: int, *, rare: bool = False, department: str = "mens", rating=4.5, reviews=100):
    return {
        "parent_asin": f"asin-{index:02d}",
        "title": f"{'rare ' if rare else ''}common blue cotton shirt {index}",
        "features": ["blue cotton", "button closure"],
        "description": ["water-resistant everyday shirt"],
        "price": 10 + index,
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Shirts"],
        "details": {"Department": department, "Pattern": "solid"},
        "average_rating": rating,
        "rating_number": reviews,
        "store": "Example Brand",
    }


def _catalogue(tmp_path) -> Catalogue:
    path = tmp_path / "catalog.jsonl"
    rows = [_product(i, rare=i < 5) for i in range(20)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return Catalogue(path)


def test_frozen_fts_constants():
    assert FTS_AND_LIMIT == 1000
    assert FTS_OR_THRESHOLD == 15
    assert KEYWORD_ROUTE_THRESHOLD == 10
    assert VECTOR_FALLBACK_LIMIT == 150
    assert FTS_BM25_WEIGHTS == (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
    assert (BUYING_FTS_OR_THRESHOLD, BUYING_KEYWORD_ROUTE_THRESHOLD) == (15, 10)
    assert (BROWSING_FTS_OR_THRESHOLD, BROWSING_KEYWORD_ROUTE_THRESHOLD) == (30, 15)


def test_and_coverage_skips_or_and_low_and_appends_weighted_or(tmp_path):
    catalogue = _catalogue(tmp_path)
    broad = catalogue.fts_route(["common"])
    assert broad.and_count == 20
    assert broad.or_count == 0
    narrow = catalogue.fts_route(["rare", "blue"])
    assert narrow.and_count == 5
    assert narrow.or_count == 20
    assert len(narrow.row_indices) == 20
    browsing = catalogue.fts_route(["common"], or_threshold=BROWSING_FTS_OR_THRESHOLD)
    assert browsing.and_count == 20
    assert browsing.or_count == 20


def test_hard_masks_and_unknown_rating_review_benefit_of_doubt(tmp_path):
    catalogue = _catalogue(tmp_path)
    catalogue.avg_ratings[0] = 0.0
    catalogue.rating_numbers[0] = 0
    catalogue.departments[1] = "women"
    state = {
        "price_max": 15.0, "target_department": "men", "min_avg_rating": 4.8,
        "min_rating_number": 500, "store": "example", "negated_terms": {"forbidden"},
    }
    eligible = catalogue.eligibility(state)
    assert eligible.mask[0]  # unknown rating/review metadata gets benefit of doubt
    assert not eligible.mask[1]  # incompatible demographic
    assert not eligible.mask[6]  # over budget
    assert allowed_departments("men") >= {"men", "unisex-adult", "unspecified", "multi-demographic"}
    assert standardize_department("Unisex Adult") == "unisex-adult"


def test_generic_negative_terms_are_ignored_but_specific_terms_filter(tmp_path):
    catalogue = _catalogue(tmp_path)
    generic = catalogue.eligibility({"negated_terms": {"clothing", "shoes", "jewelry"}})
    assert generic.mask.tolist() == [True] * 20
    specific = catalogue.eligibility({"negated_terms": {"rare"}})
    assert specific.mask.tolist() == [False] * 5 + [True] * 15
    assert specific.negative_filtered_count == 5


def test_entropy_selector_is_deterministic_and_uses_row_map(tmp_path):
    catalogue = _catalogue(tmp_path)
    candidates = list(catalogue.ids[:10])
    remaining = {"brand", "budget", "color", "material", "pattern", "waterproof"}
    first = select_best_attributes(catalogue, candidates, remaining)
    second = select_best_attributes(catalogue, candidates, remaining)
    assert first == second
    assert len(first) == 2
    assert all(value in remaining for value in first)
    assert catalogue.row_by_asin[candidates[3]] == 3


def test_entropy_empty_candidate_priority_depends_on_live_intent(tmp_path):
    catalogue = _catalogue(tmp_path)
    remaining = {"material", "brand", "style", "use_case", "budget"}
    assert select_best_attributes(catalogue, [], remaining, intent_mode="buying") == ["material", "brand"]
    assert select_best_attributes(catalogue, [], remaining, intent_mode="browsing") == ["use_case", "style"]
