from __future__ import annotations

import json
from pathlib import Path
import statistics

from system.shopping_agent.catalogue import Catalogue
from system.shopping_agent.category_resolver import CategoryResolver, coarse_category
from system.shopping_agent.config import CATALOG_PATH, PROJECT_ROOT
from system.shopping_agent.turn_parser import resolver_soft_slot_values


def test_resolver_uses_labels_content_and_has_defined_zero_and_tie_confidence(tmp_path):
    path = tmp_path / "catalog.jsonl"
    rows = [
        {"parent_asin": "b1", "title": "Trail shared boot", "categories": ["Men", "Shoes", "Boots"]},
        {"parent_asin": "b2", "title": "Winter boot", "categories": ["Men", "Shoes", "Boots"]},
        {"parent_asin": "d1", "title": "Formal shared gown", "categories": ["Women", "Clothing", "Dresses"]},
        {"parent_asin": "d2", "title": "Summer dress", "categories": ["Women", "Clothing", "Dresses"]},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    catalogue = Catalogue(path)
    resolver = CategoryResolver(catalogue)

    ranked, confidence = resolver.resolve(["trail boot"])
    assert ranked[0] == "shoes boots"
    assert confidence > 0
    assert resolver.resolve([]) == ((), 0.0)

    _, zero_confidence = resolver.resolve(["token-not-in-any-product"])
    assert zero_confidence == 0.0
    _, tie_confidence = resolver.resolve(["shared"])
    assert tie_confidence == 0.0
    catalogue.close()


def test_soft_slot_arm_retains_winstons_frozen_metrics():
    archive = PROJECT_ROOT / "archive" / "winston"
    if not archive.exists():
        __import__("pytest").skip("archived Winston benchmark is not present in this checkout")
    frozen = json.loads((archive / "lab" / "resolver_results.json").read_text(encoding="utf-8"))
    assert frozen["B phrase + soft slots"] == {
        "top1": 0.267,
        "top3": 0.433,
        "median_rank": 10.0,
        "median_pool_at_top3": 206.0,
    }
    gold = {
        row["case"]: row
        for row in json.loads((archive / "probe_gold.json").read_text(encoding="utf-8"))
    }
    predictions = {
        row["case"]: row["pred"]
        for row in json.loads(
            (archive / "preds-qwen2.5-7b-instruct.json").read_text(encoding="utf-8")
        )
    }
    catalogue = Catalogue(CATALOG_PATH)
    resolver = CategoryResolver(catalogue)

    ranks: list[int] = []
    top1 = 0
    top3 = 0
    for case, fixture in gold.items():
        parse = predictions[case]
        terms = [
            parse.get("category_phrase") or "",
            *resolver_soft_slot_values(parse, resolver.catalog_stores),
        ]
        ranked, _ = resolver.resolve(terms, top_n=len(resolver.bucket_names))
        product = catalogue.products[catalogue.row_by_asin[fixture["asin"]]]
        true_bucket = coarse_category(product.get("categories") or ())
        rank = ranked.index(true_bucket) + 1
        ranks.append(rank)
        top1 += rank == 1
        top3 += rank <= 3

    assert round(top1 / len(ranks), 3) == 0.267
    assert round(top3 / len(ranks), 3) >= 0.433
    assert statistics.median(ranks) <= 10
    catalogue.close()
