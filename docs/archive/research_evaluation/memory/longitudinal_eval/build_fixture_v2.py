"""Build the frozen v2 fixture from catalogue evidence, never from M3 ranks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE.parents[2] / "techjam-conversational-search" / "data" / "catalog.jsonl"
DEFAULT_OUTPUT = HERE / "users_40_v2.json"

TRAITS = [
    ("boots", "waterproof"), ("necklaces", "silver"), ("dresses", "floral"),
    ("backpacks", "leather"), ("jackets", "waterproof"), ("sandals", "leather"),
    ("earrings", "silver"), ("shirts", "cotton"), ("socks", "wool"),
    ("watches", "stainless steel"),
]
OVERRIDES = [
    ("dresses", "black", "red"), ("backpacks", "black", "blue"),
    ("jackets", "black", "pink"), ("boots", "black", "white"),
    ("shirts", "black", "blue"), ("sandals", "black", "pink"),
    ("necklaces", "silver", "gold"), ("earrings", "silver", "gold"),
    ("watches", "black", "blue"), ("socks", "black", "white"),
]
IRRELEVANT = [
    ("necklaces", "silver", "hiking boots"), ("hiking boots", "waterproof", "necklaces"),
    ("watches", "stainless steel", "dresses"), ("earrings", "gold", "rain jackets"),
    ("sandals", "leather", "watches"), ("backpacks", "blue", "earrings"),
    ("dresses", "floral", "work boots"), ("shirts", "cotton", "necklaces"),
    ("socks", "wool", "backpacks"), ("jackets", "waterproof", "sandals"),
]


def _singular(category: str) -> str:
    return category[:-1] if category.endswith("s") else category


def _text(product: dict[str, Any]) -> str:
    return json.dumps(product, ensure_ascii=False).casefold()


def _select(products: list[dict[str, Any]], category: str, trait: str | None = None, count: int = 5) -> list[dict[str, Any]]:
    category_term = _singular(category).casefold()
    selected = [p for p in products if category_term in _text(p) and (not trait or trait.casefold() in _text(p))]
    if len(selected) < count:
        raise RuntimeError(f"catalogue has only {len(selected)} rows for {category!r}/{trait!r}")
    return sorted(selected, key=lambda p: str(p["parent_asin"]))[:count]


def _audit(products: list[dict[str, Any]], category: str, trait: str | None) -> dict[str, Any]:
    return {str(p["parent_asin"]): {"catalogue_verified": True, "category_term": category,
            "historical_or_current_trait": trait, "title": p.get("title"),
            "categories": p.get("categories", [])} for p in products}


def _scopes(persistent: list[str] | None = None, session: list[str] | None = None) -> dict[str, list[str]]:
    return {"persistent": persistent or [], "session_specific": session or [], "unknown": []}


def _setup(sequence: int, category: str, trait: str) -> dict[str, Any]:
    return {"sequence_index": sequence, "session_role": "setup",
            "scripted_turns": [f"I'm looking for {category}.", f"For that, what matters is: {trait}."],
            "fact_scope_annotations": _scopes([trait], [category]), "setup_sequence_references": [],
            "scenario_class": None, "expected_behavior": None}


def _probe(category: str, relevant: list[dict[str, Any]], trait: str | None, *,
           messages: list[str] | None = None, sufficient: bool, refs: list[int]) -> dict[str, Any]:
    asins = [str(p["parent_asin"]) for p in relevant]
    return {"sequence_index": 2, "session_role": "probe",
            "scripted_turns": messages or [f"I'm looking for {category}."],
            "target_asin": asins[0], "relevant_asins": asins,
            "relevant_asin_audit": _audit(relevant, category, trait),
            "fact_scope_annotations": _scopes([], [category]),
            "setup_sequence_references": refs,
            "intended_historical_facts": [] if trait is None else [trait],
            "irrelevant_historical_facts": [],
            "historical_fact_leakage_terms": [] if trait is None else [trait],
            "current_query_alone_sufficient": sufficient}


def build(catalog: Path) -> dict[str, Any]:
    products = [json.loads(line) for line in catalog.open(encoding="utf-8")]
    users: list[dict[str, Any]] = []
    for index, (category, trait) in enumerate(TRAITS):
        relevant = _select(products, category, trait)
        probe = _probe(category, relevant, trait, sufficient=False, refs=[0, 1])
        users.append({"user_id": f"lp_{index:02d}", "constant_profile": {},
                      "scenario_class": "LONGITUDINAL_POSITIVE", "expected_behavior": "HELP",
                      "buyer_mode": "Buying", "memory_relation": "RELEVANT",
                      "sessions": [_setup(0, category, trait), _setup(1, category, trait), probe]})
    for index, (history_category, history_trait, current_category) in enumerate(IRRELEVANT):
        relevant = _select(products, current_category, None)
        probe = _probe(current_category, relevant, None, sufficient=True, refs=[0, 1])
        probe["irrelevant_historical_facts"] = [history_category, history_trait]
        probe["semantic_separation_audit"] = {
            "history_category": history_category, "current_category": current_category,
            "basis": "disjoint catalogue product families selected before M3 scoring"}
        users.append({"user_id": f"mi_{index:02d}", "constant_profile": {},
                      "scenario_class": "MEMORY_IRRELEVANT", "expected_behavior": "IGNORE",
                      "buyer_mode": "Buying", "memory_relation": "IRRELEVANT",
                      "sessions": [_setup(0, history_category, history_trait),
                                   _setup(1, history_category, history_trait), probe]})
    for index, (category, old, new) in enumerate(OVERRIDES):
        relevant = _select(products, category, new)
        conflicting = _select(products, category, old, count=3)
        messages = [f"I'm looking for {category}.",
                    f"Actually, ignore my earlier preference. What I need is: {new} {category}."]
        probe = _probe(category, relevant, new, messages=messages, sufficient=True, refs=[0, 1])
        probe["intended_historical_facts"] = [old]
        probe["historical_fact_leakage_terms"] = []
        probe["explicit_conflict"] = {"historical": old, "current": new}
        probe["history_matching_current_conflicting_asins"] = [p["parent_asin"] for p in conflicting]
        users.append({"user_id": f"co_{index:02d}", "constant_profile": {},
                      "scenario_class": "CURRENT_OVERRIDE", "expected_behavior": "DO_NOT_OVERRIDE",
                      "buyer_mode": "Buying", "memory_relation": "CONFLICTING",
                      "sessions": [_setup(0, category, old), _setup(1, category, old), probe]})
    for index, (category, trait) in enumerate(reversed(TRAITS)):
        relevant = _select(products, category, trait)
        probe = _probe(category, relevant, trait, sufficient=False, refs=[0, 1])
        users.append({"user_id": f"bp_{index:02d}", "constant_profile": {},
                      "scenario_class": "BROWSING_PERSONALIZATION", "expected_behavior": "PERSONALIZE",
                      "buyer_mode": "Browsing", "memory_relation": "RELEVANT",
                      "sessions": [_setup(0, category, trait), _setup(1, category, trait), probe]})
    return {"schema_version": "nickolas-longitudinal-fixture-v2",
            "frozen_before_m3": True,
            "selection_policy": "catalogue category/trait evidence only; lexicographic ASIN; no M3 outcomes",
            "users": users}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.catalog), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
