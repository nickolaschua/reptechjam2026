from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from system.shopping_agent.turn_parser import (
    CategoryResolutionError,
    SCHEMA,
    ParserRequestError,
    WinstonTurnParser,
    clean_parse,
    negation_supported,
    parsed_turn_from_raw,
    validate_raw_parse,
)
from system.shopping_agent.ollama_client import OllamaClient


class StubResolver:
    catalog_stores = frozenset({"nike", "example brand"})

    def resolve(self, terms, *, top_n=3):
        assert top_n == 3
        return ("men shoes", "women shoes", "athletic shoes")[:top_n], 0.25


def raw(*, category="boots", department=None, slots=(), price_min=None, price_max=None):
    return {
        "category_phrase": category,
        "department": department,
        "slots": list(slots),
        "price_max": price_max,
        "price_min": price_min,
        "quality_prior": "none",
        "exploring": False,
        "specificity": "type_with_requirements",
    }


def slot(attribute, value, *, declined=False, negated=False):
    return {
        "attribute": attribute,
        "value": value,
        "declined": declined,
        "negated": negated,
    }


def test_schema_is_closed_and_raw_shape_is_validated():
    assert SCHEMA["additionalProperties"] is False
    assert SCHEMA["properties"]["slots"]["items"]["additionalProperties"] is False
    validate_raw_parse(raw())
    with pytest.raises(ValueError, match="missing required"):
        validate_raw_parse({"category_phrase": "boots"})
    with pytest.raises(ValueError, match="unknown keys"):
        validate_raw_parse({**raw(), "invented": True})


def test_parser_retries_once_and_sends_schema_to_ollama():
    calls = []

    def transport(url, body, timeout):
        calls.append((url, json.loads(body), timeout))
        if len(calls) == 1:
            raise TimeoutError("slow")
        return {
            "model": "llama3.1:8b",
            "message": {"content": json.dumps(raw(category="running shoes"))},
        }

    client = OllamaClient(
        model="llama3.1:8b",
        host="http://ollama.test",
        timeout_seconds=30,
        transport=transport,
    )
    parser = WinstonTurnParser(
        StubResolver(),
        client=client,
    )
    parsed = parser.parse("running shoes", 1)
    assert parsed.category == "running shoes"
    assert parsed.resolver_candidates == ("men shoes", "women shoes", "athletic shoes")
    assert len(calls) == 2
    assert calls[-1][0] == "http://ollama.test/api/chat"
    assert calls[-1][1]["model"] == "llama3.1:8b"
    assert calls[-1][1]["messages"][0]["role"] == "user"
    assert calls[-1][1]["format"] == SCHEMA
    assert calls[-1][1]["options"] == {"temperature": 0, "num_predict": 512}
    assert calls[-1][2] == 30


def test_parser_raises_typed_error_after_the_retry():
    calls = 0

    def fail(*_):
        nonlocal calls
        calls += 1
        raise OSError("offline")

    client = OllamaClient(
        model="llama3.1:8b", timeout_seconds=1, transport=fail
    )
    parser = WinstonTurnParser(StubResolver(), client=client)
    with pytest.raises(ParserRequestError) as captured:
        parser.parse("boots", 1)
    assert calls == 2
    assert captured.value.model == "llama3.1:8b"
    assert captured.value.attempts == 2
    assert captured.value.latency_seconds >= 0


def test_resolver_failure_is_typed_and_never_silently_falls_back():
    class BrokenResolver(StubResolver):
        def resolve(self, terms, *, top_n=3):
            raise LookupError("resolver unavailable")

    client = OllamaClient(
        model="llama3.1:8b",
        timeout_seconds=1,
        transport=lambda *_: {
            "model": "llama3.1:8b",
            "message": {"content": json.dumps(raw())},
        },
    )
    parser = WinstonTurnParser(BrokenResolver(), client=client)
    with pytest.raises(CategoryResolutionError, match="resolver unavailable") as captured:
        parser.parse("boots", 1)
    assert captured.value.model == "llama3.1:8b"


def test_cleaning_drops_junk_duplicates_bad_prices_and_unsupported_department():
    parsed = clean_parse(
        raw(
            department="womens",
            price_min=-1,
            price_max=0,
            slots=(
                slot("brand", "not specified"),
                slot("color", "black"),
                slot("color", "black"),
            ),
        ),
        "show me black boots",
    )
    assert parsed["department"] is None
    assert parsed["price_min"] is parsed["price_max"] is None
    assert parsed["slots"] == [slot("color", "black")]


def test_tiers_wearer_price_negation_model_code_intent_and_message_type():
    message = "Need Nike boots for my husband over $40 and under $100, no leather, code BM8242-08E"
    parsed = parsed_turn_from_raw(
        raw(
            department="mens",
            price_min=40,
            price_max=100,
            slots=(
                slot("brand", "Nike"),
                slot("material", "waterproof"),
                slot("material", "leather", negated=True),
                slot("size", "long sleeve"),
            ),
        ),
        message,
        stores=StubResolver.catalog_stores,
        resolver_candidates=("men shoes",),
        resolver_confidence=0.2,
    )
    tiers = {(item.attribute, item.value): item.tier for item in parsed.positive_slots}
    assert tiers[("brand", "Nike")] == "hard"
    assert tiers[("material", "waterproof")] == "soft"
    assert tiers[("size", "long sleeve")] == "soft"
    assert [(item.attribute, item.value) for item in parsed.negatives] == [("material", "leather")]
    assert parsed.department == "mens"
    assert (parsed.price_min, parsed.price_max) == (40.0, 100.0)
    assert parsed.model_code == "BM8242-08E"
    assert parsed.intent == "buying"
    assert parsed.message_type == "exact"


def test_negation_must_precede_the_value_within_the_bounded_window():
    assert negation_supported("he hates anything that feels like plastic", "plastic")
    assert negation_supported("nothing with a bunch of logos", "logos")
    assert not negation_supported("brown leather that is not too heavy", "leather")


def test_compatibility_and_symptom_types_are_derived_without_model_judgment():
    compat = parsed_turn_from_raw(
        raw(category="band"),
        "I own a watch and need a band to fit my watch",
        stores=frozenset(),
    )
    symptom = parsed_turn_from_raw(
        raw(category=""),
        "my feet hurt",
        stores=frozenset(),
    )
    assert (compat.message_type, compat.intent) == ("compatibility", "buying")
    assert symptom.message_type == "symptom"


def test_archived_probe30_scoring_floor_remains_locked():
    root = Path(__file__).resolve().parents[3]
    archive = root / "archive" / "winston"
    if not archive.exists():
        pytest.skip("archived Winston benchmark is not present in this checkout")
    spec = importlib.util.spec_from_file_location("archived_winston_nlp_parse", archive / "nlp_parse.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    cases, _ = module.load_gold()
    by_case = {row["case"]: row for row in cases}
    # Historical predictions remain read-only evidence for the scoring contract.
    predictions = json.loads(
        (archive / "preds-qwen2.5-7b-instruct.json").read_text(encoding="utf-8")
    )
    rows = [
        module.score(
            row["pred"],
            by_case[row["case"]]["gold"],
            by_case[row["case"]]["discard_spans"],
            by_case[row["case"]]["utterance"],
        )
        for row in predictions
    ]
    assert len(rows) == 30
    assert sum(row["slot_f1"] for row in rows) / len(rows) >= 0.441
