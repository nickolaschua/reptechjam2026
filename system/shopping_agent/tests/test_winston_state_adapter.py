from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from system.shopping_agent.agent import Agent, _state_to_retrieval_query
from system.shopping_agent.memory_store import InMemoryVectorMemoryStore
from system.shopping_agent.category_resolver import coarse_category
from system.shopping_agent.config import CATALOG_PATH
from system.shopping_agent.turn_parser import ParsedSlot, ParsedTurn, ParserRequestError
from system.shopping_agent.vector_memory import DEFAULT_VECTOR_MEMORY_CONFIG


def turn(
    category=None,
    *,
    slots=(),
    negatives=(),
    declined=(),
    price_min=None,
    price_max=None,
    department=None,
    confidence=0.5,
    message_type="feature",
    intent="browsing",
    model_code=None,
):
    return ParsedTurn(
        category=category,
        positive_slots=tuple(slots),
        negatives=tuple(negatives),
        declined_attributes=tuple(declined),
        price_min=price_min,
        price_max=price_max,
        department=department,
        specificity="type_with_requirements" if price_min or price_max else "type_with_wishes",
        intent=intent,
        message_type=message_type,
        model_code=model_code,
        resolver_candidates=("candidate-a", "candidate-b", "candidate-c"),
        resolver_confidence=confidence,
        raw_parse={},
    )


def slot(attribute, value, tier="soft"):
    return ParsedSlot(attribute, value, tier)


class QueueParser:
    model = "llama3.1:8b"

    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    def parse(self, message, turn_number):
        self.calls.append((message, turn_number))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def bare_agent() -> Agent:
    agent = Agent.__new__(Agent)
    agent._sessions = {}
    agent._active_lifecycle = {}
    agent._ended_lifecycle = {}
    agent._forensic_capture_sessions = set()
    agent._forensic_ranking_snapshots = {}
    agent._forensic_update_sessions = set()
    agent._forensic_update_vectors = {}
    agent.memory_store = InMemoryVectorMemoryStore()
    agent.embedding_space_id = "test-space"
    agent.vector_memory_config = DEFAULT_VECTOR_MEMORY_CONFIG
    agent.instrumentation = {"semantic_queries": [], "turns": [], "agent_errors": [], "parser_calls": []}
    return agent


def ranking_agent(parser: QueueParser) -> tuple[Agent, list[str]]:
    agent = bare_agent()
    agent.turn_parser = parser
    agent._template_buckets = {"men shoes"}
    agent.catalog_ids = ["a", "b"]
    agent.catalog_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    agent.catalog_prices = np.asarray([25.0, 75.0], dtype=np.float64)
    agent.catalog_metadata = {
        "a": {"title": "Black boot", "searchable_bag": "black cotton boot"},
        "b": {"title": "Blue shoe", "searchable_bag": "blue leather shoe"},
    }
    embedded = []

    def embed(text):
        embedded.append(text)
        return np.asarray([1.0, 0.0], dtype=np.float32)

    agent.embed_dense_query = embed
    agent._call_llm = lambda *args, **kwargs: "Here are matches. Which style do you prefer?"
    agent.reset("s", {})
    return agent, embedded


def apply(agent: Agent, parsed: ParsedTurn, message: str, turn_number: int) -> bool:
    return agent._apply_parsed_turn(agent._sessions["s"], parsed, message, turn_number)


def test_multiturn_preserves_positive_facts_and_reconciles_negation():
    agent = bare_agent()
    agent.reset("s", {})
    apply(agent, turn("boots", price_max=100, confidence=0.5), "boots under $100", 1)
    ask = apply(
        agent,
        turn(
            category="product",
            slots=(slot("color", "black"),),
            negatives=(slot("material", "leather", "hard"),),
            confidence=0.0,
        ),
        "black, no leather",
        2,
    )
    state = agent._sessions["s"]
    assert ask is False
    assert state["category"] == "boots"
    assert state["price_max"] == 100
    assert state["disclosed_slots"]["color"] == {"black"}
    assert "leather" in state["negated_terms"]
    assert "leather" not in _state_to_retrieval_query(state)


def test_same_kind_replacement_requires_explicit_correction_language():
    agent = bare_agent()
    agent.reset("s", {})
    apply(agent, turn("boots", slots=(slot("color", "black"),)), "black boots", 1)
    apply(agent, turn(slots=(slot("color", "blue"),)), "blue is also fine", 2)
    assert agent._sessions["s"]["disclosed_slots"]["color"] == {"black", "blue"}
    apply(agent, turn(slots=(slot("color", "red"),)), "Actually, make it red instead", 3)
    assert agent._sessions["s"]["disclosed_slots"]["color"] == {"red"}


def test_explicit_category_shift_clears_category_facts_but_preserves_budget():
    agent = bare_agent()
    agent.reset("s", {})
    apply(
        agent,
        turn("boots", slots=(slot("color", "black"),), price_max=100),
        "black boots under 100",
        1,
    )
    apply(
        agent,
        turn("dresses", slots=(slot("style", "floral"),), confidence=0.5),
        "Actually, switch to dresses instead",
        2,
    )
    state = agent._sessions["s"]
    assert state["category"] == "dresses"
    assert state["disclosed_slots"]["style"] == {"floral"}
    assert "color" not in state["disclosed_slots"]
    assert state["price_max"] == 100
    assert state["search_epoch"] == 1


def test_declines_department_price_range_brand_duplicates_and_provenance():
    agent = bare_agent()
    agent.reset("s", {})
    parsed = turn(
        "shirts",
        slots=(
            slot("brand", "Nike", "hard"),
            slot("brand", "Nike", "hard"),
            slot("material", "cotton", "hard"),
        ),
        price_min=20,
        price_max=80,
        department="mens",
        intent="buying",
    )
    apply(agent, parsed, "Nike cotton shirts for my husband from $20 to $80", 1)
    state = agent._sessions["s"]
    assert state["target_department"] == "men"
    assert state["store"] == "Nike"
    assert (state["price_min"], state["price_max"]) == (20, 80)
    assert state["disclosed_slots"]["brand"] == {"Nike"}
    active_nike = [
        record for record in state["constraint_provenance"]
        if record["attribute"] == "brand" and record["status"] == "active"
    ]
    assert len(active_nike) == 1
    apply(agent, turn(declined=("brand",)), "I don't care about brand", 2)
    assert "brand" not in state["disclosed_slots"]
    assert state["store"] == ""
    assert "brand" in state["asked_attributes"]


def test_session_only_and_negative_facts_never_enter_ltm_text():
    agent = bare_agent()
    captured = []
    agent.embed_dense_query = lambda text: captured.append(text) or np.asarray([1.0, 0.0], dtype=np.float32)
    agent.reset("s", {}, user_id="u", sequence_index=0)
    apply(
        agent,
        turn(
            "shirts",
            slots=(slot("color", "black"), slot("material", "cotton", "hard"), slot("brand", "Nike", "hard")),
            negatives=(slot("material", "leather", "hard"),),
            price_max=90,
            department="mens",
        ),
        "black cotton Nike shirts for my husband under 90, no leather",
        1,
    )
    agent.end_session("s")
    trace = agent.get_memory_debug("s")
    assert captured == ["color: black; material: cotton"]
    assert trace["preference_text"] == "color: black; material: cotton"
    assert all(value not in trace["preference_text"] for value in ("nike", "leather", "men", "90"))


def test_low_confidence_first_turn_asks_category_and_skips_embedding_and_generation():
    parser = QueueParser(
        turn("something", slots=(slot("use_case", "winter trip"),), confidence=0.19)
    )
    agent, embedded = ranking_agent(parser)
    generated = []
    agent._call_llm = lambda *args, **kwargs: generated.append(True) or "unexpected"
    response = agent.respond("s", "I need something for a winter trip", 1, 2, debug=True)
    assert response["ask_attribute"] == "category"
    assert response["recommendations"] == []
    assert response["message"] == "What specific kind of product are you looking for?"
    assert embedded == []
    assert generated == []
    assert agent._sessions["s"]["disclosed_slots"]["use_case"] == {"winter trip"}
    assert agent.instrumentation["turns"][-1]["route"] == "category-clarification"
    assert response["debug"]["memory_trace"]["catalog_rows_scored"] == 0
    assert response["debug"]["memory_trace"]["returned"] == []


def test_pending_category_answer_resumes_retrieval_even_with_flat_confidence():
    parser = QueueParser(
        turn("something", confidence=0.1),
        turn("shoes", confidence=0.0, message_type="product_type"),
    )
    agent, embedded = ranking_agent(parser)
    first = agent.respond("s", "show me something nice", 1, 2)
    second = agent.respond("s", "shoes", 2, 2, debug=True)
    assert first["ask_attribute"] == "category"
    assert second["recommendations"]
    assert embedded
    assert agent._sessions["s"]["category"] == "shoes"
    assert agent._sessions["s"]["pending_category"] is False


def test_pending_category_reply_without_a_category_asks_again():
    parser = QueueParser(
        turn("something", confidence=0.1),
        turn(None, slots=(slot("color", "black"),), confidence=0.0),
    )
    agent, embedded = ranking_agent(parser)
    agent.respond("s", "show me something", 1, 2)
    repeated = agent.respond("s", "black would be nice", 2, 2)
    assert repeated["ask_attribute"] == "category"
    assert repeated["recommendations"] == []
    assert embedded == []
    assert agent._sessions["s"]["disclosed_slots"]["color"] == {"black"}


@pytest.mark.parametrize("confidence", [0.20, 0.8])
def test_confidence_boundary_and_high_confidence_proceed(confidence):
    parser = QueueParser(turn("boots", confidence=confidence, message_type="product_type"))
    agent, embedded = ranking_agent(parser)
    response = agent.respond("s", "boots", 1, 2)
    assert response["recommendations"]
    assert embedded


@pytest.mark.parametrize(
    "parsed",
    [
        turn("boots", slots=(slot("material", "leather", "hard"),), confidence=0.1),
        turn("watch", confidence=0.0, message_type="exact", model_code="WA1200", intent="buying"),
        turn("band", confidence=0.0, message_type="compatibility", intent="buying"),
    ],
)
def test_hard_exact_and_compatibility_low_confidence_requests_proceed(parsed):
    parser = QueueParser(parsed)
    agent, embedded = ranking_agent(parser)
    response = agent.respond("s", "specific request", 1, 2)
    assert response["recommendations"]
    assert embedded


def test_established_category_refinement_does_not_retrigger_or_accept_filler_category():
    parser = QueueParser(
        turn("boots", confidence=0.5),
        turn("product", slots=(slot("color", "black"),), confidence=0.0),
    )
    agent, _ = ranking_agent(parser)
    agent.respond("s", "boots please", 1, 2)
    refined = agent.respond("s", "make them black", 2, 2)
    assert refined["ask_attribute"] != "category"
    assert agent._sessions["s"]["category"] == "boots"


def test_parser_failure_rolls_back_every_turn_surface_and_records_telemetry():
    failure = ParserRequestError(
        "offline",
        model="llama3.1:8b",
        latency_seconds=0.125,
        attempts=2,
    )
    parser = QueueParser(failure)
    agent, _ = ranking_agent(parser)
    agent.enable_forensic_ranking("s")
    before_state = deepcopy(agent._sessions["s"])
    before_forensics = agent.get_forensic_ranking_snapshots("s")
    before_memory = agent.memory_store.get_state("u")
    with pytest.raises(ParserRequestError, match="offline"):
        agent.respond("s", "human free text", 1, 2)
    assert agent._sessions["s"] == before_state
    assert agent.get_forensic_ranking_snapshots("s") == before_forensics
    assert agent.memory_store.get_state("u") == before_memory
    parser_call = agent.instrumentation["parser_calls"][-1]
    assert parser_call["model"] == "llama3.1:8b"
    assert parser_call["error_type"] == "ParserRequestError"
    assert parser_call["rolled_back"] is True
    error = agent.instrumentation["agent_errors"][-1]
    assert error["rollback"] is True
    assert error["model"] == "llama3.1:8b"
    assert error["latency_seconds"] == 0.125


def test_all_exact_evaluator_forms_bypass_the_injected_parser():
    parser = QueueParser(AssertionError("parser must not be called"))
    agent, _ = ranking_agent(parser)
    messages = (
        "I'm looking for Men Shoes, but I'm still exploring.",
        "For that, what matters is: cotton; black.",
        "Actually, ignore my earlier preference. What I need is: waterproof.",
        "Actually, please ignore my earlier preference.",
        "I don't have a preference for material; please use your judgment.",
        "I don't have an additional preference for color.",
        "Those options are not quite right yet. Ask me about one specific attribute.",
    )
    for index, message in enumerate(messages, 1):
        agent.respond("s", message, index, 1)
    assert parser.calls == []


def test_every_catalog_bucket_matches_all_evaluator_initial_shapes():
    buckets = set()
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            buckets.add(coarse_category(product.get("categories") or ()))
    agent = bare_agent()
    agent._template_buckets = buckets
    for category in buckets:
        assert agent._can_use_fast_path(
            f"I'm looking for {category}, but I'm still exploring.", 1
        )
        assert agent._can_use_fast_path(
            f"I'm looking for {category}. A key requirement is: leather.", 1
        )
        assert agent._can_use_fast_path(
            f"I'm looking for {category}. Imported.", 1
        )


def test_human_looking_for_message_does_not_match_the_template_fast_path():
    agent = bare_agent()
    agent._template_buckets = {"men shoes"}
    assert not agent._can_use_fast_path("I'm looking for a new strap for my watch.", 1)
    assert not agent._can_use_fast_path("I'm looking for Men Shoes.", 1)
