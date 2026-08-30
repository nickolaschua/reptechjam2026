"""Phase 6 chronological and counterfactual longitudinal evaluator.

Historical memory remains observable shadow state. This module never feeds it
to query construction, retrieval, ranking, diversity, or response prompting.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "experiment_1"
SHARED_REPO = PROJECT_ROOT / "techjam-conversational-search"
DEFAULT_FIXTURE = CURRENT_DIR / "longitudinal_eval" / "users_40.json"
SMALL_FIXTURE = CURRENT_DIR / "longitudinal_eval" / "fixture_small.json"
DEFAULT_RESULTS = CURRENT_DIR / "longitudinal_eval" / "results_40.json"

for path in (CURRENT_DIR, PROJECT_ROOT, EXPERIMENT_DIR, SHARED_REPO):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

try:
    from .agent import Agent
    from .memory_store import InMemoryUserMemoryStore, MemoryStoreSnapshot
    from .longitudinal_eval.directives import (
        ShopperCallResult,
        ShopperLLMClient,
        build_directive_system_prompt,
        build_first_turn_prompt,
        established_facts_before,
        semantic_disclosure_validation,
        semantic_match,
        target_leakage,
    )
except ImportError:
    from agent import Agent
    from memory_store import InMemoryUserMemoryStore, MemoryStoreSnapshot
    from longitudinal_eval.directives import (
        ShopperCallResult,
        ShopperLLMClient,
        build_directive_system_prompt,
        build_first_turn_prompt,
        established_facts_before,
        semantic_disclosure_validation,
        semantic_match,
        target_leakage,
    )

from evaluator.local_evaluator import catalog_index, coarse_category, metric_summary, normalize_recommendations
from experiment_1.shopper_agent import call_shopper_llm, make_system_prompt, materialize_hidden_fields


MAX_TURNS = 10
TOP_K = 10
EVALUATOR_ONLY_FIELDS = {
    "expected_conflict", "expected_memory", "expected_memory_behavior",
    "intended_memory_signal", "intended_preference_signal",
    "intentionally_irrelevant_prior_indices", "longitudinal_directive",
    "memory_relevance", "negative_safe_trait_ids", "relevant_prior_sequence_indices",
    "selection_rationale", "session_role", "target_asin", "target_attribute_audit",
}


def load_fixture(path: str | Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ordered_fixture_users(fixture: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    users = fixture.get("users")
    if not isinstance(users, list):
        raise ValueError("fixture must contain a users list")
    ordered: list[dict[str, Any]] = []
    seen_users: set[str] = set()
    for raw_user in users:
        if not isinstance(raw_user, Mapping):
            raise ValueError("each user fixture must be an object")
        user_id = str(raw_user.get("user_id", ""))
        if not user_id.strip() or user_id in seen_users:
            raise ValueError("fixture user_id values must be unique and non-empty")
        seen_users.add(user_id)
        profile = raw_user.get("constant_profile")
        sessions = raw_user.get("sessions")
        if not isinstance(profile, dict):
            raise ValueError(f"constant_profile for {user_id!r} must be an object")
        if not isinstance(sessions, list) or not sessions:
            raise ValueError(f"sessions for {user_id!r} must be a non-empty list")
        copied_sessions = [dict(session) for session in sessions]
        indices = [session.get("sequence_index") for session in copied_sessions]
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError("sequence_index values must be non-negative integers")
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate sequence_index for {user_id!r}")
        if any(not str(session.get("source_sample_id", "")).strip() for session in copied_sessions):
            raise ValueError("each session requires source_sample_id")
        copied = dict(raw_user)
        copied["sessions"] = sorted(copied_sessions, key=lambda value: value["sequence_index"])
        ordered.append(copied)
    return tuple(ordered)


def make_fresh_agent(**agent_kwargs: Any) -> Agent:
    return Agent(memory_store=InMemoryUserMemoryStore(), **agent_kwargs)


def _runtime_sample(source: Mapping[str, Any], constant_profile: dict[str, Any]) -> dict[str, Any]:
    runtime = {key: value for key, value in source.items() if key not in EVALUATOR_ONLY_FIELDS}
    runtime["user_profile"] = constant_profile
    return runtime


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _memory_description(item: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(item, "id", "")), "text": str(getattr(item, "text", "")),
        "scope": getattr(item, "scope", None),
        "polarity": getattr(getattr(item, "polarity", None), "value", None),
    }


def _call_shopper(
    shopper_call: Callable[[str, str, str], Any], prompt: str, system_prompt: str,
    model_name: str, provider_name: str,
) -> ShopperCallResult:
    value = shopper_call(prompt, system_prompt, model_name)
    if isinstance(value, ShopperCallResult):
        return value
    if isinstance(value, Mapping) and "text" in value:
        return ShopperCallResult(
            str(value["text"]), str(value.get("provider", provider_name)),
            str(value.get("model", model_name)),
        )
    return ShopperCallResult(str(value), provider_name, model_name)


def _safe_debug(agent: Any, session_id: str) -> dict[str, Any]:
    getter = getattr(agent, "get_memory_debug", None)
    if getter is None:
        return {}
    try:
        value = getter(session_id)
        return dict(value) if isinstance(value, Mapping) else {}
    except Exception:
        return {}


def _visible_memories(agent: Any, session_id: str) -> tuple[Any, ...]:
    getter = getattr(agent, "get_visible_memories", None)
    if getter is None:
        return ()
    try:
        return tuple(getter(session_id))
    except Exception:
        return ()


def _annotations(session: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(session[key]) for key in EVALUATOR_ONLY_FIELDS if key in session}


def _run_session(
    agent: Any, user: Mapping[str, Any], fixture_session: Mapping[str, Any],
    samples_by_id: Mapping[str, Mapping[str, Any]], catalog_ids: set[str],
    categories: Mapping[str, Sequence[str]], products: Mapping[str, dict[str, Any]], *,
    shopper_call: Callable[[str, str, str], Any],
    system_prompt_builder: Callable[[dict, dict, str], str],
    hidden_field_builder: Callable[[dict, dict], tuple[dict, dict]],
    model_name: str, shopper_provider: str, max_turns: int, top_k: int,
    event_hook: Callable[[str, str, dict[str, Any]], None] | None,
    replay_condition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    user_id = str(user["user_id"])
    constant_profile = user["constant_profile"]
    sequence_index = int(fixture_session["sequence_index"])
    source_id = str(fixture_session["source_sample_id"])
    if source_id not in samples_by_id:
        raise ValueError(f"unknown source_sample_id {source_id!r}")
    runtime_sample = _runtime_sample(samples_by_id[source_id], constant_profile)
    target_asin = str(runtime_sample["ground_truth"]["parent_asin"])
    if fixture_session.get("target_asin") is not None and str(fixture_session["target_asin"]) != target_asin:
        raise ValueError(f"fixture target does not match {source_id!r}")
    session_id = f"{user_id}_s{sequence_index}"
    agent.reset(session_id, constant_profile, user_id=user_id, sequence_index=sequence_index)
    visible_items = _visible_memories(agent, session_id)

    card, behavior = hidden_field_builder(runtime_sample, products)
    shopper_sample = {**runtime_sample, "intent_card": card, "behavior": behavior}
    target_product = products[target_asin]
    target_category = coarse_category(categories.get(target_asin, []))
    base_system_prompt = system_prompt_builder(shopper_sample, target_product, target_category)
    directive = fixture_session.get("longitudinal_directive", {})
    established = established_facts_before(user["sessions"], sequence_index)
    is_probe = "probe" in str(fixture_session.get("session_role", "")).casefold()
    system_prompt = (
        build_directive_system_prompt(base_system_prompt, directive, established, is_probe=is_probe)
        if directive or established else base_system_prompt
    )

    override = behavior.get("override", {})
    override_applied = runtime_sample["scenario_type"] != "intent_override"
    copilot_response: dict[str, Any] = {"message": "", "ask_attribute": None, "recommendations": []}
    recommendation_metadata: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    shopper_messages: list[str] = []
    hit_turn: int | None = None
    best_rank: int | None = None
    actual_provider, actual_model = shopper_provider, model_name

    for turn in range(1, max_turns + 1):
        if turn == 1:
            prompt = "Start the conversation by telling the assistant what you are looking for."
            if runtime_sample["scenario_type"] == "intent_override":
                prompt += f" Remember to mention your initial preference for: '{override.get('old_value', '')}'."
            elif runtime_sample["scenario_type"] == "buying" and card.get("hard_constraints"):
                prompt += f" Mention your key requirement: '{card['hard_constraints'][0]}'."
            prompt = build_first_turn_prompt(prompt, directive)
            try:
                result = _call_shopper(shopper_call, prompt, system_prompt, model_name, shopper_provider)
                user_message = result.text
                actual_provider, actual_model = result.provider, result.model
            except Exception:
                user_message = f"i'm looking for {target_category}."
        elif not override_applied and turn == int(override.get("turn", 3)):
            override_applied = True
            user_message = str(override.get("message", "Actually, ignore my earlier preference."))
        else:
            prompt = f"Assistant's Response:\n{copilot_response.get('message')}\n\nAssistant's Recommendations:\n"
            for rank, product in enumerate(recommendation_metadata, start=1):
                prompt += f"{rank}. {product.get('title')} (ASIN: {product.get('parent_asin')})\n"
            prompt += (
                f"\nAssistant asked about: {copilot_response.get('ask_attribute')}\n\n"
                "What is your next message? Keep it short, natural, and stay in character. "
                f"If the target product (ASIN: {target_asin}) is in the recommendations, "
                "acknowledge it and say you want to buy it."
            )
            try:
                result = _call_shopper(shopper_call, prompt, system_prompt, model_name, shopper_provider)
                user_message = result.text
                actual_provider, actual_model = result.provider, result.model
            except Exception:
                user_message = "what options do you have?"
        shopper_messages.append(user_message)
        try:
            response = agent.respond(session_id, user_message, turn, top_k)
            if not isinstance(response, Mapping):
                raise ValueError("agent response must be an object")
            copilot_response = dict(response)
        except Exception:
            copilot_response = {"message": "", "ask_attribute": None, "recommendations": []}
        ranked = normalize_recommendations(copilot_response.get("recommendations"), catalog_ids)
        recommendation_metadata = [products[identifier] for identifier in ranked]
        target_rank = ranked.index(target_asin) + 1 if target_asin in ranked else None
        transcript.append({
            "turn": turn, "shopper": user_message,
            "agent_message": str(copilot_response.get("message", "")),
            "ask_attribute": copilot_response.get("ask_attribute"),
            "recommendations": ranked, "target_rank": target_rank,
        })
        if override_applied and target_rank is not None:
            best_rank, hit_turn = target_rank, turn
            break

    is_hit = hit_turn is not None
    scored = {
        "session_id": session_id, "user_id": user_id, "sequence_index": sequence_index,
        "source_sample_id": source_id, "target_asin": target_asin,
        "target_title": target_product.get("title"),
        "scenario_type": runtime_sample["scenario_type"],
        "session_role": fixture_session.get("session_role"),
        "shopper_provider": actual_provider, "shopper_model": actual_model,
        "hit": is_hit, "first_hit_turn": hit_turn, "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "turn_efficiency": max(0.0, (max_turns + 1 - (hit_turn or max_turns + 1)) / max_turns),
        "turns": transcript,
        "prior_visible_memory_items": [_memory_description(item) for item in visible_items],
        "embedding_space_id": getattr(agent, "embedding_space_id", None),
    }
    if replay_condition is not None:
        scored["replay_condition"] = deepcopy(dict(replay_condition))
    if event_hook is not None:
        event_hook("scored", session_id, deepcopy(scored))
    before_end = _safe_debug(agent, session_id)
    committed = agent.end_session(
        session_id,
        outcome={"hit": is_hit, "best_rank": best_rank, "reciprocal_rank": scored["reciprocal_rank"]},
        purchased_product=target_asin if is_hit else None,
    )
    committed_items = tuple(committed or ())
    after_end = _safe_debug(agent, session_id)
    if event_hook is not None:
        event_hook("ended", session_id, deepcopy(scored))
    final_memory = after_end.get("final_fast_memory", before_end.get("final_fast_memory", {}))
    validation = semantic_disclosure_validation(
        directive, shopper_messages,
        final_memory if isinstance(final_memory, Mapping) else {},
        [str(getattr(item, "text", "")) for item in committed_items],
    )
    scored.update({
        "longitudinal_directive": deepcopy(directive),
        "target_leakage": target_leakage(shopper_messages, target_asin, str(target_product.get("title", ""))),
        "final_fast_memory": _json_safe(final_memory),
        "committed_memory_items": [_memory_description(item) for item in committed_items],
        "semantic_disclosure_validation": validation,
        "historical_memory_applied": bool(after_end.get("historical_memory_applied", False)),
        "evaluation_annotations": _annotations(fixture_session),
    })
    expected = fixture_session.get("expected_memory", {})
    visible_descriptions = [_memory_description(item) for item in visible_items]

    def visible_matches(labels: Sequence[str]) -> list[str]:
        return [
            value["id"]
            for value in visible_descriptions
            if any(semantic_match(label, value["text"]) for label in labels)
        ]

    current_override = directive.get("current_override", []) if isinstance(directive, Mapping) else []
    override_rows = [value for value in validation if value["directive_type"] == "current_override"]
    scored["override_diagnostics"] = {
        "applicable": bool(current_override),
        "current_override_expressed": all(value["shopper_expressed"] for value in override_rows) if current_override else None,
        "historical_incompatible_memory_labels": list(expected.get("should_suppress", [])),
        "portable_memory_labels": list(expected.get("helpful", [])),
        "historical_incompatible_visible_memory_ids": visible_matches(expected.get("should_suppress", [])),
        "portable_visible_memory_ids": visible_matches(expected.get("helpful", [])),
        "ranking_policy_evaluated": False,
    }
    scored["distractor_diagnostics"] = {
        "helpful_visible_memory_ids": visible_matches(expected.get("helpful", [])),
        "irrelevant_visible_memory_ids": visible_matches(expected.get("irrelevant", [])),
        "history_size": len(visible_items),
        "selection_policy_evaluated": False,
    }
    scored["negative_diagnostics"] = {
        "declared_negative_labels": list(expected.get("negative", [])),
        "visible_negative_memory_ids": visible_matches(expected.get("negative", [])),
        "checked_recommendation_count": 0,
        "violation_count": None,
        "negative_preference_violation_rate": None,
        "reason": "attribute-level alternative checking is reserved for memory-active conditions",
    }
    return scored


def _metrics(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    base = metric_summary(sessions)
    efficiencies = [float(value["turn_efficiency"]) for value in sessions]
    return {
        **base,
        "mean_turn_efficiency": round(statistics.fmean(efficiencies), 6) if efficiencies else 0.0,
        "historical_memory_applied": any(value.get("historical_memory_applied") for value in sessions),
        "memory_lift": None, "memory_harm_rate": None,
        "memory_effect_status": "shadow_mode_not_interpreted",
    }


def run_longitudinal_evaluation(
    agent: Any, fixture: Mapping[str, Any], samples_by_id: Mapping[str, Mapping[str, Any]],
    catalog_ids: set[str], categories: Mapping[str, Sequence[str]],
    products: Mapping[str, dict[str, Any]], *,
    shopper_call: Callable[[str, str, str], Any] = call_shopper_llm,
    system_prompt_builder: Callable[[dict, dict, str], str] = make_system_prompt,
    hidden_field_builder: Callable[[dict, dict], tuple[dict, dict]] = materialize_hidden_fields,
    model_name: str = "llama3.1", shopper_provider: str = "legacy_auto",
    max_turns: int = MAX_TURNS, top_k: int = TOP_K,
    event_hook: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    session_results: list[dict[str, Any]] = []
    for user in ordered_fixture_users(fixture):
        for session in user["sessions"]:
            session_results.append(_run_session(
                agent, user, session, samples_by_id, catalog_ids, categories, products,
                shopper_call=shopper_call, system_prompt_builder=system_prompt_builder,
                hidden_field_builder=hidden_field_builder, model_name=model_name,
                shopper_provider=shopper_provider, max_turns=max_turns, top_k=top_k,
                event_hook=event_hook,
            ))
    return {"sessions": session_results, "metrics": _metrics(session_results)}


def _latest_route(agent: Any, session_id: str, turn: int) -> str | None:
    instrumentation = getattr(agent, "instrumentation", None)
    if not isinstance(instrumentation, Mapping):
        return None
    turns = instrumentation.get("turns", ())
    if not isinstance(turns, Sequence):
        return None
    for value in reversed(turns):
        if (
            isinstance(value, Mapping)
            and str(value.get("session_id")) == session_id
            and value.get("turn") == turn
        ):
            route = value.get("route")
            return None if route is None else str(route)
    return None


def _run_fixed_message_session(
    agent: Any,
    user: Mapping[str, Any],
    fixture_session: Mapping[str, Any],
    shopper_messages: Sequence[str],
    catalog_ids: set[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Replay shopper strings only; evaluator directives never enter Agent."""

    user_id = str(user["user_id"])
    sequence_index = int(fixture_session["sequence_index"])
    session_id = f"{user_id}_s{sequence_index}"
    target_asin = str(fixture_session["target_asin"])
    agent.reset(
        session_id,
        user["constant_profile"],
        user_id=user_id,
        sequence_index=sequence_index,
    )
    prior_items = _visible_memories(agent, session_id)
    turns: list[dict[str, Any]] = []
    agent_errors: list[dict[str, Any]] = []
    for turn, user_message in enumerate(shopper_messages, start=1):
        try:
            raw = agent.respond(session_id, str(user_message), turn, top_k)
            if not isinstance(raw, Mapping):
                raise ValueError("agent response must be an object")
            response = dict(raw)
        except Exception as exc:
            agent_errors.append(
                {"turn": turn, "type": type(exc).__name__, "message": str(exc)}
            )
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        target_rank = ranked.index(target_asin) + 1 if target_asin in ranked else None
        debug = _safe_debug(agent, session_id)
        turns.append(
            {
                "turn": turn,
                "shopper": str(user_message),
                "shopper_input_sha256": hashlib.sha256(
                    str(user_message).encode("utf-8")
                ).hexdigest(),
                "recommendations": ranked,
                "target_rank": target_rank,
                "ask_attribute": response.get("ask_attribute"),
                "route": _latest_route(agent, session_id, turn),
                "fast_memory": _json_safe(debug.get("final_fast_memory", {})),
                "agent_message": str(response.get("message", "")),
            }
        )
    before_end = _safe_debug(agent, session_id)
    committed = tuple(agent.end_session(session_id) or ())
    after_end = _safe_debug(agent, session_id)
    final_fast_memory = after_end.get(
        "final_fast_memory", before_end.get("final_fast_memory", {})
    )
    return {
        "session_id": session_id,
        "user_id": user_id,
        "sequence_index": sequence_index,
        "target_asin": target_asin,
        "shopper_inputs": [str(value) for value in shopper_messages],
        "shopper_inputs_sha256": hashlib.sha256(
            json.dumps(
                [str(value) for value in shopper_messages],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "prior_visible_memory_items": [_memory_description(item) for item in prior_items],
        "turns": turns,
        "final_fast_memory": _json_safe(final_fast_memory),
        "committed_memory_items": [_memory_description(item) for item in committed],
        "historical_memory_applied": bool(
            after_end.get("historical_memory_applied", False)
        ),
        "agent_errors": agent_errors,
    }


def _close_agent(agent: Any) -> None:
    for owner in (agent, getattr(agent, "baseline_agent", None)):
        connection = getattr(owner, "connection", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _record_agent_llm_calls(agent: Any, tape: list[dict[str, Any]]) -> bool:
    """Record evaluation-time Agent LLM results for deterministic paired replay."""

    original = getattr(agent, "_call_llm", None)
    if not callable(original):
        return False

    def recorded(
        prompt: str,
        system_prompt: str = "",
        session_id: str | None = None,
        response_json: bool = False,
    ) -> str:
        response = original(
            prompt,
            system_prompt,
            session_id=session_id,
            response_json=response_json,
        )
        tape.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "session_id": session_id,
                "response_json": bool(response_json),
                "response": response,
            }
        )
        return response

    agent._call_llm = recorded
    return True


def _replay_agent_llm_calls(
    agent: Any,
    tape: Sequence[Mapping[str, Any]],
    replay_state: dict[str, Any],
) -> bool:
    """Replay Agent LLM results and reject any paired prompt/control drift."""

    if not callable(getattr(agent, "_call_llm", None)):
        return False

    def replayed(
        prompt: str,
        system_prompt: str = "",
        session_id: str | None = None,
        response_json: bool = False,
    ) -> str:
        index = int(replay_state["index"])
        if index >= len(tape):
            raise RuntimeError("NO_HISTORY Agent made an unexpected extra LLM call")
        expected = tape[index]
        actual = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "session_id": session_id,
            "response_json": bool(response_json),
        }
        expected_call = {key: expected[key] for key in actual}
        if actual != expected_call:
            replay_state["prompt_mismatches"].append(
                {
                    "call_index": index,
                    "expected_sha256": hashlib.sha256(
                        json.dumps(expected_call, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "actual_sha256": hashlib.sha256(
                        json.dumps(actual, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                }
            )
            raise RuntimeError(
                f"Agent LLM prompt drift at paired call {index}; "
                "shadow history is influencing the Agent call path"
            )
        replay_state["index"] = index + 1
        sessions = getattr(agent, "_sessions", {})
        if session_id and session_id in sessions:
            debug = sessions[session_id].setdefault("debug_info", {})
            debug["model"] = "Paired LLM call-tape replay"
            debug["system_prompt"] = system_prompt
            debug["user_prompt"] = prompt
        return str(expected["response"])

    agent._call_llm = replayed
    return True


def run_fixed_transcript_condition(
    condition: str,
    agent_factory: Callable[[InMemoryUserMemoryStore], Any],
    fixture: Mapping[str, Any],
    captured_sessions: Sequence[Mapping[str, Any]],
    catalog_ids: set[str],
    *,
    top_k: int = TOP_K,
    configure_agent: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Replay a captured benchmark under one explicit history condition."""

    if condition not in {"SHADOW_HISTORY", "NO_HISTORY"}:
        raise ValueError("condition must be SHADOW_HISTORY or NO_HISTORY")
    captured = {str(value["session_id"]): value for value in captured_sessions}
    store = InMemoryUserMemoryStore()
    agent = agent_factory(store)
    agent_control = configure_agent(agent) if configure_agent is not None else None
    results: list[dict[str, Any]] = []
    instrumentation: dict[str, Any] = {}
    try:
        for user in ordered_fixture_users(fixture):
            for fixture_session in user["sessions"]:
                session_id = f"{user['user_id']}_s{fixture_session['sequence_index']}"
                if session_id not in captured:
                    raise ValueError(f"captured transcript missing {session_id!r}")
                if condition == "NO_HISTORY":
                    store.clear()
                    snapshot = store.export_snapshot()
                    if snapshot.records or snapshot.commits:
                        raise RuntimeError("NO_HISTORY store was not empty before reset")
                turns = captured[session_id].get("turns")
                if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
                    raise ValueError(f"captured transcript for {session_id!r} lacks turns")
                shopper_messages = [
                    str(value["shopper"])
                    for value in turns
                    if isinstance(value, Mapping) and "shopper" in value
                ]
                if len(shopper_messages) != len(turns) or not shopper_messages:
                    raise ValueError(f"captured transcript for {session_id!r} is invalid")
                replayed = _run_fixed_message_session(
                    agent,
                    user,
                    fixture_session,
                    shopper_messages,
                    catalog_ids,
                    top_k=top_k,
                )
                replayed["condition"] = condition
                replayed["store_commit_count_before_session"] = (
                    0
                    if condition == "NO_HISTORY"
                    else len(store.export_snapshot().commits) - 1
                )
                results.append(replayed)
        instrumentation_getter = getattr(agent, "get_instrumentation", None)
        instrumentation = (
            _json_safe(instrumentation_getter())
            if callable(instrumentation_getter)
            else {}
        )
    finally:
        _close_agent(agent)
    return {
        "condition": condition,
        "sessions": results,
        "instrumentation": instrumentation,
        "agent_control_enabled": bool(agent_control),
    }


def compare_fixed_transcript_conditions(
    shadow: Mapping[str, Any], no_history: Mapping[str, Any]
) -> dict[str, Any]:
    shadow_sessions = {
        str(value["session_id"]): value for value in shadow.get("sessions", ())
    }
    no_history_sessions = {
        str(value["session_id"]): value for value in no_history.get("sessions", ())
    }
    if set(shadow_sessions) != set(no_history_sessions):
        raise ValueError("parity conditions do not contain the same sessions")
    paired_turns: list[dict[str, Any]] = []
    session_checks: list[dict[str, Any]] = []
    for session_id, shadow_session in shadow_sessions.items():
        clean_session = no_history_sessions[session_id]
        shadow_turns = shadow_session["turns"]
        clean_turns = clean_session["turns"]
        if len(shadow_turns) != len(clean_turns):
            raise ValueError(f"parity turn count differs for {session_id!r}")
        same_inputs = (
            shadow_session["shopper_inputs_sha256"]
            == clean_session["shopper_inputs_sha256"]
            and shadow_session["shopper_inputs"] == clean_session["shopper_inputs"]
        )
        if not same_inputs:
            raise RuntimeError(f"shopper input drift in parity replay for {session_id!r}")
        session_checks.append(
            {
                "session_id": session_id,
                "identical_shopper_inputs": True,
                "shadow_prior_memory_count": len(
                    shadow_session["prior_visible_memory_items"]
                ),
                "no_history_prior_memory_count": len(
                    clean_session["prior_visible_memory_items"]
                ),
                "shadow_historical_memory_applied": bool(
                    shadow_session.get("historical_memory_applied")
                ),
                "no_history_historical_memory_applied": bool(
                    clean_session.get("historical_memory_applied")
                ),
            }
        )
        for shadow_turn, clean_turn in zip(shadow_turns, clean_turns):
            recommendations_identical = (
                shadow_turn["recommendations"] == clean_turn["recommendations"]
            )
            paired_turns.append(
                {
                    "session_id": session_id,
                    "turn": shadow_turn["turn"],
                    "shopper_input_sha256": shadow_turn["shopper_input_sha256"],
                    "recommendations_identical": recommendations_identical,
                    "shadow_recommendations": shadow_turn["recommendations"],
                    "no_history_recommendations": clean_turn["recommendations"],
                    "target_rank_identical": (
                        shadow_turn["target_rank"] == clean_turn["target_rank"]
                    ),
                    "shadow_target_rank": shadow_turn["target_rank"],
                    "no_history_target_rank": clean_turn["target_rank"],
                    "ask_attribute_identical": (
                        shadow_turn["ask_attribute"] == clean_turn["ask_attribute"]
                    ),
                    "fast_memory_identical": (
                        shadow_turn["fast_memory"] == clean_turn["fast_memory"]
                    ),
                    "route_identical": shadow_turn["route"] == clean_turn["route"],
                    "shadow_route": shadow_turn["route"],
                    "no_history_route": clean_turn["route"],
                    "response_prose_identical": (
                        shadow_turn["agent_message"] == clean_turn["agent_message"]
                    ),
                }
            )
    total = len(paired_turns)
    identical = sum(value["recommendations_identical"] for value in paired_turns)
    differing = total - identical
    return {
        "status": "pass" if total and differing == 0 else "fail",
        "primary_invariant": "ordered recommendation ASIN lists are identical",
        "total_paired_turns": total,
        "identical_recommendation_turns": identical,
        "differing_recommendation_turns": differing,
        "recommendation_order_parity_rate": 0.0 if not total else identical / total,
        "target_rank_difference_count": sum(
            not value["target_rank_identical"] for value in paired_turns
        ),
        "fast_memory_difference_count": sum(
            not value["fast_memory_identical"] for value in paired_turns
        ),
        "route_difference_count": sum(
            not value["route_identical"] for value in paired_turns
        ),
        "ask_attribute_difference_count": sum(
            not value["ask_attribute_identical"] for value in paired_turns
        ),
        "response_prose_difference_count": sum(
            not value["response_prose_identical"] for value in paired_turns
        ),
        "all_shopper_inputs_identical": all(
            value["identical_shopper_inputs"] for value in session_checks
        ),
        "no_history_sessions_with_prior_memory": [
            value["session_id"]
            for value in session_checks
            if value["no_history_prior_memory_count"]
        ],
        "shadow_sessions_with_prior_memory": [
            value["session_id"]
            for value in session_checks
            if value["shadow_prior_memory_count"]
        ],
        "historical_memory_applied": any(
            value["shadow_historical_memory_applied"]
            or value["no_history_historical_memory_applied"]
            for value in session_checks
        ),
        "session_checks": session_checks,
        "differing_turns": [
            value for value in paired_turns if not value["recommendations_identical"]
        ],
        "paired_turns": paired_turns,
    }


def run_strict_shadow_no_history_parity(
    agent_factory: Callable[[InMemoryUserMemoryStore], Any],
    fixture: Mapping[str, Any],
    captured_sessions: Sequence[Mapping[str, Any]],
    catalog_ids: set[str],
    *,
    top_k: int = TOP_K,
    control_agent_llm: bool = True,
) -> dict[str, Any]:
    """Replay identical turns and, when available, identical Agent LLM calls."""

    llm_tape: list[dict[str, Any]] = []
    shadow = run_fixed_transcript_condition(
        "SHADOW_HISTORY", agent_factory, fixture, captured_sessions, catalog_ids,
        top_k=top_k,
        configure_agent=(
            (lambda agent: _record_agent_llm_calls(agent, llm_tape))
            if control_agent_llm
            else None
        ),
    )
    replay_state: dict[str, Any] = {"index": 0, "prompt_mismatches": []}
    no_history = run_fixed_transcript_condition(
        "NO_HISTORY", agent_factory, fixture, captured_sessions, catalog_ids,
        top_k=top_k,
        configure_agent=(
            (lambda agent: _replay_agent_llm_calls(agent, llm_tape, replay_state))
            if control_agent_llm and shadow["agent_control_enabled"]
            else None
        ),
    )
    if no_history["agent_control_enabled"] and replay_state["index"] != len(llm_tape):
        raise RuntimeError(
            "NO_HISTORY Agent consumed a different number of paired LLM calls"
        )
    comparison = compare_fixed_transcript_conditions(shadow, no_history)
    comparison["agent_llm_call_control"] = {
        "enabled": bool(
            shadow["agent_control_enabled"] and no_history["agent_control_enabled"]
        ),
        "recorded_call_count": len(llm_tape),
        "replayed_call_count": int(replay_state["index"]),
        "prompt_mismatch_count": len(replay_state["prompt_mismatches"]),
        "prompt_mismatches": replay_state["prompt_mismatches"],
        "purpose": (
            "Hold stochastic Agent state/prose generation fixed while varying only "
            "the longitudinal store condition."
        ),
    }
    comparison["condition_summaries"] = {
        "SHADOW_HISTORY": {
            "session_count": len(shadow["sessions"]),
            "agent_error_count": sum(
                len(value["agent_errors"]) for value in shadow["sessions"]
            ),
        },
        "NO_HISTORY": {
            "session_count": len(no_history["sessions"]),
            "agent_error_count": sum(
                len(value["agent_errors"]) for value in no_history["sessions"]
            ),
        },
    }
    comparison["replays"] = {
        "SHADOW_HISTORY": shadow["sessions"],
        "NO_HISTORY": no_history["sessions"],
    }
    return comparison


def probe_replay_plan(fixture: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    configured = fixture.get("probe_replays", {})
    plans: list[dict[str, Any]] = []
    for user in ordered_fixture_users(fixture):
        probes = [value for value in user["sessions"] if "probe" in str(value.get("session_role", ""))]
        for probe in probes:
            full_size = int(probe["sequence_index"])
            sizes = list((configured.get(user["user_id"]) or {}).get("history_sizes", [0, full_size]))
            if 0 not in sizes or full_size not in sizes:
                raise ValueError(f"{user['user_id']} replay plan must include no/full history")
            for size in sizes:
                if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= full_size:
                    raise ValueError("invalid history size in replay plan")
                mode = "NO_HISTORY" if size == 0 else "FULL_HISTORY" if size == full_size else "PREFIX"
                plans.append({
                    "user_id": user["user_id"], "sequence_index": full_size,
                    "condition": f"H{size}" if user["user_id"] == "u3_distractor" else mode,
                    "history_mode": mode, "history_size": size,
                    "prior_sequence_indices": list(range(size)),
                })
    return tuple(plans)


def _session_fingerprint(user: Mapping[str, Any], session: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"user_id": user["user_id"], "constant_profile": user["constant_profile"], "session": session},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_counterfactual_probe_replays(
    agent_factory: Callable[[InMemoryUserMemoryStore], Any], history_snapshot: MemoryStoreSnapshot,
    fixture: Mapping[str, Any], samples_by_id: Mapping[str, Mapping[str, Any]],
    catalog_ids: set[str], categories: Mapping[str, Sequence[str]],
    products: Mapping[str, dict[str, Any]], *,
    shopper_call: Callable[[str, str, str], Any] = call_shopper_llm,
    system_prompt_builder: Callable[[dict, dict, str], str] = make_system_prompt,
    hidden_field_builder: Callable[[dict, dict], tuple[dict, dict]] = materialize_hidden_fields,
    model_name: str = "llama3.1", shopper_provider: str = "legacy_auto",
    max_turns: int = MAX_TURNS, top_k: int = TOP_K,
) -> dict[str, Any]:
    users = {user["user_id"]: user for user in ordered_fixture_users(fixture)}
    results: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for condition in probe_replay_plan(fixture):
        user = users[condition["user_id"]]
        probe = next(value for value in user["sessions"] if value["sequence_index"] == condition["sequence_index"])
        fingerprint = _session_fingerprint(user, probe)
        fingerprints.setdefault(user["user_id"], fingerprint)
        if fingerprints[user["user_id"]] != fingerprint:
            raise RuntimeError("counterfactual probe configuration drifted")
        selected = history_snapshot.filtered(
            user_id=user["user_id"], sequence_indices=condition["prior_sequence_indices"],
            before_sequence_index=condition["sequence_index"],
        )
        store = InMemoryUserMemoryStore()
        if condition["history_size"]:
            store.import_snapshot(selected)
        elif store.export_snapshot().commits:
            raise RuntimeError("NO_HISTORY replay did not start clean")
        scored = _run_session(
            agent_factory(store), user, probe, samples_by_id, catalog_ids, categories, products,
            shopper_call=shopper_call, system_prompt_builder=system_prompt_builder,
            hidden_field_builder=hidden_field_builder, model_name=model_name,
            shopper_provider=shopper_provider, max_turns=max_turns, top_k=top_k,
            event_hook=None,
            replay_condition={**condition, "probe_config_fingerprint": fingerprint},
        )
        results.append(scored)
    return {"sessions": results, "metrics": _metrics(results), "comparisons": summarize_probe_comparisons(results)}


def summarize_probe_comparisons(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_user: dict[str, list[Mapping[str, Any]]] = {}
    for session in sessions:
        by_user.setdefault(str(session["user_id"]), []).append(session)
    comparisons: dict[str, Any] = {}
    for user_id, values in by_user.items():
        ordered = sorted(values, key=lambda value: value["replay_condition"]["history_size"])
        no_history, full_history = ordered[0], ordered[-1]
        delta = float(full_history["reciprocal_rank"]) - float(no_history["reciprocal_rank"])
        applied = bool(full_history.get("historical_memory_applied"))
        comparisons[user_id] = {
            "same_probe_config": no_history["replay_condition"]["probe_config_fingerprint"] == full_history["replay_condition"]["probe_config_fingerprint"],
            "no_history_rr": no_history["reciprocal_rank"], "full_history_rr": full_history["reciprocal_rank"],
            "memory_lift": delta if applied else None,
            "target_rank_delta": (
                None if not applied or no_history["best_rank"] is None or full_history["best_rank"] is None
                else no_history["best_rank"] - full_history["best_rank"]
            ),
            "memory_harm": delta < 0.0 if applied else None,
            "shadow_mode_observed_rr_delta_not_a_memory_effect": delta if not applied else None,
            "history_curve": [
                {"condition": value["replay_condition"]["condition"],
                 "history_size": value["replay_condition"]["history_size"],
                 "reciprocal_rank": value["reciprocal_rank"], "best_rank": value["best_rank"]}
                for value in ordered
            ],
        }
    active = [value for value in comparisons.values() if value["memory_harm"] is not None]
    return {
        "per_user": comparisons,
        "memory_harm_rate": None if not active else sum(bool(value["memory_harm"]) for value in active) / len(active),
        "status": "shadow_mode_not_interpreted" if not active else "memory_active",
    }


def _load_public_samples(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    return {sample["sample_id"]: sample for sample in samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6 longitudinal B0 shadow evaluation")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--shopper-provider", choices=sorted(ShopperLLMClient.DEFAULT_MODELS), default="ollama")
    parser.add_argument("--shopper-model")
    parser.add_argument("--replay-probes", action="store_true")
    parser.add_argument("--allow-openai-catalog-build", action="store_true")
    args = parser.parse_args()
    catalog_path = SHARED_REPO / "data" / "catalog.jsonl"
    public_path = SHARED_REPO / "data" / "public_set.jsonl"
    catalog_ids, categories, products = catalog_index(catalog_path)
    shopper = ShopperLLMClient(args.shopper_provider, args.shopper_model)
    agent_kwargs = {
        "embedding_cache_dir": CURRENT_DIR / "embedding_cache",
        "allow_catalog_embedding": args.allow_openai_catalog_build,
    }
    agent = make_fresh_agent(**agent_kwargs)
    fixture = load_fixture(args.fixture)
    samples = _load_public_samples(public_path)
    result = run_longitudinal_evaluation(
        agent, fixture, samples, set(catalog_ids), categories, products,
        shopper_call=shopper, shopper_provider=shopper.provider, model_name=shopper.model,
    )
    if args.replay_probes:
        snapshot = agent.memory_store.export_snapshot()

        def factory(store: InMemoryUserMemoryStore) -> Agent:
            return Agent(memory_store=store, **agent_kwargs)

        result["counterfactual_probes"] = run_counterfactual_probe_replays(
            factory, snapshot, fixture, samples, set(catalog_ids), categories, products,
            shopper_call=shopper, shopper_provider=shopper.provider, model_name=shopper.model,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "compare_fixed_transcript_conditions",
    "DEFAULT_FIXTURE", "EVALUATOR_ONLY_FIELDS", "SMALL_FIXTURE", "load_fixture",
    "make_fresh_agent", "ordered_fixture_users", "probe_replay_plan",
    "run_fixed_transcript_condition", "run_strict_shadow_no_history_parity",
    "run_counterfactual_probe_replays", "run_longitudinal_evaluation",
    "summarize_probe_comparisons",
]
