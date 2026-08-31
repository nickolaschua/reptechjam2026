"""Strict, offline validator for the Phase 6 research fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping

try:
    from .directives import DIRECTIVE_KEYS
except ImportError:
    from directives import DIRECTIVE_KEYS


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_FIXTURE = HERE / "users_40.json"
DEFAULT_PUBLIC = PROJECT_ROOT / "techjam-conversational-search" / "data" / "public_set.jsonl"
DEFAULT_CATALOG = PROJECT_ROOT / "techjam-conversational-search" / "data" / "catalog.jsonl"
EVALUATOR_ONLY_KEYS = {
    "archetype",
    "expected_memory",
    "expected_memory_behavior",
    "intended_memory_signal",
    "intentionally_irrelevant_prior_indices",
    "longitudinal_directive",
    "negative_safe_trait_ids",
    "relevant_prior_sequence_indices",
    "session_role",
    "shopper_private_persona",
    "source_sample_id",
    "target_asin",
    "target_attribute_audit",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _evidence_value(product: Mapping[str, Any], source: str) -> Any:
    if source in {"title", "price", "description", "categories", "store"}:
        return product.get(source)
    match = re.fullmatch(r"features\[(\d+)\]", source)
    if match:
        features = product.get("features") or []
        index = int(match.group(1))
        return features[index] if index < len(features) else None
    if source.startswith("details."):
        return (product.get("details") or {}).get(source.split(".", 1)[1])
    return None


def validate_fixture(
    fixture: Mapping[str, Any],
    samples_by_id: Mapping[str, Mapping[str, Any]],
    products: Mapping[str, Mapping[str, Any]],
    *,
    require_research_shape: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    users = fixture.get("users")
    if not isinstance(users, list):
        return {"valid": False, "errors": ["fixture.users must be a list"], "warnings": []}
    if require_research_shape and len(users) != 4:
        errors.append("research fixture must contain exactly 4 users")

    runtime_ids: set[str] = set()
    all_sessions = 0
    for user in users:
        if not isinstance(user, Mapping):
            errors.append("every user must be an object")
            continue
        user_id = str(user.get("user_id", ""))
        profile = user.get("constant_profile")
        sessions = user.get("sessions")
        if not user_id or not isinstance(profile, Mapping) or not isinstance(sessions, list):
            errors.append(f"invalid user envelope for {user_id or '<missing>'}")
            continue
        if require_research_shape and len(sessions) != 10:
            errors.append(f"{user_id}: research fixture requires exactly 10 sessions")
        indices = [session.get("sequence_index") for session in sessions if isinstance(session, Mapping)]
        expected_indices = list(range(10)) if require_research_shape else sorted(indices)
        if sorted(indices) != expected_indices or len(indices) != len(set(indices)):
            errors.append(f"{user_id}: sequence indices are invalid")
        if EVALUATOR_ONLY_KEYS.intersection(profile):
            errors.append(f"{user_id}: evaluator-only field leaked into constant_profile")
        profile_text = json.dumps(profile, ensure_ascii=False).casefold()
        latent = (user.get("shopper_private_persona") or {}).get("latent_preferences", [])
        for preference in latent:
            for term in preference.get("leakage_terms", []):
                if re.search(rf"\b{re.escape(str(term).casefold())}\b", profile_text):
                    errors.append(f"{user_id}: tested preference term {term!r} leaked into constant_profile")

        probes = 0
        for session in sessions:
            if not isinstance(session, Mapping):
                errors.append(f"{user_id}: a session is not an object")
                continue
            all_sessions += 1
            index = session.get("sequence_index")
            runtime_id = f"{user_id}_s{index}"
            if runtime_id in runtime_ids:
                errors.append(f"duplicate runtime session ID {runtime_id}")
            runtime_ids.add(runtime_id)
            source_id = str(session.get("source_sample_id", ""))
            source = samples_by_id.get(source_id)
            source_target = (
                "" if source is None
                else str(source.get("ground_truth", {}).get("parent_asin", ""))
            )
            target = str(session.get("target_asin", source_target))
            product = products.get(target)
            if source is None:
                errors.append(f"{runtime_id}: source_sample_id does not resolve")
            if product is None:
                errors.append(f"{runtime_id}: target_asin does not resolve")
            if source is not None:
                if target != source_target:
                    errors.append(f"{runtime_id}: source target and target_asin disagree")
                if source.get("scenario_type") != "buying":
                    warnings.append(f"{runtime_id}: non-buying source row used")

            directive = session.get("longitudinal_directive")
            if directive is None and not require_research_shape:
                directive = {key: [] for key in DIRECTIVE_KEYS}
            if not isinstance(directive, Mapping) or set(directive) != set(DIRECTIVE_KEYS):
                errors.append(f"{runtime_id}: longitudinal directive keys are invalid")
            else:
                for key in DIRECTIVE_KEYS:
                    value = directive[key]
                    if not isinstance(value, list) or any(
                        not isinstance(item, str) or not item.strip() for item in value
                    ):
                        errors.append(f"{runtime_id}: directive {key} is invalid")

            audit = session.get("target_attribute_audit")
            traits = audit.get("verified_traits") if isinstance(audit, Mapping) else None
            if (not isinstance(traits, list) or not traits) and require_research_shape:
                errors.append(f"{runtime_id}: target_attribute_audit requires verified_traits")
                traits = []
            elif not isinstance(traits, list):
                traits = []
            for trait in traits:
                required = {"kind", "value", "source", "confidence", "evidence"}
                if not isinstance(trait, Mapping) or not required.issubset(trait):
                    errors.append(f"{runtime_id}: malformed verified trait")
                    continue
                if product is not None:
                    actual = _evidence_value(product, str(trait["source"]))
                    if actual is None:
                        errors.append(f"{runtime_id}: audit source {trait['source']!r} does not resolve")
                    elif str(trait["evidence"]).casefold() not in json.dumps(actual, ensure_ascii=False).casefold():
                        errors.append(f"{runtime_id}: audit evidence is absent from declared source")
                if trait.get("confidence") == "weak":
                    warnings.append(f"{runtime_id}: weak catalogue support for {trait.get('kind')}")

            for field in ("relevant_prior_sequence_indices", "intentionally_irrelevant_prior_indices"):
                values = session.get(field, [])
                if not isinstance(values, list) or any(
                    isinstance(value, bool) or not isinstance(value, int) or value >= index
                    for value in values
                ):
                    errors.append(f"{runtime_id}: {field} contains a future/invalid reference")

            if index == 9:
                probes += int("probe" in str(session.get("session_role", "")))
            signal_text = json.dumps(
                {
                    "directive": directive,
                    "expected": session.get("expected_memory"),
                    "signal": session.get("intended_memory_signal"),
                },
                ensure_ascii=False,
            ).casefold()
            if re.search(r"(?:\$\s*)?120\b|budget|under approximately", signal_text):
                price = None if product is None else product.get("price")
                if isinstance(price, bool) or not isinstance(price, (int, float)):
                    errors.append(f"{runtime_id}: budget signal lacks a numeric target price")
                if not any(trait.get("kind") == "price" for trait in traits):
                    errors.append(f"{runtime_id}: budget signal lacks audited price evidence")
        if require_research_shape and probes != 1:
            errors.append(f"{user_id}: S10 must be a probe")

        if user_id == "u2_override" and require_research_shape:
            override = sessions[9]["longitudinal_directive"]["current_override"]
            if not override or not re.search(r"bright|bold|colour|color|sport", " ".join(override), re.I):
                errors.append("u2_override S10 requires explicit bright/sporty current_override")
        if user_id == "u4_negative" and require_research_shape:
            declared = {value["id"] for value in latent}
            safe = set(sessions[9].get("negative_safe_trait_ids", []))
            if not declared.issubset(safe):
                errors.append("u4_negative S10 lacks safe evidence declarations for every negative")

    if require_research_shape:
        if all_sessions != 40:
            errors.append("research fixture must contain exactly 40 sessions")
        u3_plan = (fixture.get("probe_replays") or {}).get("u3_distractor", {})
        if u3_plan.get("history_sizes") != [0, 1, 3, 5, 9]:
            errors.append("u3_distractor must support H0/H1/H3/H5/H9")
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "user_count": len(users),
        "session_count": all_sessions,
        "runtime_session_count": len(runtime_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Phase 6 longitudinal fixture")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    samples = {value["sample_id"]: value for value in load_jsonl(args.public)}
    products = {value["parent_asin"]: value for value in load_jsonl(args.catalog)}
    result = validate_fixture(
        fixture,
        samples,
        products,
        require_research_shape=not args.allow_small,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()


__all__ = ["EVALUATOR_ONLY_KEYS", "load_jsonl", "validate_fixture"]
