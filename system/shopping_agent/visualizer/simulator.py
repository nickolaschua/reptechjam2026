"""Self-contained public-sample and shopper-simulation helpers for the dashboard."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from ..model_client import ModelClient
from ..runtime import get_runtime_providers


MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")


def _flatten(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean(value: object, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" -;,.\t\n")[:limit].rstrip()


def searchable_text(product: dict[str, Any]) -> str:
    return " ".join(part for field in SEARCH_FIELDS for part in _flatten(product.get(field))).strip()


def intent_card(product: dict[str, Any]) -> dict[str, Any]:
    title = _clean(product.get("title") or "product")
    candidates = [*_flatten(product.get("features")), *_flatten(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material: candidates.insert(0, material.group(1).lower())
    if color: candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""): candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean(item) for item in candidates if _clean(item))) or [title]
    return {"target_category": title, "hard_constraints": cleaned[:2], "soft_preferences": cleaned[2:4] or cleaned[:1]}


def behavior_for(scenario: str, card: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    behavior: dict[str, Any] = {"scenario_type": scenario}
    if scenario == "intent_override":
        hard, soft = card["hard_constraints"], card["soft_preferences"]
        old_value = soft[-1] if soft else "I prefer a different style."
        new_value = hard[0] if hard else "Please prioritize the target requirements."
        behavior["override"] = {
            "turn": rng.choice([3, 4]), "old_value": old_value, "new_value": new_value,
            "message": f"Actually, ignore my earlier preference. What I need is: {new_value}.",
        }
    return behavior


def materialize_hidden_fields(sample: dict[str, Any], products: dict[str, dict]) -> tuple[dict, dict]:
    if "intent_card" in sample and "behavior" in sample:
        return sample["intent_card"], sample["behavior"]
    product = products[str(sample["ground_truth"]["parent_asin"])]
    card = intent_card(product)
    seed = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
    return card, behavior_for(str(sample["scenario_type"]), card, random.Random(seed))


def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned = [part.strip() for value in values for part in value.split(",") if part.strip() and part.strip().lower() not in excluded]
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def make_system_prompt(sample: dict[str, Any], product: dict[str, Any], target_category: str) -> str:
    card, behavior, scenario = sample["intent_card"], sample["behavior"], sample["scenario_type"]
    description = product.get("description", "")
    if isinstance(description, list): description = " ".join(str(value) for value in description)
    details = product.get("details", {})
    details_text = " ".join(f"{key}: {value}" for key, value in details.items()) if isinstance(details, dict) else str(details)
    prompt = (
        "You are acting as a real customer shopping online.\n"
        f"Target Product Title: {product.get('title', '')}\nCategory: {target_category}\n"
        f"Hard Constraints: {', '.join(card.get('hard_constraints', []))}\n"
        f"Soft Preferences: {', '.join(card.get('soft_preferences', []))}\n"
        f"Ground Truth ASIN: {product.get('parent_asin')}\nDetails: {details_text}\nDescription: {description[:300]}\n"
        f"Scenario: {scenario}\n"
    )
    if scenario == "intent_override":
        override = behavior.get("override", {})
        prompt += f"Initially prefer '{override.get('old_value', '')}', then use the supplied override message at turn {override.get('turn', 3)}.\n"
    elif scenario == "boundary":
        prompt += "If asked about an attribute you do not care about, say so and ask the assistant to use its judgment.\n"
    return prompt + "Do not reveal the exact title or ASIN. Keep each reply natural, lowercase, and one or two sentences."


def call_shopper_llm(
    prompt: str,
    system_prompt: str,
    *,
    client: ModelClient | None = None,
    temperature: float = 0.4,
    seed: int | None = None,
) -> str:
    """Generate shopper text through the same local model used by the agent."""

    llm = client or get_runtime_providers().llm_client
    options: dict[str, Any] = {"temperature": float(temperature), "num_predict": 150}
    if seed is not None:
        options["seed"] = int(seed)
    return llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        options=options,
        role="shopper",
    )


def load_samples(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


__all__ = ["call_shopper_llm", "coarse_category", "load_samples", "make_system_prompt", "materialize_hidden_fields"]
