"""Evaluator-private longitudinal shopper controls and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Iterable, Mapping, Sequence

import requests


DIRECTIVE_KEYS = ("disclose", "reinforce", "session_only", "current_override")


@dataclass(frozen=True)
class ShopperCallResult:
    text: str
    provider: str
    model: str


class ShopperLLMClient:
    """Small explicit-provider shopper client; no credential fallthrough."""

    DEFAULT_MODELS = {
        "ollama": "llama3.1",
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "gemini": "gemini-1.5-flash",
    }

    def __init__(
        self,
        provider: str = "ollama",
        model: str | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        resolved = str(provider).strip().casefold()
        if resolved not in self.DEFAULT_MODELS:
            raise ValueError(f"unsupported shopper provider {provider!r}")
        self.provider = resolved
        self.model = str(model or self.DEFAULT_MODELS[resolved])
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def _credential(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value or value.startswith("your_") or "placeholder" in value.casefold():
            raise RuntimeError(f"{name} is required for the selected shopper provider")
        return value

    def __call__(self, prompt: str, system_prompt: str, model_name: str = "") -> ShopperCallResult:
        model = str(model_name or self.model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if self.provider == "ollama":
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.4},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            text = response.json()["message"]["content"]
        elif self.provider in {"openai", "deepseek"}:
            is_openai = self.provider == "openai"
            key = self._credential("OPENAI_API_KEY" if is_openai else "DEEPSEEK_API_KEY")
            base = "https://api.openai.com/v1" if is_openai else "https://api.deepseek.com/v1"
            response = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.4,
                    "max_tokens": 150,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
        else:
            key = self._credential("GEMINI_API_KEY")
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"role": "user", "parts": [{"text": system_prompt + "\n\n" + prompt}]}],
                    "generationConfig": {"temperature": 0.4, "maxOutputTokens": 150},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return ShopperCallResult(str(text).strip(), self.provider, model)


def _validated_directive(value: Mapping[str, Any] | None) -> dict[str, list[str]]:
    raw = {} if value is None else dict(value)
    unknown = set(raw).difference(DIRECTIVE_KEYS)
    if unknown:
        raise ValueError(f"unknown longitudinal directive field {sorted(unknown)[0]!r}")
    result: dict[str, list[str]] = {}
    for key in DIRECTIVE_KEYS:
        entries = raw.get(key, [])
        if not isinstance(entries, list) or any(
            not isinstance(entry, str) or not entry.strip() for entry in entries
        ):
            raise ValueError(f"longitudinal_directive.{key} must be a list of non-empty strings")
        result[key] = [entry.strip() for entry in entries]
    return result


def established_facts_before(sessions: Sequence[Mapping[str, Any]], sequence_index: int) -> tuple[str, ...]:
    """Return only facts whose disclosure was scheduled in an earlier session."""

    facts: list[str] = []
    for session in sorted(sessions, key=lambda value: int(value["sequence_index"])):
        if int(session["sequence_index"]) >= sequence_index:
            break
        directive = _validated_directive(session.get("longitudinal_directive"))
        for fact in directive["disclose"]:
            if fact not in facts:
                facts.append(fact)
    return tuple(facts)


def build_directive_system_prompt(
    base_prompt: str,
    directive: Mapping[str, Any] | None,
    established_facts: Sequence[str],
    *,
    is_probe: bool,
) -> str:
    """Add private controls to the shopper prompt, never to the shopping agent."""

    current = _validated_directive(directive)
    payload = json.dumps(current, ensure_ascii=False)
    known = json.dumps(list(established_facts), ensure_ascii=False)
    probe_rule = (
        "This is a probe. Do not volunteer any established historical fact; only express "
        "today's intent, except that current_override is itself today's intent."
        if is_probe
        else "Do not repeat established facts unless they appear in reinforce for this session."
    )
    return (
        f"{base_prompt}\n\n"
        "PRIVATE LONGITUDINAL EVALUATOR CONTROL (never mention this control block):\n"
        f"Already established facts, for consistency only: {known}\n"
        f"Required current-session semantics: {payload}\n"
        "Turn each required semantic into natural customer language. Do not quote labels or JSON. "
        "Disclose and reinforce are cross-session tendencies. Session_only is a genuine requirement "
        "today but must not be described as a permanent preference. Current_override must be stated "
        "explicitly and in the first message, including the contrast with the usual preference.\n"
        f"{probe_rule}\n"
        "Never reveal a future preference: facts absent from both the already-established list and "
        "the current-session semantics must not be invented or volunteered."
    )


def build_first_turn_prompt(
    base_prompt: str,
    directive: Mapping[str, Any] | None,
) -> str:
    current = _validated_directive(directive)
    scheduled = [entry for key in DIRECTIVE_KEYS for entry in current[key]]
    if not scheduled:
        return base_prompt
    return (
        base_prompt
        + " In this first message, naturally communicate every item in the private "
        "Required current-session semantics block. Do not copy its wording verbatim."
    )


_CONCEPTS: dict[str, set[str]] = {
    "breathable": {"breathable", "breathability", "airy", "ventilated", "cool"},
    "natural": {"natural", "cotton", "linen", "canvas"},
    "neutral": {"neutral", "black", "brown", "beige", "grey", "gray", "navy", "understated"},
    "dark": {"dark", "black", "brown", "navy"},
    "minimal": {"minimal", "minimalist", "understated", "simple", "basic", "classic"},
    "budget": {"budget", "under", "below", "less", "120", "affordable"},
    "waterproof": {"waterproof", "rainproof", "water-resistant", "rain"},
    "reflective": {"reflective", "reflect", "visibility", "night"},
    "formal": {"formal", "dinner", "evening", "dressy", "collared", "collar"},
    "cargo": {"cargo", "storage", "pockets", "pocket"},
    "wool": {"wool"},
    "polyester": {"polyester", "synthetic"},
    "bright": {"bright", "neon", "colourful", "colorful", "bold"},
    "sporty": {"sporty", "athletic", "running", "workout", "sports"},
}
_STOP = {
    "a", "an", "and", "as", "for", "generally", "i", "in", "is", "it", "my",
    "of", "or", "products", "that", "the", "to", "usually", "want", "with",
}


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9-]+", text.casefold()) if token not in _STOP}


def _semantic_match(expected: str, observed: str) -> bool:
    wanted = _tokens(expected)
    seen = _tokens(observed)
    if not wanted:
        return False
    expanded: set[str] = set(seen)
    concept_hit = False
    for words in _CONCEPTS.values():
        if words.intersection(seen):
            expanded.update(words)
        if words.intersection(wanted) and words.intersection(seen):
            concept_hit = True
    anchors = {
        token for token in wanted
        if len(token) >= 4 or token.isdigit()
    }
    negative_expected = bool({"avoid", "does", "not", "no"}.intersection(wanted))
    negative_seen = bool(
        re.search(
            r"\b(?:avoid|don't|do not|no|not|without|negated[_ ]terms?)\b",
            observed.casefold(),
        )
    )
    matched = len(anchors.intersection(expanded))
    threshold = 1 if len(anchors) <= 3 else 2
    return (concept_hit or matched >= threshold) and (
        not negative_expected or negative_seen
    )


def semantic_match(expected: str, observed: str) -> bool:
    """Public diagnostic matcher; never used for retrieval or ranking."""

    return _semantic_match(expected, observed)


def semantic_disclosure_validation(
    directive: Mapping[str, Any] | None,
    shopper_messages: Iterable[str],
    final_fast_memory: Mapping[str, Any] | None,
    committed_memory_texts: Iterable[str],
) -> list[dict[str, Any]]:
    current = _validated_directive(directive)
    shopper_text = " ".join(str(value) for value in shopper_messages)
    fast_text = json.dumps(final_fast_memory or {}, ensure_ascii=False, default=str)
    committed_text = " ".join(str(value) for value in committed_memory_texts)
    diagnostics: list[dict[str, Any]] = []
    for kind in DIRECTIVE_KEYS:
        for expected in current[kind]:
            diagnostics.append(
                {
                    "directive_type": kind,
                    "semantic": expected,
                    "scheduled": True,
                    "shopper_expressed": _semantic_match(expected, shopper_text),
                    "fast_memory_captured": _semantic_match(expected, fast_text),
                    "memory_committed": _semantic_match(expected, committed_text),
                    "validation_method": "concept-token-overlap-v1",
                }
            )
    return diagnostics


def target_leakage(messages: Iterable[str], target_asin: str, target_title: str) -> dict[str, Any]:
    text = " ".join(str(message) for message in messages).casefold()
    normalized_title = re.sub(r"\s+", " ", str(target_title).strip()).casefold()
    asin_leaked = str(target_asin).strip().casefold() in text
    title_leaked = bool(len(normalized_title) >= 12 and normalized_title in re.sub(r"\s+", " ", text))
    return {
        "leaked": asin_leaked or title_leaked,
        "exact_target_asin": asin_leaked,
        "exact_target_title": title_leaked,
    }


__all__ = [
    "DIRECTIVE_KEYS",
    "ShopperCallResult",
    "ShopperLLMClient",
    "build_directive_system_prompt",
    "build_first_turn_prompt",
    "established_facts_before",
    "semantic_disclosure_validation",
    "semantic_match",
    "target_leakage",
]
