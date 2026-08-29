"""Utterance-axis text: styles, modifiers, the forbidden list, and the system prompt.

Everything the LLM shopper is told lives here so the benchmark's messiness is
inspectable in one file. Styles follow the team's search-type taxonomy. Spec
section 4.
"""
from __future__ import annotations

import random
import re

TOKEN_RE = re.compile(r"[a-z]+")

# Deliberately larger than exp11's list: the forbidden list must not be padded
# with function words, and overlap must not be inflated by them.
STOPWORDS = frozenset("""
a an and are as at be but by for from i in is it me my of on or please some that
the this to want with would you looking need like get one ones also just really
something anything thing things kind sort pair set new good nice great pretty
very quite bit lot more less than then there here what which who how when where
why can could should will do does did have has had not no yes if so up out
""".split())

STYLES = ("exact", "product_type", "feature", "use_case", "symptom",
          "compatibility", "plain", "lay")
GATED = {"exact": "has_model_code", "compatibility": "compat_eligible"}   # style -> product flag it needs
INTENT_LABEL = {st: ("buying" if st == "exact" else "browsing") for st in STYLES}
MODIFIERS = ("negation", "for_other", "vague_budget", "format_noise")
MODIFIER_ONLY_FOR = {"format_noise": "exact"}                             # modifier -> the one style it applies to

STYLE_INSTRUCTIONS = {
    "exact": "You remember this from the listing: {code}. Mention it, plus one other thing you want.",
    "product_type": ("Name only the kind of item you want. Nothing else - no features, "
                     "no occasion, no brand."),
    "feature": "Name the kind of item and two or three things it must have.",
    "use_case": ("Describe the situation, event or task you need this for. Do not name "
                 "the item and do not list its attributes - let the assistant work it out."),
    "symptom": ("Describe the problem you are trying to fix, or how you want to look or "
                "feel. Do not name the item."),
    "compatibility": ("You already own a {anchor} and need this to go with it. Describe "
                      "what you own and what you need it for - do not describe the "
                      "accessory itself."),
    "plain": "Tell the assistant what you're looking for.",
    "lay": ("Describe what you want in everyday words. You must not use any of these "
            "words: {forbidden}."),
}

MODIFIER_INSTRUCTIONS = {
    "negation": "Also say one specific thing you do NOT want.",
    "for_other": "You are buying this for your {relation}.",
    "vague_budget": "Mention that price matters to you, but do not say a number.",
    "format_noise": ("Write the code with different spacing, hyphens or capitalisation "
                     "than the listing."),
}

RELATIONS = {
    "womens": ("wife", "mum", "sister", "daughter"),
    "mens": ("dad", "husband", "brother", "son"),
    "girls": ("daughter", "niece"),
    "baby-girls": ("daughter", "niece"),
    "boys": ("son", "nephew"),
    "baby-boys": ("son", "nephew"),
}


def content_words(text: str) -> list[str]:
    """Lowercase alphabetic tokens, stopwords removed, first-occurrence order, unique."""
    seen: dict[str, None] = {}
    for tok in TOKEN_RE.findall(str(text or "").lower()):
        if tok not in STOPWORDS and len(tok) > 1:
            seen.setdefault(tok, None)
    return list(seen)


def forbidden_list(product: dict, cap: int = 40) -> list[str]:
    """The listing's own vocabulary, so the `lay` style cannot paraphrase it back."""
    feats = product.get("features") or []
    text = " ".join([str(product.get("title") or ""), *map(str, feats)])
    return content_words(text)[:cap]


def relation_for(department: str | None, rng: random.Random) -> str:
    """A recipient consistent with the target's department - never a contradiction."""
    options = RELATIONS.get(department or "")
    return rng.choice(options) if options else "friend"


def build_system_prompt(product: dict, card: dict, profile: dict, style: str,
                        modifiers: list[str], code: str | None = None,
                        relation: str | None = None, anchor: str | None = None) -> str:
    """v2's shopper prompt with rule 3 removed and the style/modifier rules appended."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")
    desc = product.get("description") or ""
    if isinstance(desc, list):
        desc = " ".join(map(str, desc))
    details = product.get("details") or {}
    details_str = " ".join(f"{k}: {v}" for k, v in details.items()) if isinstance(details, dict) else str(details)

    lines = [
        "You are acting as a real customer shopping online. Your target product is:",
        f"Target Product Title: {product.get('title', '')}",
        f"Hard Constraints (Must-Haves): {', '.join(map(str, card.get('hard_constraints', [])))}",
        f"Soft Preferences (Nice-to-Haves): {', '.join(map(str, card.get('soft_preferences', [])))}",
        f"Target Product Details: {details_str}",
        f"Target Product Description: {str(desc)[:300]}",
        "",
        "Your Shopping Profile:",
        f"Purchase Frequency: {profile.get('purchase_frequency', 'Regular')}",
        f"Preference Tags: {', '.join(profile.get('preference_tags', []))}",
        f"Summary: {profile.get('summary', '')}",
        "",
        "Rules:",
        "1. Do NOT state the exact product title or any product code unless told to. Speak like a real human.",
        "2. Use synonyms and subjective phrases instead of copying the listing.",
        "3. Keep it short, informal and conversational.",
        "4. Write in lowercase with simple punctuation. No capital letters.",
        "5. Write only the opening message. One to three sentences.",
        "",
        "How to describe what you want:",
    ]
    instr = STYLE_INSTRUCTIONS[style]
    if style == "lay":
        instr = instr.format(forbidden=", ".join(forbidden_list(product)))
    elif style == "exact":
        if not code:
            raise ValueError("exact style needs a code")
        instr = instr.format(code=code)
    elif style == "compatibility":
        if not anchor:
            raise ValueError("compatibility style needs an anchor")
        instr = instr.format(anchor=anchor)
    lines.append(instr)
    for m in modifiers:
        if m not in MODIFIERS:
            raise ValueError(f"unknown modifier {m!r}")
        if m in MODIFIER_ONLY_FOR and MODIFIER_ONLY_FOR[m] != style:
            raise ValueError(f"modifier {m!r} only applies to style {MODIFIER_ONLY_FOR[m]!r}")
        text = MODIFIER_INSTRUCTIONS[m]
        if m == "for_other":
            text = text.format(relation=relation or "friend")
        lines.append(text)
    return "\n".join(lines)
