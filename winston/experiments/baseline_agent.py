"""Reference agent: category pool + IDF-weighted constraint match + popularity prior.

Pure standard library. No model, no API, no tokens. Every knob is a flag so the
ablations in exp03 can turn one thing off at a time.
"""
from __future__ import annotations

import re

from common import TOKEN_RE, get_index

CATEGORY_RE = re.compile(r"looking for (.+?)(?:\.|,\s*but)", re.I)
CONSTRAINT_RE = re.compile(
    r"(?:a key requirement is:|what matters is:|what i need is:)\s*(.+)", re.I
)
NO_PREF_RE = re.compile(r"don't have (?:an additional |a )?preference", re.I)
EXHAUSTED_RE = re.compile(r"don't have an additional preference for other", re.I)


class BaselineAgent:
    def __init__(
        self,
        *,
        use_popularity: bool = True,
        use_lexical: bool = True,
        patience: bool = True,
        fuzzy_category: bool = False,
        partial_credit: float = 0.3,
        popularity_weight: float = 1.0,
        tag_weight: float = 0.0,
        ask_attribute: str = "other",
    ) -> None:
        self.ix = get_index()
        self.use_popularity = use_popularity
        self.use_lexical = use_lexical
        self.patience = patience
        self.fuzzy_category = fuzzy_category
        self.partial_credit = partial_credit
        self.popularity_weight = popularity_weight
        self.tag_weight = tag_weight
        self.ask_attribute = ask_attribute
        self._global = sorted(self.ix.products, key=lambda a: -self.ix.popularity[a])

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.pool: list[str] | None = None
        self.constraints: list[str] = []
        self.exhausted = False
        self.tags = [
            t.split()[0].lower()
            for t in (user_profile or {}).get("preference_tags", [])
        ]

    def _absorb(self, message: str) -> None:
        if EXHAUSTED_RE.search(message):
            self.exhausted = True
        if NO_PREF_RE.search(message):
            return
        found = CONSTRAINT_RE.search(message)
        if found:
            self.constraints += [
                c.strip(" .") for c in found.group(1).split(";") if c.strip(" .")
            ]
        elif self.pool is not None and not message.startswith("Those options"):
            # intent_override turn 1 tail: "I'm looking for X. <old value>"
            tail = message.split(". ", 1)
            if len(tail) > 1 and tail[1].strip():
                self.constraints.append(tail[1].strip(" ."))

    def _resolve_pool(self, message: str) -> None:
        found = CATEGORY_RE.search(message)
        phrase = found.group(1).strip().lower() if found else message.lower()
        if self.fuzzy_category:
            keys = self.ix.resolve_bucket(phrase, top_n=3)
            self.pool = [a for k in keys for a in self.ix.buckets[k]] or self._global[:2000]
        else:
            self.pool = self.ix.buckets.get(phrase, [])

    def score(self, asin: str) -> float:
        total = 0.0
        if self.use_lexical:
            for clause in self.constraints:
                lowered = clause.lower()
                terms = [t for t in TOKEN_RE.findall(lowered) if t in self.ix.idf]
                weight = sum(self.ix.idf[t] for t in terms)
                if not weight:
                    continue
                if lowered in self.ix.text[asin]:
                    total += weight
                else:
                    total += self.partial_credit * sum(
                        self.ix.idf[t] for t in terms if t in self.ix.tokens[asin]
                    )
        if self.use_popularity:
            total += self.popularity_weight * self.ix.popularity[asin]
        if self.tag_weight:
            total += self.tag_weight * sum(
                self.ix.idf.get(t, 0.0) for t in self.tags if t in self.ix.tokens[asin]
            )
        return total

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if self.pool is None:
            self._resolve_pool(user_message)
        self._absorb(user_message)
        pool = self.pool or []
        if not pool:
            return {"message": "Could you tell me more?", "ask_attribute": self.ask_attribute,
                    "recommendations": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        ranked = sorted(((self.score(a), a) for a in pool), reverse=True)
        # Patience: the session ends on first hit AT WHATEVER RANK, so showing ten
        # items early locks in a bad rank. Show one until the card is complete.
        limit = top_k if (not self.patience or self.exhausted or turn >= 5) else 1
        return {
            "message": "Anything else that matters to you?",
            "ask_attribute": self.ask_attribute,
            "recommendations": [{"parent_asin": a} for _, a in ranked[:limit]],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
