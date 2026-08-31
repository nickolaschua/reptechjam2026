"""EXP05 - Why does experiment_1 score 0.896 on the evaluator and 0.000 on a
messy LLM-generated session?

Replays the recorded Ollama/Llama2 stress transcript against experiment_1's
parsing logic to isolate the failure modes, then runs the same messages through
whole-message bucket resolution to show the failures are bugs, not difficulty.
"""
from __future__ import annotations

import re

from common import TOKEN_RE, coarse_category, get_index, write_result

# Recorded transcript: sample public_0085, browsing, target B0C3YJMRRD.
TRANSCRIPT_TARGET = "B0C3YJMRRD"
TRANSCRIPT = [
    "I'm looking for a pair of slides that are super comfortable and have a soft, thick sole. I'd like them to be suitable for both indoor and outdoor use, and preferably made with a material that's easy to clean and doesn't absorb water.",
    "None of these options seem to match what I'm looking for. I need something with a soft, thick sole and a non-slip design, preferably for both women and men. Can you show me some options that fit these criteria?",
    "Hmm, none of these seem to match what I'm looking for. I need something with a soft, thick sole, preferably made of EVA, and it should be suitable for both indoor and outdoor use. I'd also prefer a more minimalist design, like a slide sandal.",
    "I don't see my ideal product in this list. I'm looking for something with a soft, thick sole and a heel that's about 1.57 inches high. Can you show me more options that fit those criteria?",
    "Umm, I don't see anything here that matches what I'm looking for. I need something for my feet, specifically a pair of slides with a soft, thick sole that's good for indoor and outdoor use. Can you show me some options that fit that description?",
]

# experiment_1/agent.py parsing surface, reproduced verbatim for diagnosis.
EXP1_CATEGORY_RE = re.compile(r"I'm looking for ([^.,]+)")
EXP1_TEMPLATES = {
    "a_key_requirement": r"A key requirement is: ([^.]+)\.",
    "what_matters_is": r"what matters is: ([^.]+)\.",
    "what_i_need_is": r"What I need is: ([^.]+)\.",
    "boundary_phrase": r"I don't have a preference for ([^;.]+); please use your judgment\.",
}
CHATTER = {"i", "im", "a", "an", "the", "for", "of", "to", "and", "with", "is", "are",
           "that", "be", "my", "me", "you", "can", "show", "some", "options", "fit",
           "those", "these", "criteria", "something", "looking", "need", "want", "like",
           "prefer", "preferably", "more", "see", "don", "t", "here", "what", "it", "s",
           "use", "both", "seem", "match", "none", "them", "d", "ll", "made", "cause"}


def main() -> dict:
    ix = get_index()
    out: dict = {"transcript": {"sample_id": "public_0085", "scenario": "browsing",
                                "target": TRANSCRIPT_TARGET, "turns_recorded": 10,
                                "reported_outcome": "FAILED (target never surfaced)"}}
    target_in_catalog = TRANSCRIPT_TARGET in ix.products
    out["transcript"]["target_in_catalog"] = target_in_catalog
    if target_in_catalog:
        bucket = ix.bucket_of[TRANSCRIPT_TARGET]
        pool = ix.buckets[bucket]
        by_pop = sorted(pool, key=lambda a: -ix.popularity[a])
        out["transcript"].update({
            "target_title": ix.products[TRANSCRIPT_TARGET]["title"],
            "bucket": coarse_category(ix.categories[TRANSCRIPT_TARGET]),
            "bucket_size": len(pool),
            "target_rating_number": ix.products[TRANSCRIPT_TARGET].get("rating_number"),
            "target_rank_by_popularity": by_pop.index(TRANSCRIPT_TARGET) + 1,
        })

    out["bug_1_category_regex"] = {
        "pattern": EXP1_CATEGORY_RE.pattern,
        "captures_per_turn": [
            (m.group(1) if (m := EXP1_CATEGORY_RE.search(t)) else None) for t in TRANSCRIPT
        ],
        "problem": "[^.,]+ runs to the first comma, so the whole clause becomes state['category']; "
                   "'I'm specifically looking for' does not match the anchor at all",
    }

    out["bug_2_template_only_parsing"] = {
        name: sum(1 for t in TRANSCRIPT if re.search(rx, t))
        for name, rx in EXP1_TEMPLATES.items()
    }
    out["bug_2_template_only_parsing"]["messages_tested"] = len(TRANSCRIPT)
    out["bug_2_template_only_parsing"]["problem"] = (
        "every constraint path is a template written against the simulator's literal "
        "output strings; a real shopper triggers none, so EVA / non-slip / 1.57in are discarded")

    out["bug_3_seen_asins"] = {
        "code": "if pid in state['seen_asins']: continue  /  state['seen_asins'].update(recommendations)",
        "products_removed_per_turn": 10,
        "products_removed_over_session": 100,
        "cleared_only_on": ["category change", "override regex match"],
        "problem": "neither clears here, so the agent consumes its own candidate pool; "
                   "turn 1 returns slides, turn 10 returns t-shirts",
    }

    # The same messages, with whole-message bucket resolution and no exclusion.
    def content_terms(message: str) -> list[str]:
        return [t for t in TOKEN_RE.findall(message.lower())
                if t not in CHATTER and len(t) > 2 and t in ix.idf]

    pool: list[str] | None = None
    accumulated: list[str] = []
    resolved: list[str] = []
    per_turn = []
    for turn, message in enumerate(TRANSCRIPT, 1):
        accumulated += content_terms(message)
        if pool is None:
            keys = ix.resolve_bucket(" ".join(content_terms(message)), top_n=3)
            if keys and sum(ix.idf.get(t, 0.0) for t in
                            ix.bucket_tokens[keys[0]] & set(accumulated)) > 0:
                pool = [a for k in keys for a in ix.buckets[k]]
                resolved = keys
        if not pool:
            per_turn.append({"turn": turn, "resolved_buckets": None, "target_rank": None})
            continue
        ranked = sorted(
            ((sum(ix.idf[t] for t in set(accumulated) if t in ix.tokens[a]) + ix.popularity[a], a)
             for a in pool), reverse=True)
        order = [a for _, a in ranked]
        per_turn.append({
            "turn": turn, "resolved_buckets": resolved, "pool_size": len(pool),
            "target_rank": order.index(TRANSCRIPT_TARGET) + 1 if TRANSCRIPT_TARGET in order else None,
        })
    out["whole_message_resolution"] = {
        "description": "no anchor phrase, no template regexes, no seen_asins exclusion",
        "per_turn": per_turn,
    }
    return out


if __name__ == "__main__":
    print("EXP05 agent diagnostics")
    write_result("exp05_agent_diagnostics", main())
