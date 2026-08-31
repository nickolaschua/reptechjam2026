# Harshith Retriever

This folder contains the current best local retriever experiment as a reusable component.

## What Is Here

- `warpretriever.py`: weighted lexical WARP-style retriever with a 250-candidate rerank pool.
- `WARPRetriever.search(query, top_k)`: accepts a plain query string and returns the same recommendation format expected by the TechJam evaluator:

```python
[{"parent_asin": "B000..."}, {"parent_asin": "B001..."}]
```

## How To Connect Later

Keep the competition `Agent` responsible for conversation state, prompting, and query construction. Use `WARPRetriever` only at the retrieval boundary.

```python
from harshith.warpretriever import WARPRetriever


class Agent:
    def __init__(self, catalog_path="data/catalog.jsonl"):
        self.retriever = WARPRetriever(catalog_path)
        self.sessions = {}

    def respond(self, session_id, user_message, turn, top_k):
        query = build_query_from_session_memory(session_id, user_message)
        return {
            "message": "What other details should I prioritize?",
            "ask_attribute": "other",
            "recommendations": self.retriever.search(query, top_k),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

## Notes

- The retriever builds a local pickle index from `catalog.jsonl` on first use.
- Set `WARP_INDEX_DIR` to control where that cache is stored.
- Do not put evaluator-specific hacks inside the retriever. Keep memory, override handling, and prompt policy in the agent layer.
