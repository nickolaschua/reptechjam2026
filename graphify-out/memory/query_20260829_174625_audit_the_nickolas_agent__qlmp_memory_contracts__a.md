---
type: "query"
date: "2026-08-29T17:46:25.876153+00:00"
question: "Audit the Nickolas Agent, QLMP memory contracts, and evaluator lifecycle for Phase 5 longitudinal shadow memory"
contributor: "graphify"
outcome: "useful"
source_nodes: ["Agent", "MemoryItem", "evaluate_v2"]
---

# Q: Audit the Nickolas Agent, QLMP memory contracts, and evaluator lifecycle for Phase 5 longitudinal shadow memory

## Answer

Expanded from original query via graph vocab: [agent, memory, session, reset, embedding, store, baseline, cosine, projection, evaluator, rank, query]. Agent.reset owned canonical Fast Memory only; QLMP MemoryItem is the reusable write schema; evaluate_v2 scores hit/rank before any new Nickolas end_session hook. Source verification was performed before implementation.

## Outcome

- Signal: useful

## Source Nodes

- Agent
- MemoryItem
- evaluate_v2