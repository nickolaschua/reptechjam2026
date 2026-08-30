# Fast Memory developments

## Summary

The Fast Memory in `nickolas/shopping_agent` shares the core design of
Yangxu's `experiment_1/shop_agent.py`, but the Nickolas implementation hardens
the session-state behavior and adds an explicit integration boundary for
longitudinal memory experiments.

Within one shopping conversation, both implementations maintain broadly the
same active state:

- disclosed shopping constraints;
- accumulated and superseded search terms;
- negative preferences;
- attributes that have already been asked about;
- previously recommended ASINs; and
- recent dialogue history.

Both implementations update this structured state with an LLM and fall back to
local parsing if the state update cannot be parsed.

## Developments in the Nickolas implementation

| Area | Yangxu `experiment_1` | Nickolas implementation |
| --- | --- | --- |
| State lifetime | Recreated for each session by `reset()` | Preserves the same session-local Fast Memory behavior |
| Slot representation | Some update paths can mix strings and sets | Normalizes disclosed slot values consistently into sets |
| LLM state updates | Primarily mutates the current slot mapping | Treats the returned slot mapping as the complete active mapping, allowing revoked or stale constraints to disappear |
| Route fallback | Supports baseline and hybrid routes | Snapshots state before route selection and restores it when a route falls back, preventing abandoned work from changing Fast Memory |
| Session lifecycle | No persistent cross-session memory lifecycle | Adds explicit reset/end-session lifecycle handling and final Fast Memory capture |
| Identity and ordering | State is keyed by session | Adds explicit user identity and chronological sequence metadata for longitudinal experiments |
| Isolation | Session-local isolation | Tests cross-user isolation and prevents future-session memory leakage |
| Evaluation safety | Standard evaluator behavior | Prevents evaluator annotations, targets, and purchase outcomes from becoming preference memory |
| Embeddings | Agent-specific embedding path | Uses a backend abstraction supporting the frozen OpenAI M0 and BGE variants |
| Experimental tooling | Session evaluation and instrumentation | Adds shadow history, snapshots, replay, parity checks, and longitudinal benchmark tooling |

## M0_OPENAI boundary

`M0_OPENAI` uses the Fast Memory implemented in
`nickolas/shopping_agent/agent.py`; it does not import or call Yangxu's agent at
runtime. The two implementations are closely related in design, but the
Nickolas implementation is locally owned code.

The canonical M0 baseline remains **Fast-Memory-only** for ranking. Current
session state can affect retrieval and reranking, while historical user memory
does not affect recommendations. Longitudinal records may be collected or made
visible in shadow-mode experiments, but they become ranking-active only when a
separate memory treatment, such as a QLMP strategy, is explicitly enabled.

This separation is intentional: M0 provides the frozen no-longitudinal-memory
control against which longitudinal and QLMP treatments can be measured.

## Practical interpretation

The Nickolas work is not a replacement for the original Fast Memory concept.
It is an evolution of the same within-session state model with stricter state
handling, safer routing, and the lifecycle and evaluation seams needed to test
cross-session memory without contaminating the M0 control.
