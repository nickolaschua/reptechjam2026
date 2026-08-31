# Yangxu Merge and System Audit Issues

Audit date: 2026-08-31

Scope: the active `system.shopping_agent` runtime, catalogue and FTS routing, dialogue state, clarification behavior, longitudinal memory, CLI, HTTP/SSE server, simulator, and Yangxu dashboard. The audit compared the active extraction against Yangxu commit `1404ee1419af579822c4dd867112de92595e35c0` (`1404ee1`).

The system is not clean yet. Three remaining merge regressions were missed or only partially fixed, and two additional integration concerns were identified. Previously recorded open defects in `issues.md` also remain.

## Remaining merge regressions

### YX-001 — Structured-state updates can erase valid prior constraints

- Severity: High
- Attribution: Merge regression from Yangxu's defensive merge behavior
- Location: `system/shopping_agent/agent.py:1718`

The active implementation treats `disclosed_slots` returned by the state LLM as a complete replacement and deletes every omitted soft slot.

Yangxu's `1404ee1` implementation deliberately merged returned slots into the existing state to protect against LLM forgetting. In a controlled probe, an LLM response containing only `material=cotton` erased an existing `color=black` constraint.

This can change retrieval behavior immediately and can also remove a valid preference from subsequent longitudinal-memory serialization.

### YX-002 — Clarification bookkeeping diverges in three directions

- Severity: Medium
- Attribution: Merge regression across state translation and generator prompting
- Locations: `system/shopping_agent/agent.py:1652`, `system/shopping_agent/agent.py:1738`, `system/shopping_agent/agent.py:2090`, `system/shopping_agent/agent.py:2113`

Three related divergences remain:

1. The active state update replaces the entire `asked_attributes` set. Yangxu validates and merges newly returned values, preserving questions asked on previous turns. A controlled response with `asked_attributes=[]` erased two previously asked attributes.
2. A product-category change clears constraints but does not clear `asked_attributes`. Yangxu explicitly clears them for the new category, so the active agent can incorrectly avoid relevant questions for the new product type.
3. Yangxu's latest generator prompt requires asking about both entropy-selected attributes. The active prompt permits asking about "either or both," but the code marks both attributes as asked afterward. When the model asks about only one, the other can be silently suppressed for the rest of the session.

### YX-003 — Deterministic intent precedence remains partially incorrect

- Severity: Medium
- Attribution: Partially unresolved merge regression; reopen SYS-007
- Location: `system/shopping_agent/agent.py:1309`

The fallback treats any occurrence of `just looking` as an explicit reset before checking concrete buying evidence or active hard conditions.

Confirmed misclassifications:

```text
I am not just looking; I need waterproof boots       -> browsing
Just looking, but it must be under $50               -> browsing
Just looking for a specific Nike boot under $100     -> browsing
```

This path is used only when LLM intent detection is unavailable, but it still contradicts the required rule that concrete requirements and hard conditions take precedence over ordinary exploratory language. SYS-007 should be considered partially resolved rather than closed.

## Additional integration findings

### YX-004 — Completed session state accumulates indefinitely

- Severity: Medium/Low
- Attribution: New lifecycle integration concern
- Locations: `system/shopping_agent/agent.py:357`, `system/shopping_agent/agent.py:1063`

Every ended session retains a deep copy of its complete fast-memory state in `_ended_lifecycle`. Browser sessions use unique IDs, and there is no eviction or cleanup API for ended entries.

A 100-session probe produced:

```text
active=0
session_states=0
ended_retained=100
```

This causes unbounded process-memory growth in a long-running dashboard or repeated evaluation process.

### YX-005 — Active architecture documentation describes the pre-merge runtime

- Severity: Low
- Attribution: Documentation integration drift
- Locations: `system/MEMORY_ARCHITECTURE.md:11`, `system/MEMORY_ARCHITECTURE.md:28`, `system/MEMORY_ARCHITECTURE.md:30`, `system/REPOSITORY_CLASSIFICATION.md:9`

The memory architecture document says caller-supplied mode is mandatory when stored memory exists, while live intent is now authoritative.

It also says the live path has no FTS/BM25 candidate generation and that only price and negative masks apply. The active runtime now uses FTS routing and demographic, rating, review-count, store, price, and negative eligibility masks.

The repository classification also omits the newer active browser and catalogue test modules.

## Previously recorded defects still open

The following findings from `issues.md` remain reproducible or evident.

### High severity

- SYS-001: Dashboard department filters do not separate departments.
- SYS-002: Failed JSON persistence leaves an unrepeatable phantom commit.
- SYS-003: Failed response turns permanently mutate state and consume a turn.
- SYS-004: Negation parsing creates contradictory positive and negative state and can contaminate longitudinal memory.

### Medium severity

- SYS-005: The brand parser consumes following constraints.
- SYS-006: Negative hard filters still use unsafe raw-substring matching. Only Yangxu's three generic exceptions have been restored.
- SYS-008: Dashboard values are rendered through unescaped `innerHTML`.
- SYS-009: SSE monopolizes the single-threaded HTTP server.
- SYS-010: Catalogue SQLite connections have no close lifecycle.

SYS-008 also affects the product-catalog page. Product titles, brands, and image URLs are inserted through `innerHTML`, in addition to the unsafe conversation-page rendering already recorded.

### Medium/Low severity

- SYS-012: Direct `Agent()` construction can regenerate catalogue embeddings because generation defaults to enabled.

### Low severity

- SYS-013: CLI `/mode` claims to lock a mode that live intent can override.
- SYS-014: Local `.env` values overwrite process environment variables.
- SYS-015: The unauthenticated wildcard-CORS HTTP surface is unsafe outside a trusted machine.

## Correctly integrated or intentionally different behavior

The following differences were reviewed and are not classified as merge failures:

- Buying and Browsing FTS thresholds match Yangxu's values.
- Intent-specific clarification priority orders match Yangxu.
- The generic-negative exception for `clothing`, `shoes`, and `jewelry` is restored.
- Canonical demographic handling through `target_department` is present.
- Yangxu's four server routes and split catalogue/conversation UI are preserved.
- Watches and complete-catalogue pagination are intentional extensions.
- The different profile gate, blend weights, full-matrix ordering, and lack of seen-product exclusion are documented frozen-memory decisions rather than accidental omissions.
- The HTML files contain proper Unicode. Apparent mojibake seen in terminal output was caused by PowerShell display decoding, not corrupted source files.

## Verification performed

- Compared every file changed by Yangxu commit `1404ee1` with its active extracted counterpart.
- Ran `python -m pytest system/shopping_agent/tests -q -W default`: 60 passed and 9 subtests passed.
- Observed five `ResourceWarning` instances confirming the existing unclosed SQLite connection defect.
- Ran `python -m compileall -q system`: passed.
- Checked JavaScript from both active HTML pages with `node --check`: passed.
- Ran focused adversarial intent, structured-state continuity, clarification-state, eligibility, lifecycle-retention, and catalogue probes.

The audit itself made no runtime-code changes.
