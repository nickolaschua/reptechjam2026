# System Audit Issues

Audit date: 2026-08-31

Scope: the active `system.shopping_agent` package, longitudinal memory lifecycle, catalogue and FTS routing, live intent, CLI, HTTP/SSE server, and Yangxu dashboard. This record now notes the strict merge-divergence repairs completed after the audit; inherited defects remain open unless explicitly marked otherwise.

## Merge attribution summary

The issues fall into three different groups. “Integration-related” does not always mean that Yangxu's implementation was copied incorrectly: some defects were introduced by the new longitudinal layer, while others are old Yangxu behavior that became more serious after persistence was added.

| Group | Issues | Attribution |
|---|---|---|
| Strict merge divergences (resolved) | SYS-007, SYS-011, part of SYS-006 | The live-intent/state divergences and Yangxu's dropped three-term negative exception have been restored. The inherited portion of SYS-006 remains open. |
| Integration-amplified defects | SYS-003, SYS-004 (LTM consequence), SYS-015 | Yangxu's original behavior remains, but adding automatic longitudinal persistence makes its impact worse. |
| Inherited from Yangxu | SYS-001, SYS-004 (base parser), SYS-005, most of SYS-006, SYS-008, SYS-009, SYS-010, SYS-014 | The same behavior is present in Yangxu's `1404ee1` source. These are bugs worth fixing, but they are not evidence of a failed merge. |
| Ours/new architecture only | SYS-002, SYS-012, SYS-013 | These concern the persistent store, frozen embedding cache contract, and supporting CLI, none of which existed in Yangxu's implementation in this form. They are our defects, but not failures to merge his work. |

The audit identified these divergence and integration findings (the strict-divergence items 1, 2, and 5 are now resolved):

1. The deterministic intent fallback gives browsing phrases priority over concrete buying signals (SYS-007).
2. The translated structured-state prompt contradicts itself about where demographic constraints belong (SYS-011).
3. Failed turns are not rolled back, and the new lifecycle can later persist their partial state (SYS-003).
4. Yangxu's unresolved positive/negative slot conflict can now contaminate longitudinal memory (part of SYS-004).
5. The extracted negative-mask implementation dropped Yangxu's small exception for the generic terms `clothing`, `shoes`, and `jewelry`. This is a behavioral merge divergence, although Yangxu's original raw-substring filtering was already broadly unsafe (SYS-006).

## Confirmed issues

### SYS-001 — Dashboard department filters do not separate departments

- Severity: High
- Attribution: Inherited from Yangxu's latest catalogue feature, not caused by the merge.
- Location: `system/shopping_agent/visualizer/server.py:183`, `system/shopping_agent/visualizer/server.py:213`

Department membership is determined by searching the product title plus every category string using raw substring matching. The catalogue's broad root category, “Clothing, Shoes & Jewelry,” therefore makes nearly every row match several tabs.

Measured against the active 50,000-row catalogue:

- Clothing matched 49,990 rows.
- Shoes matched 50,000 rows.
- Jewelry matched 50,000 rows.
- Watches matched 1,669 rows and included false positives such as “Overwatch” and “watching.”

Yangxu's `1404ee1` server uses the same `any(kw in combined ...)` test. The integration removed his 2,000-result truncation and added Watches, but neither change created the underlying classification bug.

### SYS-002 — Failed JSON persistence leaves an unrepeatable phantom commit

- Severity: High
- Attribution: Ours; introduced by the longitudinal JSON store rather than Yangxu's code.
- Location: `system/shopping_agent/memory_store.py:288`

`JsonFileVectorMemoryStore.commit()` mutates the in-memory store before calling `_persist()`. If disk persistence fails, the session is already marked committed and removed from the active maps, but the durable file remains stale.

A fault-injection reproduction produced:

```text
first_error OSError
active_after_failure False
commits_after_failure 1
retry_error ValueError duplicate committed session_id 's1'
disk_file_exists False
```

This contradicts the agent's retryability and atomic-commit expectations.

### SYS-003 — Failed turns permanently mutate state and consume a turn

- Severity: High
- Attribution: Integration-amplified. Yangxu already appended history and incremented manual turns before response completion; the longitudinal lifecycle makes the partial state persistable.
- Location: `system/shopping_agent/agent.py:1834`, `system/shopping_agent/agent.py:1858`, `system/shopping_agent/visualizer/server.py:258`, `system/shopping_agent/demo.py:94`

The agent appends the user message and updates constraints before embedding, scoring, and response generation finish. The browser and CLI increment their turn counters before calling the agent. There is no rollback when a provider or scorer fails.

When query embedding was forced to fail, retrying the same request increased the history count from one to two. In streamed mode, the `finally` path can commit preferences parsed from a turn whose response was never delivered.

### SYS-004 — Negation parsing creates contradictory state and can poison LTM

- Severity: High
- Attribution: Yangxu's base parser bug, amplified by the merge because positive soft slots are now committed to longitudinal memory.
- Location: `system/shopping_agent/agent.py:1184`, `system/shopping_agent/agent.py:1044`

For `I'm looking for no leather boots`, deterministic parsing produces:

```text
category = no leather boots
material = leather
negated_terms = boots, leather
```

The requested product type is excluded, while leather is simultaneously recorded as a positive material. After `I prefer leather boots` followed by `Actually no leather`, the positive `material=leather` slot remains and can be serialized into the LTM update text.

The broad negation regex and unconditional positive material extraction both exist in Yangxu's source. The persistent-memory consequence is new to the combined system.

### SYS-005 — Brand parser consumes following constraints

- Severity: Medium
- Attribution: Inherited unchanged from Yangxu.
- Location: `system/shopping_agent/agent.py:1256`

The greedy regex `brand(?:s)? like\s+([a-zA-Z0-9\s]+)` consumes text until punctuation or a nonmatching character. For:

```text
I'm looking for boots from brands like Nike under $100
```

the resulting category is `boots from brands like` and the hard store value is `nike under`. This commonly forces a zero-result hard mask.

### SYS-006 — Negative hard filters use unsafe substring matching

- Severity: Medium
- Status: Open for the inherited substring-matching defect; merge-divergence portion resolved
- Attribution: Primarily inherited from Yangxu. The catalogue-module extraction also removed his narrow exception for generic department terms.
- Location: `system/shopping_agent/catalogue.py:288`

A negative term is rejected with `term in searchable_text`, without token boundaries. Active-catalog measurements found:

- `red` falsely matched 11,708 rows through words such as “inspired,” “featured,” and “reducing.”
- `tan` falsely matched 12,189 rows through words such as “resistant” and “rectangle.”

Yangxu used the same raw substring test. His exception for `clothing`, `shoes`, and `jewelry` has now been restored in both active eligibility paths. The wider raw-substring bug remains open.

### SYS-007 — Deterministic live intent misclassifies mixed high-intent messages

- Severity: Medium
- Status: Resolved
- Attribution: Merge/integration defect in the added deterministic fallback.
- Location: `system/shopping_agent/agent.py:1309`

The fallback previously checked ordinary browsing phrases before concrete buying signals or existing hard conditions. Consequently:

```text
I need waterproof boots, show me options
```

was classified as Browsing. The deterministic fallback now gives concrete buying evidence priority, preserves an existing Buying state for ordinary option requests, and returns to Browsing only for explicit reset/exploratory behavior. Successful LLM intent detection remains authoritative.

### SYS-008 — Dashboard renders untrusted values through `innerHTML`

- Severity: Medium
- Attribution: Inherited from Yangxu's dashboard, copied unchanged as requested.
- Location: `system/shopping_agent/visualizer/conversation.html:1001`, `system/shopping_agent/visualizer/conversation.html:1129`

Manual user messages, LLM responses, product titles, brands, features, and details are interpolated into `innerHTML` without escaping. User-entered HTML containing event handlers can execute JavaScript in the local dashboard. Yangxu's `1404ee1` `conversation.html` contains the same `msgWrapper.innerHTML` and `${msg.content}` flow.

### SYS-009 — SSE monopolizes the single-threaded server

- Severity: Medium
- Attribution: Inherited from Yangxu.
- Location: `system/shopping_agent/visualizer/server.py:433`

The server uses `HTTPServer`, not `ThreadingHTTPServer`. An SSE run can occupy the sole request thread for ten turns, including model calls and deliberate delays, preventing catalog searches, session listing, or manual activity from other clients. Yangxu's server uses the same server class.

### SYS-010 — SQLite catalogue connections have no close lifecycle

- Severity: Medium
- Attribution: Inherited from Yangxu, retained after catalogue extraction.
- Location: `system/shopping_agent/catalogue.py:176`

Each `Catalogue` creates an in-memory SQLite connection but exposes no `close()` method or context-manager lifecycle, and the agent/server never closes it. Coverage runs emitted unclosed `sqlite3.Connection` warnings.

### SYS-011 — Structured-state prompt contradicts demographic handling

- Severity: Medium
- Status: Resolved
- Attribution: Merge regression while translating Yangxu's nested `hard_conditions.department` into `target_department`.
- Location: `system/shopping_agent/agent.py:1599`, `system/shopping_agent/agent.py:1606`

The prompt now gives demographics one exclusive destination, `target_department`. Defensive normalization translates legacy demographic values returned in root `department`, prefers a valid canonical `target_department` on conflict, and records the resulting gender constraint as session-only so it cannot enter longitudinal memory text.

## Lower-risk issues

### SYS-012 — Direct `Agent()` construction permits catalogue embedding regeneration

- Severity: Medium/Low
- Attribution: Ours; part of the frozen OpenAI cache integration.
- Location: `system/shopping_agent/agent.py:323`

`allow_catalog_embedding` defaults to `True`. The canonical server and CLI pass `False`, but another caller constructing `Agent()` can regenerate and write embeddings if the cache is absent or rejected. This conflicts with the documented fixed, validated 50,000-row matrix.

### SYS-013 — CLI `/mode` claims to lock a mode that live intent can override

- Severity: Low
- Attribution: Ours; the CLI predates or was not updated for Yangxu's live-intent-authoritative behavior.
- Location: `system/shopping_agent/demo.py:213`

The command prints `Mode locked`, but its value is only a caller fallback. Live detected intent remains authoritative.

### SYS-014 — Local `.env` values overwrite process environment variables

- Severity: Low
- Attribution: Inherited unchanged from Yangxu.
- Location: `system/shopping_agent/agent.py:93`

The custom loader assigns directly to `os.environ[key]`, so a checked/local `.env` silently overrides deployment-provided values. Yangxu's source contains the same assignment.

### SYS-015 — Local HTTP surface is unsafe outside a trusted machine

- Severity: Low
- Attribution: Mostly inherited from Yangxu; longitudinal mutation increases the impact.
- Location: `system/shopping_agent/visualizer/server.py:315`, `system/shopping_agent/visualizer/server.py:433`

The server binds to `0.0.0.0`, permits wildcard CORS, uses state-changing GET endpoints, and has no authentication. This was already true of Yangxu's visualizer, but those endpoints can now start and commit persistent user sessions.

## Validation status

- `python -m pytest system/shopping_agent/tests -q`: 60 passed, 9 subtests passed.
- `python -m compileall -q system`: passed.
- `git diff --check -- system`: passed.
- Coverage audit: 66% overall. Important gaps remain in server (48%), simulator (30%), CLI (36%), provider-failure rollback, disk-failure recovery, HTTP/SSE concurrency, and browser escaping.

Green tests do not cover the fault-injection and real-catalog cases above.

## Suggested repair order

1. Make JSON commits transactional and retryable.
2. Add response-turn snapshot/rollback semantics and commit only successfully completed turns.
3. Replace catalogue department substring matching with canonical catalogue metadata/token matching.
4. Reconcile negated constraints with positive slots before filtering and LTM serialization.
5. Fix token-boundary negative matching and brand/category parsing.
6. Align deterministic intent precedence and the structured-state prompt with Yangxu's live-intent rules.
7. Escape browser content and use a threaded server with an explicit shutdown/connection lifecycle.
