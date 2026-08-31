# Feasibility Slide — Cost, Scale, Latency (Design)

**Date:** 2026-09-01
**Deliverable:** paste-ready content (numbers + copy) for one slide in `TIKTOK_TECHJAM_2026.pptx`. Nothing added to the deck file itself; nothing merged into agent code.

## Goal

One slide, three blocks, answering the judges' feasibility questions: what does ASTRA cost to run, how does it scale, and is it fast enough for a live (online) and offline session.

## Measurement method

- Standalone harness script in the session scratchpad — **zero edits to repo source** (`agent.py` / `catalogue.py` carry uncommitted edits owned by other sessions on `bench`).
- Harness imports the shopping agent, wraps the LLM client, embedding backend, and retrieval entry points with wall-clock timers and token counters, then drives scripted multi-turn sessions from `system/shopping_agent/demo_scenarios.json`.
- Runs fully local on Ollama (`ollama serve` started for the run; default `TEST_MODE=false` path).
- Token counts are measured from real prompts/responses; **cost is priced at current DeepSeek `deepseek-chat` published rates** (verified at execution time, source cited). Token counts are model-agnostic, so the math holds even though the local run generates with Llama 3.1.
- Optional add-on if a `DEEPSEEK_API_KEY` is provided: ~10 real calls against `api.deepseek.com` for true online RTT and exact usage (< $0.01). Otherwise online latency = measured local pipeline with the state-tracking stage replaced by DeepSeek's cited/typical API RTT, source noted on the slide.

## Slide content

### Block 1 — API & cost per turn
- Measured LLM calls per turn (expected: 1 cloud state-tracking + 1 local question/response generation).
- Avg input/output tokens for the state-tracking call → $/turn → **$/session headline** (avg turns per session taken from the scenario runs, ~6 expected).
- Embedding + retrieval marginal cost: $0 (local fine-tuned bge-base, precomputed catalogue vectors).

### Block 2 — Scalability
- *Cost to serve:* extrapolation table — $/1k sessions, $/10k sessions from measured per-turn tokens.
- *Ease of scale:* stateless per-turn pipeline (session state = one JSON blob) → trivial horizontal scaling; retrieval is local matrix ops over precomputed vectors → no API bottleneck; at ~1M items the only two components that change are brute-force dot product → ANN index (e.g. FAISS) and SQLite FTS5 → hosted DB.

### Block 3 — Latency
- Per-stage breakdown from real runs: state-tracking LLM / FTS5 retrieval / embedding rank / entropy question-selection / generation, plus total per turn.
- Verdict line: online hosted ≈ X s/turn, offline all-Ollama ≈ Y s/turn — both conversational.

## Out of scope

- Editing the pptx directly, charts/images, changes to agent source, devpost edits (the DeepSeek-vs-OpenAI code discrepancy is flagged separately and is Winston's call).

## Success criteria

- Every number on the slide is reproducible from the harness JSON output.
- Slide copy fits one slide: three blocks, ≤ ~5 short lines each.
