# Evaluation and Ablation Plan

We have enough infrastructure to build a convincing evaluation story, but we cannot yet make clean claims that every feature improves the live system.

The strongest evidence today is for the confidence gate. Long-term memory currently has trustworthy but negative evidence. Entropy is implemented and tested, but it is not independently ablatable or evaluated against a reasonable control.

## What already exists

| Capability | Current status | What it means |
|---|---|---|
| Official-style evaluator | Ready | The 200-session public evaluator reports Hit@10, MRR, MTTC, efficiency, scenario slices, and token usage. |
| Weak baseline | Ready | The provided BM25 baseline scores Hit@10 `0.125`, MRR `0.0680`, and MTTC `9.81`. |
| Current-agent result | Available | A stored 200-session result reports Hit@10 `0.955`, MRR `0.6518`, and MTTC `2.61`, but it is not yet a controlled feature-by-feature comparison. |
| Confidence-gate ablation | Ready and already run | On 200 sessions, the gate preserved Hit@10 at `0.98`, raised MRR from `0.5566` to `0.6116`, and raised composite score by `0.010`; MTTC worsened by `0.325` turns. |
| Four-user longitudinal fixture | Exists | `users_40.json` contains four users with ten session definitions each. |
| Trustworthy memory fixture | Exists | `users_40_v2.json` contains 40 isolated timelines, each with two setup sessions and one probe: 120 records total. |
| Memory evaluation artifacts | Complete but archived | The artifacts include manifests, forensic records, hashes, paired reconstruction, slice metrics, and deterministic verification. |
| Memory improvement | Not demonstrated | Relevant-set MRR fell from `0.02145` without memory to `0.01933` with memory. The mechanism was classified `HARMFUL`. |
| Entropy feature | Implemented | Shannon entropy and gain-ratio selection are active and deterministic. |
| Entropy ablation | Missing | There is no runtime switch or evaluator comparison against a fixed-priority or random-question control. |
| General feature flags | Missing | Most features are hardwired rather than selected through a shared experiment configuration. |

Relevant sources include:

- `system/MEMORY_EVALUATION_STATUS.md`
- `system/shopping_agent/evaluate_confidence_gate.py`
- `system/shopping_agent/clarification.py`
- `techjam-conversational-search/evaluator/local_evaluator.py`

One important correction: the four-user file is a set of session specifications, not a frozen collection of 40 complete LLM conversation transcripts. The v2 results contain 80 deterministic setup-update records and 40 probe records. We therefore have evaluation fixtures and forensic outputs, but not a general-purpose corpus of reusable LLM dialogue logs.

## Experiments to run

We should not make one giant cumulative ladder across incompatible datasets. Use three paired evaluation suites, each designed around the feature it can genuinely exercise.

## 1. Single-session competition suite

Use the released 200 sessions and report the judges' metrics exactly:

- Hit Rate@10
- MRR
- MTTC
- Efficiency and composite score
- Token usage and latency
- Results by buying, browsing, override, and boundary scenarios

Run these variants against identical sessions:

| Variant | Purpose |
|---|---|
| Official weak starter | External, recognizable baseline |
| Current retrieval without confidence gate | Internal control |
| Current retrieval with confidence gate | Isolated confidence-gate effect |
| Fixed-priority clarification control | Reasonable entropy control |
| Entropy clarification | Isolated entropy effect |
| Full live system | Final end-to-end result |

For entropy, "off" should not mean asking no question. The fair control is the existing intent-specific fixed priority order. This isolates whether information gain chooses better questions than a reasonable heuristic.

Entropy-specific intermediate metrics should include:

- Target hit on the next turn after clarification
- Reduction in eligible candidate count
- Attribute-answer coverage
- Number of clarification turns
- Repeated or irrelevant-question rate
- Ultimate Hit@10, MRR, and MTTC

The existing confidence-gate experiment is already a good judge-facing result:

- MRR: `+0.055`
- Composite score: `+0.010`
- Hit@10: unchanged
- MTTC: `0.325` turns worse

This is credible evidence because it includes the tradeoff instead of hiding it.

## 2. Longitudinal-memory suite

Use the 40-probe v2 fixture, not the ordinary public evaluator, because independent public sessions cannot exercise long-term memory.

Required paired variants:

| Variant | Configuration |
|---|---|
| M0 | No prior memory |
| M1 | History stored, memory read disabled |
| M2 | Memory read with relevance gate disabled |
| M3 | Relevance gate with fixed update |
| M4 | Relevance gate with adaptive update, matching the current live mechanism |
| M5, optional | Fact-level or selected memory if the representation is improved |

Report:

- Relevant-set and exact-target MRR
- Rank delta
- Help, harm, and unchanged rates
- Gate activation rate
- Irrelevant-history harm
- Current-override harm
- Browsing-personalization lift
- Relevance-gate ROC AUC
- Dormant-interest retention for the adaptive updater

The honest current conclusion is that the relevance detector is promising, with AUC `0.97`, but the downstream blending mechanism does not yet produce reliable overall improvement. The adaptive updater has not been evaluated on the missing dormant-interest slice, so it cannot replace that conclusion yet.

## 3. State-understanding robustness suite

The public evaluator uses evaluator-shaped messages that follow the deterministic parsing path. It does not adequately exercise the LLM state editor used for natural free-text messages.

Create paraphrased and messy versions of a held-out subset, including:

- Corrections such as "actually, make that..."
- Negations such as "anything except wool"
- Implicit budget language
- Category changes
- Contradictory preferences
- No-preference answers
- Casual and incomplete phrasing

Compare deterministic regex-only parsing against the LLM state editor. Measure:

- State-slot precision and recall
- Contradiction handling
- Rollback and error rate
- Downstream Hit@10, MRR, and MTTC

## Can we run this immediately?

Partially.

- The 200-session evaluator is available.
- The confidence-gate runner is operational and already has complete results.
- Ollama is reachable locally.
- The default live BGE deployment cache is missing. Only the OpenAI embedding cache is present, and there is currently no OpenAI credential configured. A fresh live end-to-end run is therefore not immediately reproducible in the current environment.
- The archived v2 longitudinal runner is broken after relocation: importing it fails because it looks for `archive.research_evaluation.memory.scripts.agent`.
- Long-term memory can be controlled programmatically through `VectorMemoryConfig`, but the active system has no clean CLI-level `memory_enabled` switch.
- The confidence threshold is switchable.
- Entropy, retrieval routing, and several other features have no clean experiment switches.
- There is no unified runner guaranteeing that every variant uses the same commit, fixture, model, seed, cache, and evaluator.

## Experiment infrastructure to add

The missing piece is a small experiment layer, not another evaluator:

```text
ExperimentConfig
|-- memory_mode: off | fixed | adaptive
|-- relevance_gate_enabled
|-- confidence_gate_enabled
|-- clarification_policy: fixed | entropy
|-- retrieval_mode: lexical | vector | hybrid
|-- state_editor_mode: regex | llm
`-- seed / model / cache identifiers
```

One runner should then:

1. Load the dataset once.
2. Replay every variant on the same sessions.
3. Reset memory and agent state between variants.
4. Save turn-level traces and aggregate metrics.
5. Produce paired deltas, bootstrap confidence intervals, and win/harm counts.
6. Record the commit, configuration, model, dataset hash, and artifact hashes.

For the four-user fixture, do not treat 40 sessions as 40 independent observations; they are clustered within only four users. Either report user-clustered intervals or use the v2 fixture's 40 independent timelines. This detail will materially improve judge trust.

## Recommended order of work

1. Add the unified feature configuration and entropy control.
2. Restore or port the longitudinal evaluator to the active agent.
3. Provision and verify the BGE cache.
4. Re-run all variants from the same commit.
5. Only put features on the "measurably improves performance" slide if their paired results support the claim.

Today, we can confidently demonstrate the confidence gate. We can demonstrate that memory is rigorously evaluated, but not yet that it improves performance. Entropy is the most straightforward next feature for which to generate clean positive-or-negative ablation evidence.
