# Phase 6.1 — B0 longitudinal baseline validation

## A. Current run configuration

- Run: `B0_LONGITUDINAL_40`
- Shopper: `openai` / `gpt-4.1-nano`
- Agent: M0_OPENAI / B0, Fast Memory enabled, longitudinal memory shadow-only
- Embeddings: `openai-text-embedding-3-large` / `text-embedding-3-large`
- Sessions: 40
- Independent repeats: 1
- Benchmark wall time: 972.414025 seconds

## B. B0_LONGITUDINAL_40 results

| HR@10 | MRR | Mean turns | No hits | Agent errors | Leak flagged |
| --- | --- | --- | --- | --- | --- |
| 0.9 | 0.620377 | 3.825 | 4 | 0 | 0 |

### By user

| User | Sessions | HR@10 | MRR | Mean turns | No hits |
| --- | --- | --- | --- | --- | --- |
| U1 | 10 | 0.9 | 0.683333 | 3.1 | 1 |
| U2 | 10 | 0.7 | 0.494444 | 5.0 | 3 |
| U3 | 10 | 1.0 | 0.572778 | 3.4 | 0 |
| U4 | 10 | 1.0 | 0.730952 | 3.8 | 0 |

### By sequence index

| Sequence | Sessions | HR@10 | MRR | Mean turns | No hits |
| --- | --- | --- | --- | --- | --- |
| S1 | 4 | 1.0 | 1.0 | 3.75 | 0 |
| S2 | 4 | 0.75 | 0.348214 | 4.75 | 1 |
| S3 | 4 | 1.0 | 0.381944 | 2.75 | 0 |
| S4 | 4 | 0.75 | 0.361111 | 5.5 | 1 |
| S5 | 4 | 1.0 | 0.508333 | 2.75 | 0 |
| S6 | 4 | 1.0 | 1.0 | 2.25 | 0 |
| S7 | 4 | 0.75 | 0.541667 | 5.75 | 1 |
| S8 | 4 | 1.0 | 0.8125 | 4.0 | 0 |
| S9 | 4 | 0.75 | 0.5 | 5.25 | 1 |
| S10 | 4 | 1.0 | 0.75 | 1.5 | 0 |

### Probe sessions

| Probe | Hit@10 | Best rank | RR | First hit turn |
| --- | --- | --- | --- | --- |
| U1 S10 | True | 1 | 1.0 | 1 |
| U2 S10 | True | 2 | 0.5 | 1 |
| U3 S10 | True | 2 | 0.5 | 1 |
| U4 S10 | True | 1 | 1.0 | 3 |

Full probe transcripts, final Fast Memory, and committed MemoryItems are in `probe_sessions.json`.

## C. Disclosure diagnostics

| Scheduled | Shopper expressed | Fast Memory captured | MemoryItem committed |
| --- | --- | --- | --- |
| 27 | 27 (100.0%) | 27 (100.0%) | 24 (88.9%) |

Failed-chain sessions: u1_stable_s3, u1_stable_s6, u4_negative_s3.

## D. Leakage diagnostics

- Exact target ASIN leakage: 0 sessions (none)
- Exact normalized target-title leakage: 0 sessions (none)
- Sessions were not silently excluded.

## E. Shadow-memory parity

| Paired turns | Identical rankings | Different rankings | Parity | Target-rank differences | Fast Memory differences | Route differences |
| --- | --- | --- | --- | --- | --- | --- |
| 149 | 149 | 0 | 100.0% | 0 | 0 | 0 |

Identical shopper inputs: `True`. Historical memory applied: `False`.

Agent LLM call-tape control: `True`; recorded/replayed calls: 298/298; prompt mismatches: 0. This evaluation-only control holds stochastic state/prose generation fixed while varying only the longitudinal store. An initial uncontrolled paired run is retained in `shadow_parity_uncontrolled.json`; its divergence began in U1 S1 with both stores empty, demonstrating GPT-4o-mini sampling variance rather than a history read.

## F. Old 200 vs new 40 calibration

| Metric | M0 public 200 | M0 buying 80 | B0 longitudinal 40 |
| --- | --- | --- | --- |
| HR@10 | 0.92 | 0.8875 | 0.9 |
| MRR | 0.611359 | 0.568834 | 0.620377 |
| Mean turns | 3.335 | 3.6375 | 3.825 |
| No-hit rate | 8.0% | 11.3% | 10.0% |

The frozen run records `llama3.1` but not which credential-fallthrough provider actually answered. Raw comparison is therefore confounded by dataset and shopper-provider differences.

## G. Interpretation

The curated 40-session result characterizes benchmark difficulty; it is not a memory-effect estimate. If strict parity is 100%, Phase-6 history plumbing is behaviorally inert for B0. Later-sequence scores must not be interpreted as memory improvement while history remains shadow-only. Future memory effects require B0/B1/B2/B3 comparison on this identical fixture.

## H. Tests

- Shopping-agent: 97 passed.
- QLMP: 79 passed.
- Unit tests made no OpenAI calls and did not download BGE.

## I. Result files

The result directory contains all required JSON metrics/diagnostics, `run_summary.md`, `full_run.json`, controlled and uncontrolled parity artifacts, and both execution logs. The frozen M0 artifact hashes are recorded in `config.json` and remained unchanged.

## J. Scope verification

No `experiment_1`, official `techjam-conversational-search`, or QLMP-math changes; no Graphify run, dependency change, or assistant-created commit.
