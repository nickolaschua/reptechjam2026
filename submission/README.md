# ASTRA TechJam submission — BGE/Ollama release

This directory is the self-contained participant solution bundle. It exports the
required `starter.agent.Agent`, snapshots only the active runtime modules, and is
frozen to:

- chat replies: `llama3.1:8b` served by Ollama
- state parsing and intent detection: DeepSeek API (`deepseek-chat`) when `DEEPSEEK_API_KEY` is set, with automatic per-call fallback to Ollama; fully local when unset
- query and catalogue embeddings: `BAAI/bge-base-en-v1.5`
- embedding dimension: 768, L2-normalized
- catalogue cache: precomputed, validated, never generated during evaluation
- hosted APIs: optional DeepSeek for state parsing only; none required

Every path used by this bundle resolves inside this directory. Nothing reads
from a parent directory or a sibling project.

## 0. Self-contained layout

Four directories are populated locally and are never committed (see
`.gitignore`). Everything else in this bundle is version-controlled source.

| Path | Contents | How to obtain |
| --- | --- | --- |
| `data/catalog.jsonl` | organizer catalogue, 50,000 rows | participant kit, §0.1 |
| `data/public_set.jsonl` | 200 labeled public sessions | participant kit, §0.1 |
| `kit/evaluator/` | unmodified organizer evaluator | participant kit, §0.1 |
| `artifacts/*.npz` | precomputed BGE catalogue cache | download or rebuild, §3 |
| `models/bge-base-en-v1.5/` | BGE weights for query encoding | automatic, §0.2 |

### 0.1 Organizer participant kit

Download the kit and catalogue from the organizer release, then unpack them into
this bundle:

```bash
# https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
unzip techjam-participant-kit.zip -d /tmp/kit
gunzip -c catalog.jsonl.gz > data/catalog.jsonl
cp /tmp/kit/data/public_set.jsonl data/
mkdir -p kit && cp -R /tmp/kit/evaluator kit/
```

Published organizer checksums:

| File | SHA-256 |
| --- | --- |
| `catalog.jsonl.gz` | `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8` |
| `techjam-participant-kit.zip` | `b3d7e283b835343b42c4919ea2ca90f2fb5a2aa2b10537f14dcf42f03e5b38ae` |

### 0.2 Model weights

Nothing to do: if `models/bge-base-en-v1.5/` is absent, Sentence Transformers
downloads `BAAI/bge-base-en-v1.5` on first use. To pin it offline instead, place
the model snapshot at that path; `model_id` and `embedding_space_id` are
identical either way, so the catalogue cache stays valid.

`models/` lets the agent encode queries with no Hugging Face download and no
network access. If it is absent the backend falls back to the hub id and will
download on first use; `model_id` and `embedding_space_id` are identical either
way, so the catalogue cache stays valid.

Run every command below from inside this `submission/` directory.

## 1. Requirements

Validated development environment:

- Python `3.13.14`
- Ollama `0.33.2`
- NumPy `2.4.3`
- Sentence Transformers `5.6.0`
- PyTorch `2.12.1`

```bash
python -m venv .venv
source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Start the local model

```bash
ollama pull llama3.1:8b
ollama serve
```

Defaults are documented in `.env.example`. This release forcibly sets
`TEST_MODE=false` and `ALLOW_CATALOG_EMBEDDING=false`; it cannot switch to the
OpenAI branch through an inherited shell variable.

## 3. Install the BGE catalogue cache

Required path:

```text
artifacts/catalog_cache_bge-base-en-v1.5.npz
```

Install a published asset atomically:

```bash
python scripts/install_artifact.py \
  --url "https://github.com/nickolaschua/reptechjam2026/releases/download/bge-cache-v1/catalog_cache_bge-base-en-v1.5.npz" \
  --sha256 "a05b1dcee3c40bb254ccf73ab437e8d08fc33d28d444a32c812158842526191f"
```

While the repository is private, that URL requires repository access; anyone
with access can instead run `gh release download bge-cache-v1 -D artifacts/`.
The URL serves anonymously once the repository is public. Then verify the
installed cache against the organizer catalogue:

```bash
python scripts/verify_artifact.py --catalog data/catalog.jsonl
```

If no release asset is available, rebuild the cache locally instead. This uses
the same production code path as the download, so the fingerprints match:

```bash
python scripts/build_artifact.py --catalog data/catalog.jsonl
```

Expect roughly 50 minutes for 50,000 rows on CPU. Then verify it as above.

Expected identities for the frozen release:

| Value | SHA-256 |
| --- | --- |
| organizer catalogue | `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67` |
| BGE catalogue cache | `a05b1dcee3c40bb254ccf73ab437e8d08fc33d28d444a32c812158842526191f` |

The verifier checks row count, exact ASIN order, catalogue fingerprint, product
text fingerprint/version, model and embedding-space identity, dimension,
normalization, cache schema, and SHA-256.

## 4. Run tests

Bundle contract tests:

```bash
PYTHONPATH=. python -m pytest tests -q       # PowerShell: $env:PYTHONPATH="."
```

Verify the allowlisted runtime snapshot:

```bash
python scripts/verify_bundle.py
```

Real BGE/Ollama smoke test, after installing the cache and starting Ollama:

```bash
python scripts/smoke_test.py --catalog data/catalog.jsonl
```

## 5. Run the unmodified official evaluator

```bash
python scripts/run_official_evaluator.py
```

The launcher does not edit or copy the evaluator. It puts this bundle first on
`sys.path`, then executes `kit/evaluator/local_evaluator.py` unchanged against
`data/catalog.jsonl` and `data/public_set.jsonl`. Output is written to
`results/results.json`.

Reference metrics for this release are committed at
`results/eval_v1_results.json`; a rerun of the command above reproduces the
reported evaluation on the public set.

Every path is overridable:

```bash
python scripts/run_official_evaluator.py \
  --kit /path/to/official-kit \
  --catalog /path/to/catalog.jsonl \
  --dataset /path/to/public_set.jsonl \
  --output /path/to/results.json
```

## 6. Required Agent contract

`starter/agent.py` exports `Agent`. The organizer calls:

```python
agent = Agent(catalog_path)
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k)
```

The adapter accepts the organizer's positional catalogue path, routes the cache
to `artifacts/` by default, disables online catalogue embedding, and inherits the
active response contract. Official sessions do not pass `user_id`, so
longitudinal memory is not committed across evaluator sessions.

## 7. Runtime environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | *(unset)* | Optional: routes state parsing to DeepSeek; unset = fully local |
| `DEEPSEEK_TIMEOUT_SECONDS` | `3` | DeepSeek call budget before falling back to Ollama |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama service endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Frozen local chat/state model |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Per-request transport timeout; the client retries once |
| `CONFIDENCE_SIMILARITY_THRESHOLD` | `0.40` | Inclusive current-query confidence gate |
| `TECHJAM_CATALOG_PATH` | `data/catalog.jsonl` | Optional catalogue location override |
| `TECHJAM_BGE_CACHE_DIR` | `artifacts/` | Optional cache directory override |

`OPENAI_API_KEY` is not read by this release path. No credentials are required; `DEEPSEEK_API_KEY` is optional and read only for state parsing.

## 8. Failure behavior

- A missing, reordered, stale, malformed, or wrong-model cache fails startup.
- Startup never embeds the 50,000-row catalogue.
- Ollama transport and invalid-response failures are retried once, then raised as
  typed model errors.
- DeepSeek failures (timeout, non-JSON, transport) fall back to Ollama on that
  call; a session never fails because DeepSeek is unavailable.
- Failed turns restore the pre-turn state and do not advance the successful-turn
  lifecycle.
- There is no hosted-provider chat fallback; Ollama availability is required.
- Failed turns roll back session state; conversation state is isolated by
  evaluator session ID.
- Provider selection is frozen to BGE/Ollama in the evaluator entry module, and
  the official evaluator is executed unchanged by the included launcher.

## 9. Before the Devpost freeze

1. Publish the BGE cache, insert its URL above, and verify it from a clean
   download.
2. Rebuild `bundle_manifest.json` from the final clean source commit; the current
   manifest truthfully records that the first bundle was built from a dirty
   development worktree.
3. Run the bundle tests, bundle verification, and the real smoke test.
4. Run the unmodified public evaluator and refresh the Public evaluation and
   Feasibility sections below from the frozen commit.
5. Review `git status`, ensure no `.env`, `.npz`, results, model weights, or
   secret is staged, then record the full submission commit SHA.
6. After the final package is released, check out that SHA and do not modify the
   Agent, prompts, indexes, cache, model configuration, or other solution files.

## 10. Limitations and future work

- Ranking, not retrieval, is the bottleneck: the hidden target lands in the
  top 10 almost every session but often mid-list. A learned reranker over the
  confidence-gate survivors is the clearest next win.
- `ask_attribute` is recovered by parsing the agent's own generated reply, so a
  paraphrased question degrades it to `other`. Emitting it directly from the
  entropy selector would make clarification deterministic and remove the main
  per-turn LLM latency.
- Reply generation is the latency floor: one local `llama3.1:8b` call per turn.
  State parsing is already deterministic on template inputs and sub-second via
  DeepSeek on free text.
- Long-term memory ships with a full lifecycle (load at session start, commit at
  end, relevance-gated recall) but official sessions never pass `user_id`, so
  the evaluation cannot exercise it; its one-centroid representation also cannot
  hold distinct user interests at once.
- The embedder is stock `BAAI/bge-base-en-v1.5`. Continual finetuning with a
  cosine-distance loss over the user-prompt database is designed but not part
  of this release.
- `respond()` does not yet report per-turn token usage, so the evaluator's token
  accounting reads zero.
- Heuristics are tuned to the official evaluator's fixed English templates;
  robustness to free-form paraphrase rests on the LLM fallback path.
- Keyword scoring contains handcrafted weights with known substring-matching
  edge cases, and the confidence gate can improve rank quality while delaying
  the first hit.

## 11. Team contributions

| Team member | Owned |
| --- | --- |
| Nickolas | Long-term memory cache: vector store, session lifecycle, relevance gating |
| Yang Xu | Entropy-based querying: attribute selection, clarification strategy |
| Winston | Intent detection and state routing, DeepSeek/Ollama parsing plumbing |
| Judith | Short-term session state: constraints, provenance, override and boundary handling |
| Harshith | Retrieval and filtering: FTS5 keyword layer, categorical masks, vector similarity, confidence gate |

## 12. Method

ASTRA is a multi-turn conversational product-retrieval agent for the frozen
50,000-product Clothing, Shoes and Jewelry catalogue. It maintains structured
session state, re-evaluates Buying versus Browsing intent each turn, routes
through an FTS5 keyword path when lexical evidence is sufficient, and otherwise
uses a 150-row dense fallback. Hard eligibility constraints are applied before
ranking. Previously returned products are removed, a fixed top-10 pool is gated
by current-query BGE cosine similarity, and surviving products receive a
rank-preserving diversity pass. When confidence is insufficient, the agent asks
an entropy-selected structured clarification question.

State parsing and intent detection attempt the DeepSeek API first when
`DEEPSEEK_API_KEY` is set and fall back to local Ollama per call; chat replies
always stay on the local model.

The demonstration runtime contains a gated single-centroid longitudinal memory,
but the official evaluator supplies independent anonymous sessions without the
identity and sequence information needed to commit or read cross-session
memory, so evaluator performance does not rely on hidden identity inference.

```text
official Agent contract
  -> structured state update and live intent (deterministic -> DeepSeek -> Ollama)
  -> FTS5 AND / weighted-OR candidate routing
  -> session-local hard eligibility masks
  -> BGE current-query scoring and gated memory equations
  -> handcrafted keyword ranking OR s3 dense fallback
  -> seen removal -> fixed top-10 confidence gate -> diversity
  -> recommendations and/or entropy-selected clarification
```

The catalogue/query vectors share one validated embedding-space identifier.
Startup verifies the exact catalogue hash, ASIN row order, product-text
fingerprint, dimension, normalization, and cache metadata. It cannot silently
regenerate or accept a mismatched cache.

## 13. Public evaluation

Reference results (`results/eval_v1_results.json`, 200 public sessions through
the unmodified organizer evaluator):

| Metric | Value |
| --- | ---: |
| Hit Rate@10 | 1.000 |
| MRR | 0.6820 |
| MTTC | 1.785 |
| Efficiency | 0.9215 |
| TechnicalScore | 0.8889 |

| Scenario | n | Hit@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 1.000 | 0.7006 | 1.34 |
| Browsing | 80 | 1.000 | 0.6474 | 1.43 |
| Intent override | 30 | 1.000 | 0.7845 | 4.00 |
| Boundary | 10 | 1.000 | 0.5028 | 1.60 |

Controlled evidence: the confidence-gate ablation preserved Hit Rate@10 at
`0.98`, improved MRR from `0.5566` to `0.6116`, and improved the composite score
by `0.010`, while worsening MTTC by `0.325` turns; the tradeoff is presented
alongside the gain. Long-term-memory evidence is not claimed as a performance
improvement: archived evaluation found relevant-set MRR `0.021452` without
memory versus `0.019327` with memory at the frozen configuration (relevance
gate ROC AUC `0.97`), and the adaptive update has not passed the required
dormant-interest evaluation.

## 14. Feasibility disclosure

- Hardware and operating system: Apple Silicon Mac, macOS 15 (Darwin 24.6)
- Ollama model disk size: 4.9 GB (`llama3.1:8b`); BGE cache: 136 MB
- Catalogue SHA-256: `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`
- BGE cache SHA-256: `a05b1dcee3c40bb254ccf73ab437e8d08fc33d28d444a32c812158842526191f`
  (release asset `bge-cache-v1`)
- API/model cost: `0` in API fees on the local path; DeepSeek parsing, when
  enabled, bills per state-update call
- Agent initialization time, response-latency percentiles, peak memory, total
  run time, and the frozen submission commit SHA: **TODO — fill from the final
  frozen-commit run**

## 15. Demonstration

One complete multi-turn session showing customer messages, structured
`ask_attribute` values, ordered recommendations, and the target hit, without
exposing the hidden target or intent card to the Agent.

Demo recording/link: **TODO**
