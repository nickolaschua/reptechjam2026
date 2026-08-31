# ASTRA TechJam submission — BGE/Ollama release

This directory is the self-contained participant solution bundle. It exports the
required `starter.agent.Agent`, snapshots only the active runtime modules, and is
frozen to:

- chat/state model: `llama3.1:8b` served by Ollama
- query and catalogue embeddings: `BAAI/bge-base-en-v1.5`
- embedding dimension: 768, L2-normalized
- catalogue cache: precomputed, validated, never generated during evaluation
- hosted APIs: none

The organizer catalogue and the large BGE cache are external release artifacts
and are intentionally not committed here.

## 1. Requirements

Validated development environment:

- Python `3.13.14`
- Ollama `0.33.2`
- NumPy `2.4.3`
- Sentence Transformers `5.6.0`
- PyTorch `2.12.1`

Create an isolated environment from the repository root:

```powershell
python -m venv .venv-submission
.\.venv-submission\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r submission\requirements.txt
```

On Linux/macOS, activate with `source .venv-submission/bin/activate`.

## 2. Start the local model

```powershell
ollama pull llama3.1:8b
ollama serve
```

Defaults are documented in `.env.example`. This release forcibly sets
`TEST_MODE=false` and `ALLOW_CATALOG_EMBEDDING=false`; it cannot switch to the
OpenAI branch through an inherited shell variable.

## 3. Install the BGE catalogue cache

Required filename:

```text
submission/artifacts/catalog_cache_bge-base-en-v1.5.npz
```

The existing root-level `catalog_cache_Users_...npz` is an older two-array cache
and is incompatible with the release validator. Do not rename or ship it.

Build the production artifact with `colab/bge_pipeline.ipynb` at the candidate
commit, or publish the verified output as a repository release asset. Install a
published asset atomically with:

```powershell
python submission\scripts\install_artifact.py `
  --url "REPLACE_WITH_RELEASE_ASSET_URL" `
  --sha256 "REPLACE_WITH_64_CHARACTER_SHA256"
```

Before freezing the submission, replace both placeholders above with the real
release URL and checksum. Verify the installed cache against the exact organizer
catalogue:

```powershell
python submission\scripts\verify_artifact.py `
  --catalog techjam-conversational-search\data\catalog.jsonl
```

The verifier checks row count, exact ASIN order, catalogue fingerprint, product
text fingerprint/version, model and embedding-space identity, dimension,
normalization, cache schema, and SHA-256.

## 4. Run tests

Bundle contract tests:

```powershell
$env:PYTHONPATH = (Resolve-Path submission)
python -m pytest submission\tests -q
```

Verify the allowlisted runtime snapshot:

```powershell
python submission\scripts\verify_bundle.py
```

Canonical implementation tests:

```powershell
python -m pytest system\shopping_agent\tests -q
```

Real BGE/Ollama smoke test after installing the cache and starting Ollama:

```powershell
python submission\scripts\smoke_test.py `
  --catalog techjam-conversational-search\data\catalog.jsonl
```

## 5. Run the unmodified official evaluator

From the repository root:

```powershell
python submission\scripts\run_official_evaluator.py
```

The launcher does not edit or copy the evaluator. It puts this bundle first on
`sys.path`, then executes
`techjam-conversational-search/evaluator/local_evaluator.py` unchanged with the
organizer catalogue and public set. Output is written to
`submission/results/results.json`.

Alternative paths are explicit:

```powershell
python submission\scripts\run_official_evaluator.py `
  --kit C:\path\to\official-kit `
  --catalog C:\path\to\catalog.jsonl `
  --dataset C:\path\to\public_set.jsonl `
  --output C:\path\to\results.json
```

## 6. Required Agent contract

`starter/agent.py` exports `Agent`. The organizer calls:

```python
agent = Agent(catalog_path)
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k)
```

The adapter accepts the organizer's positional catalogue path, routes the cache
to `submission/artifacts` by default, disables online catalogue embedding, and
inherits the active response contract. Official sessions do not pass `user_id`,
so longitudinal memory is not committed across evaluator sessions.

## 7. Runtime environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama service endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Frozen local chat/state model |
| `OLLAMA_TIMEOUT_SECONDS` | `30` | Per-request transport timeout; the client retries once |
| `CONFIDENCE_SIMILARITY_THRESHOLD` | `0.40` | Inclusive current-query confidence gate |
| `TECHJAM_CATALOG_PATH` | `submission/data/catalog.jsonl` | Optional standalone catalogue location |
| `TECHJAM_BGE_CACHE_DIR` | `submission/artifacts` | Optional cache directory override |

`OPENAI_API_KEY` is not read by this release path. No credentials are required.

## 8. Failure behavior

- A missing, reordered, stale, malformed, or wrong-model cache fails startup.
- Startup never embeds the 50,000-row catalogue.
- Ollama transport and invalid-response failures are retried once, then raised as
  typed model errors.
- Failed turns restore the pre-turn state and do not advance the successful-turn
  lifecycle.
- There is no hosted-provider fallback; Ollama availability is required.

## 9. Before the Devpost freeze

1. Publish the BGE cache, insert its URL and SHA-256 above, and verify it from a
   clean download.
2. Rebuild `bundle_manifest.json` from the final clean source commit; the current
   manifest truthfully records that this first bundle was built from a dirty
   development worktree.
3. Run both test commands, bundle verification, and the real smoke test.
4. Run the unmodified public evaluator and complete `REPORT.md` with results,
   latency, hardware, and team contributions.
5. Review `git status`, ensure no `.env`, `.npz`, results, model ZIP, or secret is
   staged, then record the full submission commit SHA.
6. After the final package is released, check out that SHA and do not modify the
   Agent, prompts, indexes, cache, model configuration, or other solution files.
