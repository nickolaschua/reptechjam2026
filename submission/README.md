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
- There is no hosted-provider fallback; Ollama availability is required.

## 9. Before the Devpost freeze

1. Publish the BGE cache, insert its URL above, and verify it from a clean
   download.
2. Rebuild `bundle_manifest.json` from the final clean source commit; the current
   manifest truthfully records that the first bundle was built from a dirty
   development worktree.
3. Run the bundle tests, bundle verification, and the real smoke test.
4. Run the unmodified public evaluator and complete `REPORT.md` with results,
   latency, hardware, and team contributions.
5. Review `git status`, ensure no `.env`, `.npz`, results, model weights, or
   secret is staged, then record the full submission commit SHA.
6. After the final package is released, check out that SHA and do not modify the
   Agent, prompts, indexes, cache, model configuration, or other solution files.
