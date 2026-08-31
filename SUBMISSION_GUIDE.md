# TechJam submission readiness guide

Updated: 2026-09-01

Official participant-kit version reviewed: TechJam2026/techjam-conversational-search
commit `9c9e7c9ff6705142d6ab386dc1c432fc529df893` on `main`.

## Current verdict

The active `system.shopping_agent` demo is implemented and its active test suite
passes, but this repository is **not yet submission-ready**.

Release blockers:

1. The official evaluator imports `starter.agent.Agent`; the newest agent is
   `system.shopping_agent.agent.Agent`. The current
   `techjam-conversational-search/starter/agent.py` is a different, older agent.
2. There is no frozen submission directory or root dependency manifest for the
   active agent.
3. The chosen production provider is not frozen. Local mode needs Ollama, BGE
   model weights, and the external BGE catalogue cache. OpenAI mode needs
   `OPENAI_API_KEY` and the matching OpenAI catalogue cache.
4. The required large cache delivery/download procedure is not yet part of a
   submission README.
5. The repository has uncommitted code changes, generated Graphify changes, an
   untracked notebook, and an untracked 379 MB model ZIP. Do not submit or tag
   this working tree as-is.
6. A final official-evaluator run of the packaged active agent, with retained
   `results.json`, has not yet been recorded.
7. The short report still needs frozen latency, token, cost, limitations, and
   team-contribution disclosures from the exact submitted commit.

Validation completed during this audit:

- `python -m pytest system/shopping_agent/tests -q`
- latest result: `152 passed, 9 subtests passed`
- active `reset(session_id, user_profile)` and
  `respond(session_id, user_message, turn, top_k)` calls are interface-compatible;
  their additional arguments are keyword-only and optional
- secrets and `.npz` cache files are ignored by Git

## Official rules that control the release

- Submit one Python entry file exporting `Agent`, its local helper modules, setup
  instructions, a short method/model/limitations report, and latency/token/cost
  disclosure.
- The response must contain a string `message`, an allowed `ask_attribute` or
  `null`, and ordered recommendations. Only the first 10 valid unique
  `parent_asin` values are scored.
- Final evaluation uses 800 sessions released after the Devpost deadline. Run
  the unmodified official evaluator in your own environment using the Git commit
  submitted before the deadline.
- The submitted commit is the code freeze. After the final package is released,
  do not change the Agent, prompts, indexes, model configuration, or other
  solution components.
- Network access and external APIs are allowed. An offline fallback is optional.
  Disclose credentials by environment-variable name only, network dependencies,
  service limits, fallback behavior, latency, token use, and estimated cost.
- There is no standardized organizer CPU, RAM, GPU, startup-time, package-size,
  or per-response limit. Large artifacts should still use documented,
  reproducible download instructions instead of normal Git commits.
- Retain the final `results.json`, submitted commit hash, environment and hardware
  details, execution command, and relevant logs.
- A UI is optional. The runnable Python Agent and one complete multi-turn
  demonstration are required.

The synchronized local copies are in
`techjam-conversational-search/docs/submission_rules.md`,
`competition_specification.md`, and `final_evaluation_faq.md`.
The separate `techjam-conversational-search-participant-kit/` directory is an
untouched older snapshot; use it for evaluator mechanics, not current policy.

## Prepare the submission in this order

### 1. Freeze one runtime configuration

Choose exactly one primary final-evaluation path before packaging:

| Choice | Required runtime | Required catalogue artifact | Main disclosure |
| --- | --- | --- | --- |
| Local | Python, NumPy, Sentence Transformers/BGE, Ollama | `catalog_cache_bge-base-en-v1.5.npz` | model downloads, hardware, local latency |
| OpenAI | Python, NumPy, network access, OpenAI API | matching `catalog_cache_openai-...-d1536.npz` | `OPENAI_API_KEY`, models, tokens, cost, API failure behavior |

Do not let `TEST_MODE` silently choose the release behavior. State its required
value in the submission README and verify the cache model, dimensions, row order,
fingerprints, and embedding-space ID on a clean machine.

### 2. Build an isolated bundle

Recommended shape:

```text
submission/
  starter/
    __init__.py
    agent.py                 # exports the active Agent
  shopping_agent/           # only active runtime modules
  requirements.txt          # pinned or bounded versions
  README.md                  # clean-machine setup and evaluator command
  REPORT.md                  # method and required disclosures
  scripts/
    download_artifacts.*     # optional, reproducible large-asset install
    verify_artifacts.*
```

Keep the official evaluator and evaluation data outside the solution code. The
entry module must work with the official import statement without changing the
evaluator. Avoid depending on the repository's current sibling-directory layout.

Exclude archives, Graphify output, `.env`, `.demo_state`, caches not used by the
chosen provider, result files, notebooks, model-training ZIPs, and organizer data
that is not required at runtime.

### 3. Write reproducible setup instructions

The submission README must include:

- supported Python version and operating system used for the reported run
- exact dependency-install command
- artifact download location, checksum, destination, and verifier command
- required environment-variable names and safe example values
- model/provider startup steps
- one command that runs the unmodified official evaluator
- expected startup behavior and a small smoke test
- network and fallback behavior
- how to capture `results.json` and logs

Never commit a real API key. Before freezing, inspect staged files with:

```powershell
git status --short
git diff --cached --stat
git grep -n -I -E "(sk-|OPENAI_API_KEY=.+|api[_-]?key.+[=:].+)" -- .
```

Review every match manually; automated secret patterns are not exhaustive.

### 4. Validate the package, not the development tree

Use a fresh clone or clean virtual environment and run, in order:

```powershell
python -m pip install -r submission/requirements.txt
python -m pytest system/shopping_agent/tests -q
# install/verify the chosen external catalogue cache
# run the unmodified official evaluator against submission/starter/agent.py
```

Add a contract smoke test that calls `reset` once, calls `respond` for turns
1-10, validates the response schema, checks unique catalog-valid ASINs, and
confirms that two session IDs cannot leak conversational state.

For the 200 public sessions, retain:

- Hit Rate@10, MRR, MTTC, Efficiency, and TechnicalScore
- scenario slices: Buying, Browsing, Intent Override, Boundary
- initialization and per-turn latency distribution
- prompt/completion token totals and per-turn distribution
- estimated API cost, or a clear zero-API-cost local-model statement
- provider/model/cache identifiers and artifact checksums
- the exact Git commit and command

Do not present the old `0.955` result as the final active-agent score until the
packaged `system.shopping_agent` entry point reproduces it. The confidence-gate
ablation is usable evidence with its tradeoff disclosed. Long-term memory should
not be claimed as a measured performance improvement: the trustworthy archived
evaluation found negative downstream impact despite a strong relevance signal.

### 5. Prepare the report and demo

`REPORT.md` should contain:

1. Problem and approach in one paragraph.
2. Architecture: state editor, intent-aware FTS route, hard eligibility,
   embedding fallback, confidence gate, clarification, and longitudinal memory.
3. Exact models, embedding dimensions, provider, external services, and assets.
4. Public-set metrics and controlled ablations from the frozen commit.
5. Latency, token usage, estimated cost, hardware, and Python/dependency versions.
6. Reliability: retry, rollback, cache validation, session isolation, and failure
   behavior.
7. Honest limitations: external cache provisioning, provider dependence, single
   memory centroid, negative memory evidence, parser/model failure modes, and
   known open retrieval defects.
8. Team contributions with a person-to-work mapping. Do not leave this generic.

Record one complete multi-turn demonstration. Show the user message,
`ask_attribute`, recommendations, and the target hit without exposing hidden
labels to the Agent.

### 6. Freeze and preserve evidence

Immediately before the Devpost deadline:

1. Make the working tree clean and review the final diff.
2. Run tests and the public evaluator from the candidate commit.
3. Create the submission commit and record its full SHA.
4. Submit that exact commit on Devpost.
5. Archive checksums, public `results.json`, environment capture, report, demo,
   and commands without committing secrets.

After the 800-session package is released, check out the submitted SHA, install
from its instructions, run the unmodified evaluator, and do not patch the
solution. Preserve final `results.json`, logs, commit SHA, hardware/environment
details, artifact hashes, and the exact execution command.

## Final go/no-go checklist

- [ ] Official evaluator imports the packaged active Agent without path hacks.
- [ ] Clean-machine setup succeeds from the README.
- [ ] Dependencies and Python version are declared.
- [ ] Required large artifacts have reproducible downloads and verified hashes.
- [ ] Required environment-variable names are documented; no secret values are tracked.
- [ ] The chosen provider/model/cache configuration is frozen.
- [ ] Active tests and contract smoke tests pass.
- [ ] The unmodified public evaluator completes and `results.json` is retained.
- [ ] Metrics, latency, tokens, cost, hardware, and limitations are reported.
- [ ] One complete multi-turn demo is recorded.
- [ ] Team contributions are named.
- [ ] The Git working tree is clean and the submitted full commit SHA is recorded.
- [ ] The post-deadline no-modification rule is understood by every teammate.
