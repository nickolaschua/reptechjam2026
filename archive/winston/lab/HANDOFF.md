# Handoff — Winston's NLP / benchmark lane

Written 2026-08-30 for a fresh chat to resume with zero prior context. Everything
below is either in the repo or was measured this session; nothing is aspirational
unless marked.

## Who / what / constraints

- Repo: `/Users/winstonyang/Desktop/Coding/Hackathons/Techjam 2026/reptechjam2026`
  (GitHub `nickolaschua/reptechjam2026`). TikTok TechJam 2026, conversational
  shopping-search agent over a frozen 50k-product Amazon clothing catalog.
  Problem statement: `problem_statement.md`. Team brief: `nickolas/CURRENT_BEST_ARCHITECTURE.md`.
- User is **Winston** (NLP lane). Teammates: Nickolas (retrieval/routing/memory),
  Yang Xu (shop + shopper agents, `experiment_1/`), Harshith (EDA), Judith (case).
- **Hard constraints from Winston:** local LLM only (Ollama), no external API
  calls, no external vector DB (in-memory numpy is fine — that is what the brief
  asks for). Machine: 16 GB MacBook. Note: teammates' code still contains
  OpenAI/DeepSeek paths and the slide says "LLM Call 1: DeepSeek API" — unresolved
  contradiction, Winston should confirm which is current.
- Mode: `/ponytail full` (lazy/minimal), superpowers workflow (brainstorm → spec →
  plan → subagent-driven execution with spec review then quality review per task).

## Git state

- Branch **`bench`**, tracking `origin/bench`. All Winston work is committed here.
- `origin/main` (9 new commits from Nickolas, `3caa910..dc814f2`) was **merged into
  `bench`** cleanly at `74779f5`. No conflicts — main touched nothing under `winston/`
  and nothing the benchmark imports (`nickolas/experiments/experiment_11_candidate_agent.py`,
  `evaluator/`, `data/` all unchanged).
- **Local `bench` is 10 commits ahead of `origin/bench`** (the merge + main's 9).
  The merge has NOT been pushed — Winston asked commit→push→pull, not a second push.
  `git push` when ready.
- Commit chain on bench: `618e561` prompts.py → `81009ab` tokenizer/plural/docstring
  fix → `7afae9c` prompt tests → `be9f22e` covariates.py → `501a889` S925/UV400 fix →
  `e103e78` sample.py + products.jsonl → `f2d5595` rest of winston/ (32 files) →
  `74779f5` merge main.
- `.cache/` and `__pycache__/` are gitignored at repo root; `winston/experiments/.cache/`
  (210 MB index pickle) and `winston/lab/bench/.cache/` (18 MB covariates) are
  local-only and rebuild automatically.

## What exists (all under `winston/`)

| path | what | status / key number |
|---|---|---|
| `nlp_parse.py` | messy-utterance → typed JSON via Ollama `qwen2.5:7b-instruct` with a **constrained JSON schema** (`format=SCHEMA`); deterministic post-validation | slot F1 **0.441** on the 30-probe set (started 0.084 — most of the gain was fixing the *scorer*, not the model) |
| `probe_set.md` / `probe_gold.json` | 30 hand-written messy cases, 7 strata A–G, with expected parses | the only messy-input instrument until the benchmark exists; n=30 → ±0.11 F1 noise floor |
| `preds-qwen2.5-7b-instruct.json` | cached model outputs for the 30 probes | re-scoring is offline; `main()` never clobbers a fuller cache |
| `experiments/exp06_vocabulary_resolution.py` | lexical bucket resolution on paraphrases | caps ~55–65 % recovery; needs 600+ pool |
| `experiments/exp07_turn1_category_channel.py` | the simulator's turn-1 template hands over the target's bucket verbatim | **200/200** exact lookup, median pool 182 |
| `lab/pipeline.py` | stage-2 category resolver: score all 1,115 buckets from `category_phrase` + **soft slots** | true-bucket median rank **20 → 8.5** (top3 0.333 → 0.433); hard slots make it worse |
| `lab/ARCHITECTURE.md` | 5-stage design: parse → resolve → retrieve (lexical AND dense, neither gates) → agree → rank+ask | stages 1–2 built; 3–5 need nickolas' harness (torch, ~2 GB, NOT installed) |
| `PARSE_AND_ROUTING.md` | answers to Winston's 5 design questions with citations | see "Key findings" |
| `lab/specs/2026-08-29-messy-benchmark-design.md` | **the benchmark spec** (approved, revised for the team taxonomy) | source of truth for what to build |
| `lab/plans/2026-08-29-messy-benchmark.md` | **the implementation plan**, 10 tasks, complete code per step | Tasks 1–4 done; plan text was patched after each review so it matches the code |
| `lab/bench/` | the benchmark code | see next section |

## Benchmark build status (`winston/lab/bench/`)

Purpose: ~1,700 messy shopper utterances with **free ground truth** (the LLM shopper
is given the target product; the asin is the label), stratified by product
covariates × the team's search-type taxonomy, scored single-turn against three
systems plus first-question quality. Evaluate on THIS, not the public set — the
public set is saturated (HitRate 1.000) and exp09 proved typed state moves nothing there.

| task | file | status |
|---|---|---|
| 1 | `prompts.py` — 8 styles, 4 modifiers, stopwords, `content_words`, forbidden list, `build_system_prompt` | ✅ done, reviewed, fixed, re-approved |
| 2 | prompt-contract tests | ✅ |
| 3 | `covariates.py` — 14 per-product scores for all 50k, cached | ✅ done, reviewed, fixed (S925/UV400), re-approved; model-code precision 20/20 |
| 4 | `sample.py` → `products.jsonl` — 269 products, 120 base + 30 × 5 pools | ✅ implemented, **spec review passed, quality review NOT run** (machine slept mid-review, then user redirected to git) |
| 5 | `generate.py` — case plan, overlap metric, resumable Ollama loop, `--dry-run` | ⬜ not started; full code is in the plan |
| 6 | live check: 2 real generations | ⬜ |
| 7 | `score.py` — template / fair-lexical / resolver ranks + `ask_attribute` + `question_hit` | ⬜; full code in plan |
| 8 | `report.py` — bootstrap-CI tables by style × intent × generator × covariate quartile × modifier | ⬜; full code in plan |
| 9 | full generation run (~4–6 h background, resumable) | ⬜ |
| 10 | full scoring + first report | ⬜ |

Tests: `cd winston/lab/bench && python3 -m unittest test_bench -v` → **16 pass** on the
merged tree. Plan expects 24 by the end.

Environment ready: Ollama models `llama3.1:8b`, `gemma2:9b`, `mistral:7b` (generators,
never the parser's qwen), `qwen2.5:7b-instruct` (parser). `wordfreq` installed
(`pip3 install --break-system-packages`). Ollama **dies when the laptop sleeps** —
`ollama serve &` to restart; every long loop is resumable.

**To resume execution:** re-run the Task 4 Stage-B quality review (prompt is in the
plan's "Task 4" + spec §3.1/§7), then Task 5 onward. Dispatch pattern used: haiku for
mechanical tasks with complete code, sonnet for multi-file integration, opus for
reviews; spec review before quality review; implementer fixes then re-review.

## Key findings (measured, cite these)

- Best team agent (exp11 clean FTS5): **0.8995 TechnicalScore, HitRate 1.000**, no LLM, no
  embeddings. **No headroom on clean input.** Same agent under paraphrase: 0.57–0.60.
  Yang Xu's messy `eval_v2`: **0.489 / HitRate 0.575**. That gap is the only headroom.
- Dense retrieval as a *route* (exp08): 0 rescues, **21 regressions**. Never switch to dense; only fuse.
  Under paraphrase lexical degrades 0.25–0.28, dense 0.06–0.12 → dense's value is stability, a fusion weight.
- Nickolas' cascade ("keyword first, if it works return, else dense"): the "works" signal is
  **candidate count** — exp07 data: 1–9 all-phrase matches → 97.6 % right, 100+ → 8.4 %.
  Blind spot on clean input 2.2 %; unmeasured per-turn under paraphrase. **Winston was assigned
  this gate.** It does NOT yet exist in code — Nickolas' "routing change" that landed in main
  is state routing + memory hooks; retrieval is explicitly `_legacy_recommendations` (unchanged).
- Catalog taxonomy: 1,628 leaf paths / 1,115 evaluator buckets (last-2 segments) / 863 labels.
  **191 of 203 L2 nodes are promo/test slots** (`Westlake` 1,136 products, `Boot Shop`…) — 5.9 %
  of products have no real category. Category lookup should be layer 1, not 3.
- Failure modes nobody tests: silent-on-material (**28 % of public targets**), near-duplicate
  sibling (**14 %**; exp11's Jaccard dedup could suppress the target), negation (exp11 has none),
  unpopular targets (brief's own #1 threat), vague budget, compound requests. The benchmark
  over-samples the first two + low-popularity + model-code + compat-eligible.
- Parser lessons: constrained decoding constrains **structure not length** (case 09 looped
  "not too preppy" ×90 until `maxLength` was added); description-only fixes on a 7B do
  nothing (department 10/12 → 10/12); validate hard claims against the catalog instead
  (`hard_claim_holds`: brand ∈ 19,749 stores, material ∈ substance list, size grammar) —
  13 → 0 bogus hard filters.
- 67 of 147 probe gold slots never appear in the utterance (query-expansion targets, not
  extraction) → recall ceiling 0.544; scorer reports both. 12–13 of 30 gold departments are
  inferred from product type, contradicting probe rule "must stay unset".

## The integration target (from main, `techjam-conversational-search/memory/`)

`fast_memory.py` defines the hook `nlp_parse.py` should plug into:

```python
class SemanticParser(Protocol):
    def parse(self, message: str, turn: int) -> FastMemoryUpdate | None: ...

@dataclass(frozen=True)
class FastMemoryUpdate:
    category: str | None; intent: str | None            # "buying"/"browsing"
    hard_constraints: tuple[TypedConstraint, ...]; soft_preferences: ...; negatives: ...
    topic_override: bool; replace_constraints: bool; confidence: float | None

@dataclass(frozen=True)
class TypedConstraint:
    value: str; kind: ConstraintKind; hard: bool; negated: bool; explicit: bool
    strength: float; confidence: float; source_turn: int; source: str; intent_epoch: int

ConstraintKind = category|budget|material|color|size|style|brand|use_case|feature
```

Mapping from `nlp_parse` output: `category_phrase`→`category`; `exploring`→`intent`;
slots with `tier_of(s)=="hard"`→`hard_constraints`, `"soft"`→`soft_preferences`;
`price_max`→a `budget` TypedConstraint; `ConstraintKind` == `ALLOWED_ATTRIBUTES` minus `other`.
**Gap:** the team's state has `negatives` ("don't want X"); my schema has only `declined`
("don't care about X"). Negation is a top failure mode and needs a slot. `update_state(...,
semantic_parser=)` returns `None` → deterministic fallback, so the LLM stays upside-only.

Also in main: `MEMORY_ARCHITECTURE.md` (fast/slow memory keyed by `user_id` — the evaluator
gives no cross-session user id, so M1 == M0 there); `qlmp_contract.md` (the PCA/SVD memory-
steering vector op Nickolas described, not built); Nickolas' last commit: "2 outstanding: QLMP
for long term memory and its integration; eval set for long term memory".

## Open questions for Winston / team

1. Is the DeepSeek API call on the slide real, or is it local-only? (Winston's stated constraint says local-only.)
2. Slide retrieval box is "vector similarity" only — exp08 says dense-only loses 0.711 vs 0.846. Needs a lexical arm.
3. Score the benchmark against exp11 (deterministic, current plan) or the deployed `starter/agent.py` (now with memory hooks)? Both is cheap.
4. Push the merge? (`git push` on `bench`.)

## Commands

```bash
cd winston/lab/bench && python3 -m unittest test_bench -v          # 16 tests
python3 covariates.py            # rebuild 50k covariate cache (~10 s, cached)
python3 sample.py                # → products.jsonl (269, seeded)
python3 ../nlp_parse.py --model qwen2.5:7b-instruct                  # 30-probe parser run, cached
python3 ../pipeline.py           # resolver ablation, ~2 s
ollama list                      # 4 models expected
```
