# Messy-Input Benchmark (Phase A) — Design

Date: 2026-08-29. Owner: Winston. Status: approved design, not yet built.

## 1. Purpose

Produce a benchmark of **messy shopper utterances with free ground truth** so that
changes to the parser, the category resolver, and the keyword/dense gate can be
measured against the competition's own metrics instead of a 30-case hand-written
probe set with a ±0.11 noise floor.

The benchmark answers two questions per run: *given a messy opening message, where
does the target product rank?* and *does the agent's first clarifying question
target something the shopper actually cares about?* It is single-turn.
Full-session replay is Phase B and reuses this data.

Utterance styles follow the team's search-type taxonomy (exact / product type /
feature / use case / symptom / compatibility), extended with the probe set's
`plain` control and `lay` vocab-gap style.

## 2. What already exists, and why we extend it

`experiment_1/shopper_agent.py` + `run_eval_v2.py` (Yang Xu) already drive an LLM
shopper from a target product and its intent card. On the 200 public targets it
scores the current agent at **0.489 TechnicalScore, HitRate@10 0.575** — versus
0.889 / 1.000 on the templated simulator. That gap is the headroom in the project.

v2 cannot serve as the benchmark as-is:

| limitation | consequence |
|---|---|
| Rule 3 tells the shopper to *"drop descriptive hints from the product details"* | utterances paraphrase the listing — the opposite of the probe set's "do not use catalog vocabulary" |
| Rule 3 forbids *"new preferences or use-cases (like 'biking' or 'travel')"* | occasion-driven and goal-driven utterances (probe strata A and C) are banned by construction |
| Results are aggregates only | no per-case target rank, no candidate counts — cannot calibrate a gate or stratify failures |
| Uses OpenAI / DeepSeek / Gemini when a key is present | violates the team's local-only constraint; generator is unrecorded |
| n = 200, the public targets | popularity-skewed; no product covariates |

We keep v2's skeleton (target + intent card → LLM shopper → utterance) and its
`materialize_hidden_fields` / profile injection, and replace the rules, the
generator selection, and the output.

## 3. Product axis

### 3.1 Sampling

Seeded (`SEED = 20260829`). Target **~260 unique products**:

1. 120 drawn uniformly from the catalog.
2. Over-sample the three failure modes that are large enough to matter and tested
   nowhere: **+30 silent-on-material**, **+30 with a near-duplicate sibling**,
   **+30 in the bottom quartile of catalog `rating_number`**. Drawn uniformly
   within each pool; deduplicated against the base and each other.
3. **+30 with `has_model_code`** — not a failure mode, but model codes are 3.9% of
   the catalog, so without this the `exact` style would have ~8 cases.
4. **+30 with `compat_eligible`** — same reason; genuine compatibility targets are
   ~350 products (0.7%).
5. Exclude products with an empty title. Promo-bucket products are *included*
   (they are a failure mode) but not over-sampled.

### 3.2 Covariates — computed once per sampled product, never binned before reporting

| name | definition | source |
|---|---|---|
| `descriptiveness` | Σ IDF over unique tokens in `features` | `common.py` `ix.idf` |
| `title_richness` | Σ IDF over unique tokens in `title` | same |
| `jargon` | fraction of title+feature tokens with catalog df ≥ 10 **and** `wordfreq.zipf_frequency(t, "en") < 3.0` | `wordfreq` (offline after install); `null` if unavailable |
| `bucket_size` | size of the product's evaluator bucket (`coarse_category`, last two path segments) | `ix.buckets` |
| `popularity` | `rating_number` | catalog |
| `has_model_code` | title matches `\b[A-Z]{1,4}-?\d{3,}[A-Z0-9-]*\b`, excluding the material grades `316L 925 14K 18K 10K 585 750` and pure digits | regex |
| `silent_on_material` | no material word (`cotton polyester nylon leather wool spandex silk rayon denim suede`) anywhere in title+features+details+description | regex |
| `has_near_duplicate` | another product in the same bucket has title-token Jaccard > 0.6 | computed |
| `price_present` | `price is not None` | catalog |
| `promo_bucket` | `categories[1]` not in the 12 canonical departments | catalog |
| `category_depth` | `len(categories)` | catalog |
| `compat_eligible` | bucket ∈ {`watches watch bands`, `shoe care & accessories shoelaces`, `charms bead`, `charms & charm bracelets charms`, `shoe care & accessories shoe decoration charms`} — things bought *for* an item the shopper already owns | `ix.bucket_of` |
| `compat_anchor` | the owned item, from bucket: watch bands → "watch"; shoelaces → "sneakers or boots"; any charms bucket → "charm bracelet"; else `null` | derived |
| `department` | `normalize_department(details.Department)` — needed by the `for_other` modifier | `nlp_parse.py` |

Validated by random sampling on 2026-08-28: the boilerplate and sparse rules
overlap 99% and are replaced here by the continuous `descriptiveness`; the old
model-code regex fired on material grades and is tightened as above.

## 4. Utterance axis

### 4.1 Base system prompt

v2's `make_system_prompt`, made single-turn. Dropped: rule 1 (lead with hard
constraints — contradicts the `use_case`/`symptom` styles), rule 3 (paraphrase the
listing — the opposite of messy), rule 5 (recognise the product and end the chat —
multi-turn only), and the Ground Truth ASIN and Category lines (leakage). Kept:
rules 2, 4, 6, 7. Added: *"Write only the opening message. One to three sentences."*
The intent card, profile, title, details and description are injected as v2 does.

### 4.2 Styles — one utterance per style per product

| style | search type | instruction appended to the system prompt | intent label | probe stratum |
|---|---|---|---|---|
| `exact` | exact search | "You remember this from the listing: {code}. Mention it, plus one other thing you want." Only when `has_model_code`; `{code}` is the first regex match | **buying** | D |
| `product_type` | product type | "Name only the kind of item you want. Nothing else — no features, no occasion, no brand." | browsing | — (under-specified) |
| `feature` | feature | "Name the kind of item and two or three things it must have." | browsing | — |
| `use_case` | use case / problem | "Describe the situation, event or task you need this for. Do not name the item and do not list its attributes — let the assistant work it out." | browsing | A |
| `symptom` | symptom / problem | "Describe the problem you are trying to fix, or how you want to look or feel. Do not name the item." | browsing | C |
| `compatibility` | compatibility | "You already own a {anchor} and need this to go with it. Describe what you own and what you need it for — do not describe the accessory itself." Only when `compat_eligible` | browsing | — (anchor confusion) |
| `plain` | — | "Tell the assistant what you're looking for." Unconstrained control | browsing | G |
| `lay` | — | "Describe what you want in everyday words. You must not use any of these words: {forbidden}." `{forbidden}` = content words of title + features, capped at 40 | browsing | B |

`intent_label` is recorded per case from this table. It is the first labelled
buying/browsing signal the project has for messy input.

### 4.3 Modifiers — each applied independently with probability 0.2, recorded as flags

| modifier | instruction appended | failure mode |
|---|---|---|
| `negation` | "Also say one specific thing you do NOT want." | negated term treated as positive |
| `for_other` | "You are buying this for your {relation}." | department taken from the speaker, not the recipient |
| `vague_budget` | "Mention that price matters to you, but do not say a number." | budget with no numeric value |
| `format_noise` | "Write the code with different spacing, hyphens or capitalisation than the listing." **`exact` style only**, probability 0.3 | exact match breaks on formatting; embeddings destroy the code either way |

`{relation}` is drawn to be consistent with the product's normalized department
(`womens` → wife/mum/sister/daughter; `mens` → dad/husband/brother/son; `girls`/
`baby-girls` → daughter/niece; `boys`/`baby-boys` → son/nephew; unisex or unknown →
friend). A contradiction between relation and target department is a labelling
error, not a test case.

### 4.4 Generators

Round-robin over `llama3.1:8b`, `gemma2:9b`, `mistral:7b` via Ollama at
`temperature 0.7`. **Never `qwen2.5:7b-instruct`** — that is the parser's model
and sharing it would let the parser learn its own dialect. The model tag is
recorded per case and is a covariate in reporting. All calls go to
`localhost:11434`; the script reads no API keys.

### 4.5 Messiness control

For every utterance compute
`overlap = |content(utterance) ∩ content(title + features)| / |content(utterance)|`,
where `content()` = lowercase alphanumeric tokens (single characters dropped)
minus the stopword list in `prompts.py` — the same function that builds the `lay`
forbidden list, which additionally drops pure-digit tokens. Overlap is bag-of-words
with no stemming: "shoes" does not match "shoe", and a negated word counts as
overlap. Both are reported caveats, not corrections.
If `overlap > 0.5` regenerate once; if still above, keep and set `overlap_flag`.
Overlap is reported as a distribution and used as a reporting covariate. It is
**never** an outcome filter — rejecting cases the current agent already solves
would bias the set toward whatever the current agent is bad at.

## 5. Output

`winston/lab/bench/`

```
products.jsonl   one row per sampled product: asin + all covariates from §3.2
cases.jsonl      one row per utterance
manifest.json    seed, model tags, counts per style/modifier/generator, wall time
```

`cases.jsonl` row:

```json
{"case_id": "c0001", "asin": "B00KZIV0Q0", "utterance": "...",
 "style": "use_case", "intent_label": "browsing", "modifiers": ["negation"],
 "generator": "gemma2:9b", "overlap": 0.18, "overlap_flag": false}
```

Ground truth is `asin`. There is no hand-written gold parse. Expected size:
~260 products × 6 open styles + `exact` and `compatibility` where eligible
≈ **1,600–1,700 cases**. Generation is resumable and runs in the background.

## 6. Scoring (single-turn)

`score.py` feeds each utterance as turn 1 to two systems and records the target's
rank in each:

1. **Current best agent** — `nickolas/experiments/experiment_11_candidate_agent.py`
   `respond(session_id, utterance, turn=1, top_k=50)`. Deterministic, no LLM.
   Rank within the returned list; `null` if absent. Also records the response's
   `ask_attribute` and **`question_hit`**: true if that attribute equals
   `classify_constraint(c)` for any `c` in the target's `intent_card`
   `hard_constraints`. A hit means the first question would have elicited a
   real disclosure — the cheapest available measure of clarification quality,
   and the direct test of "can the agent narrow a browser down".
2. **Parser → resolver** — `winston/nlp_parse.py` `parse_with_ollama` then
   `winston/lab/pipeline.py` `resolve()`. Records the true bucket's rank among
   the 1,115 buckets and the resolver confidence.

`report.py` produces HitRate@10, MRR, bucket-rank, and question-hit rate, sliced
by style × intent label × generator × each covariate's quartile × each modifier
flag, every cell with a **bootstrap 95% CI** (2,000 resamples, seeded). Cells
with n < 20 are shown but marked.

## 7. Success criteria — for the benchmark itself

- ≥ 1,000 cases; every open style ≥ 200 cases; `exact` ≥ 30; `compatibility` ≥ 30.
- Each over-sampled pool ≥ 25 products after deduplication.
- `lay` style median overlap < 0.30 — proves the forbidden list works.
- `use_case` and `symptom` median overlap < 0.35.
- `product_type` median utterance length < 8 content words — proves it stayed broad.
- Re-running with the same seed reproduces `products.jsonl` exactly.
  (Utterances vary with temperature and are not expected to reproduce.)
- Runs with no network and no API keys.
- Full generation ≤ 6 hours on the 16 GB laptop (≈ 1,700 calls at ~8–12 s), resumable.

## 8. Out of scope

- Full-session replay (Phase B). Reuses `cases.jsonl` as turn 1 with v2's loop.
- Tuning any gate threshold. The benchmark produces the data; the gate is
  Nickolas's routing layer and is being replaced.
- Fixing v2's intent-override simulation (its MTTC of 8.7 indicates the LLM
  shopper is not triggering overrides on the scripted turn).
- Any change to `nlp_parse.py` or the resolver.

## 9. Dependencies

- Ollama with `llama3.1:8b`, `gemma2:9b`, `mistral:7b` pulled (≈ 15 GB total; one
  resident at a time).
- `winston/experiments/common.py` (cached catalog index, IDF, buckets).
- `wordfreq` — optional; `jargon` is `null` without it.
- `experiment_11_candidate_agent.py` for scoring; its own dependencies are
  SQLite FTS5 and numpy per `nickolas/CURRENT_BEST_ARCHITECTURE.md`.

## 10. Files

```
winston/lab/bench/
  prompts.py       style and modifier text from §4, plus the forbidden-list builder
  covariates.py    §3.2 → products.jsonl
  sample.py        §3.1 → the product list (calls covariates.py)
  generate.py      §4  → cases.jsonl + manifest.json
  score.py         §6  → results.jsonl (one row per case per system)
  report.py        §6  → report.md with the CI tables
```
