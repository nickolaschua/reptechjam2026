# Lab: Parse → Resolve → Fuse

A sandbox that **imports** the existing infra rather than forking it. Nothing here
modifies the submission agent, nickolas' experiments, or the evaluator.

---

## The one-paragraph version

The team's best agent scores **0.8995 with HitRate@10 = 1.000** using no LLM and no
embeddings. There is no headroom left on clean input. But the same agent scores
**0.57–0.60 on paraphrased input** — that 0.30 gap is the only headroom in the
project, and it is a *language understanding* gap, which is this lane's job. So we
build a pipeline that turns a messy message into typed state, resolves the category
from **all** the evidence instead of one guessed phrase, and runs lexical and dense
retrieval **together** rather than choosing between them.

---

## Why this and not something else

Three things are already settled by measurement. We build around them, not into them.

| Already tested | Result | What it means for us |
|---|---|---|
| Dense as a **route** (exp08) | 0 rescues, **21 regressions** | Never switch to dense. Only ever fuse. |
| Typed state on public set (exp09) | bit-for-bit identical, 0 change | Don't evaluate on clean sessions. Nothing can move. |
| Lexical vs dense under paraphrase (exp08) | lexical degrades **0.25–0.28**, dense **0.06–0.12** | Dense's value is *stability*, not recall. That is a fusion weight, not a gate. |

And one thing is newly measured, in `pipeline.py`, on the 30 probe cases:

| resolver input | top1 | top3 | median rank of true bucket |
|---|--:|--:|--:|
| `category_phrase` only (today) | 0.167 | 0.333 | 20.0 |
| **+ soft slots** | **0.267** | **0.433** | 10.0 |
| + hard slots too | 0.233 | 0.433 | **7.5** |
| slots only, no phrase | 0.100 | 0.167 | 25.0 |

(Re-run 2026-08-30 after the soft→hard tier fix in `nlp_parse.py`; the earlier
table predated it.) The parser was already extracting the evidence and discarding
it. Feeding slots back in **halves the median rank** of the correct bucket. With
corrected tiers, hard slots no longer hurt — B vs C is within n=30 noise; the
benchmark's `bucket_rank` settles it. That is stage 2, and it works today with no
new dependencies.

---

## The five stages

```
        message
           │
  ┌────────▼────────┐
  │ 1. PARSE        │  template regex first (free, 100% on the simulator)
  │                 │  local LLM w/ constrained schema only when regex misses
  └────────┬────────┘  → typed slots: category_phrase, department, slots[], price
           │
  ┌────────▼────────┐
  │ 2. RESOLVE      │  score all 1,115 buckets from phrase + soft slots
  │                 │  → ranked buckets + confidence (top-1 vs top-2 margin)
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │ 3. RETRIEVE     │  lexical (BM25) AND dense (MiniLM), both scored
  │                 │  neither one gates. Nothing gets excluded here.
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │ 4. AGREE        │  overlap(lexical top-50, dense top-50) → agreement ∈ [0,1]
  └────────┬────────┘
           │
  ┌────────▼────────┐
  │ 5. RANK + ASK   │  RRF fusion weighted by agreement and confidence
  │                 │  + popularity prior
  │                 │  low agreement → widen pool AND spend a clarification turn
  └────────┬────────┘
           │
      top-10 + ask_attribute
```

### Stage 1 — Parse
**Owner: Winston. Status: built** (`../nlp_parse.py`, slot F1 0.333 on probes).

The template regex from exp07 is the fast path — on the simulator's grammar it
hands over the bucket verbatim, 200/200, for free. The LLM only runs when the
regex fails. That keeps token cost near zero on clean input and reserves the model
for the case it is actually needed.

**Known defect blocking stage 3:** the parser promotes 13 soft tags to hard slots
(`material: "waterproof"`, `size: "roomy enough to fit a laptop"`). Must be fixed
before any hard filter is wired up, or recall dies.

### Stage 2 — Resolve
**Owner: Winston. Status: built and measured** (`pipeline.py`).

The important idea: **category is a resolution step, not a field the model guesses.**
Ranking 1,115 short bucket strings is a retrieval problem, and the parser's job is
only to supply evidence. Confidence = the relative margin between the top two
buckets, and it feeds stage 5 — a flat top-2 means the resolver is guessing and the
pipeline should not act as if it knows the category.

Next: encoder fallback over the 1,115 bucket strings when confidence is low. That
is exp06's verdict and it is still the right call — but only in the low-confidence
branch, so it costs nothing on the easy majority.

### Stage 3 — Retrieve
**Owner: open. Status: not started. Needs nickolas' harness.**

```python
from nickolas.experiments.harness import Harness, replay_policy
lex_ids, lex_scores = harness.lexical.ranked(query)
dns_ids, dns_scores = harness.dense.ranked(query)
```

Both routes always run. Bucket restriction from stage 2 is a **boost, not a filter**
— the resolver is right 43% of the time at top-3, so hard-filtering on it would
throw away the target more than half the time.

### Stage 4 — Agree
**Owner: open. Status: `agreement()` written in `pipeline.py`, untested.**

One number: how much do the two routes' top-50 overlap. High = query is
well-specified. Low = vague or paraphrased, i.e. the exp06 regime.

### Stage 5 — Rank and ask
**Owner: open. Status: not started.**

RRF fusion, weighted by agreement and resolver confidence, plus the popularity
prior (which is doing real work — target median `rating_number` ≈ 6,846 vs catalog
median ≈ 12).

The payoff beyond ranking: **agreement tells you when to ask.** The current best
agent picks its clarification attribute from the turn number alone, and the
architecture brief flags this as a limitation — *"It can ask irrelevant
questions."* Low agreement is a principled trigger, and MTTC is 20% of the score.

---

## How we know if it worked

**Evaluate on paraphrased sessions, not clean ones.** exp09 proved clean sessions
cannot move. exp08 already built the perturbation harness:

```python
from nickolas.experiments.experiment_08_intent_routed_dense_browsing import transform_message
# synonym_substitution | clause_reordering | lexical_compression
```

Success = beat these, without regressing clean sessions:

| transform | lexical baseline (TechnicalScore) |
|---|--:|
| synonym substitution | 0.570 |
| clause reordering | 0.600 |
| lexical compression | 0.599 |

Two gates, both required:
1. **Paraphrase TechnicalScore goes up** against the numbers above.
2. **Clean 200-session score does not regress** below 0.8995.

Gate 2 matters because a pipeline that only helps on messy input and hurts on clean
input is not shippable — the private set is 800 sessions generated the same way as
the public 200.

---

## Build order

Ranked by value per unit of work. Stages 1–2 need nothing installed.

1. ~~Feed soft slots to the resolver~~ — **done**, median rank 20 → 8.5.
2. **Fix the soft→hard tier bug** in `nlp_parse.py`. Blocks stage 3 entirely.
   Likely a prompt fix: "one attribute per slot" (8 values are also hitting the
   40-char cap by cramming several attributes into one).
3. **Encoder fallback in stage 2**, low-confidence branch only. ~2MB over 1,115
   strings.
4. **Stage 3 + 4**: wire `replay_policy` with a fused ranker, record agreement.
5. **Stage 5**: agreement-triggered clarification.
6. Memory layer — ablate against a **popularity-only control** or the result is
   uninterpretable (see PARSE_AND_ROUTING.md §6).

---

## Setup

Stages 1–2 run now:

```bash
cd winston/lab
python3 pipeline.py          # ~2s, uses the cached catalog index
```

Stages 3–5 need the harness (~2GB, torch):

```bash
pip install -r ../../nickolas/experiments/requirements.txt
```

## Files

| file | what |
|---|---|
| `pipeline.py` | the five stages; 1–2 implemented, 3–5 marked with contracts |
| `ARCHITECTURE.md` | this |
| `resolver_results.json` | stage 2 measurements, regenerated on each run |
| `.cache/` | bucket content profiles (gitignored, rebuilt in ~30s) |

Depends on, does not modify: `winston/nlp_parse.py`, `winston/experiments/common.py`,
`nickolas/experiments/harness.py`, `techjam-conversational-search/evaluator/`.
