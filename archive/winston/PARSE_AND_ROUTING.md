# Parse → Route: what is measured, what is designed, what is unbuilt

Last updated: 2026-08-28. Owner: Winston (NLP lane).

Scope: how a user message becomes structured state, and how that state is split
across the retrieval routes. Every number here is reproduced by a script in this
repo; nothing is asserted from intuition. Sources are cited inline.

---

## 0. The context that changes everything

`main` now contains `nickolas/CURRENT_BEST_ARCHITECTURE.md`. The team's best
observed agent scores **TechnicalScore 0.8995** with:

> "no LLM call, embedding model, GPU, model training, web request, or external
> service"

and **HitRate@10 = 1.000** on all 200 public sessions.

Three consequences for this lane, and they are not optional:

1. **There is no headroom on the public evaluator.** HitRate is saturated. A
   better parser cannot show a gain on a benchmark that is already perfect.
2. **Typed state was already tested and changed nothing.** Experiment 9
   reproduced all 2,000 Experiment 7 turn rankings *bit-for-bit* with typed
   structured state — 0 rescues, 0 regressions.
3. **Dense retrieval was already tested as a route and lost badly.** See §1.

So the parser cannot be justified by public score. It can only be justified on
**messy input**, and `probe_set.md` is currently the only instrument that
measures that. That is the argument for this lane — not a weakness in it.

The architecture brief's own registered next experiment #6 is *"Typed state and
negation: parse material, color, size, budget, brand, style, use case,
positive/negative polarity, hard/soft strength, and override lineage."* That is
exactly `nlp_parse.py`. This lane is the designated next step; it just has to be
evaluated on the right instrument.

---

## 1. Status against the five questions

| # | Question | Status |
|---|---|---|
| 1 | Vague category via adjacent keywords (cross-training) | **Measured, not solved.** Root cause identified — it is not the disagreement you expected. |
| 2 | Goal-not-object descriptions (boilerplate stratum) | **Measured, not solved.** Slots extract well; category resolution is what fails. |
| 3 | Parsing + agreement/disagreement logic | Parsing **built and measured**. Agreement logic **not started** — but the design space just narrowed sharply. |
| 4 | Browsing/buying score | **Solved for the template, unbuilt for messy input.** Currently a boolean, and untested. |
| 5 | Four-layer routing | **Partly measured, and the proposed ordering is wrong.** Layer 3 should be layer 1; layer 1 was measured harmful as a route. |

---

## 2. Q1 — the cross-training case. The disagreement is not where you think.

You predicted heavy disagreement between the keyword filter and the intent
filter. There is disagreement, but the parser fails *before* either route sees
the query.

Probe case 02, actual output from `qwen2.5:7b-instruct`:

```
utterance      "a shoe that is good for all purpose usage, like if i do a mixture
                of sports like lifting weights, running, maybe racket sports.
                i'm a woman."
gold category   cross-training-shoes
pred category   shoe                        <-- collapsed
pred slots      use_case: all purpose usage
                use_case: sports
                use_case: lifting weights
                use_case: running
                use_case: racket sports
```

**The model extracted every piece of category evidence correctly and typed it
correctly — then discarded it at category resolution.** "Cross-training" is
precisely the conjunction of those five `use_case` slots. No route can recover
this, because `category_phrase = "shoe"` is what gets handed downstream.

### What the measurements say

Three independent sources, all pointing the same way:

- **`exp06_vocabulary_resolution.json`** — lexical bucket resolution over 51
  plausible paraphrasings: `top1 0.196`, `top3 0.490`, `top10 0.549`, needing a
  **625-product median pool** to get there. Caps at 55–65%.
- **`exp07_turn1_category_channel.json`** — on the *templated* public set the
  same problem is trivial: 200/200 exact bucket lookup. The template hands you
  the answer. Messy input does not.
- **Per-stratum F1** from the current run (`preds-qwen2.5-7b-instruct.json`):

  | stratum | n | slot F1 | category ok |
  |---|--:|--:|--:|
  | A situation/occasion | 6 | 0.123 | 0.500 |
  | B jargon vocab gap | 5 | 0.467 | 0.800 |
  | C boilerplate-only | 5 | 0.542 | **0.400** |
  | D model-code/spec | 4 | 0.327 | 0.750 |
  | E sparse text | 4 | 0.173 | 0.500 |
  | F crowded + priced | 3 | 0.244 | 0.333 |
  | G public-set control | 3 | 0.489 | 0.333 |

  Stratum A — your vague-description case — is the **worst** at 0.123.

### Proposal: make category a resolution step, not a parse field

Stop asking the parser to name the category. Ask it to emit evidence, then
resolve against the 1,115 catalog buckets as a separate, measurable step.

```
parse    -> category_phrase (weak) + use_case slots (strong, already correct)
resolve  -> score all 1,115 buckets against phrase + use_case slots together
            lexical (exp06 combined_1_1) when confident
            encoder over 1,115 short strings when margin is low
```

`exp06`'s `combined_1_1` (label + content profile) already beats label-only
(`top3 0.490` vs `0.392`). It has never been run with the use_case slots added
as query terms, which is the cheapest untested improvement available.

**This is the single highest-value thing to build next**, because case 02 shows
the evidence is already being captured and thrown away.

---

## 3. Q2 — goal-not-object descriptions

Same root cause, and the stratum table proves it: **stratum C has the best slot
F1 (0.542) and the worst category resolution (0.400)**. The parser understands
what the user wants to *achieve*; it cannot turn that into a catalog bucket.

This is not a separate problem from Q1. It is the same fix: resolve the category
from the full slot set rather than from a single phrase the model guessed.

Where it differs: goal descriptions produce `use_case` and `feature` slots with
**no lexical purchase on the catalog at all**. Probe case 30 (Crocs) is the pure
version — the user says "foam clog things with the holes", and the catalog says
"croslite", "ethylene". Zero content-word overlap. `probe_set.md`'s own note is
correct: only the popularity prior finds that product.

So: goal-only descriptions need the encoder, and they need the prior. They are
the case that cannot be solved lexically, and they are ~5/30 of the probe set.

---

## 4. Q3 — the agreement/disagreement logic

The design space narrowed a lot this week, because **routing was tested and it
lost**.

`experiment_08_intent_routed_dense_browsing` routed browsing sessions to dense
(MiniLM-L6-v2 cosine) and buying to lexical. Intent routing accuracy was
**100%**. Result on the 140-session held-out set:

| | TechnicalScore | MRR | rescues | regressions |
|---|--:|--:|--:|--:|
| Experiment 7 (lexical) | **0.846** | **0.648** | — | — |
| Intent-routed dense | 0.711 | 0.538 | **0** | **21** |

Zero rescues. Twenty-one regressions. Routing to dense is a dead end *as a
route*.

### But look at the paraphrase suite — this is the important table

| transform | method | TechnicalScore | HitRate@10 | **TS degradation** |
|---|---|--:|--:|--:|
| synonym substitution | lexical | 0.570 | 0.722 | **0.281** |
| synonym substitution | dense | 0.384 | 0.456 | **0.058** |
| clause reordering | lexical | 0.600 | 0.733 | **0.250** |
| clause reordering | dense | 0.386 | 0.467 | **0.057** |
| lexical compression | lexical | 0.599 | 0.744 | **0.252** |
| lexical compression | dense | 0.315 | 0.389 | **0.125** |

**Lexical is higher everywhere. Dense degrades 2–4× less.**

Neither dominates. Lexical wins on level, dense wins on *stability*. That is the
signature of two signals that should be **fused**, not switched between — and
fusion is exactly what has not been tested. Experiment 8 tested `if browsing:
use dense else: use lexical`. It did not test `always use both, weighted`.

### Concrete proposal

Disagreement is a *feature*, not a routing decision:

```
agreement = overlap( top-50 lexical candidates, top-50 dense candidates )

high agreement  -> the query is well-specified. Trust lexical ranking, tighten
                   the slate, consider early termination.
low agreement   -> the query is vague or paraphrased. This is the exp06 regime.
                   Widen the pool, down-weight hard filters, and ASK — this is
                   the highest-information moment to spend a clarification turn.
```

That gives the clarification policy a real trigger. The current best agent picks
its question from the **turn number alone** (architecture brief §9) and is
explicitly flagged as a limitation: *"It can ask irrelevant questions."*
Disagreement is a cheap, principled signal for what to ask and when.

**Untested. This is the second thing to build.**

---

## 5. Q4 — the browsing/buying score

Two very different situations, and it matters not to conflate them:

**On the template it is already solved and worth nothing.** Experiment 8 hit
**100% routing accuracy** because the simulator emits a literal marker:

```
"I'm looking for {category}, but I'm still exploring."     -> browsing
"I'm looking for {category}. A key requirement is: {c}."   -> buying
```

A regex gets 100%. An embedder cannot beat 100%.

**On messy input it is unmeasured.** `nlp_parse.py` emits `exploring: bool`, and
**all 30 probe cases have `exploring = false`** — your probe set contains no
browsing utterances, so the field is completely untested. That is a gap in the
instrument, not in the model.

### Proposal

You are right that it should be a score, not a boolean, and right that it is not
keyword work. But before building an embedder for it:

1. **Add browsing cases to the probe set.** The public set has 80 browsing
   sessions to borrow phrasing from. Without them there is nothing to measure.
2. Change `exploring: bool` → `specificity: 0.0–1.0` in the schema, defined
   against your own definition: *how clear is the user's direction?*
3. A cheap proxy exists and should be the baseline before any encoder: **count
   the hard slots**. Your tier rule already separates filterable from scoreable.
   `n_hard = 0` with several soft tags is browsing; `n_hard >= 2` is buying. Test
   that first — it is free, and an encoder has to beat it to earn its place.

---

## 6. Q5 — the four-layer plan, revised against the measurements

Your proposed order was: **vector (coarse) → keyword (fine) → category (medium)
→ memory**. Three of the four assignments are contradicted by data.

### Layer 3 (category) should be layer 1

`exp07` on the public set: parse the carrier phrase, look the string up in the
1,115 buckets → **100% hit, target inside the bucket 100%, median pool 182
products.** It is the cheapest and highest-precision cut available, and it runs
before any scoring. Putting a vector filter ahead of it spends compute on 50,000
products to reach a set that a dict lookup already gives you.

On messy input it degrades to the exp06 regime (55–65%) — which is the argument
for the resolver in §2, not for demoting the layer.

### Layer 1 (vector) is not a coarse filter — it is a fusion component

Measured harmful as a route (§4): 0 rescues, 21 regressions. Its measured value
is **variance reduction under paraphrase**, not recall at the top of the funnel.
Use it as a scored signal alongside lexical, never as a gate that can exclude
products.

### Layer 2 (keyword hard filter) — do not wire this up yet

You are right that it is "extracting hards from parsed input". The problem is
what the parser currently puts there. From the current run:

> **13 soft tags were promoted to hard filters**, including
> `material: "waterproof"`, `material: "stretchy"`, `size: "long sleeve"`,
> `size: "roomy enough to fit a laptop"`, `brand: "UPF 50"`.

Every one of those would filter the catalog on a term that should only *score*.
`material: "waterproof"` excludes every product whose `details` does not say
waterproof — which, per `exp01`, is most of them, and probe case 04's own note
says "waterproof" is an inference the product text never contains.

**A hard filter built on today's parser output would wreck recall.** Fix the
tier assignment before connecting this layer. Keep three-valued matching
(match / contradict / **silent**) — silent must never exclude.

### Layer 4 (memory) — no evidence either way, and one caution

The best agent scores 0.8995 while **explicitly not using the profile**
(architecture brief §2: *"deliberately does not use the user profile"*). So
memory is not required for the current score. That does not make it worthless —
it makes it unmeasured.

Your weighting idea (memory counts more in buying, less in browsing) is sound
and testable. But note the confound: `exp03` already measured preference-tag
lift, and the public targets are heavily popularity-skewed (target median
`rating_number` ≈ 6,846 vs catalog median ≈ 12). A memory layer that helps may
just be re-deriving the popularity prior. Ablate against a popularity-only
control or the result will not be interpretable.

### Revised pipeline

```
1. CATEGORY      exact bucket lookup -> resolver (lexical + encoder fallback)
                 free on template, exp06 regime on messy input
                 gives ~182 candidates on turn 1
2. FUSION        lexical (BM25/FTS5) + dense, both scored, neither gating
                 record agreement as the disagreement signal
3. SOFT SCORING  soft slots + popularity prior + memory (weighted by specificity)
4. HARD FILTER   only high-confidence hard slots, three-valued, silent never excludes
                 currently unsafe - see layer 2 above
```

Category first, filters last, nothing that can exclude on weak evidence.

---

## 7. What to build next, ranked

1. **Category resolution from the full slot set.** Feed `use_case` slots into
   `exp06`'s `combined_1_1` resolver alongside `category_phrase`. Case 02 shows
   the evidence is already captured. Highest value, lowest cost, directly fixes
   strata A and C.
2. **Fix the soft→hard tier assignment.** 13 errors today; blocks layer 2
   entirely. Likely a prompt fix ("one attribute per slot" — 8 values are also
   hitting the 40-char cap by cramming).
3. **Agreement signal.** Lexical/dense top-50 overlap, wired to the
   clarification policy. Replaces turn-number question selection.
4. **Browsing cases in the probe set**, then `specificity` score with the
   `n_hard` count as the baseline to beat.
5. Memory layer, ablated against a popularity-only control.

## 8. Open problems in the instrument itself

These limit what can be concluded and should be fixed in `probe_set.md`:

- **67 of 147 gold slots (46%) never appear in the utterance** (`quick-dry`,
  `flattering`, `hooded`, `plus-size`). They are query *expansion* targets, not
  extraction targets, and they cap any extractor at **0.544 recall**. Suggest
  splitting them into an `expand:` key so parser and expander are scored
  separately.
- **12–13 of 30 gold departments are inferred from product type**
  (`bathing suit` → womens, `high heels` → womens) which contradicts case 04's
  own rule: *"user never states gender... Must stay unset."* Note that fixing
  this makes the current score *worse* (0.600 → 0.400) because the model emits
  `null` only 1 time in 30 — it reaches for `unisex-*` instead. Both sides need
  work; the gold is not the only problem.
- **No browsing utterances**, so `exploring` is untestable (§5).

## 9. Reproducing everything here

```bash
cd winston
python3 nlp_parse.py                                # gold conversion + self-check
python3 nlp_parse.py --model qwen2.5:7b-instruct    # full run, caches predictions
cd experiments && python3 exp06_vocabulary_resolution.py
                 python3 exp07_turn1_category_channel.py
```

Predictions are cached to `preds-<model>.json`; re-scoring never needs the model
again. Teammates' results are under `nickolas/results/` and
`harshith/analysis/`.

Harshith's independent category analysis agrees exactly with `exp07` — 863
labels, 1,628 full paths, 1,832 nodes, max depth 8, identical depth
distribution. That number is settled.
