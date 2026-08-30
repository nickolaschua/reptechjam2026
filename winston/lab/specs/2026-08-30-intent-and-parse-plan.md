# After the benchmark: intent axes and ambiguous-prompt parsing

Date: 2026-08-30. Owner: Winston. Status: plan, sequenced for a short clock.

## 0. The constraint that shapes everything

The keyword-matching layer (BM25 + template regex) is not modified. Everything here
is a **bolt-on that fires only when a message falls through that layer's regex** —
i.e. only on human-written input. On the simulator's deterministic template the
bolt-on never runs, so the public score cannot regress by construction. The
fall-through trigger applies on every turn (`INITIAL_RE`, `DISCLOSURE_RE`,
`OVERRIDE_RE` all missing), not only turn 1.

Integration hook that already exists in main: `fast_memory.SemanticParser.parse()`
returning `None` hands control back to the keyword layer. The retrieval/ask half
has no hook yet (`_legacy_recommendations` is unchanged) — one function to ask
Nickolas for, not a rewrite.

## 1. What the probes settled (2026-08-29, scratchpad scripts)

| finding | consequence |
|---|---|
| BM25 pool size / OR-entropy / NQC do not track specificity on this catalog (`shoes`, `women` sit in every product's root categories; AND-pool is 0 for every messy utterance) | drop retrieval-geometry as an intent feature |
| resolver confidence separates top-3 hit vs miss (median 0.27 vs 0.09, n=30) but overlaps heavily; posterior entropy is saturated (~5.5/5.64) | confidence is a weak *resolvable* feature; entropy is dead |
| `women's running shoes` ± `waterproof` ± `black size 7 under $100` give **identical** resolver output | constraint density is invisible to category resolution — it lives in the parse |

So intent is **two orthogonal axes**, not one score:

```
specificity  <- parse only: n_hard, n_soft, price_stated, exp08 stance cues
resolvable   <- stage 2/4: resolver confidence, lexical/dense agreement (untested)
```

| | resolvable | not resolvable |
|---|---|---|
| **specific** | tight slate; ask highest-value within-bucket attribute | vocab gap (Crocs). Dense weight up; ask a category-confirming question |
| **vague** | template-browsing case. Popularity slate; ask top attribute | cross-category (Japan winter). Diverse slate over top-3 buckets; ask category |

Prior art to cite, not claim: this is query performance prediction (Cronen-Townsend
2002 clarity; Shtok 2012 NQC) applied as a fusion weight + clarification trigger.

## 2. Labels — all free, none hand-written

| label | source | measures |
|---|---|---|
| `lexical_rank <= 10` | `score.py` | retrievability: can the keyword layer find it from this text alone |
| `card_hard_said` | `score.py` (intent-card hard constraints voiced in the utterance) | specificity, grounded in what the shopper said |
| `intent_label` | style prior (`exact` = buying, else browsing) | weak second opinion only; `feature` style is buying by our own definition |

Every approach below is evaluated against the first two. The decision that matters
is *ask vs show*, so AUC / Brier against `lexical_rank <= 10` is the primary number.

## 3. Part A — browsing vs buying: approaches, cheapest first

| id | approach | needs | cost | stop rule |
|---|---|---|---|---|
| A0 | baselines: style prior; exp08 `EXPLORATORY_CUES`/`BUYING_CUES` regex; `card_hard_said >= 1`; `n_hard >= 2` | lexical pass (+ resolver pass for `n_hard`) | 0 | these are the floor every later approach must beat |
| A1 | two-axis rule: specificity = `n_hard + price_stated + cue_hit`; resolvable = `resolver_confidence >= 0.2`; report the 2x2 with HitRate per cell | resolver pass | 0 | ship if the 2x2 separates lexical hit from miss with non-overlapping CIs |
| A2 | calibrated LR over ~7 features (`n_hard`, `n_soft`, `price_stated`, cue_hit, `resolver_confidence`, content-word count, `overlap`) -> P(lexical hit) | A1 data | 20 lines, no new deps | only if A1 leaves room; score on Brier/ECE, not accuracy |
| A3 | LLM self-report: add `specificity: 0-1` to the parser's constrained schema — **same call, zero extra latency** | schema change before the resolver pass | 5 lines | compare to A1 on the same labels |
| A4 | MiniLM embedding + LR | torch (~2 GB, not installed) | high | last resort; exp08 says dense buys stability, not level |

## 4. Part B — parsing ambiguous prompts: experiments, each on `parses.jsonl`

| id | experiment | benchmark slice that tests it | measure |
|---|---|---|---|
| B1 | **negation slot**: `negated: bool` per slot (matches the team's `TypedConstraint.negated`); exclude negated terms from the lexical query | `negation` modifier (~20 % of cases) | `lexical_rank` with vs without the negated term |
| B2 | category from evidence (built: resolver + soft slots, probe median rank 20 -> 8.5); low-confidence fallback = qwen picks among the resolver's top-10 buckets (constrained choice, no torch) | `use_case`, `symptom`, `lay` | `bucket_rank`, split by `resolver_confidence` |
| B3 | **parse -> keyword-layer state**: map `category_phrase` + slots onto exp11's `{category, constraints[]}` and call its own `_rank`. The parse *replaces the regex*; retrieval is untouched. This is the bolt-on in its least invasive form | all | `parsed_rank` vs `lexical_rank` vs `template_rank` — add as a 4th system in `score.py` |
| B4 | query expansion for the vocab gap: qwen emits 3-5 catalog-vocabulary synonyms per soft slot, added as low-weight OR terms | `lay` (forbidden list forces the gap by construction) | `parsed_rank` with vs without expansion |
| B5 | department from recipient, not speaker | `for_other` modifier | parsed department == product department |
| B6 | clarification policy: ask for whichever of {material, color, size, budget, feature} is **absent** from the parse, instead of the turn-number sequence | all | `question_hit` vs exp11's fixed sequence |

B3 is the one that decides whether the bolt-on is worth shipping. Run it first.

## 5. Sequence, given the clock

```
now (generation running, ~3.5 h)
  1. schema: add `negated` per slot and `specificity` 0-1 to nlp_parse.SCHEMA   (A3, B1)
  2. re-run the 30-probe set; slot F1 must not drop below 0.441
  3. score.py: add `parsed_rank` (B3) and read `negated` (B1)
  4. lexical pass on whatever cases.jsonl holds (minutes) -> first report: A0 baselines, question_hit
+40 min
  5. resolver pass on the first 300 cases (stratified prefix; ~8 s/case) -> A1, A3, B1, B2, B3, B5
  6. decide: does parsed_rank beat lexical_rank? If no, the bolt-on is the clarification policy (B6) only
generation done
  7. full lexical pass; resolver pass continues in the background on the rest
  8. B4, B6, A2 if A1 left room
```

Schema changes (step 1) must land **before** step 5, or the pass is re-run.

## 5a. Two gates every parser change must pass (added 2026-08-31)

| gate | command | pass condition | why |
|---|---|---|---|
| probe F1 | `python3 nlp_parse.py --model qwen2.5:7b-instruct` | slot F1 >= 0.441 (baseline 2026-08-30) | messy-input extraction did not regress |
| template | `python3 lab/bench/template_check.py 30` | exact browsing/buying forms: parsed rank == regex rank case for case; paraphrase: parsed hit@10 == buying-template regex hit@10 | the bolt-on must be a no-op on the simulator's own text and a full recovery when the organizer paraphrases it |

The template gate exists because of what it caught first time: on a browsing
template with nothing to extract, the 7B padded slots with the schema's own
literals (`scenario_only`, `reputable/trusted`) and dropped hit@10 from 0.67 to
0.17. Any literal in SCHEMA or PROMPT is a candidate padding value; `JUNK_VALUES`
must cover them all (the self-check asserts the enum lists do).

## 6. Out of scope

Anything that touches the keyword layer's ranking on template input. Dense
retrieval until torch is installed. Memory/profile (unmeasured; ablate against a
popularity-only control if ever attempted).
