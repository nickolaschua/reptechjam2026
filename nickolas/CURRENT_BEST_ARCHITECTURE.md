# Current Best Architecture Research Brief

Last updated: 2026-08-27

## Executive summary

The highest-scoring architecture observed in this repository is the Experiment 11 **clean FTS5 agent with specific clarification questions and global pagination**, identified as:

```text
clean_specific_global_pagination
```

On the 200-session public evaluator it achieved:

- TechnicalScore: **0.899530**
- HitRate@10: **1.000000**
- MRR: **0.717101**
- Mean turns to conversion (MTTC): **1.780000**
- Efficiency: **0.922000**

This is an experimental result, not the currently deployed submission architecture. The submission agent remains the Experiment 7 exact-match/BM25-RRF system, which scored **0.840846** on the same public evaluator.

There is also an important experimental-selection distinction:

- **Best observed after examining all results:** `clean_specific_global_pagination`, full score **0.899530**.
- **Selected using only the frozen 60-session calibration partition:** `clean_specific_query_pagination`, full score **0.894362** and 140-session evaluation-partition score **0.896425**.
- **Production promotion authorized:** no. The public dataset and the nominal evaluation partition had already been inspected during the preceding audit. Private or newly generated validation is required.

The central finding is that a small, deterministic, CPU-only SQLite FTS5 architecture currently beats the more complicated exact/BM25 and XTR/WARP experiments on this public simulator. Much of the gain appears to come from a combination of catalog lexical retrieval, a weak product-popularity prior, deterministic diversity, clarification-question choice, and never repeating already shown products.

## Task and evaluator context

The task is conversational product retrieval over a fixed catalog of **50,000 products**. An agent receives one user message at a time and must return:

```python
{
    "message": str,
    "ask_attribute": str | None,
    "recommendations": [{"parent_asin": str}, ...],
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
    },
}
```

The evaluator requests up to 10 recommendations per turn and simulates at most 10 turns. The 200 public sessions contain:

- 80 buying sessions
- 80 browsing sessions
- 30 intent-override sessions
- 10 boundary sessions

The target product, scenario label, hidden intent card, future replies, and conversion result are evaluator-only information. The agent receives only the current user message, turn number, session ID, catalog, and a user profile at reset time. The clean FTS5 agent deliberately does not use the user profile.

The evaluator stops a session when the target product appears in the Top 10, except that appearances before the required intent override do not count. A miss is assigned turn 11 for MTTC.

The score is:

```text
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 * HitRate@10
               + 0.30 * MRR
               + 0.20 * Efficiency
```

This means recall is the largest component, but placing the target high in the slate and finding it in fewer dialogue turns are also important.

## Architecture at a glance

```text
50,000-product JSONL catalog
        |
        v
In-memory SQLite FTS5 index
  title, categories, features,
  details, store, description
        |
        v
Observable dialogue parser
  category + active constraints
        |
        v
Tokenize, remove stopwords, deduplicate,
cap query at 45 terms
        |
        v
FTS candidate generation (maximum 1,000)
  1. all-term AND query
  2. weighted OR expansion if fewer than 30
        |
        v
Deterministic reranking
  lexical bag matches
  + weak rating-count prior
  + candidate-order prior
        |
        v
Brand/title diversification
        |
        v
Global seen-product exclusion
        |
        v
Top 10 recommendations
+ deterministic specific question
```

There is no LLM call, embedding model, GPU, model training, web request, or external service in the current best observed agent.

## 1. Catalog index

At initialization, the agent loads every JSONL product into an in-memory SQLite database and creates this FTS5 schema:

```sql
CREATE VIRTUAL TABLE products USING fts5(
    parent_asin UNINDEXED,
    title,
    categories,
    features,
    details,
    store,
    description,
    tokenize='unicode61 remove_diacritics 2'
)
```

The following metadata is also retained in Python for reranking and diversification:

- `title`
- normalized store/brand
- `searchable_bag`: title + categories + only the first three features
- `rating_number`

The measured index construction time in the latest full run was **15.901 seconds** on the local Windows CPU environment.

## 2. Session state

Each session keeps:

```text
category          coarse product category parsed from turn 1
constraints       ordered, deduplicated active preference strings
override_seed     the original preference that may later be revoked
shown_global      products shown anywhere in the session
shown_by_query    products shown for each normalized query state
history           raw messages seen so far
profile_present   whether a profile was supplied; profile content is not used
```

Constraint deduplication is case-insensitive and whitespace-normalized.

## 3. Dialogue parsing and intent overrides

The parser recognizes the official evaluator's message forms with case-insensitive regular expressions. It accepts straight and curly apostrophes.

Supported forms include:

```text
I'm looking for CATEGORY, but I'm still exploring.
I'm looking for CATEGORY. A key requirement is: CONSTRAINT.
I'm looking for CATEGORY. INITIAL_PREFERENCE.
For that, what matters is: CONSTRAINT; CONSTRAINT.
Actually, ignore my earlier preference. What I need is: NEW_PREFERENCE.
```

On an override:

1. Only the stored initial `override_seed` is removed.
2. Preferences disclosed after the initial turn remain active.
3. The new preference is appended.
4. The override seed is cleared.
5. Both global and query-scoped shown-product sets are cleared because the user's effective intent changed.

This corrects a problem in the original Yang implementation: the clean version does not keep or boost the stale preference after the evaluator says to ignore it.

No-preference replies do not add constraints, so the existing state remains unchanged.

## 4. Query construction

The retrieval query is:

```text
category + all active constraints
```

If no category has been parsed, the fallback category text is `clothing item`.

Tokenization:

- extracts alphanumeric tokens;
- lowercases them;
- removes a fixed list of common stopwords;
- removes duplicates while keeping first occurrence order;
- caps the query at 45 unique terms.

All constraints are currently treated as positive free-text evidence. The agent does not perform typed slot extraction, explicit negation modeling, numeric budget filtering, recency weighting, or semantic rewriting.

## 5. FTS5 candidate generation

Candidate generation has two stages:

### Stage A: strict all-term retrieval

The agent first builds a quoted FTS expression:

```text
"term1" AND "term2" AND ...
```

It takes at most 1,000 matching product IDs. This query is ordered by SQLite row ID, which is effectively catalog insertion order.

### Stage B: weighted soft expansion

If the strict query returns fewer than 30 products, the agent runs:

```text
"term1" OR "term2" OR ...
```

The weighted FTS5 BM25 fields are:

| Field | Weight |
|---|---:|
| title | 6.0 |
| categories | 4.0 |
| features | 2.5 |
| details | 2.5 |
| store | 1.5 |
| description | 1.0 |

The OR results are ordered by FTS5 BM25 score and then ascending `parent_asin`. New IDs are appended until the combined pool reaches at most 1,000.

If retrieval produces no candidates, the system falls back to the complete catalog.

An important implementation detail for future research is that weighted BM25 directly orders only the OR expansion. When the AND stage already returns at least 30 products, its candidates retain catalog insertion order before the Python reranker. This is a promising target for a clean ranking ablation.

## 6. Deterministic reranking

Each candidate receives:

```text
score = -0.001 * candidate_position
      + 0.3 * count(query terms found in searchable_bag)
      + 0.02 * rating_number^0.1
```

Where:

- `candidate_position` is the zero-based position from candidate generation;
- `searchable_bag` contains title, categories, and the first three features;
- the rating prior is based on review/rating count, not average rating;
- ties are broken by ascending `parent_asin`.

The popularity exponent of `0.1` makes the prior deliberately weak, but it is still consequential because the public targets are extremely popularity-skewed. In the audit, the catalog median `rating_number` was about 12 while the target median was about 6,846; 162 of 200 targets were in the catalog's top 5% by rating count and 173 were in the top 10%. This may not generalize to a private or differently sampled test set.

## 7. Diversity control

The reranker greedily scans products in score order and applies two soft diversity rules:

1. Select no more than two products from the same normalized store/brand.
2. Reject a product when the Jaccard overlap of its whitespace-split title tokens with an already selected title is greater than 0.6.

If these rules leave fewer than the requested number of recommendations, the agent backfills from the scored list without the diversity restrictions. Diversity therefore changes ordering and coverage but never intentionally returns a short slate when enough candidates exist.

## 8. Pagination behavior

The best observed configuration uses **global pagination**:

- every recommended product is added to `shown_global`;
- products shown earlier in the session are excluded on later turns, even when new constraints change the query;
- the seen set is cleared after an explicit intent override.

This behavior forces exploration through the ranked candidate set. It materially improved public-evaluator performance, especially compared with no pagination.

The calibration-selected alternative uses **query-scoped pagination**:

- recommendations are remembered under the normalized `category + constraints` query;
- an unchanged query advances to unseen products;
- a changed query can revisit a previously shown product;
- all query histories are cleared after an override.

Query-scoped behavior is arguably safer for real users because a newly disclosed constraint can make an earlier item relevant again. Global pagination scored better on the inspected public set, but calibration selected query-scoped pagination by a narrow margin.

## 9. Clarification-question policy

The best architecture asks one specific attribute per turn using a deterministic cycle:

```text
feature -> material -> color -> size -> style
        -> use_case -> budget -> brand -> category
```

The attribute is selected only from the turn number; it is not based on candidate entropy or detected missing fields.

This works well with the public evaluator because the simulator classifies hidden constraints into these attribute buckets and discloses up to two matching undisclosed constraints. If no hidden constraint matches the requested attribute, the simulator returns a no-preference message.

A broad `other` policy was also evaluated. It scored lower overall than the best specific/global configuration, despite `other` being able to disclose any remaining generated constraint.

## 10. Response format and cost

Every response uses a static message:

```text
Here are the strongest matches. What should I refine next?
```

It returns the selected clarification attribute, Top-K ASINs, and reports zero prompt/completion tokens. The architecture has no per-turn inference cost beyond SQLite queries and Python ranking.

Latest measured latency for the best observed global-pagination configuration:

| Statistic | Latency |
|---|---:|
| Count | 356 calls |
| Mean | 154.105 ms |
| Median | 16.528 ms |
| p95 | 585.306 ms |
| p99 | 1,224.768 ms |
| Maximum | 1,599.005 ms |

These tail measurements varied noticeably between otherwise score-identical runs, so they should be treated as local end-to-end measurements rather than stable microbenchmarks.

## Experimental results

### Full 200-session comparison

| Architecture | TechnicalScore | Hit@10 | MRR | MTTC | Efficiency |
|---|---:|---:|---:|---:|---:|
| Current submission: exact + conditional BM25 RRF | 0.840846 | 0.955 | 0.651821 | 2.610 | 0.8390 |
| Yang Experiment 1 original | 0.896387 | 1.000 | 0.704290 | 1.745 | 0.9255 |
| Clean, specific questions, global pagination | **0.899530** | **1.000** | **0.717101** | 1.780 | 0.9220 |
| Clean, specific questions, query pagination | 0.894362 | 1.000 | 0.702206 | 1.815 | 0.9185 |
| Clean, broad `other`, query pagination | 0.889043 | 0.995 | 0.691476 | 1.795 | 0.9205 |
| Clean, specific questions, no pagination | 0.862272 | 0.960 | 0.678575 | 2.065 | 0.8935 |

The best observed architecture improved TechnicalScore by **0.058684** over the current submission and by **0.003143** over Yang's original.

Compared with the current submission across all 200 sessions, it produced:

- 9 hard-failure rescues
- 0 hit-to-miss regressions
- 86 sessions converted earlier
- 12 sessions converted later

### Best observed architecture by scenario

| Scenario | Sessions | Hit@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Boundary | 10 | 1.000 | 0.503889 | 1.600 |
| Browsing | 80 | 1.000 | 0.668244 | 1.475 |
| Buying | 80 | 1.000 | 0.737862 | 1.350 |
| Intent override | 30 | 1.000 | 0.863095 | 3.800 |

Intent-override MTTC is structurally higher because target appearances before the evaluator's override turn are not eligible conversions.

### Frozen split results and selection discipline

The existing Experiment 6 split contains 60 calibration sessions and 140 evaluation sessions.

| Clean configuration | Calibration score | Evaluation-partition score | Full score |
|---|---:|---:|---:|
| Specific + global pagination | 0.887597 | **0.904645** | **0.899530** |
| Specific + query pagination | **0.889548** | 0.896425 | 0.894362 |
| `other` + query pagination | 0.871270 | 0.896660 | 0.889043 |
| Specific + no pagination | 0.827750 | 0.877068 | 0.862272 |

The preregistered selection rule chooses calibration TechnicalScore, then calibration MRR, then method name. It therefore selected `clean_specific_query_pagination`. That selected candidate passed every retrospective diagnostic gate against the current submission on the evaluation partition:

- TechnicalScore higher: yes
- HitRate@10 not lower: yes
- MRR not lower: yes
- regressions no greater than rescues: yes
- evaluation rescues: 4
- evaluation regressions: 0
- evaluation sessions accelerated: 60
- evaluation sessions delayed: 12

However, the split is not an unbiased holdout because the public data and this partition had already been examined. The global version is accurately called the **best observed** configuration, not a cleanly selected winner.

## Relationship to the currently deployed submission agent

The current submission agent has not been replaced. It uses the Experiment 7 architecture:

1. stateful category and exact phrase constraints;
2. exact scoring based primarily on the count of phrases occurring verbatim;
3. conditional BM25 fallback when exact evidence is weak or ambiguous;
4. equal-weight reciprocal rank fusion with `RRF_K=60` and depth 1,000;
5. deterministic ASIN tie-breaking;
6. a fixed `other` clarification policy.

Its saved and freshly reproduced public score is **0.840846**. Experiment 11 intentionally leaves this file unchanged until new-data validation can justify a production promotion.

## Leakage and safety boundary

The clean candidate's retrieval path uses only:

- catalog fields;
- current and previous user-visible messages;
- turn number;
- session-local shown-item history.

It does not receive or use:

- ground-truth target ASIN;
- scenario type;
- sample ID;
- hidden intent card;
- simulator behavior object;
- future turns;
- conversion result;
- private profile contents.

Oracle data is used only by the external evaluator after recommendations are produced to calculate ranks, hits, and scores.

## What appears to be driving the gain

The ablations and implementation audit suggest several interacting causes rather than one isolated innovation:

1. **Forced exploration matters.** Removing pagination reduced the clean score from 0.899530 to 0.862272, though this comparison also uses the globally best observed branch.
2. **Question choice matters.** Specific questions paired with global pagination beat broad `other` with query pagination, although this is not a perfectly isolated one-variable comparison.
3. **Popularity is useful on this public set.** Targets are far more popular than typical catalog products, making even a weak review-count prior valuable.
4. **Lexical field weighting is strong.** Titles and categories receive the largest FTS weights, with feature/detail evidence next.
5. **Diversification increases slate coverage.** Brand and near-duplicate-title controls prevent the Top 10 from collapsing into redundant products.
6. **Correct state handling preserves override performance.** Removing only the revoked seed avoids stale-intent contamination without throwing away later valid constraints.
7. **Simple deterministic components are enough.** Dense retrieval and the XTR/WARP index did not beat this public score.

The current experiment does not causally isolate every pairwise interaction. A factorial ablation on fresh data would be required for strong causal attribution.

## Known limitations and threats to validity

### Evaluation overfitting

The largest limitation is that the public sessions and their target distribution were inspected before Experiment 11. The reported score is excellent evidence that the architecture fits this simulator, but it is not sufficient evidence of private-test or real-user generalization.

### Popularity-distribution dependence

The rating-count prior exploits a strong public target-popularity skew. It could become neutral or harmful if private targets are sampled uniformly, emphasize long-tail products, or use another marketplace distribution.

### Template-oriented parsing

The parser is robust to case, whitespace, punctuation, and straight/curly apostrophes for known evaluator templates. It is not a general natural-language state tracker. Arbitrary paraphrases, negation, corrections phrased differently, and multiple semantic clauses may be missed.

### Untyped constraints

All active preferences are stored as positive strings. There is no structured representation for color, material, size, budget, brand, style, exclusions, strength, or hard-versus-soft preference.

### Candidate truncation and insertion-order effects

The strict AND stage takes the first 1,000 matches in catalog row order. Weighted BM25 is used only for the soft OR expansion when fewer than 30 strict candidates exist. Better full-pool scoring may improve quality or reveal that some current gain is catalog-order-dependent.

### Global pagination can hide newly relevant products

Global seen-item suppression performed best publicly, but a real user may add a constraint that makes a previously shown product the best answer. Query-scoped pagination avoids this failure mode and was the calibration-selected candidate.

### Static clarification sequence

The specific question policy ignores the current candidate set and disclosed slots. It can ask irrelevant questions or revisit attribute types. Candidate-entropy or expected-information-gain selection could be more data-efficient if evaluated without leakage.

### Limited product features in Python reranking

The reranker's `searchable_bag` uses only title, categories, and the first three features, even though FTS indexes more fields. Details, store, later features, and description influence FTS expansion but not the term-match component of the final score.

### No price-specific handling

There is no dedicated numeric price column, normalized budget representation, or hard budget filter in the clean agent.

### Local latency variability

Median response time is low, but the measured p95/p99 varied across repeated runs. The current code executes Python loops over as many as 1,000 candidates and performs per-item string checks, leaving room for optimization.

## Recommended next research experiments

The next experiments should be registered before looking at new labels and evaluated on private or newly generated sessions.

1. **Validate generalization first.** Freeze both global and query-scoped variants and compare them on new sessions without further tuning.
2. **Remove or recalibrate popularity.** Test weights `0`, `0.005`, `0.01`, and `0.02`, stratified by target popularity decile.
3. **Rank every candidate with FTS5 BM25.** Eliminate insertion-order ranking from the strict AND stage and compare against the current implementation.
4. **Factorial pagination/question ablation.** Evaluate global/query/none crossed with specific/other questions so their interaction is identifiable.
5. **Adaptive clarification.** Choose the attribute with greatest candidate entropy or expected rank improvement while preventing repeated/unavailable questions.
6. **Typed state and negation.** Parse material, color, size, budget, brand, style, use case, positive/negative polarity, hard/soft strength, and override lineage.
7. **Numeric budget filtering.** Normalize prices and compare hard filtering, soft penalties, and no budget logic.
8. **Long-tail validation.** Construct a target set balanced by rating-count decile to measure how much performance comes from popularity skew.
9. **Parser robustness suite.** Test paraphrases, capitalization, Unicode punctuation, reordered clauses, multiple constraints, negations, and corrections.
10. **Latency optimization.** Move more scoring into SQLite, pre-tokenize metadata, cache normalized query results, and benchmark initialization plus p50/p95 under controlled conditions.
11. **Hybrid retrieval only after clean controls.** Compare FTS5 alone with FTS5 + dense or WARP RRF using identical state, question, pagination, popularity, and diversity policies.
12. **Real interaction policy.** Evaluate whether global non-repetition frustrates users or improves discovery when relevance changes over turns.

## Reproduction and verification

From the repository root:

```powershell
python -m unittest discover -s nickolas/experiments/tests -v
python -m nickolas.experiments.run_all --only 11 --skip-baseline
```

Verification status from the latest run:

- 41 tests passed.
- Current submission control reproduced its saved score exactly.
- All Experiment 11 diagnostic gates passed.
- Candidate source snapshot matches the evaluated source hash.
- No Experiment 11 failure artifact exists.
- The starter agent was not modified by Experiment 11.

## Source and artifact map

- Candidate implementation: `nickolas/experiments/experiment_11_candidate_agent.py`
- Experiment runner: `nickolas/experiments/experiment_11_clean_fts5_candidate.py`
- Candidate tests: `nickolas/experiments/tests/test_experiment_11.py`
- Full metrics: `nickolas/results/experiment_11_clean_fts5_candidate/metrics.json`
- Human summary: `nickolas/results/experiment_11_clean_fts5_candidate/summary.md`
- Per-session outcomes: `nickolas/results/experiment_11_clean_fts5_candidate/sessions.json`
- Current-agent comparisons: `nickolas/results/experiment_11_clean_fts5_candidate/comparisons.json`
- Latency measurements: `nickolas/results/experiment_11_clean_fts5_candidate/latency.json`
- Comparison chart: `nickolas/results/experiment_11_clean_fts5_candidate/agent_comparison.png`
- Evaluated candidate snapshot: `nickolas/results/experiment_11_clean_fts5_candidate/candidate_agent_snapshot.py`
- Current deployed submission: `techjam-conversational-search/starter/agent.py`
- Official local evaluator: `techjam-conversational-search/evaluator/local_evaluator.py`

## Context to give another research model

The following is the concise framing another model should use:

> I am researching a conversational product-retrieval agent over 50,000 catalog items. My best observed public-set architecture is a deterministic CPU-only SQLite FTS5 agent scoring 0.899530. It parses category and active free-text constraints, correctly removes a revoked initial preference on intent override, generates up to 1,000 lexical candidates, applies weighted field retrieval, reranks using term coverage plus a weak `rating_number^0.1` prior, diversifies by brand/title similarity, globally suppresses already shown products, and cycles through specific clarification attributes. It achieves 100% HitRate@10, 0.717101 MRR, and 1.78 MTTC on 200 public sessions. A query-scoped pagination version was selected on a frozen calibration partition and scored 0.894362 overall. The public data were inspected, targets are heavily popularity-skewed, and no production promotion is authorized without new-data validation. Help me design leakage-safe ablations and improvements that distinguish genuine retrieval/state/policy gains from evaluator-template and popularity-distribution overfitting.

