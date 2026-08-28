# Revised Technical Understanding and System Direction

## Executive summary

The provided evaluator changes how we should understand this challenge. It initially appears to be a broad conversational-commerce problem: identify whether a customer is buying or browsing, ask intelligent natural-language questions, track changing intent, and recommend suitable products.

The implemented task is more specific:

> Given a coarse product category and progressively disclosed snippets derived from a hidden catalog item, maintain the correct session state and retrieve the exact target `parent_asin` in the Top 10 as early and as highly ranked as possible.

The simulated customer is deterministic. No model judges whether the natural-language question is intelligent, relevant, or well written. The evaluator uses the structured `ask_attribute` field to decide what information to reveal next. The real technical problem is therefore a combination of:

1. active information acquisition;
2. stateful constraint management;
3. exact product retrieval and reranking.

Natural-language generation and psychological intent classification are secondary to these three capabilities.

## The old and revised problem models

| Initial interpretation | Evaluator-aligned interpretation |
|---|---|
| Understand open-ended shopping intent | Recover a hidden product from progressively disclosed evidence |
| Judge whether someone is Buying or Browsing | Recognize how much target evidence is currently available |
| Generate high-quality conversational questions | Select a useful structured `ask_attribute` action |
| Recommend generally appropriate products | Return the one exact hidden catalog ASIN |
| Treat the conversation as unstructured text | Maintain structured active, stale, and disclosed constraints |
| Use an LLM as the central reasoner | Use deterministic state logic and strong lexical/hybrid retrieval |
| Optimize helpfulness and naturalness | Optimize HitRate@10, reciprocal rank, and time to conversion |

This is best viewed as a partially observable product-identification problem. The conversation is the protocol through which evidence about the hidden product is released.

```text
Hidden target product
        ↓
Evaluator derives hidden constraints from its catalog record
        ↓
Scenario controls which evidence is initially disclosed
        ↓
Agent retrieves candidates and requests more evidence
        ↓
Evaluator deterministically reveals matching constraints
        ↓
Agent updates state, retrieves again, and ranks the exact ASIN
```

## What the evaluator secretly constructs

Each sample contains a ground-truth `parent_asin`. If a hidden intent card is not already present, the local evaluator derives one directly from the target product's catalog metadata.

It gathers candidate constraints from product features and details, then searches the full product text for recognized material and color terms. A product price becomes a budget candidate. After cleaning and deduplication:

- the first two candidates become hard constraints;
- candidates three and four become soft preferences;
- if too few candidates exist, earlier values may be reused.

A simplified hidden card might look like:

```text
Hard constraints:
1. leather
2. color: black

Soft preferences:
3. waterproof construction
4. Department: womens
```

These are often close to, or exactly copied from, the target product record. This makes disclosed constraint text extremely useful for lexical retrieval. The hidden intent is not necessarily a natural, independently authored customer preference; it may be a long raw feature string from the catalog.

## What the agent sees

At the start of a session, `reset()` receives:

- a random session ID;
- an aggregate user profile.

The user profile contains purchase-frequency and rating summaries plus preference tags. It may be used as a weak personalization or ranking signal, but it does not control the simulator's replies.

On every turn, `respond()` receives:

- the session ID;
- the current customer message;
- the turn number;
- `top_k`, always set to 10.

The agent returns:

```json
{
  "message": "What requirements matter most to you?",
  "ask_attribute": "other",
  "recommendations": [
    {"parent_asin": "B000000000"}
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0
  }
}
```

The `message` must be a string, but its semantics are not evaluated. If the response is not a dictionary or `message` is not a string, the local evaluator replaces the entire response with an empty fallback, causing both the question and recommendations to be lost for that turn.

## How questions actually work

There is no semantic question-quality assessor. These responses are operationally equivalent:

```json
{
  "message": "Do you prefer a particular material?",
  "ask_attribute": "material"
}
```

```json
{
  "message": "Any words can be placed here.",
  "ask_attribute": "material"
}
```

Both request material information because only `ask_attribute` controls the deterministic reply.

The allowed values are:

```text
category, material, color, size, style, brand,
budget, feature, use_case, other, null
```

After an unsuccessful recommendation turn, the evaluator uses that field as follows:

| `ask_attribute` | Customer behavior |
|---|---|
| `null` or a non-string | Asks the agent to request one specific attribute |
| Recognized value with matches | Reveals up to two undisclosed matching constraints |
| Recognized value without matches | States there is no additional preference for it |
| Unknown string in the local evaluator | Converts it to `other` |
| `other` | Reveals up to two undisclosed constraints regardless of classification |

Unknown strings should not be deliberately used because the official schema only permits the enumerated values. The explicitly allowed `other` value already provides the broad behavior.

### Constraint classification

Hidden constraint strings are classified by ordered keyword rules:

1. `budget`
2. `material`
3. `color`
4. `size`
5. `style`
6. `use_case`
7. otherwise `feature`

The first matching rule wins. A long constraint containing both `cotton` and `size`, for example, is treated as material.

There are several important quirks:

- `category` and `brand` are accepted question values, but the classifier never assigns a generated constraint to either class.
- Some colors can be extracted while constructing the intent card but are not recognized by the later constraint classifier unless the string also says `color`.
- `feature` is the fallback classification for unmatched text.
- `other` bypasses classification and selects the first two remaining constraints.

In the supplied evaluator, `other` is consequently the broadest and usually most reliable information request.

## Order of operations on every turn

The evaluator does not answer the agent's question before checking its recommendations. Instead, it follows this order:

1. Call `Agent.respond()` with the current message.
2. Validate and normalize the returned recommendations.
3. Check whether the target is present and currently allowed to score.
4. End the session immediately on a valid hit.
5. If this was turn 10, end as a miss.
6. Otherwise, construct the next customer message using the scenario and `ask_attribute`.

Therefore, information requested on turn 1 only becomes available for retrieval on turn 2. The agent should not choose between asking and recommending: it should do both on every useful turn.

```json
{
  "message": "What matters most to you?",
  "ask_attribute": "other",
  "recommendations": [
    {"parent_asin": "current-best-1"},
    {"parent_asin": "current-best-2"}
  ]
}
```

The recommendations create an immediate chance to score, while the structured question improves the next turn if the current list misses.

## Scenario mechanics and implications

### Buying: 40%

The initial message contains a coarse category and the first hard constraint:

```text
I'm looking for <category>. A key requirement is: <first hard constraint>.
```

The target can score from turn 1. The disclosed hard constraint is marked as already known and will not be repeated in later replies.

Practical behavior:

1. Extract the category and hard-constraint phrase.
2. Immediately run precise category-aware retrieval.
3. Return the best 10 candidates on turn 1.
4. Ask `other` at the same time to acquire additional evidence if the first list misses.

Buying should generally be the easiest scenario because useful target-derived text is available before the first recommendation.

### Browsing: 40%

The initial message only provides a coarse category:

```text
I'm looking for <category>, but I'm still exploring.
```

The target can technically score on turn 1, but there is no disclosed constraint to distinguish it from other products in the same category.

Practical behavior:

1. Run broad category retrieval and return 10 candidates rather than wasting the turn.
2. Ask `other` simultaneously.
3. Parse the target-derived constraints in the reply.
4. On turn 2, search using the category plus the revealed phrases.
5. Continue accumulating evidence if needed.

Browsing is not fundamentally a different retrieval engine. It is the same product-identification process beginning with less evidence.

### Intent Override: 15%

The session starts with a category and an old soft preference. A forced message on deterministic turn 3 or 4 replaces it with the first hard constraint:

```text
Actually, ignore my earlier preference. What I need is: <new hard constraint>.
```

The override turn is selected from 3 or 4 by a random generator seeded with the sample ID and scenario type, so it remains stable for a particular sample.

The critical scoring rule is that the target cannot score before the override is delivered. Even an exact rank-1 recommendation on an earlier turn is ignored. It must be returned again after scoring becomes active.

The forced override also replaces the normal answer to the question immediately preceding it. That question may therefore acquire no information.

Practical behavior:

1. Maintain old preferences separately from stable category context.
2. Detect the explicit override phrase.
3. Delete or strongly downweight the stale preference.
4. Make the new requirement authoritative.
5. Rerun retrieval immediately on the override turn.
6. Recommend promising candidates again, even if they appeared before the override.

This is primarily a state-transition problem rather than nuanced psychological intent detection.

### Boundary: 5%

Boundary begins with the same vague message as Browsing. The first time the agent returns any string-valued `ask_attribute`, the simulator refuses:

```text
I don't have a preference for <attribute>; please use your judgment.
```

This is a one-time refusal. It does not disclose or consume any hidden constraint. Subsequent structured questions use the normal matching logic.

If `ask_attribute` is `null`, the boundary event is not consumed; the evaluator simply asks the agent to request a specific attribute.

Practical behavior:

1. Return broad category recommendations and ask `other` on turn 1.
2. Recognize the refusal as a boundary event, not proof that no constraints exist.
3. Continue recommending.
4. Ask `other` again; the later request can reveal hidden constraints.

## Buying and Browsing do not require complex intent classification

The evaluator does not directly award points for correctly labelling a session Buying or Browsing. Those labels mainly determine the initial evidence level:

```text
Buying   = category + one hard constraint
Browsing = category only
```

A simple rule-based parser can recognize the supplied templates:

```python
if "A key requirement is:" in message:
    evidence_state = "hard_constraint_available"
elif "but I'm still exploring" in message:
    evidence_state = "needs_clarification"
```

Both can then use one retrieval stack. The system should route based on available evidence, not attempt to infer a deep psychological state.

For robustness, these rules should tolerate paraphrases rather than depend solely on exact strings, because organizer documentation allows natural-language paraphrasing as long as it does not decide correctness.

## Recommendation normalization

Only exact catalog IDs score. The evaluator normalizes recommendation output by:

1. reading `parent_asin` from each dictionary;
2. stripping and converting it to a string;
3. dropping empty or non-catalog IDs;
4. removing duplicates while preserving the first occurrence;
5. taking the first 10 valid unique IDs.

The optional numeric recommendation `score` is ignored. List order is the score-bearing rank.

The agent should always return valid, unique catalog ASINs ordered from best to worst. Candidate inclusion matters, but ordering also has substantial value through MRR.

## Technical score

The evaluator calculates:

```text
HitRate@10 = successful sessions / N
MRR        = mean reciprocal target rank, with misses equal to 0
MTTC       = mean first-hit turn, with misses assigned turn 11
Efficiency = clip((11 - MTTC) / 10, 0, 1)

TechnicalScore = 0.50 * HitRate@10
               + 0.30 * MRR
               + 0.20 * Efficiency
```

All sessions are included equally in these overall metrics. The fixed scenario proportions are:

| Scenario | Weight in dataset | Public count |
|---|---:|---:|
| Buying | 40% | 80 of 200 |
| Browsing | 40% | 80 of 200 |
| Intent Override | 15% | 30 of 200 |
| Boundary | 5% | 10 of 200 |

The evaluator reports per-scenario metrics for diagnosis, but does not separately add them to the total. With a fixed mix, the overall score is equivalent to the proportionally weighted average of scenario scores before rounding.

The contribution of a single successful session before dataset averaging can be expressed as:

```text
0.50 + 0.30 / target_rank + 0.02 * (11 - hit_turn)
```

A miss contributes zero under this equivalent formulation.

| Outcome | Per-session value |
|---|---:|
| Rank 1, turn 1 | 1.00 |
| Rank 1, turn 3 | 0.96 |
| Rank 1, turn 4 | 0.94 |
| Rank 10, turn 4 | 0.67 |
| Miss | 0.00 |

The optimization priority should be:

1. turn misses into Top-10 hits;
2. move the target toward rank 1;
3. reduce the number of turns.

Aggressive filtering that produces a beautiful ranking on easy sessions but removes the target on harder sessions is likely harmful. Constraints should often be strong boosts rather than unconditional filters because catalog-derived constraint strings can be noisy.

## What does not directly change the score

- Natural-language question quality is not semantically assessed.
- Optional recommendation score values are ignored.
- Token usage is reported but does not enter the technical score.
- The profile does not control simulator replies.
- Asking a question has no direct reward.
- Correctly naming a scenario has no direct reward.

Questions and state handling matter only insofar as they improve target inclusion, target rank, or conversion turn.

## Revised system architecture

The system should be built around a structured session state and a strong retrieval stack:

```text
Customer message
      ↓
Deterministic/tolerant message parser
      ↓
Session state
  ├─ category
  ├─ active hard constraints
  ├─ active soft preferences
  ├─ stale/overridden preferences
  ├─ negative or no-preference signals
  ├─ boundary refusal status
  └─ override status
      ↓
Candidate generation
  ├─ category-aware BM25
  ├─ exact phrase and rare-token retrieval
  └─ optional dense retrieval for recall
      ↓
Constraint-aware reranker
      ↓
Top 10 valid unique ASINs
      ↓
Deterministic question policy
```

### Candidate generation

Index all useful catalog fields:

- title;
- features;
- description;
- details;
- categories;
- store;
- price where applicable.

BM25 or another lexical method should be a central retriever because disclosed phrases often originate from the target record itself. Dense retrieval can be added as a complementary recall route, particularly for paraphrased messages, but should not erase exact-match signals.

### Constraint-aware reranking

A sensible ranking priority is:

1. coverage of active hard constraints;
2. overlap with rare disclosed phrases;
3. correct category;
4. coverage of the newest override constraint;
5. soft-preference coverage;
6. aggregate-profile compatibility;
7. rating/popularity as weak tie-breakers.

Old preferences should not compete equally with the new intent after an override.

### Question policy

For the supplied evaluator, a simple policy is likely competitive:

```python
if useful_constraints_may_remain:
    ask_attribute = "other"
else:
    ask_attribute = None
```

Every such response should still contain the current best recommendations.

If a boundary refusal occurs, the agent should not conclude that the user has no hidden requirements. It can request `other` again on a later turn.

### Role of an LLM

An LLM may help with:

- robust constraint extraction from paraphrased messages;
- synonym normalization;
- semantic candidate recall;
- reranking ambiguous products;
- customer-facing wording.

It is not obviously necessary for:

- deciding that Buying contains an initial constraint;
- generating the question prose;
- selecting `other`;
- detecting the explicit override;
- maintaining session state.

A deterministic offline baseline using parsing, BM25, constraint scoring, and a small state machine should be developed first. Additional model complexity should be accepted only when measured public-set experiments improve HitRate, MRR, or MTTC.

## Recommended per-turn policy

```text
On reset:
    create clean structured state for the session

On each customer message:
    1. Parse category and newly disclosed constraints.
    2. Detect a boundary refusal.
    3. Detect an override and invalidate stale preference state.
    4. Build a query from the category and active evidence.
    5. Generate a broad candidate set.
    6. Rerank by active constraint coverage and exact overlap.
    7. Return the best 10 valid unique ASINs.
    8. Request more evidence with ask_attribute = "other" when useful.
```

For Intent Override:

```text
If the message says to ignore an earlier preference:
    mark the old preference stale
    promote the new requirement to authoritative hard evidence
    reretrieve immediately
    resubmit strong candidates even if they appeared earlier
```

For Boundary:

```text
If the message says there is no preference for the requested attribute:
    record the refusal
    do not add it as a positive constraint
    continue retrieval
    request other again later if evidence is still needed
```

## Concrete implementation priorities

1. Establish a reproducible local evaluation baseline and retain per-session results.
2. Build field-aware lexical indexes over the frozen catalog.
3. Implement reliable parsing of initial constraints and deterministic replies.
4. Maintain structured state by session ID.
5. Add explicit override invalidation.
6. Return recommendations on every turn while requesting more evidence.
7. Use `other` as the default broad clarification in local experiments.
8. Add phrase-overlap and constraint-coverage reranking.
9. Analyze results separately for Buying, Browsing, Intent Override, and Boundary.
10. Add semantic or LLM components only where ablations show measurable gains.

## Risks and cautions

The strongest local-evaluator exploit is that `ask_attribute: "other"` reveals up to two arbitrary undisclosed constraints. We can use this behavior because `other` is explicitly allowed by the API contract, but the entire system should not depend on exact response wording. The organizer may paraphrase messages, and the private set contains different targets.

The robust general strategy is therefore:

- acquire structured evidence efficiently;
- parse target-derived text tolerantly;
- keep active state separate from stale state;
- use lexical and semantic evidence together;
- always preserve exact-ASIN recall.

We should also avoid overestimating the meaning of the public intent cards. Since they are generated from product metadata, public-set improvements may reward exact feature matching more heavily than real-world conversational generalization. This is acceptable for the competition objective, but it should be acknowledged when describing broader product applicability.

## Final position

Our model of the challenge should shift from a general conversational shopping assistant to a stateful hidden-item retrieval agent.

The winning loop is:

> Recommend immediately, request structured evidence at the same time, treat disclosed phrases as high-value target clues, invalidate stale intent on override, and repeatedly rerank the exact catalog until the hidden ASIN appears near the top.

Conversation remains the interface, but retrieval, state correctness, and evidence handling are the actual scoring engine.
