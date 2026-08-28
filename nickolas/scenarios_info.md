# How the Evaluation Scenarios Actually Work

This document describes the behavior implemented by the provided local evaluator. It is based on `evaluator/local_evaluator.py`, the evaluator tests, the API contract, the evaluation config, and the competition specification.

## Short answer: is the response deterministic?

Yes. The simulated customer's behavior is deterministic for a given sample.

There is no LLM or semantic judge deciding whether the agent's natural-language question is good. The evaluator does not interpret the meaning of the agent's `message`. It uses the structured `ask_attribute` value to select a predefined reply.

A response such as:

```json
{
  "message": "Would you prefer cotton or leather?",
  "ask_attribute": "material",
  "recommendations": [{"parent_asin": "B000000000"}]
}
```

is processed as a request for `material`. The wording in `message` does not determine what the customer reveals.

The natural-language `message` still must be a string. If the response is not a dictionary, or `message` is not a string, the evaluator replaces the entire response with an empty fallback. That loses the question and all recommendations for that turn.

## What the evaluator knows

For each session, the evaluator knows:

- the exact target `parent_asin`;
- the target product's catalog record;
- the scenario type;
- a hidden intent card containing up to two hard constraints and up to two soft preferences;
- for Intent Override, the override turn and old/new preferences.

The agent does not receive the target or hidden intent card. At reset it receives only:

- a random session ID;
- the aggregate user profile.

On each turn it receives only:

- the session ID;
- the current deterministic customer message;
- the turn number;
- `top_k`, which is always 10.

## How the hidden intent card is constructed

When the public sample does not already contain hidden fields, the evaluator derives them from the target product.

It gathers candidates from the product's `features` followed by its `details`. It also searches the full product text for a recognized material and color:

1. The first recognized material is inserted at the front.
2. The first recognized color is inserted immediately after it.
3. A budget constraint is appended when the product has a price.
4. Duplicate strings are removed while preserving order.
5. Each constraint is cleaned and truncated to 180 characters.
6. The first two candidates become hard constraints.
7. Candidates three and four become soft preferences.

If no candidates exist, the product title is used. If there are no third and fourth candidates, the first candidate is reused as a soft preference.

This construction matters because the hidden constraints can be raw, long feature strings rather than neat human-authored slots.

## What makes a question "good"

The evaluator only considers `ask_attribute`. Its allowed values are:

```text
category, material, color, size, style, brand,
budget, feature, use_case, other, null
```

After a normal turn, the evaluator collects all hard constraints followed by all soft preferences. It removes constraints already disclosed, finds up to two that match the requested attribute, marks them disclosed, and returns them.

A useful question is therefore one whose `ask_attribute` matches one or more undisclosed constraints. There is no direct score for asking a good question. A useful question only helps indirectly by improving later recommendations.

### Deterministic reply table

| Agent output | Deterministic customer behavior |
|---|---|
| `ask_attribute: null` or a non-string value | `Those options are not quite right yet. Ask me about one specific attribute.` |
| A recognized attribute with a match | Reveals up to two undisclosed matching constraints. |
| A recognized attribute without a match | Says there is no additional preference for that attribute. |
| An unrecognized string | The evaluator converts it to `other`. |
| `ask_attribute: "other"` | Reveals up to two undisclosed constraints of any classified type. |

Under the API schema, an official response should use only the allowed values. Although the local evaluator converts an unknown string to `other`, relying on this is unsafe because schema validation may reject the response before evaluation. Use the explicitly allowed `other` value instead.

### How constraints are classified

Classification is rule-based and checked in this order:

1. `budget`: contains `budget`, or a pattern such as `$50`, `under 50`, or `<=50`.
2. `material`: contains cotton, polyester, nylon, leather, wool, spandex, silk, rayon, or fabric.
3. `color`: contains `color`, black, white, blue, red, pink, or green.
4. `size`: contains size, sizing, width, wide, or narrow.
5. `style`: contains department, style, fit, sleeve, or neck.
6. `use_case`: contains hiking, running, gym, winter, outdoor, or work.
7. Anything else becomes `feature`.

The first matching rule wins. For example, a long feature containing both `cotton` and `size` is classified as `material` because material is checked first.

Important implementation quirks:

- `category` and `brand` are allowed questions, but `classify_constraint()` never assigns either classification. They normally reveal nothing.
- Brown, gray, grey, purple, yellow, and orange can be extracted when the intent card is created, but the later constraint classifier does not recognize those words as colors unless the constraint also contains the literal word `color`.
- `feature` catches everything that does not match an earlier rule.
- `other` bypasses classification and reveals the first two undisclosed constraints. In this local evaluator, it is the broadest information request.

## Turn loop shared by every scenario

Every session follows this loop for at most 10 turns:

1. The evaluator sends the current customer message to `Agent.respond()`.
2. The agent returns `message`, `ask_attribute`, and ranked recommendations.
3. Recommendations are normalized.
4. If scoring is currently allowed and the target appears, the session immediately succeeds and stops.
5. If this is turn 10, the session stops as a miss.
6. Otherwise, the evaluator creates the next customer message from the scenario rules and `ask_attribute`.

Recommendations are therefore evaluated before the customer's reply to the question is generated. Information obtained by a question can only improve recommendations on the following turn.

## Scenario behavior

### Buying: 40% of sessions

Initial message:

```text
I'm looking for <coarse category>. A key requirement is: <first hard constraint>.
```

The first hard constraint is immediately marked as disclosed. The target can score starting on turn 1.

After an unsuccessful turn, the normal deterministic question/reply logic applies. Asking `other` will reveal up to two remaining undisclosed constraints; it will not repeat the constraint already present in the initial message.

Buying is usually the easiest scenario because the first request contains category information and a target-derived hard constraint.

### Browsing: 40% of sessions

Initial message:

```text
I'm looking for <coarse category>, but I'm still exploring.
```

No hidden constraint is initially disclosed. The target can still score on turn 1 if retrieval from the category/profile happens to find it.

If the turn does not hit, the evaluator answers according to `ask_attribute`. This is the main scenario where clarification strategy matters: an informative question on turn 1 reveals information for the turn-2 recommendation.

### Intent Override: 15% of sessions

The evaluator creates:

- `old_value`: normally the last soft preference;
- `new_value`: normally the first hard constraint;
- override turn: deterministically selected as turn 3 or turn 4 from a random generator seeded with `sample_id` and `scenario_type`.

Initial message:

```text
I'm looking for <coarse category>. <old value>
```

Most importantly, scoring is disabled until the override is sent. If the target appears in recommendations before the override, it is ignored: no hit, rank, or early conversion is recorded.

Immediately before the configured override turn, the evaluator skips the normal answer to the previous `ask_attribute` and instead sends:

```text
Actually, ignore my earlier preference. What I need is: <new value>.
```

The new value is marked disclosed. Scoring is enabled on the turn when the agent receives this message. Thus:

- override at turn 3 means turns 1 and 2 cannot score, and turn 3 is the earliest hit;
- override at turn 4 means turns 1 through 3 cannot score, and turn 4 is the earliest hit.

The evaluator itself does not delete the old preference from the agent's memory. Correct override handling is the agent's responsibility. After the override turn, normal deterministic question/reply behavior resumes.

Questions immediately before the override are effectively wasted from an information-gathering perspective because the forced override message replaces the normal answer. Recommendations on those turns are also unable to score.

### Boundary: 5% of sessions

Initial message:

```text
I'm looking for <coarse category>, but I'm still exploring.
```

The first time the agent supplies any string-valued `ask_attribute`, the simulator refuses to give a preference:

```text
I don't have a preference for <attribute>; please use your judgment.
```

This consumes the one-time boundary behavior even when the attribute is unknown. It does not reveal or mark any hidden constraint as disclosed.

After that first refusal, subsequent questions use the normal matching logic. If the first response uses `ask_attribute: null`, the boundary event is not consumed; the normal prompt asking for a specific attribute is returned instead.

The target can score on any turn, including turn 1. The boundary response only happens after an unsuccessful recommendation turn.

## How recommendations are normalized and scored

The evaluator accepts a recommendation as either a dictionary containing `parent_asin` or, locally, a raw value. The official schema requires dictionaries.

It then:

1. converts the value to a stripped string;
2. discards empty IDs;
3. discards IDs absent from the frozen catalog;
4. removes duplicates while preserving the first occurrence;
5. keeps the first 10 valid unique IDs.

The optional numeric `score` field is ignored. Only list order matters.

If the target is present, its position in this normalized list determines reciprocal rank. Since invalid IDs and duplicates are removed before ranking, their original positions do not count, but submitting them is still unhelpful.

## Technical score

Across all sessions:

```text
HitRate@10 = successful sessions / number of sessions
MRR        = average reciprocal rank; a miss contributes 0
MTTC       = average first-hit turn; a miss contributes turn 11
Efficiency = clip((11 - MTTC) / 10, 0, 1)

TechnicalScore = 0.50 * HitRate@10
               + 0.30 * MRR
               + 0.20 * Efficiency
```

There is no separate question-quality metric. Asking questions affects the score only through whether the later recommendation hits, how highly it ranks, and how many turns it takes.

The fixed scenario mix is:

| Scenario | Share | Public sessions out of 200 |
|---|---:|---:|
| Buying | 40% | 80 |
| Browsing | 40% | 80 |
| Intent Override | 15% | 30 |
| Boundary | 5% | 10 |

All sessions enter the overall metrics equally. Per-scenario metrics are reported for diagnosis; they are not separately added to the score. Because of the fixed proportions, the overall score is equivalently the weighted average of the four scenario scores before rounding.

For one session, its contribution before averaging can be viewed as:

```text
successful session: 0.50 + 0.30 / rank + 0.02 * (11 - hit_turn)
miss:               0
```

Examples:

| Result | Per-session value before dataset averaging |
|---|---:|
| Rank 1 on turn 1 | 1.00 |
| Rank 1 on turn 3 | 0.96 |
| Rank 1 on turn 4 | 0.94 |
| Rank 10 on turn 4 | 0.67 |
| Miss | 0.00 |

Intent Override therefore cannot achieve the turn-1 maximum because scoring is deliberately locked until turn 3 or 4.

## What does not affect the core score

- The evaluator does not semantically grade `message`.
- The optional recommendation `score` numbers are ignored.
- Reported token usage is accumulated but does not change the technical score.
- The aggregate user profile does not change the simulator's deterministic replies. It is merely information the agent may use for retrieval or ranking.
- A question has no direct reward or penalty. Its only cost is potentially using a turn without enough retrieval benefit.

## Practical implications for an agent

1. Always return a valid string in `message`, even if the retrieval system does not need natural-language generation.
2. Recommend products on every turn. A question and recommendations can be returned together, and recommendations are checked before the next reply.
3. In Buying, exploit the disclosed hard constraint immediately.
4. In Browsing, `other` is the broadest local-evaluator question and can reveal up to two constraints at once.
5. Do not expect `category` or `brand` to reveal generated constraints under the current classifier.
6. In Boundary, expect the first actual question to be refused; continue retrieval and ask again later if needed.
7. In Intent Override, recognize the explicit `Actually, ignore my earlier preference` message, remove or strongly downweight stale state, and rerank immediately.
8. Do not rely on a pre-override target recommendation being remembered by the evaluator. It is ignored and must be recommended again after the override.
9. Rank quality matters in addition to inclusion: rank 1 gives much more MRR credit than rank 10.
10. Earlier success matters, but a later hit is substantially better than a miss.

## Simplified pseudocode

```python
for sample in samples:
    reset_agent(profile)
    build_hidden_intent_from_target_product()
    message = scenario_initial_message()
    scoring_allowed = scenario != "intent_override"

    for turn in range(1, 11):
        response = agent.respond(message, turn, top_k=10)
        ranked = first_10_valid_unique_catalog_ids(response.recommendations)

        if scoring_allowed and target in ranked:
            record_hit(turn, rank)
            break

        if turn == 10:
            record_miss_as_turn_11()
            break

        if override_is_due_next_turn:
            scoring_allowed = True
            message = forced_override_message
        else:
            message = deterministic_reply_to(response.ask_attribute)
```

## Bottom line

The customer is deterministic, and question wording is not judged. The structured `ask_attribute` controls which hidden constraint—if any—is revealed. Question quality is therefore an information-acquisition decision: choose an attribute that exposes useful target-derived text, then use that text to place the exact target `parent_asin` in the Top 10 as early and as highly as possible.
