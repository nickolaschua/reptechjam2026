# TechJam 2026 Evaluator and Starter Agent Briefing

This is a self-contained technical briefing for anyone brainstorming or designing our solution. It documents what the checked-in evaluator actually does, how the starter behaves, the score we reproduced locally, and the most important consequences for agent design.

## 1. Challenge in one paragraph

Build a Python conversational shopping agent that identifies one hidden target product from a frozen catalog of 50,000 Amazon `Clothing_Shoes_and_Jewelry` products. Each evaluation session lasts at most 10 turns. On every turn the agent may ask for one structured attribute and may return an ordered list of product IDs. A session ends as soon as the hidden target appears in the scored Top 10 (except for a special intent-override rule described below). The public development set contains 200 labeled sessions; the organizer has 800 private sessions. Public and private splits use different users and targets.

## 2. Source of truth in this repository

The runnable participant kit is under `techjam-conversational-search-participant-kit/`:

- `starter/agent.py`: editable weak BM25 baseline.
- `evaluator/local_evaluator.py`: deterministic local simulator and scorer.
- `data/catalog.jsonl`: the 50,000-product catalog (downloaded locally and ignored in this kit).
- `data/public_set.jsonl`: 200 public labeled sessions.
- `docs/evaluation_config.json`: metric constants and weights.
- `docs/agent_api_contract.json`: request/response schema.
- `docs/baseline_results.json`: published weak-baseline result.
- `results.json`: generated local evaluation details (ignored by Git).

Run from the participant-kit directory with:

```bash
python -m evaluator.local_evaluator
```

The evaluator imports `Agent` directly from `starter.agent`, evaluates all sessions, writes `results.json`, and prints aggregate results.

## 3. Exact agent interface

The evaluator constructs the agent once, then calls `reset` for each new randomized session ID:

```python
class Agent:
    def __init__(self, catalog_path="data/catalog.jsonl") -> None:
        ...

    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000...", "score": 0.9}
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
            },
        }
```

`top_k` is always 10. Allowed `ask_attribute` values are:

```text
category, material, color, size, style, brand, budget,
feature, use_case, other, or null
```

Important validation behavior:

- `message` must exist and be a string. If the response is not a dictionary, or `message` is not a string, the entire response is replaced with an empty invalid response.
- Recommendations may technically contain up to 100 entries under the JSON schema, but the evaluator keeps only the first 10 valid, unique catalog IDs.
- A recommendation may be an object containing `parent_asin` or, in the local evaluator's normalization function, a raw ID value. Use the documented object format for official compatibility.
- Duplicate IDs and IDs absent from the frozen catalog are silently discarded without consuming a scored position.
- The optional numeric `score` is ignored. List order is the ranking.
- Nonnegative integer token counts are accumulated when supplied. Token use does not affect the core technical score.
- Any exception from `respond` is caught and treated as an empty response for that turn. Do not rely on exceptions being visible.

## 4. What the agent can see

At reset, the agent receives only a safe aggregate profile with:

- `purchase_frequency`
- `average_prior_rating`
- `rating_style`
- `preference_tags`
- `summary`

At each turn it receives only the random `session_id`, current simulated `user_message`, 1-based `turn`, and `top_k=10`.

The agent is **not** passed the scenario label, target ASIN, difficulty bucket, hidden intent card, or simulator state. The public JSONL contains labels for offline evaluation and analysis, but using those labels as a lookup table would overfit the public set and cannot generalize to the 800 private sessions.

Catalog fields available for retrieval are:

```text
parent_asin, title, features, description, price, categories,
details, average_rating, rating_number, store
```

Only exact `parent_asin` equality is scored.

## 5. Exact deterministic conversation simulation

There is no LLM customer in the local evaluator. Customer text is generated deterministically from metadata belonging to the hidden target.

### 5.1 Hidden intent-card construction

If a session does not already contain hidden intent fields, the evaluator derives them from the target catalog product:

1. It builds searchable text from `title`, `features`, `details`, `description`, `categories`, and `store`.
2. It flattens all `features`, followed by all `details`, into candidate constraint strings.
3. If a recognized material occurs anywhere in the searchable text, the first material match is inserted at the front.
4. If a recognized color occurs, `color: <color>` is inserted immediately after that.
5. If price exists, `budget around $<price>` is appended at the end.
6. It cleans, de-duplicates, and truncates each constraint to 180 characters.
7. The first two constraints become `hard_constraints`.
8. The next two become `soft_preferences`; if none exist, the first constraint is reused.

Recognized material terms are:

```text
cotton, polyester, nylon, leather, wool, spandex, silk, rayon, fabric
```

Recognized colors for insertion are:

```text
black, white, blue, red, pink, green, brown, gray, grey,
purple, yellow, orange
```

Because material and color are inserted before raw feature/detail strings, they often become the earliest and therefore hard constraints.

### 5.2 Initial message

The initial category is a coarse string made from the last two non-generic category components of the target product.

- **Buying:** `I'm looking for <category>. A key requirement is: <first hard constraint>.` That first hard constraint is marked disclosed.
- **Browsing:** `I'm looking for <category>, but I'm still exploring.`
- **Boundary:** same vague initial form as Browsing.
- **Intent Override:** `I'm looking for <category>. <old soft preference>`

### 5.3 How questions actually work

The evaluator does **not** inspect or understand the agent's natural-language `message`. Only the structured `ask_attribute` controls the next customer reply. The prose still needs to be a valid string, but eloquent question wording has no scoring effect in this local harness.

For ordinary replies:

- If `ask_attribute` is null, empty, or not a string, the customer says: `Those options are not quite right yet. Ask me about one specific attribute.`
- If the string is not an allowed attribute, the evaluator silently treats it as `other`.
- The evaluator combines hard constraints followed by soft preferences, removes already disclosed constraints, and selects at most two whose rule-based classification matches the requested attribute.
- If matches exist, the reply is: `For that, what matters is: <constraint 1>; <constraint 2>.`
- If no match exists, the reply is: `I don't have an additional preference for <attribute>.`

Constraint classification is simplistic:

- `budget`: contains `budget`, or a dollar/under/`<=` numeric pattern.
- `material`: contains one of the recognized material terms.
- `color`: contains `color`, black, white, blue, red, pink, or green. Note that several colors recognized during card construction are not included here unless the constraint literally contains the word `color`.
- `size`: contains size/sizing/width/wide/narrow.
- `style`: contains department/style/fit/sleeve/neck.
- `use_case`: contains hiking/running/gym/winter/outdoor/work.
- Everything else: `feature`.

No constraint is classified as `category` or `brand` by this implementation. Asking those will normally produce no new preference. Asking `other` is special: it matches any undisclosed constraint and reveals up to two, regardless of classification. This makes `other` a high-information action in the current local evaluator, although an official implementation could enforce the documented semantics more strictly.

Repeatedly asking an attribute whose matching constraints have already been disclosed yields no additional information.

### 5.4 Boundary behavior

On a Boundary session, the first turn on which the agent supplies any nonempty string `ask_attribute` always gets:

```text
I don't have a preference for <attribute>; please use your judgment.
```

This consumes the one-time boundary response without revealing a constraint. Later questions use the ordinary disclosure rules. If the first question is null, the special boundary response is not consumed yet.

### 5.5 Intent Override behavior

For each Intent Override session, the evaluator deterministically schedules the override for turn 3 or 4 based on sample ID and scenario. The initial message contains an old soft preference. Just before the scheduled turn, the simulator ignores the preceding `ask_attribute` and sends:

```text
Actually, ignore my earlier preference. What I need is: <new hard constraint>.
```

The new value is the first hard constraint and is marked disclosed. Most importantly, **the session is forbidden from converting before the override is delivered**. Even if the agent recommends the exact target on turn 1 or 2, the evaluator ignores that hit. After the override arrives, a target recommendation can score normally.

Therefore an agent must detect override wording, remove or downweight the contradicted old preference, incorporate the new requirement, and rerank. It should still recommend every turn because the scenario label is hidden and ordinary sessions can convert immediately.

## 6. Exactly how recommendations are scored

On each turn, the evaluator normalizes the recommendations and checks the first 10 valid unique catalog ASINs in order. On the first eligible occurrence of the target:

- `first_hit_turn` is the current turn.
- `best_rank` is its 1-based position in that turn's normalized list.
- Reciprocal rank is `1 / best_rank`.
- The session immediately ends.

Despite the field name `best_rank`, the evaluator never compares ranks across later turns because it stops at the first hit. If the target first appears at rank 10 on turn 1, the session finishes with reciprocal rank 0.1; it cannot later improve to rank 1. This creates a genuine recall-versus-ranking tradeoff when deciding what to expose early.

A miss after all 10 turns receives no reciprocal-rank credit and is assigned turn 11 for MTTC.

Metrics over all sessions:

```text
HitRate@10 = successful sessions / N
MRR = mean(1 / target_rank), with misses = 0
MTTC = mean(first_hit_turn), with misses = 11
Efficiency = clip((11 - MTTC) / 10, 0, 1)

TechnicalScore =
    0.50 * HitRate@10
  + 0.30 * MRR
  + 0.20 * Efficiency
```

Interpretation:

- 50% rewards getting the exact target anywhere in Top 10 within 10 turns.
- 30% rewards placing it nearer rank 1 on the first successful turn.
- 20% rewards finding it earlier.
- Ranking position does not change Hit Rate or MTTC, only MRR.
- Turn does not change Hit Rate or MRR, only MTTC/Efficiency.
- Recommendations and a clarification can be returned together, so asking a question does not require withholding a Top-10 list.

Metrics are also broken out by scenario. The fixed mix is:

- Buying: 40% (80 of 200 public sessions)
- Browsing: 40% (80)
- Intent Override: 15% (30)
- Boundary: 5% (10)

## 7. What the starter agent actually does

The starter is a stateless, standard-library-only sparse retrieval baseline:

- On construction, it loads all catalog products into an in-memory SQLite FTS5 table.
- Indexed text columns are title, categories, features, details, store, and description.
- It lowercases alphanumeric query tokens, removes a small stopword list and one-character tokens, preserves first occurrence order, and keeps at most 40 unique terms.
- It constructs an OR query, so matching any retained term is enough to retrieve a product.
- It orders results with SQLite `bm25` using column weights:
  - `parent_asin`: 0.0 (unindexed)
  - title: 6.0
  - categories: 4.0
  - features: 2.5
  - details: 2.5
  - store: 1.5
  - description: 1.0
- It returns the first `top_k` results.
- It always says `Here are the closest matches I found.`
- It always returns `ask_attribute: null`.
- It reports zero token usage.
- `reset` stores only the session ID. It ignores the user profile and retains no messages, constraints, prior results, or conversation state.

Consequences: the starter independently searches only the latest customer message. It forgets the initial category as soon as the simulator sends a generic correction, never elicits useful constraints, cannot combine evidence across turns, cannot handle overrides deliberately, and has no semantic or personalized ranking. Because a null question causes a generic reply after a miss, later turns usually repeat an unhelpful retrieval rather than refine it.

## 8. Reproduced starter performance

We ran the unmodified starter locally on all 200 public sessions. It completed successfully and exactly matched the published baseline:

| Metric | Result |
|---|---:|
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |
| Technical Score | 0.10671 |
| Reported tokens | 0 |

Per-scenario results:

| Scenario | N | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.2375 | 0.126508 | 8.625 |
| Browsing | 80 | 0.025 | 0.004514 | 10.75 |
| Intent Override | 30 | 0.133333 | 0.104167 | 10.066667 |
| Boundary | 10 | 0.0 | 0.0 | 11.0 |

The baseline technical score calculation is:

```text
0.50(0.125) + 0.30(0.068034) + 0.20(0.119)
= 0.1067102
= 0.10671 after rounding
```

Buying is much easier for the starter because the first message exposes a target-derived hard constraint. Browsing and Boundary begin vague, and the starter never asks a usable question. The nonzero Intent Override result comes from BM25 retrieval after the forced target-derived override message, not from actual state handling.

## 9. High-value design implications

These are reasoned implications of the verified evaluator, not additional competition rules:

1. **Always recommend and ask together.** A useful Top-10 can earn an immediate hit while a structured question sets up better evidence after a miss.
2. **Conversation memory is essential.** Accumulate the coarse category and every revealed constraint; do not search only the latest message.
3. **Optimize retrieval first.** Hit Rate carries the largest weight, and a candidate cannot be reranked into Top 10 if retrieval never finds it.
4. **Then optimize first-hit order.** Since the session stops on its first hit, rank the most plausible target as high as possible on every list rather than expecting later reranking credit.
5. **Ask high-yield attributes.** In this local simulator, `other` reveals up to two arbitrary undisclosed constraints. A robust implementation can use it as a fallback while still supporting semantically appropriate attribute selection for official evaluation.
6. **Account for the intent-card extraction rules.** Material, color, early feature entries, and details are disproportionately likely to be hidden constraints. Price is appended late and may not enter the four retained constraints for rich products.
7. **Handle negative answers.** A no-preference reply is evidence that the requested attribute should not be a hard filter. Boundary sessions deliberately test this.
8. **Handle overrides explicitly.** Detect `Actually, ignore my earlier preference`, invalidate the obsolete value, keep compatible category/context, and prioritize the new hard constraint.
9. **Use profile data cautiously.** Preference tags and summary can help tie-break or choose questions, but they describe historical aggregate tendencies, not guaranteed target requirements. They should not overpower explicit current-session constraints.
10. **Do not confuse natural dialogue quality with simulator control.** `message` matters for validity and demos, while `ask_attribute` is the actual control channel in local scoring.
11. **Do not exploit public ground truth.** Use labels for evaluation, error analysis, and tuning general rules—not sample-ID-to-ASIN memorization.
12. **Build an offline path.** Official scoring may disable network access. Any API/LLM layer must fail safely into a deterministic local retriever and dialogue manager.

## 10. Suggested experiment loop

For every meaningful agent revision:

1. Run evaluator unit tests:

   ```bash
   python -m unittest discover -s tests
   ```

2. Run the full public evaluator:

   ```bash
   python -m evaluator.local_evaluator
   ```

3. Record overall and per-scenario Hit Rate, MRR, MTTC, Efficiency, and Technical Score.
4. Inspect session-level failures in `results.json`, joining them to public labels and catalog metadata for analysis only.
5. Compare against the frozen baseline above.
6. Check initialization time, per-turn latency, memory use, token use, and offline behavior.
7. Avoid tuning a rule solely to individual public ASINs; prefer improvements that follow from catalog and simulator structure and should transfer to private targets.

## 11. Submission and operational constraints

- Submit one Python agent entry point exporting `Agent`, plus required helper modules, dependency manifest, setup instructions, and method/limitations report.
- Do not edit the evaluator or public labels when claiming a score.
- Do not mutate the frozen catalog or emit fabricated/non-catalog IDs.
- Never commit API keys or secrets; use environment variables.
- Declare model choice, approximate cost, token use, latency, network requirements, and fallback behavior.
- Expect possible CPU, memory, timeout, and network restrictions during final judging.
- A paid LLM is optional; reproducibility and a reliable offline fallback matter more than elaborate prose generation.

## 12. Current status

So far, the workspace has been placed in the GitHub repository `nickolaschua/reptechjam2026`, the evaluator has been run successfully on the untouched starter, and development is occurring on the `nickolas` branch. No improved agent result has yet been established in this conversation; `0.10671` is the verified reference score to beat.
