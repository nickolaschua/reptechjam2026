# Pipeline changes: what the message path did before, and what it does now

Written 2026-08-31. Every number in this file is reproduced by a command listed in
section 6. "Deployed agent" means `techjam-conversational-search/starter/agent.py`,
which is the competition-interface agent; the merge of 2026-08-31 did not change it.

---

## 1. The pipeline before (deployed agent, one message)

```
user message
     |
     v
[_update_state]  starter/agent.py:396
     |   three literal string checks, nothing else:
     |   1. turn 1, starts with "i'm looking for "
     |        text before ", but I'm still exploring" or before the first "."
     |          -> state.category
     |        text after "a key requirement is:" -> state.constraints
     |   2. starts with "for that, what matters is:"
     |        payload split on "; " -> append each piece to state.constraints
     |   3. starts with "actually, ignore my earlier preference. what i need is:"
     |        remove the remembered seed constraint, append the new value
     |   4. ANY OTHER MESSAGE: the function returns without changing anything.
     v
[query build]  respond(), starter/agent.py:447
     |   query terms = (state.category or "clothing item") + state.constraints
     v
[retrieval]  _legacy_recommendations -> _sqlite_recommendations
     |   SQLite FTS5, OR-query over up to 80 unique terms,
     |   ranked by bm25 with field weights
     |   title 6.0, categories 4.0, features 2.5, details 2.5, store 1.5, description 1.0
     v
[top-k slate]  +  [fixed question text]
     |   "I am using N stated preferences. What other requirement should I consider?"
```

Behaviour on the two kinds of input:

- Evaluator message ("I'm looking for Women Dresses, but I'm still exploring."):
  check 1 fires, category = "Women Dresses". This is what the 0.87 TechnicalScore
  is built on.
- Human message ("need one of those long sleeve shirts for fishing, the kind that
  blocks the sun. gotta dry fast"): no check fires. category stays "", the query
  becomes the literal words "clothing item", and retrieval ranks products against
  those two words. Measured over 1,685 human-style messages: target in the top 10
  in 4.3% of cases (MRR 0.011). The score is not 0 only because some generated
  messages happen to start with "i'm looking for", which makes check 1 swallow the
  whole first sentence as the category.

---

## 2. The pipeline now (deployed agent + bolt-on)

One method is overridden (`_update_state`). Every other box is the same object
code as section 1.

```
user message
     |
     v
[is_template?]  winston/lab/bolt_on.py:147
     |   the message matches one of the evaluator's sentence shapes
     |   AND the extracted category is one of the 1,115 real catalog bucket names
     |
     +-- yes --> [the EXACT _update_state from section 1]   (byte-identical path)
     |
     +-- no ---> [parse]  winston/nlp_parse.py:parse_with_ollama
     |               one call to qwen2.5:7b-instruct on localhost, temperature 0.
     |               output is FORCED by a JSON grammar (Ollama format=SCHEMA) to:
     |                 category_phrase   string, max 40 chars
     |                 department        enum {womens, mens, girls, boys, ...} or null
     |                 slots             max 8 of {attribute in {color, material, size,
     |                                   brand, style, use_case, feature, budget, other},
     |                                   value max 40 chars, declined bool, negated bool}
     |                 price_max, price_min, quality_prior, exploring, specificity
     |               the model cannot emit anything outside this shape.
     |
     |           [deterministic validation]  bolt_on.py:clean_parse + nlp_parse.py
     |               runs in Python, not in the model:
     |               - slot values on the junk list are deleted
     |                 ("not specified", "any", the schema's own label words,
     |                  stance words like "exploring")
     |               - negated=true survives only if a negation word actually
     |                 appears next to that value in the user's message
     |               - department is set to null unless the message contains a
     |                 wearer word (women, men, wife, daughter, son, ...)
     |               - decline cues clear the value; duplicates collapse;
     |                 price <= 0 becomes null
     |
     |           [state fill]  winston/lab/bench/plug_check.py:BoltOnAgent
     |               state.category = category_phrase
     |               every surviving slot value that is neither declined nor
     |               negated goes through the EXISTING _append_constraint()
     |               negated values are never appended, so "no polyester" cannot
     |               make retrieval search FOR polyester
     v
[query build]     unchanged
[retrieval]       unchanged (same FTS5, same weights)
[slate, question] unchanged
```

Behaviour, measured:

| input | before | after | evidence |
|---|--:|--:|---|
| 60 full evaluator sessions | TechnicalScore 0.872409 | identical on every metric | a parser stub that raises proves the LLM is never called on template text |
| 1,685 human-style messages, turn 1 | hit@10 0.043, MRR 0.011 | hit@10 0.196, MRR 0.103 | paired: 553 better, 50 worse, 1,082 tie |
| Yang Xu's hybrid agent (FTS5 + BGE fallback + LLM tracker), same 400 cases | hit@10 0.050, MRR 0.030 | hit@10 0.182, MRR 0.088 | paired 116 better, 18 worse, 266 tie; his router sent 271/400 to its no-LLM matcher (0.000 there); where his LLM ran (129) he scores 0.155 vs the bolt-on's 0.225 on the same cases |
| our reader in front of HIS retriever (his FTS5 + BGE fallback + post-scoring, LLM tracker replaced), same 400 | 0.050 as shipped | 0.150, MRR 0.084 | his BGE fallback fired on 3/400 once the state was clean - dense helps only as a parallel fused route, not as a fallback; `shop_agent_plug.py` |

Same message trace as section 1: the parse returns category_phrase
"long sleeve shirts" and slots use_case "fishing", feature "blocks the sun",
feature "fast-drying"; the unchanged FTS5 runs on those words instead of
"clothing item".

Cost of the new box: 0 ms on a template turn (the parser is not called);
one local model call (~10 s on the M4 laptop, ~2-4 s on an RTX 4090) on a
human turn. No network, no API key.

Parser model choice, measured on the 30 hand-written probes (same gold, same
scorer): qwen2.5:7b slot F1 0.461 (recall 0.532) vs llama3.1:8b 0.260 (recall
0.288) vs qwen2.5:3b 0.267. Recall is what matters here - a missed slot is a
lost query term. llama3.1 also could never be scored on the benchmark itself,
since it generates half of its utterances.

Fair per-track version of the same comparison (both arms on exp11's retrieval,
so the parse is the only variable): raw utterance 0.143 [0.127, 0.160] ->
parsed 0.233 [0.211, 0.253], 191 rescues, 40 regressions. Per style:
exact 0.39->0.71, feature 0.29->0.49, compatibility 0.36->0.45, plain 0.24->0.36,
product_type 0.23->0.32, negation cases 0.10->0.21, gift-for-someone-else
0.12->0.22, and lay/symptom/use_case stuck at 0.01->0.02-0.06 (see section 3).

---

## 3. Built and measured, not yet wired into the deployed agent

The rows above show keyword retrieval cannot reach the vocabulary-gap styles
(the shopper cannot name the item: "those foam clog things with the holes").
For those, a second retrieval route was built and measured on the same benchmark:

```
                        +--> [dense route]  BGE embedder, RAW sentence as query
                        |       cosine against 50k precomputed catalog vectors.
                        |       the raw sentence, NOT the parse: parsed queries
                        |       measurably hurt this route (0.438 -> 0.367)
user message --+--------+
               |        +--> [lexical route]  parse -> state -> the same FTS5
               |        |
               |        +--> [category route]  resolver's top-3 buckets from the
               |                parse, members ranked by popularity, weighted by
               |                resolver confidence
               v
            [parse] ------ also supplies to the fusion step:
                             - per-route weights (from slot counts + confidence)
                             - a contradiction filter (a product that contradicts
                               a stated hard slot is sunk, silence is never sunk)
                                     |
                                     v
                        [weighted reciprocal-rank fusion] -> top-k
```

Measured (single turn, hit@10):

| system | number | scope |
|---|--:|---|
| dense, stock BGE, raw sentence | 0.303 | n=584, no fine-tune, no leakage possible |
| fusion with stock BGE | 0.325 | n=584 |
| dense, BGE fine-tuned on benchmark pairs | 0.438 | 53 held-out products only, n=128 |
| fusion with fine-tuned BGE + parse-driven weights | 0.444 | held-out only, n=135 |

Leakage disclosure: the fine-tune's own result files aggregate to 0.502/0.517,
but 209 of 262 evaluated products were in its training pairs; the held-out
columns above are the defensible numbers. Style detail (whole set, includes
trained products, so read as an upper bound): lay 0.45, symptom 0.26,
use_case 0.17 - against 0.01-0.05 for any keyword variant.

Also built on the classification front (measured, not yet deployed):

- `bolt_on.intent_of`: buying/browsing from an explicit-cue check plus a count of
  actionable slots. Chosen over the LLM's own `exploring` flag (wrong on half the
  probes it fired on) and over candidate-pool size (3/8 on the problem statement's
  own examples).
- `bench/intent_clf.py`: a logistic-regression distillation of that rule needing
  no LLM, so a browsing turn can skip the ~10-18 s parse and go straight to the
  dense route on the raw sentence.
- The ask-don't-guess signal: with hard-slot count and resolver confidence the
  1,685 cases split into show-a-tight-slate (hit@10 0.41, 8% of cases) down to
  ask-a-question (hit@10 0.17, 51% of cases).

---

## 4. What did NOT change

- The FTS5 index, its field weights, and the ranking code.
- The question policy and all customer-facing message text.
- The evaluator, the simulator, and every file under
  `techjam-conversational-search/` (the memory hook `SemanticParser` already
  existed there and is what the bolt-on implements).
- Multi-turn state logic on template sessions.
- `system/shopping_agent/` (the team's new runtime): none of this is wired into
  it yet; its own LLM state tracker is separate and unmodified.

## 5. Known limits

Single-turn on messy input (multi-turn messy is untested). The messy utterances
are generated by two local 8-9B models under our style prompts, not typed by real
shoppers. The fine-tune needs one rerun with the product split fixed from the
start before its numbers go in the report. The vocabulary-gap styles remain
unsolved in the deployed (lexical-only) configuration.

## 6. Where every number comes from

```
cd winston/lab/bench
python3 plug_check.py --public 60       # section 2 row 1: identical evaluator metrics
python3 plug_check.py --bench           # section 2 row 2: 0.043 -> 0.196
python3 report.py                       # section 2 fair per-track tables (report.md)
python3 template_check.py --report      # template/paraphrase gate
python3 dense_rank.py                   # section 3 dense rows (DENSE_MODEL=... for fine-tuned)
python3 fuse_rank.py                    # section 3 fusion rows
python3 intent_clf.py                   # section 3 classifier report
```

Dataset: 1,685 utterances = 269 catalog products x 8 styles, generators
llama3.1:8b and gemma2:9b, seed 20260829, ground truth = target ASIN, no hand
labels. Construction and success criteria: `specs/2026-08-29-messy-benchmark-design.md`.
