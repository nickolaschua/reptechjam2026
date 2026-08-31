# The case for the fall-through parser

Measured 2026-08-30/31 on this repo. Every number reproduces with the commands at the end.

## Claim

Keep the keyword layer exactly as it is. Add one thing: when a message does NOT match
the evaluator's template (the regex misses), a local 7B parse fills the same
`{category, constraints}` state the keyword layer already reads. Retrieval, ranking and
question policy are untouched. This strictly dominates the existing infra:

| input regime | existing infra | + bolt-on | evidence |
|---|--:|--:|---|
| evaluator sessions (templated, multi-turn, n=60) | TechnicalScore 0.872409 | **identical to the digit** | `plug_check.py --public 60`; an asserting parser proves the LLM is never invoked on template text |
| human-typed messy input (n=1,685, turn 1) | hit@10 **0.043** / MRR 0.011 | **0.196** / 0.103 (4.6x) | `plug_check.py --bench`; paired 553 better / 50 worse / 1082 tie |
| exp11 retrieval, fair per-track (n=1,685) | hit@10 **0.143** [.127,.160] | **0.233** [.211,.253] | `report.py`; 191 rescues / 40 regressions |

The baselines that matter are the ones the team built: exp11's FTS5 (above) and Yang Xu's
hybrid agent (FTS5 + BGE fallback + LLM tracker) - the latter is being run on the same
1,685 cases (`shop_agent_baseline.py`) and its row lands here. The paraphrase result
(regex 0.03 -> parsed 0.77, = the un-paraphrased regex score 30/30) is kept as a gate
in `template_check.py`, not a headline: the regex was never designed to see paraphrases.

## Where the gain comes from (fair per-track comparison, exp11 retrieval, n=1,685)

raw utterance 0.143 [0.127, 0.160] -> parsed 0.233 [0.211, 0.253]; 191 rescues / 40 regressions.

| style | raw | parsed |
|---|--:|--:|
| exact (model code) | 0.39 | **0.71** |
| feature | 0.29 | **0.49** |
| compatibility | 0.36 | **0.45** |
| plain | 0.24 | **0.36** |
| product_type | 0.23 | **0.32** |
| negation modifier (n=356) | 0.10 | **0.21** (negated slot: 70% recall, 1.7% false) |
| bought-for-someone-else (n=369) | 0.12 | **0.22** (department evidence veto) |
| lay / symptom / use_case | 0.01 | 0.02-0.06 |

The last row is the honest boundary: vocabulary-gap input (the shopper cannot name the
item) is not fixable by a better keyword query - the true category ranks in the resolver's
top-3 only ~15% there. Roadmap: constrained category pick from the resolver's top-10 (B2),
catalog-vocabulary expansion (B4), and ask-instead-of-show - the parse's own hard-slot
count and resolver confidence split hit@10 0.41 vs 0.17 (2x2), which is the trigger.

## Cost

Template turn: 0 extra ms (parser never runs). Messy turn: one local qwen2.5:7b call,
~10 s on an M4 laptop, ~2-4 s on the 4090. No API, no network, $0.

## Reproduce

```bash
cd winston/lab/bench
python3 plug_check.py --public 60      # identical metrics, stock vs plugged
python3 plug_check.py --bench          # 1,685 messy cases through the deployed agent
python3 template_check.py 30           # template + paraphrase gate
python3 report.py                      # full CI tables (report.md)
python3 frontend_bolt_on.py            # Yang Xu's live UI driving the bolt-on (port 8081)
```

Dataset: 269 products x 8 utterance styles, two local generators (llama3.1:8b, gemma2:9b),
ground truth = the target asin, seed 20260829. Spec: `../specs/2026-08-29-messy-benchmark-design.md`.
