# Experiments

Every claim in the findings summary is reproduced by one of these. Each script
writes a JSON file to `results/`; nothing is hand-transcribed.

```bash
cd winston/experiments
python3 run_all.py                    # everything (~10 min, evaluator runs dominate)
python3 exp03_scoring_ablation.py     # or one at a time
```

Requires `techjam-conversational-search/data/catalog.jsonl` (50,000 rows — download
from the participant-kit release, it is gitignored). Standard library plus the kit's
evaluator; no third-party packages, no API keys, no network.

| script | question it answers | key output |
|---|---|---|
| `exp01_catalog_profile.py` | What is in the catalog, and which attributes can be *filtered* rather than merely scored? | field coverage, details schema, selectivity, exclusivity, three-valued match/contradict/silent, literal-filter false negatives |
| `exp02_simulator_leakage.py` | How much does the public simulator give away? | turns to full card by ask policy, `intent_card()` inversion, constraint document frequency |
| `exp03_scoring_ablation.py` | Which scoring components earn their place? | full evaluator runs per ablation, rank histograms, preference-tag lift, field weighting |
| `exp04_robustness.py` | What happens when the customer stops copy-pasting catalog text? | reply perturbations, category perturbations, fuzzy-lookup recovery, flat vs hierarchy |
| `exp05_agent_diagnostics.py` | Why does `experiment_1` score 0.896 here and 0.000 on a messy session? | the three parsing bugs, plus whole-message resolution on the same transcript |

## Shared modules

- `common.py` — catalog loading, IDF, popularity, bucket index, `resolve_bucket()`.
  Pickles to `.cache/` because building IDF over 50k products takes ~30s.
- `baseline_agent.py` — the reference agent (category pool + IDF match + popularity
  prior + patience policy). Every component is a constructor flag so ablations can
  disable exactly one thing.

## Reading the results

`technical_score` is the competition's `0.50·HitRate + 0.30·MRR + 0.20·Efficiency`.
The published weak-BM25 baseline is **0.10671**; it is included in
`exp03_scoring_ablation.json` for reference.

Two cautions when quoting these numbers:

- HitRate saturates at 1.000 on the public set. Anything measured there is measuring
  a benchmark whose customer utterances are copy-pasted product text.
- `exp02`'s card inversion depends on the private set generating intent cards with
  the same function the public evaluator falls back to. The private samples ship
  their own precomputed cards, so it may not transfer. `exp04` is the honest test.
