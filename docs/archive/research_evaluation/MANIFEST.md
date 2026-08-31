# Research and evaluation archive

Archived on 2026-08-30 while preparing the Nickolas TechJam demo. These files are preserved evidence and are excluded from active imports and pytest discovery.

## Memory evaluation

- `memory/longitudinal_eval/`: fixture builders, evaluator v2, 40-user fixtures, microbenchmarks, calibration/sweep scripts, and frozen result bundles.
- `memory/scripts/`: former `run_longitudinal_eval.py` and `run_longitudinal_eval_v2.py` entry points.
- `tests/`: evaluator-specific tests renamed with `.py.txt` so they cannot be discovered accidentally.

The `memory/longitudinal_eval/results/v2/` bundle remains the trustworthy 40-probe evidence. Its manifests and hashes were not regenerated or rewritten during demo cleanup. Threshold calibration and blend-weight sweep outputs remain beside it.

## Embedding and M0 baselines

`embedding_and_baselines/` preserves the embedding bakeoff, M0 runner/configuration, baseline sessions/results, benchmark outputs, wrapper agents, and milestone documentation.

## Retrieval research

`retrieval/` preserves experiments 1–11, reports and raw results, model/cache evidence, Colab material, and the pre-demo scenario notes.

## Runtime boundary

The active demo does not import this directory. Archived programs may need historical paths restored before rerunning; they are retained for provenance, not maintained as live entry points. See `nickolas/MEMORY_EVALUATION_STATUS.md` for the concise honest conclusion.
