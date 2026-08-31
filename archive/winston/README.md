# Winston bolt-on research archive

This directory preserves Winston's parser, resolver, benchmark, probe, dense/fusion,
fine-tuning, LoRA, report, plan, handoff, and test artifacts as research provenance.
It was moved from the repository-root `winston/` tree on 2026-08-31 when the completed
parts of the bolt-on were integrated into `system.shopping_agent.Agent`, the sole live
deployment target.

## Frozen evidence and claims

- `preds-qwen2.5-7b-instruct.json` and `probe_gold.json` retain the 30-probe Qwen run.
  With the archived scorer, its mean slot F1 is `0.4614` (the acceptance floor is
  `0.441`).
- `lab/resolver_results.json` records the completed `category_phrase + soft slots`
  arm at top-1 `0.267`, top-3 `0.433`, and median true-bucket rank `10`.
- Dense fusion, agreement-driven weighting, contradiction reranking, `make_ranker()`,
  BGE fine-tuning, and LoRA work remain experiments. They are not active runtime code.
- `lab/bench/plug_check.py`, `lab/bench/bolt_on_agent.py`, and
  `lab/bench/frontend_bolt_on.py` are baseline-specific proof artifacts. The frontend
  launcher is obsolete: it imports the removed `experiment_1` server and does not
  expose a callable `run_server`.

## Production replacements

The maintained implementations are:

- `system/shopping_agent/turn_parser.py`: constrained schema/Ollama call, validation,
  cleaning, tier demotion, negation/wearer checks, intent/message type, and typed turns.
- `system/shopping_agent/category_resolver.py`: the completed lexical resolver built
  directly from the already-loaded live `Catalogue`, without this archive's cache.
- `system/shopping_agent/agent.py`: state adaptation, category clarification, telemetry,
  and full-turn rollback.

The resolver supplies ambiguity telemetry only; it does not filter, boost, fuse, or
rerank products.

## Reproduction

From the repository root:

```powershell
python archive/winston/nlp_parse.py
python archive/winston/lab/pipeline.py
python -m pytest system/shopping_agent/tests/test_winston_turn_parser.py -q
python -m pytest system/shopping_agent/tests/test_winston_category_resolver.py -q
```

The tracked benchmark scripts discover the repository catalogue from this archived
location. Running model-backed or dense experiments still requires their documented
local models and Python dependencies.

The following artifacts were absent from the checkout at archival time and are not
part of the evidence: `parses.jsonl`, `lab/.cache/bge_ft3/`, and
`PIPELINE_CHANGES.md`. Ignored `.cache/`, `__pycache__/`, and model-weight directories
must not be committed.
