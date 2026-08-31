# M0_OPENAI freeze manifest

- Baseline: M0_OPENAI
- Fast Memory: enabled
- Slow Memory: disabled
- Embedding backend: OpenAI
- Embedding model: `text-embedding-3-large`
- Embedding space: `openai-text-embedding-3-large:text-embedding-3-large:dimensions=3072:normalization=l2:query=symmetric-v1`
- Evaluator: existing `experiment_1.run_eval_v2` evaluator v2
- Test result: passed (35 tests)
- Timestamp/run ID: 2026-08-29T17:22:30.570406+00:00
- Git commit: `a954d771c23eade42d797630d9001c5e9dce3cb9`
- `nickolas/shopping_agent` has uncommitted changes: true
- Result files: `config.json`, `metrics.json`, `latency.json`, `results_v2.json`, `run_summary.md`, `eval_sessions/`

## Source SHA-256

- `run_m0.py`: `2764d43c78b5ad06648d46a5188cc53500e689dff56178fa36d7867616d34dc0`
- `agent.py`: `d742fe4b469582485ea027e993a9b1d944dc6d1d17ffdfb85adec1f6570229b4`
- `embedding_backends.py`: `3f1eb4bd286b703fa05084bc79cfa9065ee6d24f87fef6d8e9282d1437bd93ea`
- `m0_openai.json`: `9cc74c7769213b5cb7ce99a2837d94ab292a74e4e4ba9894a04ca8263ddbca28`
