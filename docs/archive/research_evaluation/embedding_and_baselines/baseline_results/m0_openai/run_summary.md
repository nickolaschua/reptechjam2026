# M0_OPENAI run summary

- Fast Memory: enabled (Patch-2 semantics)
- Slow Memory: disabled and not implemented
- Embedding backend: OpenAI `text-embedding-3-large` for both product and query vectors
- HR@10: 0.92
- MRR: 0.611359
- Mean turns to hit/conversion: 3.335
- Failed sessions: 16
- Agent errors: 0
- Fast/full turns: 33 / 618
- Dense retrieval invocations: 1

OpenAI embeddings are used intentionally as the development baseline. Phase 3
found BGE stronger on the controlled embedding benchmark, so this run does not
claim OpenAI is the final or competition-best embedder. BGE remains available
for later final-system reruns.
