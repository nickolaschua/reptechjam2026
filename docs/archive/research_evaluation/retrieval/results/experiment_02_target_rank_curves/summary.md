# Experiment 2 — Target-rank curves

> **AGENT-REALISTIC EVALUATION.** Queries contain only the current message or category plus constraints disclosed by that turn. Target ASINs and undisclosed intent fields never enter a query.

The best early-termination policy is **exact_phrase**, with technical score **0.816917**, HitRate@10 **0.930**, and MRR **0.623**. Full ten-turn traces are retained for diagnostic curves even when a normal evaluation would have stopped on a hit.

`not_retrieved` means the lexical or truncated-RRF ranker did not retrieve the target; it is not silently converted to rank 50,001. Dense ranks cover the full catalog.
