Experiment 10 XTR/WARP Colab output
=======================================

This archive contains a converted 4-bit WARP index and frozen Top-1000
rankings for 600 distinct agent-visible queries. Retrieval used CPU mode,
nprobe=32, one Torch thread, and score-descending/PID-ascending ties.

Apply xtr_warp_compat.patch to xtr-warp revision cca97613e6f969ac89f259946b976f8c5a6f1399 before
loading warp_index/techjam26-products-xtr-warp.nbits=4. The patch pins model revision
f40cd399e67dfc8ec974e922ad828610e3c83a36 and supports the compact portable index layout.

No model weights were trained or fine-tuned. Rankings contain no targets or session labels.
Verify SHA256SUMS and manifest.json before importing any result.
