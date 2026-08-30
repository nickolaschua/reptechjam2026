"""Fine-tune BGE on the benchmark's own (messy utterance -> target product) pairs.

Mirrors experiment_1/finetune_embedder.py (MNRL, same product text, same-bucket hard
negatives) but trains on GENERATED messy utterances instead of templated ones, and
holds out every product that appears in the evaluation cases (parses.jsonl), so the
dense/fusion numbers after this are on unseen products.

    DENSE_DEVICE=cpu python3 finetune_bge.py          -> .cache/bge_ft/
"""
from __future__ import annotations

import inspect
import json
import os
import random
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss

LAB = Path(__file__).resolve().parent
KIT = LAB.parent.parent / "techjam-conversational-search"
sys.path.insert(0, str(KIT)); sys.path.insert(0, str(LAB))
from dense_rank import QUERY_PREFIX, catalog_text  # noqa: E402
from evaluator.local_evaluator import catalog_index, coarse_category  # noqa: E402

OUT = LAB / ".cache" / os.environ.get("FT_OUT", "bge_ft")
NEG_PER_ANCHOR = int(os.environ.get("FT_NEG", 2))
EPOCHS = float(os.environ.get("FT_EPOCHS", 3))
HOLDOUT = os.environ.get("FT_HOLDOUT", "parsed")     # parsed | random20


def main() -> None:
    rng = random.Random(20260830)
    _, cats, products = catalog_index(KIT / "data" / "catalog.jsonl")
    buckets: dict[str, list[str]] = {}
    for a, c in cats.items():
        buckets.setdefault(coarse_category(c), []).append(a)

    cases = [json.loads(l) for l in (LAB / "bench" / "cases.jsonl").open() if l.strip()]
    if HOLDOUT == "random20":
        # a fixed fifth of the PRODUCTS is never trained on; evaluate on whichever of
        # their cases have a parse, now or later - decoupled from the parse pass
        asins = sorted({c["asin"] for c in cases})
        eval_asins = set(rng.sample(asins, len(asins) // 5))
    else:
        eval_ids = {json.loads(l)["case_id"] for l in (LAB / "bench" / "parses.jsonl").open() if l.strip()}
        eval_asins = {c["asin"] for c in cases if c["case_id"] in eval_ids}
    train = [c for c in cases if c["asin"] not in eval_asins]

    anchors, positives, negatives = [], [], []
    for c in train:
        pos = catalog_text(products[c["asin"]])
        pool = [a for a in buckets[coarse_category(cats[c["asin"]])] if a != c["asin"]] or \
               [a for a in products if a != c["asin"]]
        for neg in rng.sample(pool, min(NEG_PER_ANCHOR, len(pool))):
            anchors.append(QUERY_PREFIX + c["utterance"])      # exactly what inference embeds
            positives.append(pos)
            negatives.append(catalog_text(products[neg]))
    print(f"cases {len(cases)} | held-out products {len(eval_asins)} | train cases {len(train)} | triplets {len(anchors)}")
    (LAB / ".cache").mkdir(exist_ok=True)
    (LAB / ".cache" / f"{OUT.name}_heldout.json").write_text(json.dumps(sorted(eval_asins)))

    import os
    device = os.environ.get("DENSE_DEVICE", "cpu")          # mps once the GPU is free
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    model.max_seq_length = 256
    params = set(inspect.signature(SentenceTransformerTrainingArguments).parameters)
    args = {k: v for k, v in dict(
        output_dir=str(LAB / ".cache" / "bge_ft_runs"), num_train_epochs=EPOCHS, per_device_train_batch_size=16,
        gradient_accumulation_steps=2, learning_rate=2e-5, warmup_steps=10, use_cpu=(device == "cpu"),
        dataloader_num_workers=0, logging_steps=10, save_strategy="no", report_to="none", seed=0,
    ).items() if k in params}
    trainer = SentenceTransformerTrainer(
        model=model, args=SentenceTransformerTrainingArguments(**args),
        train_dataset=Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives}),
        loss=MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save_pretrained(str(OUT))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
