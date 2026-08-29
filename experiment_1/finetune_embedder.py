import os
import sys
import json
import random
import argparse
from pathlib import Path
from datasets import Dataset

# Setup paths relative to this script
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent / "techjam-conversational-search"
sys.path.insert(0, str(repo_root))

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    intent_card,
    materialize_hidden_fields
)

try:
    from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
except ImportError:
    print("[Error] sentence-transformers library not installed. Please run:")
    print("pip install sentence-transformers datasets")
    sys.exit(1)


def _clean_constraint(text: str) -> str:
    text = text.strip()
    # Keep it under 5 words (representing clean attribute values) and avoid promo characters/links
    if len(text.split()) > 5:
        return ""
    if any(char in text for char in ["♥", "★", "☀", "✔", "http", "www", "©", "♥"]):
        return ""
    return text


def _get_bucket(cats: list[str]) -> str:
    if not cats:
        return "generic"
    # Use the second-to-last category node if category path is deep (e.g. "watches", "shoes")
    if len(cats) > 2:
        return cats[-2].lower()
    return cats[-1].lower()


def generate_triplets(catalog_path, dataset_path, num_negatives=2):
    print("[Finetune] Loading catalog and dataset...")
    catalog_ids, categories, products = catalog_index(catalog_path)
    
    # Group products by category bucket for hard negative sampling
    cat_buckets = {}
    for pid, p in products.items():
        # Get category bucket (using second-to-last node for tighter subcategory alignment)
        cats = p.get("categories") or []
        bucket = _get_bucket(cats)
        if bucket not in cat_buckets:
            cat_buckets[bucket] = []
        cat_buckets[bucket].append(pid)
    
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
        
    anchors = []
    positives = []
    negatives = []
    
    print(f"[Finetune] Materializing triplets from {len(samples)} public-set samples...")
    
    for idx, sample in enumerate(samples):
        target_asin = sample["ground_truth"]["parent_asin"]
        if target_asin not in products:
            continue
            
        target_product = products[target_asin]
        card = intent_card(target_product)
        
        # Positive product string (matches our runtime embedder text format)
        title = target_product.get("title") or ""
        cats_str = ", ".join(target_product.get("categories") or [])
        feats = "; ".join((target_product.get("features") or [])[:3])
        positive_text = f"Product: {title}. Categories: {cats_str}. Features: {feats}.".strip()
        
        # Extract constraints and make queries (applying cleaning filter)
        raw_hard = card.get("hard_constraints", [])
        raw_soft = card.get("soft_preferences", [])
        hard_constraints = [c for c in (_clean_constraint(x) for x in raw_hard) if c]
        soft_preferences = [s for s in (_clean_constraint(x) for x in raw_soft) if s]
        coarse_cat = coarse_category(categories.get(target_asin, []))
        
        # If filtering left us empty, fall back to the first raw constraint (capped at 4 words)
        if not hard_constraints and raw_hard:
            hard_constraints = [" ".join(raw_hard[0].split()[:4])]
        if not soft_preferences and raw_soft:
            soft_preferences = [" ".join(raw_soft[0].split()[:4])]
            
        hard_str = ", ".join(hard_constraints)
        soft_str = ", ".join(soft_preferences)
        
        # Build natural dialogue queries
        queries = []
        if hard_str and soft_str:
            queries.append(f"i'm looking for {coarse_cat}. it must be {hard_str}. ideally, it should also be {soft_str}.")
            queries.append(f"hi, can you find me a {coarse_cat}? a key requirement is {hard_str}. i prefer {soft_str}.")
        elif hard_str:
            queries.append(f"i need a {coarse_cat} with {hard_str}.")
            queries.append(f"looking for {coarse_cat}. key requirement: {hard_str}.")
        else:
            queries.append(f"i'm looking for {coarse_cat}.")
            queries.append(f"can you show me options for {coarse_cat}?")
            
        # Get category bucket of this target product for sampling hard negatives
        cats = target_product.get("categories") or []
        bucket = _get_bucket(cats)
        bucket_candidates = cat_buckets.get(bucket, list(products.keys()))
        
        # Filter target out of candidates
        bucket_candidates = [pid for pid in bucket_candidates if pid != target_asin]
        if not bucket_candidates:
            bucket_candidates = [pid for pid in products.keys() if pid != target_asin]
            
        for query in queries:
            # Sample hard negatives from same category bucket
            sampled_neg_ids = random.sample(bucket_candidates, min(num_negatives, len(bucket_candidates)))
            for neg_id in sampled_neg_ids:
                neg_product = products[neg_id]
                neg_title = neg_product.get("title") or ""
                neg_cats_str = ", ".join(neg_product.get("categories") or [])
                neg_feats = "; ".join((neg_product.get("features") or [])[:3])
                negative_text = f"Product: {neg_title}. Categories: {neg_cats_str}. Features: {neg_feats}.".strip()
                
                # Append triplet
                anchors.append(query)
                positives.append(positive_text)
                negatives.append(negative_text)
                
    print(f"[Finetune] Generated {len(anchors)} triplet pairs for training.")
    return Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    })


def main():
    parser = argparse.ArgumentParser(description="Finetune BGE Embedder on TechJam Catalog")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="Base model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8 to prevent MPS OOM)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU instead of MPS/GPU")
    args = parser.parse_args()
    
    catalog_path = repo_root / "data/catalog.jsonl"
    dataset_path = repo_root / "data/public_set.jsonl"
    output_dir = current_dir / "model_finetuned"
    
    # 1. Prepare Triplet Dataset
    train_dataset = generate_triplets(catalog_path, dataset_path)
    
    # 2. Load Base embedder model
    print(f"[Finetune] Loading pretrained model: {args.model}...")
    model = SentenceTransformer(args.model)
    
    # 3. Setup Loss (Multiple Negatives Ranking Loss with in-batch negatives)
    loss = MultipleNegativesRankingLoss(model)
    
    # 4. Training Arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Recreates effective batch size of 32
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=False,  # Disabled as PyTorch MPS AMP is unstable/leaky
        use_cpu=args.cpu,
        dataloader_num_workers=0,  # Safer multi-processing on macOS
        logging_steps=10,
        save_strategy="no"
    )
    
    # 5. Initialize Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    
    # 6. Execute training
    print("[Finetune] Starting SentenceTransformer training...")
    trainer.train()
    
    # 7. Save final model
    print(f"[Finetune] Saving final model to: {output_dir}")
    model.save_pretrained(str(output_dir))
    print("[Finetune] Done! The custom agent is already configured to automatically load this model.")

if __name__ == "__main__":
    main()
