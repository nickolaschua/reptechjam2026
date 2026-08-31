"""LoRA on Qwen2.5-{0.5,1.5,3}B-Instruct: utterance -> parser JSON.

    python3 train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --out adapter-1.5b
    python3 train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --out /tmp/dry --max-steps 2   # smoke

Prompt tokens are masked (-100); only the JSON completion is learned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

HERE = Path(__file__).resolve().parent
import inspect  # noqa: E402
_TA_PARAMS = set(inspect.signature(TrainingArguments).parameters)


def encode(tok, row: dict, max_len: int) -> dict:
    msgs = [{"role": "user", "content": row["prompt"]}]
    # transformers 5.x: tokenize=True returns a BatchEncoding, so render text then tokenize
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tok(row["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + completion_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


class Collate:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        return {"input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
                "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="adapter")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if cuda else torch.float32)
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in (HERE / "sft_train.jsonl").open() if l.strip()]
    dev = [json.loads(l) for l in (HERE / "sft_dev.jsonl").open() if l.strip()]
    train_ds = [encode(tok, r, args.max_len) for r in rows]
    dev_ds = [encode(tok, r, args.max_len) for r in dev] or None
    print(f"train {len(train_ds)}  dev {len(dev_ds or [])}  max tokens {max(len(x['input_ids']) for x in train_ds)}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**{k: v for k, v in dict(
            output_dir=str(HERE / "runs"), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=args.lr,
            warmup_steps=5, lr_scheduler_type="cosine", logging_steps=5, save_strategy="no",
            eval_strategy="epoch" if dev_ds else "no", bf16=cuda, report_to="none", seed=0,
        ).items() if k in _TA_PARAMS}),   # ponytail: transformers 4.x/5.x both accepted
        train_dataset=train_ds, eval_dataset=dev_ds, data_collator=Collate(tok.pad_token_id))
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"adapter -> {args.out}")


if __name__ == "__main__":
    main()
