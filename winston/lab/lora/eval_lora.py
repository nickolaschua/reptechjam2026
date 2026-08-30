"""Score a base model (+ optional LoRA adapter) on the 30 hand-gold probes and the
held-out teacher dev set, with the parser's own scorer. Unconstrained decoding:
a malformed JSON counts as an empty parse, and is reported.

    python3 eval_lora.py --model Qwen/Qwen2.5-1.5B-Instruct [--adapter adapter-1.5b]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
WINSTON = HERE.parent.parent
sys.path.insert(0, str(WINSTON))
from nlp_parse import score  # noqa: E402

EMPTY = {"category_phrase": "", "department": None, "slots": [], "price_max": None,
         "price_min": None, "quality_prior": "none", "exploring": False, "specificity": None}


def generate(model, tok, prompt: str, max_new: int = 400) -> dict:
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   tokenize=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        parsed = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict) or "slots" not in parsed:
        return {**EMPTY, "_malformed": True}
    parsed.setdefault("category_phrase", ""); parsed.setdefault("department", None)
    parsed.setdefault("price_max", None); parsed.setdefault("price_min", None)
    parsed.setdefault("quality_prior", "none")
    parsed["slots"] = [s for s in parsed["slots"] if isinstance(s, dict) and "attribute" in s and "value" in s]
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if cuda else torch.float32,
                                                 device_map="cuda" if cuda else None)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for name in ("sft_probe", "sft_dev"):
        rows = [json.loads(l) for l in (HERE / f"{name}.jsonl").open() if l.strip()][:args.limit]
        if not rows:
            continue
        t0, res, bad = time.time(), [], 0
        for r in rows:
            pred = generate(model, tok, r["prompt"])
            bad += pred.pop("_malformed", False)
            gold = json.loads(r["completion"])
            utt = r["prompt"].split("Message: ", 1)[-1]
            res.append(score(pred, gold, None, utt))
        n = len(res)
        f1 = sum(x["slot_f1"] for x in res) / n
        print(f"{name}: n={n}  slot F1={f1:.3f}  P={sum(x['slot_precision'] for x in res)/n:.3f}  "
              f"R={sum(x['slot_recall'] for x in res)/n:.3f}  dept={sum(x['department_ok'] for x in res)/n:.3f}  "
              f"cat={sum(x['category_overlap'] for x in res)/n:.3f}  price={sum(x['price_ok'] for x in res)/n:.3f}  "
              f"malformed={bad}  {(time.time()-t0)/n:.1f}s/case")


if __name__ == "__main__":
    main()
