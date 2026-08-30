import sys
import json
import time
import uuid
import random
import requests
import argparse
from pathlib import Path
import numpy as np

# Setup paths relative to this script
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent / "techjam-conversational-search"
sys.path.insert(0, str(repo_root))

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    intent_card,
    behavior_for,
    normalize_recommendations,
    metric_summary
)
from shop_agent import Agent
from shopper_agent import (
    ShopperIntentState,
    call_shopper_llm,
    make_system_prompt,
    materialize_hidden_fields,
)

MAX_TURNS = 10
TOP_K = 10

def evaluate_v2(agent, samples, catalog_ids, categories, products, model_name="llama3.1", max_samples=200):
    sessions = []
    t_start = time.time()
    
    selected_samples = samples[:max_samples]
    print(f"[Evaluator v2] Starting batch evaluation over {len(selected_samples)} sessions using LLM model: {model_name}...")
    
    for idx, sample in enumerate(selected_samples):
        session_id = f"stress_test_sim_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        
        # Materialize intents
        card, behavior = materialize_hidden_fields(sample, products)
        sample = {**sample, "intent_card": card, "behavior": behavior}
        
        target_asin = sample["ground_truth"]["parent_asin"]
        target_product = products[target_asin]
        coarse_cat = coarse_category(categories.get(target_asin, []))
        
        shopper_intent_state = ShopperIntentState.from_sample(sample)
        system_prompt = make_system_prompt(
            sample,
            target_product,
            coarse_cat,
            shopper_intent_state,
        )
        user_message = ""
        override = behavior.get("override", {})
        override_applied = sample["scenario_type"] != "intent_override"
        
        hit_turn = None
        best_rank = None
        recs_meta = []
        
        # Collect dialog lines
        dialog_lines = []
        dialog_lines.append(f"Target Product Title: {target_product.get('title')}\n")
        dialog_lines.append(f"Target ASIN: {target_asin}\n")
        dialog_lines.append(f"Scenario: {sample['scenario_type']}\n")
        dialog_lines.append(f"Constraints: {card.get('hard_constraints')}\n")
        dialog_lines.append("="*60 + "\n")
        
        # Run dialogue turns
        for turn in range(1, MAX_TURNS + 1):
            if turn == 1:
                prompt = "Start the conversation by telling the assistant what you are looking for."
                if sample["scenario_type"] == "intent_override":
                    old_val = override.get("old_value", "")
                    prompt += f" Remember to mention your initial preference for: '{old_val}'."
                elif sample["scenario_type"] == "buying" and card.get("hard_constraints"):
                    hard_val = card["hard_constraints"][0]
                    prompt += f" Mention your key requirement: '{hard_val}'."
                try:
                    user_message = call_shopper_llm(prompt, system_prompt, model_name)
                except Exception:
                    user_message = f"I'm looking for {coarse_cat}."
            elif not override_applied and turn == int(override.get("turn", 3)):
                override_applied = True
                shopper_intent_state.apply_override(
                    str(override.get("new_value") or ""),
                    turn,
                )
                system_prompt = make_system_prompt(
                    sample,
                    target_product,
                    coarse_cat,
                    shopper_intent_state,
                )
                user_message = override.get("message", "Actually, ignore my earlier preference.")
            else:
                prompt = (
                    f"Assistant's Response:\n{copilot_response.get('message')}\n\n"
                    f"Assistant's Recommendations:\n"
                )
                for i, rec in enumerate(recs_meta):
                    prompt += f"{i+1}. {rec.get('title')} (ASIN: {rec.get('parent_asin')})\n"
                prompt += (
                    f"\nAssistant asked about: {copilot_response.get('ask_attribute')}\n\n"
                    "What is your next message? Keep it short, natural, and stay in character. "
                    "If the target product (ASIN: " + target_asin + ") is in the recommendations, "
                    "acknowledge it and say you want to buy it."
                )
                try:
                    user_message = call_shopper_llm(prompt, system_prompt, model_name)
                except Exception:
                    user_message = "What options do you have?"
                    
            # Get Copilot Response
            try:
                copilot_response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                copilot_response = {"message": "", "ask_attribute": None, "recommendations": []}
                
            ranked = normalize_recommendations(copilot_response.get("recommendations"), catalog_ids)
            recs_meta = [products[pid] for pid in ranked]
            
            # Log turn dialog
            dialog_lines.append(f"--- TURN {turn} ---\n")
            dialog_lines.append(f"Shopper: {user_message}\n")
            dialog_lines.append(f"Copilot: {copilot_response.get('message')}\n")
            dialog_lines.append(f"Copilot Attribute Query: {copilot_response.get('ask_attribute')}\n")
            dialog_lines.append("Recommendations:\n")
            for r_idx, rec in enumerate(recs_meta):
                dialog_lines.append(f"  {r_idx+1}. {rec.get('title')} (ASIN: {rec.get('parent_asin')})\n")
            dialog_lines.append("\n")
            
            # Check success
            if override_applied and target_asin in ranked:
                best_rank = ranked.index(target_asin) + 1
                hit_turn = turn
                break
                
        is_hit = hit_turn is not None
        dialog_lines.append(f"=== RESULT: {'SUCCESS (Turn ' + str(hit_turn) + ')' if is_hit else 'FAILED'} ===\n")
        
        # Save session file
        session_file = Path("eval_sessions") / f"{sample['sample_id']}.txt"
        session_file.parent.mkdir(exist_ok=True)
        session_file.write_text("".join(dialog_lines), encoding="utf-8")
        
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": is_hit,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
        
        print(f"  Session {idx+1}/{len(selected_samples)}: {sample['sample_id']} | Scenario: {sample['scenario_type']} | {'SUCCESS (Turn ' + str(hit_turn) + ')' if is_hit else 'FAILED'}")
        
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    
    # Compute scenario breakdown
    grouped = {}
    for s in sessions:
        sc = s["scenario_type"]
        if sc not in grouped:
            grouped[sc] = []
        grouped[sc].append(s)
        
    scenario_metrics = {name: metric_summary(grouped[name]) for name in sorted(grouped)}
    
    metrics = {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "scenario_metrics": scenario_metrics
    }
    
    # Save results to a file
    output_path = Path("results_v2.json")
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("EVALUATOR V2 BATCH RESULTS:")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print(f"Total time: {time.time() - t_start:.2f} seconds")
    print("=" * 60)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluator v2 (Ollama LLM Shopper)")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name (default: llama3.1)")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to evaluate (default: 200 for full run)")
    args = parser.parse_args()
    
    agent = Agent()
    catalog_path = repo_root / "data/catalog.jsonl"
    dataset_path = repo_root / "data/public_set.jsonl"
    
    catalog_ids, categories, products = catalog_index(catalog_path)
    
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                
    evaluate_v2(agent, samples, set(catalog_ids), categories, products, args.model, args.samples)

if __name__ == "__main__":
    main()
