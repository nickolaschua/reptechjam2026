import os
import re
import sys
import json
import uuid
import random
import argparse
import requests
from pathlib import Path

# Setup paths relative to this script
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent / "techjam-conversational-search"
sys.path.insert(0, str(repo_root))

# Helper to automatically parse .env file at repo root if exists
def load_env_file():
    env_path = current_dir.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # Strip spaces and optional quotes
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    # Set in process environment if not already set
                    if key and val and not os.environ.get(key):
                        os.environ[key] = val

load_env_file()

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    intent_card,
    behavior_for,
    normalize_recommendations
)
from agent import Agent

MAX_TURNS = 10
TOP_K = 10

# ANSI colors for terminal prints
COLOR_USER = "\033[92m"       # Green
COLOR_COPILOT = "\033[94m"    # Blue
COLOR_SYSTEM = "\033[93m"     # Yellow
COLOR_RESET = "\033[0m"

def materialize_hidden_fields(sample: dict, products: dict[str, dict]) -> tuple[dict, dict]:
    if "intent_card" in sample and "behavior" in sample:
        return sample["intent_card"], sample["behavior"]
    target = str(sample["ground_truth"]["parent_asin"])
    product = products[target]
    card = intent_card(product)
    seed_source = f"{sample.get('sample_id', '')}\0{sample.get('scenario_type', '')}"
    rng = random.Random(seed_source)
    behavior = behavior_for(str(sample["scenario_type"]), card, rng)
    return card, behavior

def call_llm(prompt: str, system_prompt: str = "") -> str:
    gemini_key = os.environ.get("GEMINI_API_KEY")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    
    # Clean placeholders
    if gemini_key and (gemini_key.startswith("your_") or "placeholder" in gemini_key.lower()):
        gemini_key = None
    if deepseek_key and (deepseek_key.startswith("your_") or "placeholder" in deepseek_key.lower()):
        deepseek_key = None
        
    if not gemini_key and not deepseek_key:
        raise ValueError("Neither GEMINI_API_KEY nor DEEPSEEK_API_KEY is set in environment.")
        
    # Prioritize DeepSeek if it starts with sk- or if Gemini key is invalid/OAuth token
    use_deepseek = False
    if deepseek_key and (deepseek_key.startswith("sk-") or not gemini_key or not gemini_key.startswith("AIzaSy")):
        use_deepseek = True
        
    if not use_deepseek and gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}]
        }
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        if res.status_code != 200:
            print(f"\n{COLOR_SYSTEM}[System API Error] Status Code: {res.status_code}")
            print(f"Response Body: {res.text}{COLOR_RESET}\n")
        res.raise_for_status()
        data = res.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    else:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {deepseek_key}"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4
        }
        res = requests.post(url, headers=headers, json=payload, timeout=20)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()

def make_system_prompt(sample: dict, product: dict, target_category: str) -> str:
    card = sample["intent_card"]
    behavior = sample["behavior"]
    scenario = sample["scenario_type"]
    
    prompt = (
        "You are acting as a real customer shopping online. Your target product you want to find is:\n"
        f"Target Product Title: {product.get('title')}\n"
        f"Category: {target_category}\n"
        f"Hard Constraints (Must-Haves): {', '.join(card.get('hard_constraints', []))}\n"
        f"Soft Preferences (Nice-to-Haves): {', '.join(card.get('soft_preferences', []))}\n"
        f"Ground Truth ASIN: {product.get('parent_asin')}\n\n"
        f"Your Shopping Profile:\n"
        f"Purchase Frequency: {sample['user_profile'].get('purchase_frequency', 'Regular')}\n"
        f"Preference Tags: {', '.join(sample['user_profile'].get('preference_tags', []))}\n"
        f"Summary: {sample['user_profile'].get('summary', '')}\n\n"
        f"Active Scenario: {scenario.upper()}\n"
    )
    
    if scenario == "intent_override":
        override = behavior.get("override", {})
        prompt += (
            f"Instruction for Intent Override:\n"
            f"- For the first few turns, you prefer the soft style: '{override.get('old_value')}'\n"
            f"- When the turn counter hits {override.get('turn')}, you must override your preference. "
            f"You will say: '{override.get('message')}' and pivot your search to require: '{override.get('new_value')}'\n\n"
        )
    elif scenario == "boundary":
        prompt += (
            "Instruction for Boundary Case:\n"
            "- If the shopping assistant asks about a specific attribute that you don't have a preference for, "
            "you should say you don't have a preference for it and ask them to use their judgment.\n\n"
        )
        
    prompt += (
        "Rules of Dialogue:\n"
        "1. Prioritize your Hard Constraints (must-haves) first. Disclose your Soft Preferences (nice-to-haves) using softer, negotiable language (e.g., 'ideally...', 'I'd also prefer...').\n"
        "2. Do NOT state the product title or ASIN directly to the assistant. Speak like a real human.\n"
        "3. Answer the assistant's questions in a natural conversational way. Use synonyms or subjective phrases instead of copy-pasting.\n"
        "4. Review the assistant's suggestions. If the correct product is recommended in the list, you should recognize it and state that you want to buy/select it (which will end the conversation).\n"
        "5. Keep your responses short and conversational (1-2 sentences)."
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description="LLM-based Shopper Simulator stress test")
    parser.add_argument("--asin", help="Specify a targeted Ground Truth ASIN to run")
    parser.add_argument("--index", type=int, help="Index of the sample in the 200 public set (0-199)")
    parser.add_argument("--scenario", choices=["buying", "browsing", "intent_override", "boundary"], help="Scenario filter")
    args = parser.parse_args()

    # Check for keys
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("DEEPSEEK_API_KEY"):
        print(f"{COLOR_SYSTEM}[ERROR] Neither GEMINI_API_KEY nor DEEPSEEK_API_KEY is found in the environment.{COLOR_RESET}")
        print("Please export your API key before running this script.")
        return

    # Load dataset & catalog
    dataset_path = repo_root / "data/public_set.jsonl"
    catalog_path = repo_root / "data/catalog.jsonl"
    
    print(f"{COLOR_SYSTEM}[System] Loading catalog and public set...{COLOR_RESET}")
    catalog_ids, categories, products = catalog_index(catalog_path)
    
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Selection Logic
    sample = None
    if args.asin:
        target_asin = args.asin.strip()
        if target_asin not in products:
            print(f"{COLOR_SYSTEM}[ERROR] ASIN {target_asin} not found in catalog.{COLOR_RESET}")
            return
        product = products[target_asin]
        card = intent_card(product)
        sample = {
            "sample_id": "custom_asin",
            "scenario_type": args.scenario or "buying",
            "ground_truth": {"parent_asin": target_asin},
            "user_profile": {
                "summary": "Targeted stress test search.",
                "preference_tags": ["fit", "comfort"]
            },
            "intent_card": card,
            "behavior": behavior_for(args.scenario or "buying", card, random.Random())
        }
    elif args.index is not None:
        if args.index < 0 or args.index >= len(samples):
            print(f"{COLOR_SYSTEM}[ERROR] Index {args.index} out of bounds.{COLOR_RESET}")
            return
        sample = samples[args.index]
    elif args.scenario:
        matched = [s for s in samples if s["scenario_type"].lower() == args.scenario.lower()]
        if not matched:
            print(f"{COLOR_SYSTEM}[ERROR] No samples found matching scenario: {args.scenario}{COLOR_RESET}")
            return
        sample = random.choice(matched)
    else:
        sample = random.choice(samples)

    # Materialize intents
    card, behavior = materialize_hidden_fields(sample, products)
    sample = {**sample, "intent_card": card, "behavior": behavior}
    
    target_asin = sample["ground_truth"]["parent_asin"]
    target_product = products[target_asin]
    coarse_cat = coarse_category(categories.get(target_asin, []))

    print(f"\n{COLOR_SYSTEM}==================================================")
    print("STRESS TEST SIMULATION STARTED")
    print(f"Sample ID: {sample.get('sample_id')}")
    print(f"Scenario: {sample['scenario_type']}")
    print(f"Target ASIN: {target_asin}")
    print(f"Target Title: {target_product.get('title')}")
    print(f"Target Details: {coarse_cat}")
    print(f"Constraints: {card.get('hard_constraints', [])} | {card.get('soft_preferences', [])}")
    print(f"=================================================={COLOR_RESET}\n")

    # Initialize Copilot Agent under test
    agent = Agent(catalog_path)
    session_id = f"stress_test_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    # LLM User state and prompts setup
    system_prompt = make_system_prompt(sample, target_product, coarse_cat)
    history = []
    
    # Run loop
    success = False
    hit_turn = -1
    best_rank = -1
    
    # Turn 1 initial greeting from Copilot or user starts
    user_message = ""
    override = behavior.get("override", {})
    override_applied = sample["scenario_type"] != "intent_override"

    for turn in range(1, MAX_TURNS + 1):
        print(f"{COLOR_SYSTEM}--- TURN {turn} ---{COLOR_RESET}")
        
        # 1. Shopper Action
        if turn == 1:
            # Generate initial query
            prompt = "Start the conversation by telling the assistant what you are looking for."
            if sample["scenario_type"] == "intent_override":
                old_val = override.get("old_value", "")
                prompt += f" Remember to mention your initial preference for: '{old_val}'."
            elif sample["scenario_type"] == "buying" and card.get("hard_constraints"):
                hard_val = card["hard_constraints"][0]
                prompt += f" Mention your key requirement: '{hard_val}'."
                
            user_message = call_llm(prompt, system_prompt)
        elif not override_applied and turn == int(override.get("turn", 3)):
            # Override turn
            override_applied = True
            user_message = override.get("message", "Actually, ignore my earlier preference.")
        else:
            # Standard conversational turn
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
            user_message = call_llm(prompt, system_prompt)

        print(f"{COLOR_USER}Shopper: {user_message}{COLOR_RESET}")
        history.append({"role": "user", "content": user_message})

        # 2. Copilot Response
        try:
            copilot_response = agent.respond(session_id, user_message, turn, TOP_K)
        except Exception as e:
            print(f"{COLOR_SYSTEM}[ERROR] Agent crashed: {e}{COLOR_RESET}")
            break

        print(f"{COLOR_COPILOT}Copilot: {copilot_response.get('message')}{COLOR_RESET}")
        if copilot_response.get("ask_attribute"):
            print(f"{COLOR_COPILOT}Copilot Attribute Query: {copilot_response.get('ask_attribute')}{COLOR_RESET}")
            
        ranked = normalize_recommendations(copilot_response.get("recommendations"), catalog_ids)
        recs_meta = [products[pid] for pid in ranked]
        
        # Display recommendations in terminal
        print(f"{COLOR_COPILOT}Copilot Recommendations:{COLOR_RESET}")
        for i, rec in enumerate(recs_meta):
            print(f"  {i+1}. {rec.get('title')[:75]}... (ASIN: {rec.get('parent_asin')})")
            
        # 3. Check Success condition (Target recommended)
        if override_applied and target_asin in ranked:
            success = True
            hit_turn = turn
            best_rank = ranked.index(target_asin) + 1
            print(f"\n{COLOR_USER}[Success] Target ASIN found at Rank {best_rank} on Turn {hit_turn}!{COLOR_RESET}\n")
            break
            
    if success:
        print(f"{COLOR_SYSTEM}=== Stress Test Status: SUCCESS (Turn {hit_turn}, Rank {best_rank}) ==={COLOR_RESET}")
    else:
        print(f"{COLOR_SYSTEM}=== Stress Test Status: FAILED (Target not found in {MAX_TURNS} turns) ==={COLOR_RESET}")

if __name__ == "__main__":
    main()
