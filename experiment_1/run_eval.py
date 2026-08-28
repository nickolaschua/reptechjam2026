import json
import time
import sys
import statistics
from collections import defaultdict
from pathlib import Path
from agent import Agent

# Setup paths to import evaluator
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent / "techjam-conversational-search"
sys.path.insert(0, str(repo_root))

from evaluator.local_evaluator import (
    catalog_index,
    materialize_hidden_fields,
    coarse_category,
    initial_message,
    customer_reply,
    normalize_recommendations,
    metric_summary
)

def evaluate_custom(agent, samples, catalog_ids, categories, products):
    sessions = []
    catalog_ids_set = set(catalog_ids)
    failed_logs = []
    successful_logs = defaultdict(list)
    
    for sample in samples:
        session_id = f"public_sim_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        
        effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
        
        disclosed = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
        
        history = []
        hit_turn = None
        best_rank = None
        
        for turn in range(1, 11): # MAX_TURNS = 10
            response = agent.respond(session_id, user_message, turn, 10) # TOP_K = 10
            recs = [item.get("parent_asin") if isinstance(item, dict) else item for item in (response.get("recommendations") or [])]
            ranked = normalize_recommendations(recs, catalog_ids_set)
            
            # Log turn dialog and recommendation rankings
            history.append({
                "turn": turn,
                "customer": user_message,
                "copilot_ask": response.get("ask_attribute"),
                "copilot_msg": response.get("message") or "Here are the top matches.",
                "recommendations": ranked
            })
            
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
                
            if turn == 10:
                break
                
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                )
                
        is_hit = hit_turn is not None
        sessions.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": is_hit,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        })
        
        # Generate formatted log string
        p_target = products[target]
        log = []
        log.append("="*80)
        log.append(f"{'SUCCESSFUL' if is_hit else 'FAILED'} SESSION: {sample['sample_id']} | Scenario: {sample['scenario_type']}")
        log.append(f"Target Product: [{p_target.get('store', 'Unknown')}] {p_target.get('title')}")
        log.append(f"Target ASIN:    {target}")
        log.append(f"Target Constraints:")
        log.append(f"  Hard Constraints: {effective_intent_card.get('hard_constraints')}")
        log.append(f"  Soft Preferences: {effective_intent_card.get('soft_preferences')}")
        log.append("-" * 80)
        
        for turn_log in history:
            log.append(f"[Turn {turn_log['turn']}]")
            log.append(f"Customer: \"{turn_log['customer']}\"")
            log.append(f"Copilot: \"{turn_log['copilot_msg']}\"")
            log.append(f"Copilot requested attribute: {turn_log['copilot_ask']}")
            log.append(f"Copilot Top Recommendations:")
            for idx, r_asin in enumerate(turn_log['recommendations']):
                p_rec = products[r_asin]
                is_target = " <--- (TARGET WAS RECOMMENDED!)" if r_asin == target else ""
                log.append(f"  {idx+1}. [{p_rec.get('store', 'Unknown')}] {p_rec.get('title')[:75]}...{is_target}")
            log.append("")
        
        log.append("="*80 + "\n\n")
        log_str = "\n".join(log)
        
        if is_hit:
            successful_logs[sample["scenario_type"]].append(log_str)
        else:
            failed_logs.append(log_str)
            
    # Calculate overall metrics
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    
    grouped = defaultdict(list)
    for s in sessions:
        grouped[s["scenario_type"]].append(s)
        
    metrics = {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
    }
    
    # Save logs to a dedicated 'sessions' subdirectory
    sessions_dir = current_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    # Save failed logs
    with open(sessions_dir / "failed_sessions.txt", "w", encoding="utf-8") as f:
        f.write("".join(failed_logs))
        
    # Save successful logs by scenario
    scenarios = ["buying", "browsing", "boundary", "intent_override"]
    for scen in scenarios:
        logs_list = successful_logs[scen]
        filename = f"successful_{scen}.txt"
        with open(sessions_dir / filename, "w", encoding="utf-8") as f:
            f.write(f"=== SUCCESSFUL SESSIONS FOR SCENARIO: {scen.upper()} ({len(logs_list)} total successes) ===\n\n")
            f.write("".join(logs_list))
        
    return metrics

def main():
    print("[Experiment 1] Initializing Agent...")
    t0 = time.time()
    agent = Agent()
    print(f"[Experiment 1] Agent initialized in {time.time() - t0:.2f}s!")
    
    # Load catalog and dev set
    catalog_path = repo_root / "data/catalog.jsonl"
    dataset_path = repo_root / "data/public_set.jsonl"
    
    print("[Experiment 1] Loading catalog index...")
    catalog_ids, categories, products = catalog_index(catalog_path)
    
    print("[Experiment 1] Loading public set samples...")
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            
    print(f"[Experiment 1] Running evaluator over {len(samples)} sessions...")
    t_start = time.time()
    metrics = evaluate_custom(agent, samples, catalog_ids, categories, products)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 EVALUATION RESULTS:")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print(f"Total time: {time.time() - t_start:.2f} seconds")
    print(f"Average time per session: {(time.time() - t_start) / len(samples) * 1000:.1f} ms")
    print(f"Failed sessions saved to: experiment_1/sessions/failed_sessions.txt")
    print(f"Successful sessions saved to: experiment_1/sessions/ (e.g., successful_buying.txt)")
    print("=" * 60)

if __name__ == "__main__":
    main()
