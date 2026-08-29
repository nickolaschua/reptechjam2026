import sys
import json
import time
import os
import re
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Setup paths to import agent
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent / "techjam-conversational-search"
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(repo_root))

from shop_agent import Agent
from shopper_agent import make_system_prompt, call_ollama, coarse_category
from evaluator.local_evaluator import catalog_index, materialize_hidden_fields, normalize_recommendations

GLOBAL_AGENT = None
GLOBAL_CATALOG_IDS = None
GLOBAL_CATEGORIES = None
GLOBAL_PRODUCTS = None
active_manual_sessions = {}

class VisualizerHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from the visualizer directory
        super().__init__(*args, directory=str(current_dir), **kwargs)

    def do_GET(self):
        url_parsed = urllib.parse.urlparse(self.path)
        if url_parsed.path == "/stream":
            self.handle_stream(url_parsed.query)
        elif url_parsed.path == "/sessions_list":
            self.handle_sessions_list()
        elif url_parsed.path == "/manual_start":
            self.handle_manual_start(url_parsed.query)
        elif url_parsed.path == "/manual_step":
            self.handle_manual_step(url_parsed.query)
        else:
            super().do_GET()

    def send_json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
        
    def send_json_error(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode("utf-8"))

    def handle_sessions_list(self):
        dataset_path = repo_root / "data/public_set.jsonl"
        
        if not dataset_path.exists():
            self.send_json_error(404, "dataset file not found")
            return
            
        samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
                
        list_data = []
        for s in samples:
            target = str(s["ground_truth"]["parent_asin"])
            prod = GLOBAL_PRODUCTS.get(target, {})
            list_data.append({
                "sample_id": s["sample_id"],
                "scenario": s["scenario_type"],
                "target_title": prod.get("title", "Unknown"),
                "target_brand": prod.get("store") or prod.get("details", {}).get("Manufacturer") or "Unknown"
            })
            
        self.send_json_response(list_data)

    def handle_manual_start(self, query):
        params = urllib.parse.parse_qs(query)
        sample_id = params.get("sample_id", ["public_0001"])[0]
        
        dataset_path = repo_root / "data/public_set.jsonl"
        sample = None
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                if s["sample_id"] == sample_id:
                    sample = s
                    break
                    
        if not sample:
            self.send_json_error(404, "Sample not found.")
            return
            
        target_asin = str(sample["ground_truth"]["parent_asin"])
        target_product = GLOBAL_PRODUCTS[target_asin]
        card, behavior = materialize_hidden_fields(sample, GLOBAL_PRODUCTS)
        
        session_id = f"manual_sim_{sample_id}"
        GLOBAL_AGENT.reset(session_id, sample["user_profile"])
        
        active_manual_sessions[session_id] = {
            "turn": 0,
            "target_asin": target_asin,
            "sample": sample,
            "card": card,
            "behavior": behavior
        }
        
        self.send_json_response({
            "session_id": sample["sample_id"],
            "scenario": sample["scenario_type"],
            "target_asin": target_asin,
            "target_title": target_product.get("title"),
            "target_brand": target_product.get("store") or target_product.get("details", {}).get("Manufacturer") or "Unknown",
            "hard_constraints": card.get("hard_constraints", []),
            "soft_preferences": card.get("soft_preferences", [])
        })

    def handle_manual_step(self, query):
        params = urllib.parse.parse_qs(query)
        sample_id = params.get("sample_id", ["public_0001"])[0]
        message = params.get("message", [""])[0]
        
        session_id = f"manual_sim_{sample_id}"
        if session_id not in active_manual_sessions:
            self.send_json_error(400, "Session not initialized.")
            return
            
        session_state = active_manual_sessions[session_id]
        session_state["turn"] += 1
        turn = session_state["turn"]
        target_asin = session_state["target_asin"]
        
        # Respond
        copilot_response = GLOBAL_AGENT.respond(session_id, message, turn, 10)
        ranked = normalize_recommendations(copilot_response.get("recommendations"), GLOBAL_CATALOG_IDS)
        
        recs_meta = [GLOBAL_PRODUCTS[pid] for pid in ranked]
        recs_data = []
        for i, rec in enumerate(recs_meta):
            recs_data.append({
                "rank": i+1,
                "brand": rec.get("store") or rec.get("details", {}).get("Manufacturer") or "Unknown",
                "title": rec.get("title"),
                "asin": rec.get("parent_asin"),
                "is_target": rec.get("parent_asin") == target_asin
            })
            
        is_hit = target_asin in ranked
        
        self.send_json_response({
            "message": copilot_response.get("message"),
            "ask_attribute": copilot_response.get("ask_attribute"),
            "recommendations": recs_data,
            "turn": turn,
            "success": is_hit,
            "debug": copilot_response.get("debug")
        })

    def handle_stream(self, query):
        params = urllib.parse.parse_qs(query)
        target_sample_id = params.get("sample_id", ["public_0001"])[0]
        
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        dataset_path = repo_root / "data/public_set.jsonl"
        sample = None
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                if s["sample_id"] == target_sample_id:
                    sample = s
                    break
                    
        if not sample:
            self.send_sse("error", "Sample not found.")
            return
            
        target_asin = str(sample["ground_truth"]["parent_asin"])
        target_product = GLOBAL_PRODUCTS[target_asin]
        coarse_cat = coarse_category(GLOBAL_CATEGORIES.get(target_asin, []))
        
        card, behavior = materialize_hidden_fields(sample, GLOBAL_PRODUCTS)
        
        # Send initial target product info
        self.send_sse("target", {
            "session_id": sample["sample_id"],
            "scenario": sample["scenario_type"],
            "target_asin": target_asin,
            "target_title": target_product.get("title"),
            "target_brand": target_product.get("store") or target_product.get("details", {}).get("Manufacturer") or "Unknown",
            "hard_constraints": card.get("hard_constraints", []),
            "soft_preferences": card.get("soft_preferences", [])
        })
        
        # Initialize Agent under test
        session_id = f"web_sim_{target_sample_id}"
        GLOBAL_AGENT.reset(session_id, sample["user_profile"])
        
        effective_sample = {**sample, "intent_card": card, "behavior": behavior}
        system_prompt = make_system_prompt(effective_sample, target_product, coarse_cat)
        user_message = ""
        override = behavior.get("override", {})
        override_applied = sample["scenario_type"] != "intent_override"
        
        recs_meta = []
        
        for turn in range(1, 11):
            # 1. Shopper Turn (Calls Ollama)
            if turn == 1:
                prompt = "Start the conversation by telling the assistant what you are looking for."
                if sample["scenario_type"] == "intent_override":
                    old_val = override.get("old_value", "")
                    prompt += f" Remember to mention your initial preference for: '{old_val}'."
                elif sample["scenario_type"] == "buying" and card.get("hard_constraints"):
                    hard_val = card["hard_constraints"][0]
                    prompt += f" Mention your key requirement: '{hard_val}'."
                user_message = call_ollama(prompt, system_prompt, "llama3.1")
            elif not override_applied and turn == int(override.get("turn", 3)):
                override_applied = True
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
                user_message = call_ollama(prompt, system_prompt, "llama3.1")
                
            # Send customer message live to browser
            self.send_sse("msg", {"role": "customer", "content": user_message, "turn": turn})
            
            # 2. Copilot Response
            time.sleep(0.5)
            copilot_response = GLOBAL_AGENT.respond(session_id, user_message, turn, 10)
            ranked = normalize_recommendations(copilot_response.get("recommendations"), GLOBAL_CATALOG_IDS)
            recs_meta = [GLOBAL_PRODUCTS[pid] for pid in ranked]
            
            recs_data = []
            for i, rec in enumerate(recs_meta):
                recs_data.append({
                    "rank": i+1,
                    "brand": rec.get("store") or rec.get("details", {}).get("Manufacturer") or "Unknown",
                    "title": rec.get("title"),
                    "asin": rec.get("parent_asin"),
                    "is_target": rec.get("parent_asin") == target_asin
                })
                
            # Send Copilot message live to browser
            self.send_sse("msg", {
                "role": "copilot",
                "content": copilot_response.get("message"),
                "ask": copilot_response.get("ask_attribute"),
                "recommendations": recs_data,
                "turn": turn,
                "debug": copilot_response.get("debug")
            })
            
            # 3. Check Success condition
            if override_applied and target_asin in ranked:
                best_rank = ranked.index(target_asin) + 1
                self.send_sse("status", {
                    "success": True,
                    "turn": turn,
                    "rank": best_rank
                })
                break
                
            if turn == 10:
                self.send_sse("status", {
                    "success": False,
                    "turn": 10
                })

    def send_sse(self, event, data):
        try:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
            self.wfile.flush()
        except Exception as e:
            print(f"[Server SSE Error] {e}")

def run_server(port=8080):
    global GLOBAL_AGENT, GLOBAL_CATALOG_IDS, GLOBAL_CATEGORIES, GLOBAL_PRODUCTS
    catalog_path = repo_root / "data/catalog.jsonl"
    print("[Server] Pre-loading catalog index and embeddings on startup...")
    GLOBAL_CATALOG_IDS, GLOBAL_CATEGORIES, GLOBAL_PRODUCTS = catalog_index(catalog_path)
    GLOBAL_AGENT = Agent(catalog_path)
    
    server = HTTPServer(("0.0.0.0", port), VisualizerHTTPHandler)
    print(f"[Server] Live Conversational Simulator running at: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Server] Shutting down.")
        server.server_close()

if __name__ == "__main__":
    run_server(8080)
