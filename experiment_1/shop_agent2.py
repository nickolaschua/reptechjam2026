import json
import re
import sqlite3
import requests
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent / "techjam-conversational-search"

class Agent:
    """Category Route Agent (shop_agent2)
    Uses NumPy boolean bitmasks to filter the catalog by categories,
    departments, and price limits, ranking matches by popularity.
    """
    def __init__(self, catalog_path: str | Path = None) -> None:
        if catalog_path is None:
            catalog_path = repo_root / "data/catalog.jsonl"
        self.catalog_path = Path(catalog_path)
        
        self._sessions = {}
        self._build_category_index()
        
    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "llama3.1",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.4
            }
        }
        try:
            res = requests.post(url, json=payload, timeout=5)
            res.raise_for_status()
            data = res.json()
            return data["message"]["content"].strip()
        except Exception:
            return "Here are the top matches based on your preferences."

    def _build_category_index(self) -> None:
        print("[Agent 2] Indexing Category and Price metadata...")
        self.catalog_ids = []
        self.catalog_prices = []
        self.catalog_departments = []
        self.catalog_categories_set = []  # List of sets of categories for each product
        self.catalog_popularity = []      # rating_number as a popularity proxy
        
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = str(p["parent_asin"])
                self.catalog_ids.append(pid)
                
                # Extract price
                price_val = p.get("price")
                try:
                    price_float = float(str(price_val).replace("$", "").replace(",", "").strip()) if price_val is not None else 9999.0
                except ValueError:
                    price_float = 9999.0
                self.catalog_prices.append(price_float)
                
                # Extract categories and department
                cats = p.get("categories") or []
                self.catalog_categories_set.append(set(c.lower() for c in cats))
                
                # Department is at Index 2 of category hierarchy (Clothing, Shoes, Jewelry, etc.)
                dept = ""
                if len(cats) > 2:
                    dept = cats[2].strip().lower()
                elif cats:
                    dept = cats[-1].strip().lower()
                self.catalog_departments.append(dept)
                
                # Popularity
                self.catalog_popularity.append(float(p.get("rating_number") or 0))

        # Convert to NumPy arrays for fast bitmask filtering
        self.catalog_ids = np.array(self.catalog_ids)
        self.catalog_prices = np.array(self.catalog_prices)
        self.catalog_departments = np.array(self.catalog_departments)
        self.catalog_popularity = np.array(self.catalog_popularity)
        print(f"[Agent 2] Indexing complete. Indexed {len(self.catalog_ids)} items.")

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "category": "clothing",          # Current specific category search
            "department": "",                # e.g., clothing, shoes, jewelry
            "price_max": 9999.0,            # e.g., budget limits
            "seen_asins": set(),
            "history": []                    # Dialogue history
        }

    def _extract_category_and_budget(self, session_id: str, message: str) -> None:
        state = self._sessions[session_id]
        msg_lower = message.lower()
        
        # 1. Extract category search (e.g. "I'm looking for t-shirts/boots")
        cat_match = re.search(r"i'm looking for ([^.,]+)", msg_lower)
        if cat_match:
            state["category"] = cat_match.group(1).strip()
            
        # 2. Heuristically classify high-level department
        if any(w in msg_lower for w in ["shoe", "boot", "sandal", "slide", "sneaker", "clog"]):
            state["department"] = "shoes"
        elif any(w in msg_lower for w in ["ring", "necklace", "earring", "bracelet", "jewelry"]):
            state["department"] = "jewelry"
        elif any(w in msg_lower for w in ["shirt", "pant", "hoodie", "jacket", "underwear", "socks", "onesie", "dress", "leotard"]):
            state["department"] = "clothing"
            
        # 3. Extract budget constraints (e.g. "under $80", "budget of $50")
        budget_match = re.search(r"(?:under|below|max|budget of)\s*\$?(\d+(?:\.\d+)?)", msg_lower)
        if budget_match:
            state["price_max"] = float(budget_match.group(1))
        elif "price" in msg_lower:
            # fallback if they mention cheap/expensive without numbers
            if "cheap" in msg_lower or "budget" in msg_lower:
                state["price_max"] = 40.0

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions[session_id]
        
        # Parse inputs for categories and budget limits
        self._extract_category_and_budget(session_id, user_message)
        
        # Build numpy bitmask
        mask = np.ones(len(self.catalog_ids), dtype=bool)
        
        # Filter by department if detected
        if state["department"]:
            mask &= (self.catalog_departments == state["department"])
            
        # Filter by price if set
        if state["price_max"] < 9999.0:
            mask &= (self.catalog_prices <= state["price_max"])
            
        # Filter by specific category token intersection
        target_cat = state["category"]
        if target_cat:
            cat_tokens = set(target_cat.split())
            cat_match_mask = []
            for item_cats in self.catalog_categories_set:
                # True if any search token intersects with the product categories
                cat_match_mask.append(bool(cat_tokens & item_cats))
            mask &= np.array(cat_match_mask)
            
        # Get matching indices
        matching_indices = np.where(mask)[0]
        
        # If mask is too restrictive (0 matches), relax category filter and keep department/price
        if len(matching_indices) == 0:
            mask_relaxed = np.ones(len(self.catalog_ids), dtype=bool)
            if state["department"]:
                mask_relaxed &= (self.catalog_departments == state["department"])
            if state["price_max"] < 9999.0:
                mask_relaxed &= (self.catalog_prices <= state["price_max"])
            matching_indices = np.where(mask_relaxed)[0]
            
        # Fallback to entire catalog if still empty
        if len(matching_indices) == 0:
            matching_indices = np.arange(len(self.catalog_ids))
            
        # Rank matching candidates by popularity (rating_number)
        match_popularity = self.catalog_popularity[matching_indices]
        sorted_local_indices = np.argsort(match_popularity)[::-1]
        sorted_global_indices = matching_indices[sorted_local_indices]
        
        # Slice top candidates, filtering out seen ASINs
        recommendations = []
        for idx in sorted_global_indices:
            asin = str(self.catalog_ids[idx])
            if asin not in state["seen_asins"]:
                recommendations.append(asin)
                if len(recommendations) == top_k:
                    break
                    
        # Fill if empty due to unseen filtering
        if len(recommendations) < top_k:
            for idx in sorted_global_indices:
                asin = str(self.catalog_ids[idx])
                if asin not in recommendations:
                    recommendations.append(asin)
                    if len(recommendations) == top_k:
                        break
                        
        # Record dialogue history
        state["history"].append({"role": "user", "content": user_message})
        
        state["seen_asins"].update(recommendations)
        
        # System prompt for Llama 3.1
        sys_prompt = (
            "You are a helpful e-commerce shopping copilot. The user is looking for a product.\n"
            f"Active search filters: Category: {state['category']}, Department: {state['department']}, Max Price: {state['price_max']}\n"
            "Based on the conversation, write a very short (1-2 sentences), natural response to the user. "
            "Acknowledge their request politely, present the recommendations, and ask a follow-up clarifying question to narrow down the search (e.g. asking about style, material, color, or brand if they haven't specified it yet)."
        )
        
        # Format recent history
        history_str = ""
        for msg in state["history"][-4:]:
            role = "Customer" if msg["role"] == "user" else "Copilot"
            history_str += f"{role}: {msg['content']}\n"
            
        prompt = f"Dialogue history:\n{history_str}\n\nCopilot Response:"
        
        agent_message = self._call_ollama(prompt, sys_prompt)
        state["history"].append({"role": "assistant", "content": agent_message})
        
        return {
            "message": agent_message,
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": r} for r in recommendations]
        }
