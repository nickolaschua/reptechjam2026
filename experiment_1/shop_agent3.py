import json
import re
import os
import time
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent / "techjam-conversational-search"

class Agent:
    """Vector Route Agent (shop_agent3)
    Uses a local SentenceTransformer model (MiniLM) and Maximum Inner Product
    Search (MIPS) in RAM to retrieve the closest semantic product matches.
    """
    def __init__(self, catalog_path: str | Path = None) -> None:
        if catalog_path is None:
            catalog_path = repo_root / "data/catalog.jsonl"
        self.catalog_path = Path(catalog_path)
        
        # 1. Load the SentenceTransformer model
        finetuned_model_path = current_dir.parent / "yangxu/model_finetuned"
        if finetuned_model_path.exists():
            self.model_path = str(finetuned_model_path)
            print(f"[Agent 3] Loading fine-tuned model: {self.model_path}")
        else:
            self.model_path = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"[Agent 3] Loading base model: {self.model_path}")
            
        self.model = SentenceTransformer(self.model_path)
        self._sessions = {}
        self._build_vector_index()
        
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

    def _build_vector_index(self) -> None:
        # Load catalog titles/texts
        self.catalog_ids = []
        self.catalog_texts = []
        self.catalog_products = {}
        
        print("[Agent 3] Loading catalog metadata...")
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = str(p["parent_asin"])
                self.catalog_ids.append(pid)
                self.catalog_products[pid] = p
                
                title = p.get("title") or ""
                cats = ", ".join(p.get("categories") or [])
                feats = "; ".join((p.get("features") or [])[:3])
                text = f"Product: {title}. Categories: {cats}. Features: {feats}.".strip()
                self.catalog_texts.append(text)

        # Cache file configuration to avoid re-encoding
        model_name_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", self.model_path)
        cache_path = current_dir / f"catalog_cache_{model_name_clean}.npz"
        
        if cache_path.exists():
            print(f"[Agent 3] Loading pre-computed embeddings from cache: {cache_path.name}...")
            data = np.load(cache_path)
            self.catalog_embeddings = data["embeddings"]
            cached_ids = list(data["ids"])
            if cached_ids == self.catalog_ids:
                print("[Agent 3] Cached embeddings loaded successfully.")
                return
            else:
                print("[Agent 3] Cache ID mismatch, re-encoding...")

        print(f"[Agent 3] Encoding {len(self.catalog_texts)} products. This will take a moment...")
        t0 = time.time()
        embeddings = self.model.encode(self.catalog_texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
        # L2 Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.catalog_embeddings = embeddings / np.maximum(norms, 1e-12)
        
        # Save to cache
        np.savez_compressed(cache_path, embeddings=self.catalog_embeddings, ids=self.catalog_ids)
        print(f"[Agent 3] Encoding complete and cached in {time.time() - t0:.2f}s!")

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "accumulated_queries": [],   # Dialog turns text
            "seen_asins": set(),
            "history": []                # Dialogue history
        }

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions[session_id]
        
        # Accumulate conversational search context
        state["accumulated_queries"].append(user_message)
        
        # Construct target query string by merging dialog inputs
        query_text = " ".join(state["accumulated_queries"])
        
        # Compute normalized query vector
        query_emb = self.model.encode(query_text, convert_to_numpy=True)
        q_norm = np.linalg.norm(query_emb)
        query_emb_normalized = query_emb / max(q_norm, 1e-12)
        
        # Compute dot product (Cosine Similarity since both vectors are normalized)
        scores = np.dot(self.catalog_embeddings, query_emb_normalized)
        
        # Retrieve ranked matches
        sorted_global_indices = np.argsort(scores)[::-1]
        
        recommendations = []
        for idx in sorted_global_indices:
            asin = self.catalog_ids[idx]
            if asin not in state["seen_asins"]:
                recommendations.append(asin)
                if len(recommendations) == top_k:
                    break
                    
        # Fallback fill
        if len(recommendations) < top_k:
            for idx in sorted_global_indices:
                asin = self.catalog_ids[idx]
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
