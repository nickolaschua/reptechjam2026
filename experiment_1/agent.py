from __future__ import annotations
import os
import re
import sys
import json
import time
import requests
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent / "techjam-conversational-search"

# Allowed attributes set by the evaluator
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other"
}

# Standard keyword lists for local parsing
COLORS = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"}
MATERIALS = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "color", "budget", "around", "think", "mistake", "options", "items",
    "guess", "sure", "actually", "sorry", "thanks", "yes", "no", "look",
    "need", "prefer", "like", "find", "search", "show", "give", "bring",
    "tell", "ask", "reapply", "filters", "results", "suggestions", "pairs",
    "original", "um", "umm", "uh", "ignore", "earlier", "preference",
    "preferences", "character", "buy", "see", "seem", "seems", "get", "got",
    "make", "made", "done", "run", "doing"
}

def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)

def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]

class Agent:
    """Unified Hybrid Agent (DP1/DP2 Cascade)
    1. Tries Keyword string matching (FTS5) first.
    2. Falls back to Category (NumPy bitmask) and Vector (MIPS semantic search) routes if FTS5 has low confidence.
    3. Uses local Llama 3.1 to generate conversational clarifying questions.
    """
    def __init__(self, catalog_path: str | Path = None) -> None:
        if catalog_path is None:
            catalog_path = repo_root / "data/catalog.jsonl"
        self.catalog_path = Path(catalog_path)
        
        self.connection = sqlite3.connect(":memory:")
        self._sessions = {}
        
        # Build indexes
        self._build_fts5_index()
        self._build_category_index()
        
        # Load SentenceTransformer model
        finetuned_model_path = current_dir.parent / "yangxu/model_finetuned"
        if finetuned_model_path.exists():
            self.model_path = str(finetuned_model_path)
            print(f"[Hybrid Agent] Loading fine-tuned model: {self.model_path}")
        else:
            self.model_path = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"[Hybrid Agent] Loading base model: {self.model_path}")
            
        self.model = SentenceTransformer(self.model_path)
        self._build_vector_index()

    def _build_fts5_index(self) -> None:
        print("[Hybrid Agent] Indexing SQLite FTS5 database...")
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch = []
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                product = json.loads(line)
                batch.append((
                    str(product["parent_asin"]),
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        print("[Hybrid Agent] FTS5 index built.")

    def _build_category_index(self) -> None:
        print("[Hybrid Agent] Building Category metadata...")
        self.catalog_ids = []
        self.catalog_prices = []
        self.catalog_departments = []
        self.catalog_categories_set = []
        self.catalog_popularity = []
        self.catalog_metadata = {}
        
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = str(p["parent_asin"])
                self.catalog_ids.append(pid)
                
                price_val = p.get("price")
                try:
                    price_float = float(str(price_val).replace("$", "").replace(",", "").strip()) if price_val is not None else 9999.0
                except ValueError:
                    price_float = 9999.0
                self.catalog_prices.append(price_float)
                
                cats = p.get("categories") or []
                self.catalog_categories_set.append(set(c.lower() for c in cats))
                
                dept = ""
                if len(cats) > 2:
                    dept = cats[2].strip().lower()
                elif cats:
                    dept = cats[-1].strip().lower()
                self.catalog_departments.append(dept)
                
                pop = float(p.get("rating_number") or 0)
                self.catalog_popularity.append(pop)
                
                # Metadata dictionary for fast post-retrieval scoring
                title = p.get("title") or ""
                brand = p.get("store") or p.get("details", {}).get("Manufacturer") or ""
                search_bag = (title + " " + " ".join(cats) + " " + " ".join(p.get("features") or [])).lower()
                
                self.catalog_metadata[pid] = {
                    "title": title,
                    "brand": brand.strip().lower(),
                    "rating_number": pop,
                    "searchable_bag": search_bag
                }
                
        self.catalog_ids_arr = np.array(self.catalog_ids)
        self.catalog_prices = np.array(self.catalog_prices)
        self.catalog_departments = np.array(self.catalog_departments)
        self.catalog_popularity = np.array(self.catalog_popularity)
        print("[Hybrid Agent] Category metadata loaded.")

    def _build_vector_index(self) -> None:
        self.catalog_texts = []
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                title = p.get("title") or ""
                cats = ", ".join(p.get("categories") or [])
                feats = "; ".join((p.get("features") or [])[:3])
                text = f"Product: {title}. Categories: {cats}. Features: {feats}.".strip()
                self.catalog_texts.append(text)

        model_name_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", self.model_path)
        cache_path = current_dir / f"catalog_cache_{model_name_clean}.npz"
        
        if cache_path.exists():
            print(f"[Hybrid Agent] Loading pre-computed embeddings: {cache_path.name}")
            data = np.load(cache_path)
            self.catalog_embeddings = data["embeddings"]
            return

        print(f"[Hybrid Agent] Encoding {len(self.catalog_texts)} products...")
        embeddings = self.model.encode(self.catalog_texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.catalog_embeddings = embeddings / np.maximum(norms, 1e-12)
        np.savez_compressed(cache_path, embeddings=self.catalog_embeddings, ids=self.catalog_ids)
        print("[Hybrid Agent] Vector index built and cached.")

    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        if os.environ.get("FAST_EVAL") == "1":
            return "Here are the top matches based on your preferences."
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

    def _classify_constraint_locally(self, val: str) -> str:
        """Categorize a value into one of the evaluator's allowed attributes."""
        val_lower = val.lower()
        if "budget" in val_lower or re.search(r"(?:\$|<=|under)\s*\d", val_lower):
            return "budget"
        if any(m in val_lower for m in MATERIALS):
            return "material"
        if any(c in val_lower for c in COLORS):
            return "color"
        if any(word in val_lower for word in ("size", "sizing", "width", "wide", "narrow")):
            return "size"
        if any(word in val_lower for word in ("sole", "heel", "wedge", "cushion", "rubber", "flat")):
            return "sole"
        if any(word in val_lower for word in ("department", "style", "fit", "sleeve", "neck", "combat", "fashion", "riding", "chelsea", "casual", "dressy", "western", "cowboy", "rain", "snow", "bootie")):
            return "style"
        return "feature"

    def _erase_attribute_memory(self, state: dict, attr: str) -> None:
        """Purge the attribute from active slot memory, stash its keywords, and update terms."""
        old_val = state["disclosed_slots"].pop(attr, None)
        if old_val:
            val_terms = []
            if isinstance(old_val, set):
                for val in old_val:
                    val_terms.extend(_terms(val))
            else:
                val_terms.extend(_terms(old_val))
                
            for w in val_terms:
                if w not in state["stashed_terms"]:
                    state["stashed_terms"].append(w)
                if w in state["accumulated_terms"]:
                    try:
                        state["accumulated_terms"].remove(w)
                    except ValueError:
                        pass

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = {
            "disclosed_slots": {},
            "accumulated_terms": [],
            "stashed_terms": [],
            "seen_asins": set(),
            "history": [],
            "negated_terms": set(),
            "asked_attributes": set(),
            # Slot attributes
            "category": "clothing",
            "department": "",
            "price_max": 9999.0,
        }

    def _parse_message_locally(self, session_id: str, message: str) -> None:
        state = self._sessions[session_id]
        msg_lower = message.lower()
        
        # 1. Check for Intent Override
        if "what i need is:" in msg_lower:
            idx = msg_lower.find("what i need is:")
            val_part = message[idx + len("what i need is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            attr = self._classify_constraint_locally(val_part)
            
            # Erase all prior active slots and stash them to avoid query pollution
            for a in list(state["disclosed_slots"].keys()):
                self._erase_attribute_memory(state, a)
                
            # Move all current accumulated terms to stashed_terms
            for term in state["accumulated_terms"]:
                if term not in state["stashed_terms"]:
                    state["stashed_terms"].append(term)
            state["accumulated_terms"] = []
            
            # Set the new slot
            state["disclosed_slots"][attr] = val_part
            state["seen_asins"].clear()
            
            # Re-seed accumulated terms with category + new message terms
            for term in _terms(state["category"]):
                if term not in state["accumulated_terms"]:
                    state["accumulated_terms"].append(term)
            for term in _terms(val_part):
                if term not in state["accumulated_terms"] and term not in state["negated_terms"]:
                    state["accumulated_terms"].append(term)
            return

        # 2. Check for Boundary Case
        if "i don't have a preference for" in msg_lower:
            boundary_match = re.search(r"i don't have a preference for ([^;.]+)", msg_lower)
            if boundary_match:
                attr = boundary_match.group(1).strip().lower()
                if "asked_attributes" not in state:
                    state["asked_attributes"] = set()
                state["asked_attributes"].add(attr)
                self._erase_attribute_memory(state, attr)
                return

        # 3. Extract Negated Terms
        negated_terms = set()
        negation_matches = re.finditer(r"\b(not|no|except|without|instead of)\b\s*([a-zA-Z0-9\s,]+)", msg_lower)
        for match in negation_matches:
            segment = match.group(2)
            words = re.split(r"\b(or|and|,)\b", segment)
            for w in words:
                cleaned = w.strip()
                cleaned = re.sub(r"[^a-z0-9]", "", cleaned)
                if len(cleaned) > 2 and cleaned not in STOPWORDS:
                    negated_terms.add(cleaned)
                    
        if "negated_terms" not in state:
            state["negated_terms"] = set()
        state["negated_terms"].update(negated_terms)
        
        # 4. Extract Category, Department, Budget slots
        cat_match = re.search(r"i'm looking for ([^.,]+)", msg_lower)
        if cat_match:
            cand_cat = cat_match.group(1).strip()
            if len(cand_cat.split()) <= 4:
                state["category"] = cand_cat
            else:
                state["category"] = " ".join(cand_cat.split()[:4])
            
        if any(w in msg_lower for w in ["shoe", "boot", "sandal", "slide", "sneaker", "clog"]):
            state["department"] = "shoes"
        elif any(w in msg_lower for w in ["ring", "necklace", "earring", "bracelet", "jewelry"]):
            state["department"] = "jewelry"
        elif any(w in msg_lower for w in ["shirt", "pant", "hoodie", "jacket", "underwear", "socks", "onesie", "dress", "leotard"]):
            state["department"] = "clothing"
            
        budget_match = re.search(r"(?:under|below|max|budget of)\s*\$?(\d+(?:\.\d+)?)", msg_lower)
        if budget_match:
            state["price_max"] = float(budget_match.group(1))

        # 5. Extract standard attributes (matters is, key requirement is)
        if "a key requirement is:" in msg_lower:
            idx = msg_lower.find("a key requirement is:")
            val_part = message[idx + len("a key requirement is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            attr = self._classify_constraint_locally(val_part)
            state["disclosed_slots"][attr] = val_part
            
        if "what matters is:" in msg_lower:
            idx = msg_lower.find("what matters is:")
            val_part = message[idx + len("what matters is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            values = [v.strip() for v in val_part.split(";")]
            for val in values:
                attr = self._classify_constraint_locally(val)
                state["disclosed_slots"][attr] = val

        # Extract brand
        brand_match = re.search(r"brand(?:s)? like\s+([a-zA-Z0-9\s]+)", msg_lower)
        if brand_match:
            b_val = brand_match.group(1).strip().lower()
            state["disclosed_slots"]["brand"] = b_val

        # Extract material
        materials_found = [m for m in ["leather", "wool", "cotton", "polyester", "nylon", "silk", "pvc", "resin", "denim", "canvas"] if m in msg_lower]
        if materials_found:
            state["disclosed_slots"]["material"] = materials_found[0]
            
        # Extract sole
        sole_found = [s for s in ["rubber", "flat", "heel", "wedge", "cushion"] if s in msg_lower]
        if sole_found:
            state["disclosed_slots"]["sole"] = sole_found[0]
            
        # Extract style
        style_found = [st for st in ["combat", "fashion", "riding", "chelsea", "casual", "dressy", "western", "cowboy", "rain", "snow", "bootie"] if st in msg_lower]
        if style_found:
            state["disclosed_slots"]["style"] = style_found[0]
            
        # Extract color
        color_found = [c for c in ["black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "gold", "silver"] if c in msg_lower]
        if color_found:
            state["disclosed_slots"]["color"] = color_found[0]

        # Accumulate terms from the message
        new_terms = _terms(message)
        for term in new_terms:
            if term not in state["accumulated_terms"] and term not in state["negated_terms"]:
                state["accumulated_terms"].append(term)
                
        # Clean any previously accumulated terms that are now negated
        state["accumulated_terms"] = [t for t in state["accumulated_terms"] if t not in state["negated_terms"]]

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions[session_id]
        state["history"].append({"role": "user", "content": user_message})
        
        # 1. Parse current message local slots and terms
        self._parse_message_locally(session_id, user_message)
        
        # 2. Always compute Category Route Filtering Mask
        cat_mask = np.ones(len(self.catalog_ids), dtype=bool)
        if state["department"]:
            cat_mask &= (self.catalog_departments == state["department"])
        if state["price_max"] < 9999.0:
            cat_mask &= (self.catalog_prices <= state["price_max"])
        
        target_cat = state["category"]
        if target_cat:
            cat_tokens = set(t.rstrip('s') for t in _terms(target_cat))
            if cat_tokens:
                cat_tok_mask = [bool(cat_tokens & set(c.rstrip('s') for c in item_cats)) for item_cats in self.catalog_categories_set]
                if any(cat_tok_mask):
                    cat_mask &= np.array(cat_tok_mask)
                
        cat_indices = np.where(cat_mask)[0]
        category_asin_set = set(self.catalog_ids_arr[cat_indices])
        
        if not category_asin_set:
            cat_mask = np.ones(len(self.catalog_ids), dtype=bool)
            if state["department"]:
                cat_mask &= (self.catalog_departments == state["department"])
            if state["price_max"] < 9999.0:
                cat_mask &= (self.catalog_prices <= state["price_max"])
            cat_indices = np.where(cat_mask)[0]
            category_asin_set = set(self.catalog_ids_arr[cat_indices])

        candidate_ids = []
        unique_terms = state["accumulated_terms"][:45]
        
        # Cascade Route 1: Keyword Route FTS-Matching
        if unique_terms:
            # Check FTS5 AND
            expression_and = " AND ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT 1000",
                (expression_and,)
            ).fetchall()
            candidate_ids = [str(r[0]) for r in rows]
            
        # Check FTS5 OR if AND yielded low coverage
        if len(candidate_ids) < 15 and unique_terms:
            expression_or = " OR ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) as score "
                "FROM products WHERE products MATCH ? ORDER BY score LIMIT 1000",
                (expression_or,)
            ).fetchall()
            for r in rows:
                asin = str(r[0])
                if asin not in candidate_ids:
                    candidate_ids.append(asin)
                    
        # Apply Category Route mask to FTS5 candidates
        candidate_ids = [pid for pid in candidate_ids if pid in category_asin_set]
                    
        # Cascade Route 2 & 3: Category / Vector Fallback if Keyword Route fails
        if len(candidate_ids) < 10:
            # Vector Route MIPS (Top 100)
            query_text = " ".join([m["content"] for m in state["history"] if m["role"] == "user"])
            query_emb = self.model.encode(query_text, convert_to_numpy=True)
            q_norm = np.linalg.norm(query_emb)
            query_emb_normalized = query_emb / max(q_norm, 1e-12)
            
            scores = np.dot(self.catalog_embeddings, query_emb_normalized)
            sorted_vec_indices = np.argsort(scores)[::-1][:100]
            vector_asins = [self.catalog_ids[idx] for idx in sorted_vec_indices]
            
            # Hybrid Pool Fusion: merge Category filtering with Vector similarity matches
            fallback_candidates = []
            # Keep vector candidates that satisfy category/department/price criteria first
            for v_asin in vector_asins:
                if v_asin in category_asin_set:
                    fallback_candidates.append(v_asin)
            
            # Fill with remainder of category matches or vector matches
            for v_asin in vector_asins:
                if len(fallback_candidates) >= 150:
                    break
                if v_asin not in fallback_candidates:
                    fallback_candidates.append(v_asin)
                    
            for cat_asin in category_asin_set:
                if len(fallback_candidates) >= 200:
                    break
                if cat_asin not in fallback_candidates:
                    fallback_candidates.append(cat_asin)
                    
            # Filter all candidates to ensure they satisfy the category mask
            candidate_ids = [pid for pid in fallback_candidates if pid in category_asin_set]
            
        # If still empty (extremely rare), fallback to complete catalog
        if not candidate_ids:
            candidate_ids = list(category_asin_set) if category_asin_set else self.catalog_ids
            
        # 4. Post-Retrieval Scoring
        scored_candidates = []
        brand_constraints = state["disclosed_slots"].get("brand")
        if brand_constraints:
            if isinstance(brand_constraints, str):
                brand_constraints = {brand_constraints}
        else:
            brand_constraints = set()
        
        for idx, pid in enumerate(candidate_ids):
            if pid in state["seen_asins"]:
                continue
                
            meta = self.catalog_metadata[pid]
            
            # Exclude negated terms
            has_negated = False
            for neg in state.get("negated_terms", set()):
                if neg.lower() in meta["searchable_bag"]:
                    has_negated = True
                    break
            if has_negated:
                continue
                
            score = -0.001 * idx # penalty for retrieval ranking order
            
            # Exact brand filter check
            if brand_constraints:
                if not any(bc.lower() in meta["brand"] for bc in brand_constraints):
                    score -= 10.0
                    
            # Boost exact keyword matches
            for val in state["accumulated_terms"]:
                if val.lower() in meta["searchable_bag"]:
                    score += 0.3
                    
            # Boost stashed keyword matches
            for val in state.get("stashed_terms", []):
                if val.lower() in meta["searchable_bag"]:
                    score += 0.05
                    
            # Popularity scaling
            score += 0.02 * (meta["rating_number"] ** 0.1)
            scored_candidates.append((score, pid))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 5. Diversification
        recommendations = []
        chosen_brands = {}
        chosen_titles = []
        
        def get_jaccard_similarity(t1, t2):
            s1 = set(t1.lower().split())
            s2 = set(t2.lower().split())
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)
            
        for score, pid in scored_candidates:
            meta = self.catalog_metadata[pid]
            brand = meta["brand"]
            title = meta["title"]
            
            if brand and chosen_brands.get(brand, 0) >= 2:
                continue
                
            is_too_similar = False
            for chosen_title in chosen_titles:
                if get_jaccard_similarity(title, chosen_title) > 0.8:
                    is_too_similar = True
                    break
            if is_too_similar:
                continue
                
            recommendations.append(pid)
            if brand:
                chosen_brands[brand] = chosen_brands.get(brand, 0) + 1
            chosen_titles.append(title)
            
            if len(recommendations) == top_k:
                break
                
        # Fill rest if list is incomplete
        if len(recommendations) < top_k:
            for score, pid in scored_candidates:
                if pid not in recommendations:
                    recommendations.append(pid)
                    if len(recommendations) == top_k:
                        break
                        
        state["seen_asins"].update(recommendations)
        
        # 6. Generate Conversational Clarifying Question with Llama 3.1
        disclosed_slots_str = ", ".join(f"{k}: {v}" for k, v in state["disclosed_slots"].items())
        sys_prompt = (
            "You are a helpful e-commerce shopping copilot. The user is looking for a product.\n"
            f"Active search filters: Category: {state['category']}, Department: {state['department']}, Max Price: {state['price_max']}\n"
            f"Attributes already specified by user: {disclosed_slots_str}.\n"
            "CRITICAL RULE: Do not ask the user about any attribute they have already specified. "
            "Instead, choose a new attribute (e.g., color, size, height) that has not been specified yet.\n"
            "Based on the conversation, write a very short (1-2 sentences), natural response to the user. "
            "Acknowledge their request politely, present the recommendations, and ask a follow-up clarifying question to narrow down the search."
        )
        
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
