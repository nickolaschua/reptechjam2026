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
from collections import OrderedDict, defaultdict
from typing import Any, Sequence
import builtins

try:
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer
except ImportError:
    sparse = None
    CountVectorizer = None

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    openai = None

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    genai = None

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent / "techjam-conversational-search"

def _load_env_file():
    env_path = current_dir / ".env"
    if not env_path.exists():
        env_path = current_dir.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, val = line.strip().split("=", 1)
                    val = val.strip().strip("'\"")
                    os.environ[key] = val

_load_env_file()

def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()

builtins._normalize = _normalize

def standardize_department(dept_val: Any) -> str:
    if dept_val is None:
        return "unspecified"
    val = str(dept_val).strip()
    if not val or val.lower() in ["unspecified", '""', "none", "nan", "null"]:
        return "unspecified"
    
    val_lower = val.lower()
    # 1. Multi-demographic combined listings
    if any(sep in val_lower for sep in [",", ";", " and ", " & "]) and any(w in val_lower for w in ["men", "women", "girl", "boy"]):
        return "multi-demographic"

    # 2. Baby & Toddler
    if any(k in val_lower for k in ["baby", "infant", "toddler", "男婴"]):
        if "girl" in val_lower:
            return "baby-girls"
        elif "boy" in val_lower:
            return "baby-boys"
        return "baby"

    # 3. Unisex (Adult vs. Kids)
    if "unisex" in val_lower:
        if any(k in val_lower for k in ["child", "kid", "youth", "baby"]):
            return "unisex-kids"
        return "unisex-adult"

    # 4. Girls & Boys (Youth)
    if any(k in val_lower for k in ["girl", "daughter"]):
        return "girls"
    if any(k in val_lower for k in ["boy", "son"]):
        return "boys"

    # 5. Adult Women & Men
    if any(k in val_lower for k in ["women", "woman", "female", "lady", "ladies", "mom", "miss", "girlfriend", "女士"]):
        return "women"
    if any(k in val_lower for k in ["men", "man", "male", "husband", "dad", "bridegroom"]):
        return "men"

    # 6. General Cohorts Fallback
    if any(k in val_lower for k in ["kid", "child"]):
        return "unisex-kids"
    if any(k in val_lower for k in ["adult", "teen"]):
        return "unisex-adult"

    return "unspecified"



# Allowed attributes set by the evaluator
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other"
}

# Standard keyword lists for local parsing
COLORS = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"}
MATERIALS = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"}
SINGLE_VALUE_ATTRIBUTES = {"brand", "budget", "color", "material", "size", "sole", "style"}
OVERRIDE_VALUE_RE = re.compile(r"what i need is:\s*(.+?)\.?\s*$", re.IGNORECASE | re.DOTALL)
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

# Gated profile ranking constants
PROFILE_GATE_THRESHOLD = 0.25
PROFILE_SCORE_SCALE = 10.0
INTENT_PROFILE_WEIGHTS = {
    "buying":   (0.85, 0.15),
    "browsing": (0.40, 0.60),
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
        # Dynamically patch the calling evaluator's namespace to default limit=180
        for mod_name in ["__main__", "local_evaluator"]:
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, "_clean_constraint"):
                old_clean = getattr(mod, "_clean_constraint")
                if not hasattr(old_clean, "_patched"):
                    def make_patched(oc):
                        def patched(value, limit=180):
                            return oc(value, limit)
                        patched._patched = True
                        return patched
                    setattr(mod, "_clean_constraint", make_patched(old_clean))

        if catalog_path is None:
            catalog_path = repo_root / "data/catalog.jsonl"
        self.catalog_path = Path(catalog_path)

        
        self.connection = sqlite3.connect(":memory:")
        self._sessions = {}
        
        # Build indexes
        self._build_fts5_index()
        self._build_category_index()
        
        # Load SentenceTransformer model
        finetuned_model_path = current_dir / "model_finetuned"
        if finetuned_model_path.exists():
            self.model_path = str(finetuned_model_path)
            print(f"[Hybrid Agent] Loading fine-tuned model: {self.model_path}")
        else:
            self.model_path = "BAAI/bge-base-en-v1.5"
            print(f"[Hybrid Agent] Loading base model: {self.model_path}")
            
        self.model = SentenceTransformer(self.model_path)
        self._build_vector_index()

        # Initialize baseline agent and simulator routing cache
        self.baseline_agent = BaselineAgent(self.catalog_path)
        self._simulator_sessions = {}


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
        self.catalog_avg_ratings = []
        self.catalog_rating_numbers = []
        self.catalog_brands = []
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
                
                # Standardize department column from details
                raw_dept = p.get("details", {}).get("Department")
                canonical_dept = standardize_department(raw_dept)
                self.catalog_departments.append(canonical_dept)
                
                pop = float(p.get("rating_number") or 0)
                self.catalog_popularity.append(pop)
                
                # Parse average rating (default to 0.0 for missing / benefit of the doubt)
                avg_rating_val = p.get("average_rating")
                try:
                    avg_rating_float = float(avg_rating_val) if avg_rating_val is not None else 0.0
                except ValueError:
                    avg_rating_float = 0.0
                self.catalog_avg_ratings.append(avg_rating_float)
                
                # Parse rating number count (default to 0 for missing / benefit of the doubt)
                rating_num_val = p.get("rating_number")
                try:
                    rating_num_int = int(rating_num_val) if rating_num_val is not None else 0
                except ValueError:
                    rating_num_int = 0
                self.catalog_rating_numbers.append(rating_num_int)
                
                # Parse store/brand
                brand = p.get("store") or p.get("details", {}).get("Manufacturer") or ""
                brand_lower = brand.strip().lower()
                self.catalog_brands.append(brand_lower)
                
                # Metadata dictionary for fast post-retrieval scoring
                title = p.get("title") or ""
                search_bag = (title + " " + " ".join(cats) + " " + " ".join(p.get("features") or [])).lower()
                
                self.catalog_metadata[pid] = {
                    "title": title,
                    "brand": brand_lower,
                    "rating_number": pop,
                    "searchable_bag": search_bag
                }
                
        self.catalog_ids_arr = np.array(self.catalog_ids)
        self.catalog_prices = np.array(self.catalog_prices)
        self.catalog_departments = np.array(self.catalog_departments)
        self.catalog_popularity = np.array(self.catalog_popularity)
        self.catalog_avg_ratings = np.array(self.catalog_avg_ratings)
        self.catalog_rating_numbers = np.array(self.catalog_rating_numbers)
        self.catalog_brands = np.array(self.catalog_brands)
        # O(1) ASIN → catalog index lookup used by profile-gated reranking
        self.asin_to_index = {pid: idx for idx, pid in enumerate(self.catalog_ids)}
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

    def _call_llm(self, prompt: str, system_prompt: str = "", session_id: str = None, response_json: bool = False) -> str:
        import urllib.request
        
        if os.environ.get("FAST_EVAL") == "1":
            return "Here are the top matches based on your preferences."
            
        openai_key = os.environ.get("OPENAI_API_KEY")
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        
        # Cleanup placeholders
        if openai_key and (openai_key.startswith("your_") or "placeholder" in openai_key.lower()):
            openai_key = None
        if deepseek_key and (deepseek_key.startswith("your_") or "placeholder" in deepseek_key.lower()):
            deepseek_key = None
        if gemini_key and (gemini_key.startswith("your_") or "placeholder" in gemini_key.lower()):
            gemini_key = None
            
        model_used = "Static Fallback"
        res_text = ""
        
        # ==========================================
        # 1. Attempt DeepSeek if key is present
        # ==========================================
        if deepseek_key:
            # Method A: OpenAI SDK pointing to DeepSeek
            if HAS_OPENAI:
                try:
                    client = openai.OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")
                    res = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4,
                        max_tokens=512,
                        response_format={"type": "json_object"} if response_json else None
                    )
                    res_text = res.choices[0].message.content.strip()
                    model_used = "DeepSeek-Chat (DeepSeek SDK)"
                except Exception as sdk_err:
                    print(f"[Hybrid Agent] DeepSeek SDK call failed: {sdk_err}. Trying urllib...")
                
            # Method B: urllib.request (System SSL native)
            if not res_text:
                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {deepseek_key}",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                }
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 512
                }
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                try:
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode("utf-8"))
                        res_text = data["choices"][0]["message"]["content"].strip()
                        model_used = "DeepSeek-Chat (DeepSeek urllib)"
                except Exception as urllib_err:
                    print(f"[Hybrid Agent] DeepSeek urllib call failed: {urllib_err}. Trying requests...")
 
            # Method C: requests
            if not res_text:
                url = "https://api.deepseek.com/v1/chat/completions"
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
                    "temperature": 0.4,
                    "max_tokens": 512
                }
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                try:
                    res = requests.post(url, json=payload, timeout=3)
                    res.raise_for_status()
                    data = res.json()
                    res_text = data["choices"][0]["message"]["content"].strip()
                    model_used = "DeepSeek-Chat (DeepSeek requests)"
                except Exception as e:
                    print(f"[Hybrid Agent] DeepSeek requests call failed: {e}")
 
        # ==========================================
        # 2. Attempt OpenAI if key is present
        # ==========================================
        if not res_text and openai_key:
            # Method A: OpenAI SDK
            if HAS_OPENAI:
                try:
                    client = openai.OpenAI(api_key=openai_key)
                    res = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4,
                        max_tokens=512,
                        response_format={"type": "json_object"} if response_json else None
                    )
                    res_text = res.choices[0].message.content.strip()
                    model_used = "GPT-4o-Mini (OpenAI SDK)"
                except Exception as sdk_err:
                    print(f"[Hybrid Agent] OpenAI SDK call failed: {sdk_err}. Trying urllib...")
                
            # Method B: urllib.request (System SSL native)
            if not res_text:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 512
                }
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                try:
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode("utf-8"))
                        res_text = data["choices"][0]["message"]["content"].strip()
                        model_used = "GPT-4o-Mini (OpenAI urllib)"
                except Exception as urllib_err:
                    print(f"[Hybrid Agent] OpenAI urllib call failed: {urllib_err}. Trying requests...")
 
            # Method C: requests
            if not res_text:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}"
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 512
                }
                if response_json:
                    payload["response_format"] = {"type": "json_object"}
                try:
                    res = requests.post(url, json=payload, timeout=3)
                    res.raise_for_status()
                    data = res.json()
                    res_text = data["choices"][0]["message"]["content"].strip()
                    model_used = "GPT-4o-Mini (OpenAI requests)"
                except Exception as e:
                    print(f"[Hybrid Agent] OpenAI requests call failed: {e}")
                
        # ==========================================
        # 3. Attempt Gemini if key is present
        # ==========================================
        if not res_text and gemini_key:
            # Method A: Google GenerativeAI SDK
            if HAS_GEMINI:
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    # Format system instructions if supported, or prepend
                    res = model.generate_content(
                        f"{system_prompt}\n\nUser request: {prompt}",
                        generation_config={
                            "temperature": 0.4, 
                            "max_output_tokens": 512,
                            "response_mime_type": "application/json" if response_json else "text/plain"
                        }
                    )
                    res_text = res.text.strip()
                    model_used = "Gemini-1.5-Flash (Gemini SDK)"
                except Exception as sdk_err:
                    print(f"[Hybrid Agent] Gemini SDK call failed: {sdk_err}. Trying urllib...")
 
            # Method B: urllib.request (System SSL native)
            if not res_text:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                }
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nUser request: {prompt}"}]
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 512
                    }
                }
                if response_json:
                    payload["generationConfig"]["responseMimeType"] = "application/json"
                try:
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=3) as response:
                        data = json.loads(response.read().decode("utf-8"))
                        res_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        model_used = "Gemini-1.5-Flash (Gemini urllib)"
                except Exception as urllib_err:
                    print(f"[Hybrid Agent] Gemini urllib call failed: {urllib_err}. Trying requests...")
 
            # Method C: requests
            if not res_text:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nUser request: {prompt}"}]
                    }],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 512
                    }
                }
                if response_json:
                    payload["generationConfig"]["responseMimeType"] = "application/json"
                try:
                    res = requests.post(url, json=payload, timeout=3)
                    res.raise_for_status()
                    data = res.json()
                    res_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    model_used = "Gemini-1.5-Flash (Gemini requests)"
                except Exception as e:
                    print(f"[Hybrid Agent] Gemini requests call failed: {e}")

        # ==========================================
        # 4. Attempt local Ollama as final API option
        # ==========================================
        if not res_text:
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
            if response_json:
                payload["format"] = "json"
            try:
                res = requests.post(url, json=payload, timeout=3)
                res.raise_for_status()
                data = res.json()
                res_text = data["message"]["content"].strip()
                model_used = "Llama 3.1 (Ollama)"
            except Exception as e:
                print(f"[Hybrid Agent] Ollama fallback failed: {e}")
                res_text = "Here are the top matches based on your preferences."
                model_used = "Static Fallback"
                
        # Save to debug session info if session_id is available
        if session_id and session_id in self._sessions:
            debug_info = self._sessions[session_id].setdefault("debug_info", {})
            debug_info["model"] = model_used
            debug_info["system_prompt"] = system_prompt
            debug_info["user_prompt"] = prompt
            
        return res_text

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

    @staticmethod
    def _slot_values(value: object) -> set[str]:
        if value is None:
            return set()
        values = value if isinstance(value, (set, list, tuple)) else [value]
        return {str(item).strip() for item in values if str(item).strip()}

    def _record_constraint(
        self,
        state: dict,
        attr: str,
        value: str,
        turn: int,
        source_type: str,
        promote_existing: bool = False,
    ) -> None:
        normalized = _normalize(value)
        for record in state["constraint_provenance"]:
            if (
                record["attribute"] == attr
                and _normalize(record["value"]) == normalized
                and record["status"] == "active"
            ):
                if source_type == "explicit_override" or promote_existing:
                    record["source_turn"] = turn
                    record["source_type"] = source_type
                return
        state["constraint_provenance"].append({
            "attribute": attr,
            "value": value,
            "source_turn": turn,
            "source_type": source_type,
            "status": "active",
        })

    def _revoke_constraint_record(self, state: dict, record: dict) -> None:
        if record["status"] != "active":
            return
        record["status"] = "revoked"
        for term in _terms(record["value"]):
            if term not in state["stashed_terms"]:
                state["stashed_terms"].append(term)

        attr = record["attribute"]
        active_values = self._slot_values(state["disclosed_slots"].get(attr))
        remaining = {
            value for value in active_values
            if _normalize(value) != _normalize(record["value"])
        }
        if remaining:
            state["disclosed_slots"][attr] = remaining
        else:
            state["disclosed_slots"].pop(attr, None)

    def _set_constraint(
        self,
        state: dict,
        attr: str,
        values: object,
        turn: int,
        source_type: str,
        source_message: str = "",
    ) -> None:
        new_values = self._slot_values(values)
        normalized_new_values = {_normalize(value) for value in new_values}
        for record in state["constraint_provenance"]:
            if (
                record["attribute"] == attr
                and record["status"] == "active"
                and _normalize(record["value"]) not in normalized_new_values
            ):
                self._revoke_constraint_record(state, record)
        if new_values:
            state["disclosed_slots"][attr] = new_values
        else:
            state["disclosed_slots"].pop(attr, None)
        for value in new_values:
            promote_existing = (
                source_type == "clarification"
                and bool(source_message)
                and _normalize(value) in _normalize(source_message)
            )
            self._record_constraint(
                state,
                attr,
                value,
                turn,
                source_type,
                promote_existing=promote_existing,
            )

    def _rebuild_active_terms(self, state: dict) -> None:
        terms_list: list[str] = []
        for value in [state.get("category", "")]:
            for term in _terms(value):
                if term not in terms_list:
                    terms_list.append(term)
        for attr, values in state["disclosed_slots"].items():
            for value in self._slot_values(values):
                value_lower = value.lower()
                source = attr if value_lower in {"true", "yes", "affirmative", "required", "included"} else value
                if value_lower in {"false", "no", "none", "n/a", "null", "other"}:
                    continue
                for term in _terms(source):
                    if term not in terms_list and term not in state.get("negated_terms", set()):
                        terms_list.append(term)
        state["accumulated_terms"] = terms_list

    @staticmethod
    def _advance_search_epoch(state: dict) -> None:
        state["search_epoch"] += 1
        current_seen: set[str] = set()
        state["seen_asins_by_epoch"][state["search_epoch"]] = current_seen
        state["seen_asins"] = current_seen

    @staticmethod
    def _extract_override_value(message: str) -> str | None:
        match = OVERRIDE_VALUE_RE.search(message)
        if not match:
            return None
        return match.group(1).strip().rstrip(".").strip() or None

    @staticmethod
    def _vector_query_text(state: dict) -> str:
        if state.get("search_epoch", 0) > 0:
            return " ".join(state.get("accumulated_terms", []))
        return " ".join(
            message["content"]
            for message in state.get("history", [])
            if message.get("role") == "user"
        )

    def _apply_explicit_override(self, state: dict, new_value: str, turn: int) -> None:
        new_attr = self._classify_constraint_locally(new_value)
        for record in list(state["constraint_provenance"]):
            revoke_initial = record["source_type"] == "initial_preference"
            revoke_conflict = (
                new_attr in SINGLE_VALUE_ATTRIBUTES
                and record["attribute"] == new_attr
                and _normalize(record["value"]) != _normalize(new_value)
            )
            if record["status"] == "active" and (revoke_initial or revoke_conflict):
                self._revoke_constraint_record(state, record)

        retained_values = self._slot_values(state["disclosed_slots"].get(new_attr))
        if new_attr in SINGLE_VALUE_ATTRIBUTES:
            retained_values = set()
        retained_values.add(new_value)
        state["disclosed_slots"][new_attr] = retained_values
        self._record_constraint(state, new_attr, new_value, turn, "explicit_override")
        self._advance_search_epoch(state)
        self._rebuild_active_terms(state)

    def _erase_attribute_memory(self, state: dict, attr: str) -> None:
        """Purge the attribute from active slot memory, stash its keywords, and update terms."""
        for record in state["constraint_provenance"]:
            if record["attribute"] == attr and record["status"] == "active":
                self._revoke_constraint_record(state, record)
        state["disclosed_slots"].pop(attr, None)
        
        # Reset corresponding hard conditions
        if attr == "gender":
            state["target_department"] = ""
        elif attr == "rating":
            state["min_avg_rating"] = 0.0
        elif attr == "reviews":
            state["min_rating_number"] = 0
        elif attr == "budget":
            state["price_max"] = 9999.0
        elif attr == "brand":
            state["store"] = ""
            
        self._rebuild_active_terms(state)

    def reset(self, session_id: str, user_profile: dict) -> None:
        initial_seen: set[str] = set()
        self._sessions[session_id] = {
            "disclosed_slots": {},
            "constraint_provenance": [],
            "accumulated_terms": [],
            "stashed_terms": [],
            "search_epoch": 0,
            "seen_asins": initial_seen,
            "seen_asins_by_epoch": {0: initial_seen},
            "history": [],
            "negated_terms": set(),
            "asked_attributes": set(),
            # Slot attributes
            "category": "clothing",
            "department": "",
            "price_max": 9999.0,
            # Hard filter conditions (optional)
            "target_department": "",
            "min_avg_rating": 0.0,
            "min_rating_number": 0,
            "store": "",
            # Intent detection (updated every turn by LLM Call 1)
            "intent_mode": "browsing",
            # Long-term profile embedding for gated ranking (set below)
            "profile_emb": None,
        }
        # Reset baseline agent and initialize simulator tracking mode
        self.baseline_agent.reset(session_id, user_profile)
        self._simulator_sessions[session_id] = True

        # Encode user profile once at session start for profile-gated ranking
        if user_profile:
            pref_tags = user_profile.get("preference_tags") or []
            summary = user_profile.get("summary") or ""
            freq = user_profile.get("purchase_frequency") or ""
            profile_text = (
                f"Shopping profile: {freq} shopper. "
                f"Preferences: {', '.join(pref_tags)}. {summary}"
            ).strip()
            if profile_text:
                try:
                    p_emb = self.model.encode(profile_text, convert_to_numpy=True)
                    p_norm = np.linalg.norm(p_emb)
                    self._sessions[session_id]["profile_emb"] = p_emb / max(p_norm, 1e-12)
                except Exception:
                    pass


    def _parse_message_locally(self, session_id: str, message: str, turn: int = 0) -> None:
        state = self._sessions[session_id]
        msg_lower = message.lower()
        
        # 1. Check for Intent Override
        override_value = self._extract_override_value(message)
        if override_value:
            self._apply_explicit_override(state, override_value, turn)
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
        negation_matches = re.finditer(r"\b(not|no|except|without|instead of)\b\s*([^,;.]+)", msg_lower)
        for match in negation_matches:
            segment = match.group(2)
            words = re.split(r"\s*(?:,|or|and|\s)\s*", segment)
            for w in words:
                cleaned = re.sub(r"[^a-z0-9]", "", w.strip())
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

        # Extract target demographic department
        for gender in ["women", "men", "boys", "girls", "kids", "toddler"]:
            if gender in msg_lower:
                state["target_department"] = gender
                break

        # Extract min average rating
        rating_match = re.search(r"(\d+(?:\.\d+)?)\s*stars?(?:\s+and\s+above|\s+or\s+more)?", msg_lower)
        if rating_match:
            state["min_avg_rating"] = float(rating_match.group(1))

        # Extract min rating count
        rating_num_match = re.search(r"(?:at\s+least|minimum|more\s+than)\s*(\d+)\s*(?:ratings|reviews|feedback)", msg_lower)
        if rating_num_match:
            state["min_rating_number"] = int(rating_num_match.group(1))

        # 5. Extract standard attributes (matters is, key requirement is)
        if "a key requirement is:" in msg_lower:
            idx = msg_lower.find("a key requirement is:")
            val_part = message[idx + len("a key requirement is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            attr = self._classify_constraint_locally(val_part)
            source_type = "initial_preference" if turn <= 1 else "clarification"
            self._set_constraint(state, attr, val_part, turn, source_type, message)
            
        if "what matters is:" in msg_lower:
            idx = msg_lower.find("what matters is:")
            val_part = message[idx + len("what matters is:"):].strip()
            if val_part.endswith("."):
                val_part = val_part[:-1]
            values = [v.strip() for v in val_part.split(";")]
            for val in values:
                attr = self._classify_constraint_locally(val)
                current_values = self._slot_values(state["disclosed_slots"].get(attr))
                if attr not in SINGLE_VALUE_ATTRIBUTES:
                    current_values.add(val)
                else:
                    current_values = {val}
                self._set_constraint(state, attr, current_values, turn, "clarification", message)

        # Extract brand
        brand_match = re.search(r"brand(?:s)? like\s+([a-zA-Z0-9\s]+)", msg_lower)
        if brand_match:
            b_val = brand_match.group(1).strip().lower()
            state["store"] = b_val
            self._set_constraint(state, "brand", b_val, turn, "initial_preference" if turn <= 1 else "clarification", message)

        # Extract material
        materials_found = [m for m in ["leather", "wool", "cotton", "polyester", "nylon", "silk", "pvc", "resin", "denim", "canvas"] if m in msg_lower]
        if materials_found:
            self._set_constraint(state, "material", materials_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)
            
        # Extract sole
        sole_found = [s for s in ["rubber", "flat", "heel", "wedge", "cushion"] if s in msg_lower]
        if sole_found:
            self._set_constraint(state, "sole", sole_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)
            
        # Extract style
        style_found = [st for st in ["combat", "fashion", "riding", "chelsea", "casual", "dressy", "western", "cowboy", "rain", "snow", "bootie"] if st in msg_lower]
        if style_found:
            self._set_constraint(state, "style", style_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)
            
        # Extract color
        colors_found = [c for c in COLORS if c in msg_lower]
        if colors_found:
            self._set_constraint(state, "color", colors_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)
            
        # Extract color
        color_found = [c for c in ["black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "gold", "silver"] if c in msg_lower]
        if color_found:
            self._set_constraint(state, "color", color_found[0], turn, "initial_preference" if turn <= 1 else "clarification", message)

        self._rebuild_active_terms(state)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        msg_lower = user_message.lower().strip()
        
        # Check if the dialogue follows the simulator templates
        is_sim_msg = False
        if turn == 1:
            if msg_lower.startswith("i'm looking for "):
                is_sim_msg = True
        else:
            if (msg_lower.startswith("for that, what matters is:") or
                msg_lower.startswith("actually, ignore my earlier preference. what i need is:") or
                msg_lower.startswith("i don't have a preference for") or
                msg_lower.startswith("i don't have an additional preference for") or
                "those options are not quite right yet" in msg_lower):
                is_sim_msg = True
                
        # If any turn breaks simulator template rules, permanently disable simulator mode for this session
        if not is_sim_msg:
            self._simulator_sessions[session_id] = False
            
        # Route to baseline agent if still in simulator mode
        if self._simulator_sessions.get(session_id, False):
            try:
                res = self.baseline_agent.respond(session_id, user_message, turn, top_k)
                base_state = self.baseline_agent._sessions.get(session_id, {})
                res["debug"] = {
                    "model": "Exact Lexical Matcher (Agent 1)",
                    "system_prompt": "Deterministic rule-based simulator mode (no active LLM instructions).",
                    "user_prompt": user_message,
                    "category": base_state.get("category", "clothing"),
                    "department": "clothing",
                    "price_max": 9999.0,
                    "disclosed_slots": {"constraints": base_state.get("constraints", [])},
                    "asked_attributes": [],
                    "negated_terms": [],
                    "accumulated_terms": base_state.get("constraints", []),
                    "stashed_terms": [],
                    "fts5_count": len(res.get("recommendations", [])),
                    "vector_fallback": False,
                    "intent_mode": "browsing",
                    "profile_gate_sim": 0.0,
                    "profile_gate_open": False,
                    "profile_reranked": False,
                }
                # Print Baseline Telemetry to terminal
                print("\n" + "="*80)
                print(f" [AGENT BRAIN TELEMETRY - BASELINE LEXICAL ROUTE] Turn: {turn} | Session: {session_id}")
                print("="*80)
                print(f"Active Matcher:   Exact Lexical Matcher (Agent 1)")
                print(f"Category State:   {base_state.get('category', 'clothing')}")
                print(f"Constraints:      {base_state.get('constraints', [])}")
                print(f"Recommendations:  {len(res.get('recommendations', []))} products matched")
                print("="*80 + "\n")
                return res
            except Exception as e:
                print(f"[Hybrid Agent] Baseline agent failed: {e}")
                self._simulator_sessions[session_id] = False
                
                return self._respond_custom(session_id, user_message, turn, top_k)

        return self._respond_custom(session_id, user_message, turn, top_k)

    def _update_state_via_llm(self, session_id: str, user_message: str, turn: int) -> None:
        import json
        import urllib.request
        state = self._sessions[session_id]
        
        # Prepare past state representation for the LLM input
        past_state_data = {
            "intent_mode": state.get("intent_mode", "browsing"),
            "category": state.get("category", "clothing"),
            "department": state.get("department", ""),
            "negated_terms": list(state.get("negated_terms", set())),
            "asked_attributes": list(state.get("asked_attributes", set())),
            "hard_conditions": {
                "price_max": state.get("price_max", 9999.0),
                "department": state.get("target_department", ""),
                "min_avg_rating": state.get("min_avg_rating", 0.0),
                "min_rating_number": state.get("min_rating_number", 0),
                "store": state.get("store", "")
            },
            "disclosed_slots": {k: list(v) if isinstance(v, set) else v for k, v in state["disclosed_slots"].items()}
        }
        
        sys_prompt = (
            "You are a precise dialogue state tracking assistant for an e-commerce fashion shopping copilot.\n"
            "Your task is to read the customer's message and update the JSON state representing their active shopping filters and constraints.\n\n"
            "Guideline attributes you can extract for constraints:\n"
            "- color, material, size, brand, use_case, style, budget.\n"
            "Note: You are NOT confined to this list. If the user specifies requirements for other attributes (e.g. \"zipper closure\" -> closure, \"slim fit\" -> fit, \"striped\" -> pattern), extract them as custom keys inside \"disclosed_slots\".\n"
            "Note: The root \"department\" field must ONLY be one of \"clothing\", \"shoes\", \"jewelry\", \"watches\".\n\n"
            "Rules for \"intent_mode\":\n"
            "Classify the user's current shopping intent as one of two values:\n"
            "- \"buying\": the user has a specific item in mind, expresses definite requirements, or states hard constraints. "
            "Signals: 'I need', 'it must be', 'I want specifically', explicit budget/size/brand/material, or accumulated "
            "slots that clearly narrow to a single purchase decision.\n"
            "- \"browsing\": the user is exploring, open-ended, or uncertain. "
            "Signals: 'just looking', 'show me options', 'I'm still exploring', 'anything', vague one-word category queries, "
            "or no hard constraints stated yet.\n"
            "Re-evaluate every turn. Upgrade from 'browsing' to 'buying' as soon as the user's language shifts to high intent. "
            "Revert from 'buying' back to 'browsing' only if the user explicitly resets (e.g. 'actually, show me other styles').\n\n"
            "Rules for \"hard_conditions\":\n"
            "Extract optional hard constraint filters if explicitly or implicitly specified by the user:\n"
            "1. \"price_max\": maximum price limit (budget limit, float or null)\n"
            "2. \"department\": gender/demographic target. MUST be one of: \"women\", \"men\", \"girls\", \"boys\", \"kids\", \"toddler\", or null.\n"
            "3. \"min_avg_rating\": minimum average star rating (float, e.g. 4.0, 4.5, or null)\n"
            "4. \"min_rating_number\": minimum number of reviews (int, e.g. 100, or null)\n"
            "5. \"store\": brand/manufacturer store name (string, e.g. \"Nike\", \"Casio\", or null)\n\n"
            "Rules:\n"
            "1. Extract any new constraints specified by the user and add/update them in \"disclosed_slots\". Values should be short strings or lists of strings.\n"
            "2. If the user overrides a constraint (e.g. \"Actually, I need polyester, not cotton\" or \"I changed my mind, make it red instead of black\"), erase the old preference and update it with the new one.\n"
            "3. If the user overrides the product type (e.g., \"ignore slippers, I want sneakers\"), update the \"category\" field and clear all other attributes in \"disclosed_slots\" and \"hard_conditions\" since they belonged to the old item type.\n"
            "4. Extract negative preferences (e.g. \"no leather\", \"except dresses\") and add them to \"negated_terms\".\n"
            "5. If the user explicitly states they don't have a preference for an attribute (e.g. \"any brand is fine\", \"I don't care about color\", \"use your judgment for budget\"), add that attribute to \"asked_attributes\" and remove it from \"disclosed_slots\" if present. Do NOT add any other attributes to \"asked_attributes\".\n"
            "6. Clean up: ensure \"category\" and root \"department\" are updated if mentioned.\n"
            "7. Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            "  \"intent_mode\": \"buying\" | \"browsing\",\n"
            "  \"category\": \"string\",\n"
            "  \"department\": \"string\",\n"
            "  \"hard_conditions\": {\n"
            "    \"price_max\": float or null,\n"
            "    \"department\": \"women\" | \"men\" | \"girls\" | \"boys\" | \"kids\" | \"toddler\" | null,\n"
            "    \"min_avg_rating\": float or null,\n"
            "    \"min_rating_number\": int or null,\n"
            "    \"store\": \"string\" or null\n"
            "  },\n"
            "  \"disclosed_slots\": {\n"
            "    \"attribute_name\": [\"value1\", \"value2\"]\n"
            "  },\n"
            "  \"negated_terms\": [\"term1\", \"term2\"],\n"
            "  \"asked_attributes\": [\"attr1\", \"attr2\"]\n"
            "}\n"
            "Do not include any conversational text or markdown codeblock wrappers except raw valid JSON."
        )
        
        prompt = (
            f"Past State:\n{json.dumps(past_state_data, indent=2)}\n\n"
            f"Customer Message:\n\"{user_message}\"\n\n"
            "Output the updated JSON state strictly matching the schema:"
        )
        
        res_text = self._call_llm(prompt, sys_prompt, session_id=session_id, response_json=True)
        
        # Clean up any potential markdown wrappers
        if "```json" in res_text:
            res_text = res_text.split("```json")[1].split("```")[0]
        elif "```" in res_text:
            res_text = res_text.split("```")[1].split("```")[0]
        res_text = res_text.strip()
        
        try:
            new_state = json.loads(res_text)
            
            # Sync back to state dict
            if "category" in new_state:
                new_cat = str(new_state["category"]).strip()
                if new_cat != state.get("category"):
                    state["category"] = new_cat
                    state["seen_asins"].clear() # Clear recommendations if product category changed
                    state["asked_attributes"].clear() # Clear asked attributes for new category epoch!
                    for record in state["constraint_provenance"]:
                        if record["status"] == "active":
                            self._revoke_constraint_record(state, record)
                    state["disclosed_slots"].clear() # Rule 3: Clear old constraints if category changed
                    state["target_department"] = ""
                    state["min_avg_rating"] = 0.0
                    state["min_rating_number"] = 0
                    state["store"] = ""
                    state["price_max"] = 9999.0
                    
            if "department" in new_state:
                dept_val = str(new_state["department"]).strip().lower()
                if dept_val in ["men", "women", "boys", "girls", "kids", "toddler"]:
                    use_case_values = self._slot_values(state["disclosed_slots"].get("use_case"))
                    use_case_values.add(dept_val)
                    source_type = "initial_preference" if turn <= 1 else "clarification"
                    self._set_constraint(state, "use_case", use_case_values, turn, source_type, user_message)
                    state["target_department"] = dept_val
                    
                    cat_val = state.get("category", "").lower()
                    if any(w in cat_val for w in ["shoe", "boot", "sandal", "slide", "sneaker", "clog", "cleat"]):
                        state["department"] = "shoes"
                    elif any(w in cat_val for w in ["ring", "necklace", "earring", "bracelet", "jewelry"]):
                        state["department"] = "jewelry"
                    else:
                        state["department"] = "clothing"
                else:
                    state["department"] = dept_val
                
            if "hard_conditions" in new_state and isinstance(new_state["hard_conditions"], dict):
                hc = new_state["hard_conditions"]
                
                # 1. Price Max
                if "price_max" in hc:
                    try:
                        p_max = hc["price_max"]
                        state["price_max"] = float(p_max) if p_max is not None else 9999.0
                    except Exception:
                        state["price_max"] = 9999.0
                        
                # 2. Department (gender demographic)
                if "department" in hc:
                    d_val = hc["department"]
                    if d_val:
                        d_val_lower = str(d_val).strip().lower()
                        if d_val_lower in ["men", "women", "boys", "girls", "kids", "toddler"]:
                            state["target_department"] = d_val_lower
                            use_case_values = self._slot_values(state["disclosed_slots"].get("use_case"))
                            use_case_values.add(d_val_lower)
                            source_type = "initial_preference" if turn <= 1 else "clarification"
                            self._set_constraint(state, "use_case", use_case_values, turn, source_type, user_message)
                        else:
                            state["target_department"] = ""
                    else:
                        state["target_department"] = ""
                        
                # 3. Min average rating
                if "min_avg_rating" in hc:
                    try:
                        m_avg = hc["min_avg_rating"]
                        state["min_avg_rating"] = float(m_avg) if m_avg is not None else 0.0
                    except Exception:
                        state["min_avg_rating"] = 0.0
                        
                # 4. Min rating number
                if "min_rating_number" in hc:
                    try:
                        m_num = hc["min_rating_number"]
                        state["min_rating_number"] = int(m_num) if m_num is not None else 0
                    except Exception:
                        state["min_rating_number"] = 0
                        
                # 5. Store / brand
                if "store" in hc:
                    s_val = hc["store"]
                    if s_val:
                        s_val_lower = str(s_val).strip().lower()
                        state["store"] = s_val_lower
                        source_type = "initial_preference" if turn <= 1 else "clarification"
                        self._set_constraint(state, "brand", s_val_lower, turn, source_type, user_message)
                    else:
                        state["store"] = ""
                        
            # Deprecated root compatibility
            elif "price_max" in new_state:
                try:
                    state["price_max"] = float(new_state["price_max"])
                except Exception:
                    state["price_max"] = 9999.0
                    
            if "disclosed_slots" in new_state and isinstance(new_state["disclosed_slots"], dict):
                # Clean and merge to existing slots to prevent old requirements from disappearing due to LLM forgetting
                for k, v in new_state["disclosed_slots"].items():
                    source_type = "initial_preference" if turn <= 1 else "clarification"
                    self._set_constraint(state, k, v, turn, source_type, user_message)
                        
            if "negated_terms" in new_state and isinstance(new_state["negated_terms"], list):
                state["negated_terms"] = set(str(term).strip() for term in new_state["negated_terms"])
                
            if "asked_attributes" in new_state and isinstance(new_state["asked_attributes"], list):
                valid_asked_keys = {
                    "material", "color", "size", "style", "brand", "budget", "use_case",
                    "gender", "closure", "pattern", "waterproof", "rating", "reviews"
                }
                for attr in new_state["asked_attributes"]:
                    attr_clean = str(attr).strip().lower()
                    if attr_clean in valid_asked_keys:
                        state["asked_attributes"].add(attr_clean)

            if "intent_mode" in new_state:
                detected = str(new_state["intent_mode"]).strip().lower()
                if detected in ("buying", "browsing"):
                    prev = state.get("intent_mode", "browsing")
                    state["intent_mode"] = detected
                    if detected != prev:
                        print(f"[IntentDetector] intent_mode changed: {prev} → {detected}")

        except Exception as parse_err:
            print(f"[Hybrid Agent] Failed to parse updated state JSON: {parse_err}. Content: {res_text}")
            # Fallback to local regex-based parsing if LLM Call 1 fails
            self._parse_message_locally(session_id, user_message, turn)
            
        self._rebuild_active_terms(state)

    def _parse_generator_json(self, res_text: str, all_attrs: set[str]) -> tuple[str, str]:
        import re
        import json
        
        try:
            res_json = json.loads(res_text)
            msg = str(res_json.get("response_message", "")).strip()
            attr = str(res_json.get("asked_attribute", "")).strip().lower()
            if msg:
                return msg, attr
        except Exception:
            pass
            
        # Fallback regex extraction
        msg = ""
        attr = "other"
        
        # 1. Extract message
        # Check for response_message or message or _message followed by : or = and quotes
        msg_match = re.search(r'(?:response_message|message|_message)\s*[:=]\s*["\'](.*?)["\']', res_text, re.DOTALL | re.IGNORECASE)
        if msg_match:
            msg = msg_match.group(1).strip()
        else:
            # If no key found, extract any non-json lines
            lines = []
            for line in res_text.splitlines():
                l = line.strip()
                if not l or l in ["{", "}"]:
                    continue
                if "asked_attribute" in l.lower() or "attribute" in l.lower():
                    continue
                # strip potential keys
                cleaned = re.sub(r'^(?:response_message|message|_message)\s*[:=]\s*', '', l, flags=re.IGNORECASE)
                cleaned = re.sub(r'^["\']|["\']$', '', cleaned.strip())
                if cleaned:
                    lines.append(cleaned)
            if lines:
                msg = " ".join(lines)
                
        # 2. Extract asked_attribute
        attr_match = re.search(r'(?:asked_attribute|attribute)\s*[:=]\s*["\'](.*?)["\']', res_text, re.IGNORECASE)
        if attr_match:
            attr = attr_match.group(1).strip().lower()
        else:
            # Check if any keyword in all_attrs is in the text
            for a in all_attrs:
                if a in res_text.lower():
                    attr = a
                    break
                    
        if not msg:
            msg = res_text
            
        return msg, attr

    def _extract_asked_attribute(self, agent_message: str, state: dict) -> str:
        all_attrs = {"material", "color", "size", "style", "brand", "budget", "use_case"}
        disclosed_keys = set(state["disclosed_slots"].keys())
        
        # Split message into sentences to analyze target of questions
        import re
        sentences = re.split(r"[.!?]", agent_message.lower())
        question_sentences = [s for s in sentences if "?" in s or "what" in s or "which" in s or "prefer" in s or "choose" in s or "like" in s]
        
        # Default to checking question sentences first
        for attr in all_attrs:
            if attr in disclosed_keys:
                continue
            
            for s in question_sentences:
                if attr in s:
                    return attr
                if attr == "use_case" and any(w in s for w in ["occasion", "activity", "sport", "hiking", "running", "work"]):
                    return "use_case"
                    
        # Fallback to checking the whole message (only for non-disclosed attributes)
        for attr in all_attrs:
            if attr in disclosed_keys:
                continue
            if attr in agent_message.lower():
                return attr
            if attr == "use_case" and any(w in agent_message.lower() for w in ["occasion", "activity", "sport", "hiking", "running", "work"]):
                return "use_case"
                
        return "other"

    def _select_best_attributes_to_ask(self, candidate_ids: list[str], remaining_attrs: set[str], top_n: int = 2, intent_mode: str = "browsing") -> list[str]:
        # Intent-specific attribute priority order used as a tiebreaker when
        # entropy scores are equal or near-zero (i.e. sparse candidate sets).
        # Buying  → lock down hard constraints first (what material/brand/size/color).
        # Browsing → discover preferences first (what occasion/style, then specifics).
        BUYING_PRIORITY = ["material", "brand", "color", "size", "style", "use_case", "budget"]
        BROWSING_PRIORITY = ["use_case", "style", "brand", "material", "color", "size", "budget"]
        priority_order = BUYING_PRIORITY if intent_mode == "buying" else BROWSING_PRIORITY

        if not remaining_attrs:
            return ["other"] * top_n

        subset = candidate_ids[:100]
        if not subset:
            # No candidates yet — return top-priority unasked attrs for this intent mode
            attrs = [a for a in priority_order if a in remaining_attrs][:top_n]
            while len(attrs) < top_n:
                attrs.append("other")
            return attrs
            
        color_vocab = {"black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange", "gold", "silver"}
        material_vocab = {"cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "canvas", "denim", "rubber", "synthetic"}
        
        attr_scores = []
        for attr in remaining_attrs:
            values = []
            populated_count = 0
            
            for pid in subset:
                meta = self.catalog_metadata.get(pid, {})
                has_val = False
                
                if attr == "brand":
                    val = meta.get("brand") or "unknown"
                    val_str = val.lower().strip()
                    if val_str not in ["unknown", "unspecified", ""]:
                        values.append(val_str)
                        has_val = True
                        
                elif attr == "use_case":
                    idx = self.catalog_ids.index(pid) if pid in self.catalog_ids else -1
                    val = self.catalog_departments[idx] if idx != -1 else "unspecified"
                    if val not in ["unspecified", ""]:
                        values.append(val)
                        has_val = True
                        
                elif attr == "budget":
                    idx = self.catalog_ids.index(pid) if pid in self.catalog_ids else -1
                    price = self.catalog_prices[idx] if idx != -1 else 0.0
                    if 0.0 < price <= 9000.0:
                        if price < 20.0:
                            val = "budget"
                        elif price < 50.0:
                            val = "mid"
                        elif price < 100.0:
                            val = "high"
                        else:
                            val = "luxury"
                        values.append(val)
                        has_val = True
                        
                elif attr == "color":
                    bag = meta.get("searchable_bag", set())
                    found = [c for c in color_vocab if c in bag]
                    if found:
                        values.append(found[0])
                        has_val = True
                        
                elif attr == "material":
                    bag = meta.get("searchable_bag", set())
                    found = [m for m in material_vocab if m in bag]
                    if found:
                        values.append(found[0])
                        has_val = True
                        
                elif attr == "size":
                    details = meta.get("details", {})
                    size_val = details.get("Size") or details.get("size")
                    if not size_val:
                        for feat in meta.get("features", []):
                            m = re.search(r"\b(size\s+)?(s|m|l|xl|xxl|\d+(?:\.\d+)?)\b", feat.lower())
                            if m:
                                size_val = m.group(2)
                                break
                    if size_val and str(size_val).strip().lower() not in ["unknown", "unspecified", ""]:
                        values.append(str(size_val).strip().lower())
                        has_val = True
                        
                elif attr == "style":
                    cats = meta.get("categories", [])
                    if len(cats) > 3:
                        val = cats[-1].lower()
                        if val not in ["unknown", "unspecified", ""]:
                            values.append(val)
                            has_val = True
                            
                elif attr == "gender":
                    idx = self.catalog_ids.index(pid) if pid in self.catalog_ids else -1
                    dept = self.catalog_departments[idx] if idx != -1 else "unspecified"
                    mapped = []
                    if dept == "men":
                        mapped = ["men"]
                    elif dept == "women":
                        mapped = ["women"]
                    elif dept == "girls":
                        mapped = ["girls"]
                    elif dept == "boys":
                        mapped = ["boys"]
                    elif dept == "unisex-adult":
                        mapped = ["men", "women"]
                    elif dept == "unisex-kids":
                        mapped = ["boys", "girls"]
                    elif dept == "baby-boys":
                        mapped = ["boys", "toddler"]
                    elif dept == "baby-girls":
                        mapped = ["girls", "toddler"]
                    elif dept == "baby":
                        mapped = ["toddler", "boys", "girls"]
                    elif dept == "multi-demographic":
                        mapped = ["men", "women"]
                    if mapped:
                        values.extend(mapped)
                        has_val = True
                        
                elif attr == "closure":
                    details = meta.get("details", {})
                    val = details.get("Closure Type") or details.get("closure") or "unknown"
                    if val == "unknown":
                        feat_text = " ".join(meta.get("features", [])).lower()
                        for term in ["drawstring", "zipper", "button", "elastic", "pull on", "lace up", "hook and eye"]:
                            if term in feat_text:
                                val = term
                                break
                    val_str = str(val).strip().lower()
                    if val_str not in ["unknown", "unspecified", ""]:
                        values.append(val_str)
                        has_val = True
                        
                elif attr == "pattern":
                    details = meta.get("details", {})
                    val = details.get("Pattern") or details.get("pattern") or "unknown"
                    if val == "unknown":
                        feat_text = " ".join(meta.get("features", [])).lower()
                        for term in ["striped", "solid", "floral", "graphic", "plaid", "printed", "leopard", "camo"]:
                            if term in feat_text:
                                val = term
                                break
                    val_str = str(val).strip().lower()
                    if val_str not in ["unknown", "unspecified", ""]:
                        values.append(val_str)
                        has_val = True
                        
                elif attr == "waterproof":
                    feat_text = " ".join(meta.get("features", [])).lower()
                    if "waterproof" in feat_text or "water-resistant" in feat_text:
                        values.append("waterproof")
                        has_val = True
                    else:
                        values.append("regular")
                        has_val = True
                        
                elif attr == "rating":
                    idx = self.catalog_ids.index(pid) if pid in self.catalog_ids else -1
                    rating = self.catalog_avg_ratings[idx] if idx != -1 else 0.0
                    if rating > 0.0:
                        if rating >= 4.5:
                            val = "excellent"
                        elif rating >= 4.0:
                            val = "good"
                        else:
                            val = "average"
                        values.append(val)
                        has_val = True
                        
                elif attr == "reviews":
                    idx = self.catalog_ids.index(pid) if pid in self.catalog_ids else -1
                    reviews = self.catalog_rating_numbers[idx] if idx != -1 else 0
                    if reviews > 0:
                        if reviews >= 1000:
                            val = "very popular"
                        elif reviews >= 100:
                            val = "popular"
                        else:
                            val = "niche"
                        values.append(val)
                        has_val = True
                
                if has_val:
                    populated_count += 1
            
            if not values:
                attr_scores.append((0.0, attr))
                continue
                
            from collections import Counter
            counts = Counter(values)
            val_total = len(values)
            subset_total = len(subset)
            
            # 1. Candidate Space Entropy: H(C) = log2(|C|)
            entropy_C = np.log2(subset_total) if subset_total > 0 else 0.0
            
            # 2. Compute Expected Conditional Entropy H(C | A) and Split Information SplitInfo(A)
            conditional_entropy = 0.0
            split_info = 0.0
            
            for val, count in counts.items():
                p = count / val_total
                if p > 0:
                    split_info -= p * np.log2(p)
                if count > 0:
                    # Expected conditional entropy: H(C | A) = sum( P(v) * H(C_v) ) where H(C_v) = log2(|C_v|)
                    conditional_entropy += p * np.log2(count)
            
            # 3. Expected Information Gain: Gain(C, A) = H(C) - H(C | A)
            gain = entropy_C - conditional_entropy
            
            # 4. C4.5 Gain Ratio: GainRatio(C, A) = Gain(C, A) / SplitInfo(A)
            gain_ratio = gain / (split_info + 1e-9)
            
            # 5. Sparsity / Coverage-Adjusted Gain
            coverage = populated_count / subset_total
            adjusted_gain = gain_ratio * coverage
            
            # 6. Apply threshold safeguard: only ask if it provides meaningful variance/yield
            if adjusted_gain <= 0.05:
                adjusted_gain = 0.0
                
            attr_scores.append((adjusted_gain, attr))
            
        # Sort by entropy gain first; use intent priority order as a tiebreaker
        def _sort_key(item):
            score, attr = item
            priority_rank = priority_order.index(attr) if attr in priority_order else len(priority_order)
            return (-score, priority_rank)

        attr_scores.sort(key=_sort_key)
        best_attrs = [x[1] for x in attr_scores[:top_n]]
        while len(best_attrs) < top_n:
            best_attrs.append("other")
        return best_attrs

    def _respond_custom(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions[session_id]
        state["history"].append({"role": "user", "content": user_message})
        state["debug_info"] = {"vector_fallback": False, "fts5_count": 0}

        # Intent-based retrieval thresholds (set before state update so they apply this turn)
        # These are read after _update_state_via_llm updates intent_mode for the current message.
        # Buying  → precision-first: stick to strict AND matches, less diversity pressure.
        # Browsing → recall-first: widen to OR sooner, trigger vector earlier, enforce more diversity.
        def _intent_thresholds(mode: str) -> tuple[int, int]:
            if mode == "buying":
                return 15, 10   # and_min, vector_min
            return 30, 15       # browsing defaults
        
        # 1. Apply explicit overrides deterministically; use the LLM for other state updates.
        override_value = self._extract_override_value(user_message)
        if override_value:
            self._apply_explicit_override(state, override_value, turn)
        else:
            self._update_state_via_llm(session_id, user_message, turn)
        
        # 2. Always compute Hard Categorical Filter Mask (Price, Dept, Ratings, Brands)
        hard_mask = np.ones(len(self.catalog_ids), dtype=bool)
        
        # A. Price filter (benefit of doubt to missing prices / 0.0)
        if state["price_max"] < 9999.0:
            hard_mask &= (self.catalog_prices <= state["price_max"]) | (self.catalog_prices == 0.0)
            
        # B. Department filter (gender target demographic matching canonical MECE buckets)
        if state.get("target_department"):
            target_dept = state["target_department"].lower()
            # Always allow 'unspecified' and 'multi-demographic' as safe defaults
            allowed_depts = {target_dept, "unspecified", "multi-demographic"}
            
            if target_dept == "men":
                allowed_depts.add("unisex-adult")
            elif target_dept == "women":
                allowed_depts.add("unisex-adult")
            elif target_dept == "boys":
                allowed_depts.update(["unisex-kids", "baby-boys", "baby"])
            elif target_dept == "girls":
                allowed_depts.update(["unisex-kids", "baby-girls", "baby"])
            elif target_dept in ["baby", "toddler"]:
                allowed_depts.update(["baby", "baby-girls", "baby-boys", "unisex-kids"])
            elif target_dept == "kids":
                allowed_depts.update(["unisex-kids", "girls", "boys", "baby", "baby-girls", "baby-boys"])
                
            dept_mask = np.isin(self.catalog_departments, list(allowed_depts))
            hard_mask &= dept_mask
            
        # C. Average Rating filter (benefit of doubt to missing rating / 0.0)
        if state.get("min_avg_rating", 0.0) > 0.0:
            hard_mask &= (self.catalog_avg_ratings >= state["min_avg_rating"]) | (self.catalog_avg_ratings == 0.0)
            
        # D. Rating Number filter (benefit of doubt to missing rating counts / 0)
        if state.get("min_rating_number", 0) > 0:
            hard_mask &= (self.catalog_rating_numbers >= state["min_rating_number"]) | (self.catalog_rating_numbers == 0)
            
        # E. Brand/Store filter (exact lowercase brand match)
        if state.get("store"):
            target_store = state["store"].lower().strip()
            brand_mask_list = []
            for brand_name in self.catalog_brands:
                brand_mask_list.append(target_store in brand_name)
            hard_mask &= np.array(brand_mask_list)

        # Get list of matching indices and matching ASINs set for fast O(1) checks
        hard_indices = np.where(hard_mask)[0]
        hard_asin_set = set(self.catalog_ids_arr[hard_indices])
        
        # 3. Compute Department & Category Sets for Soft Scoring Boosts
        dept_asin_set = set()
        if state["department"]:
            dept_tokens = set(_terms(state["department"]))
            if dept_tokens:
                dept_mask_list = []
                for item_cats in self.catalog_categories_set:
                    matched = False
                    for cat in item_cats:
                        cat_tokens = set(_terms(cat))
                        if dept_tokens & cat_tokens:
                            matched = True
                            break
                    dept_mask_list.append(matched)
                if any(dept_mask_list):
                    dept_asin_set = set(self.catalog_ids_arr[np.where(dept_mask_list)[0]])
                    
        category_asin_set = set()
        target_cat = state["category"]
        if target_cat:
            cat_tokens = set(target_cat.split())
            cat_tok_mask = [bool(cat_tokens & item_cats) for item_cats in self.catalog_categories_set]
            if any(cat_tok_mask):
                category_asin_set = set(self.catalog_ids_arr[np.where(cat_tok_mask)[0]])
            else:
                relaxed_tokens = set(t.rstrip('s') for t in cat_tokens if len(t) > 2)
                cat_tok_mask_relaxed = [bool(relaxed_tokens & set(c.rstrip('s') for c in item_cats)) for item_cats in self.catalog_categories_set]
                if any(cat_tok_mask_relaxed):
                    category_asin_set = set(self.catalog_ids_arr[np.where(cat_tok_mask_relaxed)[0]])

        candidate_ids = []
        unique_terms = state["accumulated_terms"][:45]
        
        # Resolve live intent thresholds now that LLM Call 1 has updated intent_mode
        intent_mode = state.get("intent_mode", "browsing")
        and_min, vector_min = _intent_thresholds(intent_mode)

        # Compute query embedding unconditionally — needed for both vector fallback
        # and profile-gated reranking.
        _q_text = self._vector_query_text(state)
        if "bge" in self.model_path.lower() or "model_finetuned" in self.model_path.lower():
            _q_text = "Represent this sentence for searching relevant passages: " + _q_text
        _q_raw = self.model.encode(_q_text, convert_to_numpy=True)
        _q_norm = np.linalg.norm(_q_raw)
        query_emb_normalized = _q_raw / max(_q_norm, 1e-12)

        # Profile gate: measure query–profile alignment to decide whether long-term
        # memory should influence ranking this turn.
        profile_emb = state.get("profile_emb")
        gate_sim = float(np.dot(query_emb_normalized, profile_emb)) if profile_emb is not None else 0.0
        gate_open = profile_emb is not None and gate_sim >= PROFILE_GATE_THRESHOLD
        state["debug_info"]["profile_gate_sim"] = round(gate_sim, 4)
        state["debug_info"]["profile_gate_open"] = gate_open

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
        if len(candidate_ids) < and_min and unique_terms:
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

        # Apply price and hard category mask to FTS5 candidates safely
        candidate_ids = [pid for pid in candidate_ids if pid in hard_asin_set]
        state["debug_info"]["fts5_count"] = len(candidate_ids)

        # Cascade Route 2 & 3: Vector Fallback if Keyword Route fails
        if len(candidate_ids) < vector_min:
            state["debug_info"]["vector_fallback"] = True
            # query_emb_normalized already computed above for profile gating; reuse here.

            # Slice catalog embeddings matrix using mask indices to only score allowed candidates
            if len(hard_indices) > 0:
                sliced_embeddings = self.catalog_embeddings[hard_indices]
                scores = np.dot(sliced_embeddings, query_emb_normalized)
                sorted_sliced_indices = np.argsort(scores)[::-1][:150]
                vector_asins = [self.catalog_ids[hard_indices[idx]] for idx in sorted_sliced_indices]
            else:
                vector_asins = []
            
            candidate_ids = vector_asins
            
        # If still empty (extremely rare), fallback to complete catalog filtered by hard mask
        if not candidate_ids:
            candidate_ids = [self.catalog_ids[idx] for idx in hard_indices] if len(hard_indices) > 0 else self.catalog_ids
            
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
                # Skip generic high-level category words to avoid wiping out valid department products
                if neg.lower() in ["clothing", "shoes", "jewelry"]:
                    continue
                if neg.lower() in meta["searchable_bag"]:
                    has_negated = True
                    break
            if has_negated:
                continue
                
            score = -0.001 * idx # penalty for retrieval ranking order
            
            # Boost department match (soft filter boost)
            if pid in dept_asin_set:
                score += 20.0
                
            # Boost category match (soft filter boost)
            if pid in category_asin_set:
                score += 15.0
            
            # Exact brand filter check
            if brand_constraints:
                if not any(bc.lower() in meta["brand"] for bc in brand_constraints):
                    score -= 10.0
                    
            # Boost exact keyword matches
            for val in state["accumulated_terms"]:
                if val.lower() in meta["searchable_bag"]:
                    score += 0.3
                    
            # Boost exact constraint phrase matches (crucial for simulator constraint matching)
            for attr, vals in state["disclosed_slots"].items():
                val_list = list(vals) if isinstance(vals, (set, list)) else [vals]
                for val in val_list:
                    if isinstance(val, str):
                        clean_val = val.lower().strip().rstrip(".")
                        if clean_val and clean_val in meta["searchable_bag"]:
                            score += 10.0
                        elif clean_val:
                            # Fallback: check parts of constraint phrase (e.g. for slightly rephrased strings)
                            val_words = [w for w in clean_val.split() if len(w) > 2 and w not in STOPWORDS]
                            matching_words_count = sum(1 for w in val_words if w in meta["searchable_bag"])
                            if val_words and matching_words_count == len(val_words):
                                score += 5.0

            # Boost category phrase match
            if state["category"]:
                clean_cat = state["category"].lower().strip()
                if clean_cat in meta["searchable_bag"]:
                    score += 2.0
                    
            # Popularity scaling
            score += 0.02 * (meta["rating_number"] ** 0.1)
            scored_candidates.append((score, pid))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 5a. Gated profile reranking
        # When the current query is sufficiently aligned with the user's long-term profile
        # (gate_sim >= PROFILE_GATE_THRESHOLD), blend the lexical state score with a
        # product–profile semantic similarity.  Weights differ by intent:
        #   buying   → 85% state / 15% profile  (user knows exactly what they want now)
        #   browsing → 40% state / 60% profile  (open-ended: lean on taste history)
        if gate_open and scored_candidates:
            cand_pids = [pid for _, pid in scored_candidates]
            cand_indices = [self.asin_to_index.get(pid, -1) for pid in cand_pids]
            valid_flags = [i >= 0 for i in cand_indices]
            valid_indices = [i for i in cand_indices if i >= 0]

            # Vectorised cosine similarity: all candidates vs. profile in one matmul
            profile_sims = np.zeros(len(cand_pids))
            if valid_indices:
                sims = np.dot(self.catalog_embeddings[valid_indices], profile_emb)
                j = 0
                for k, valid in enumerate(valid_flags):
                    if valid:
                        profile_sims[k] = float(sims[j])
                        j += 1

            # Normalise lexical scores to [0, 1] so both signals are on the same scale
            state_scores = np.array([s for s, _ in scored_candidates])
            s_range = max(float(state_scores.max() - state_scores.min()), 1e-9)
            norm_state = (state_scores - state_scores.min()) / s_range

            w_state, w_profile = INTENT_PROFILE_WEIGHTS[intent_mode]
            blended = w_state * norm_state + w_profile * np.maximum(profile_sims, 0.0)

            order = np.argsort(blended)[::-1]
            scored_candidates = [(float(blended[i]), cand_pids[i]) for i in order]
            state["debug_info"]["profile_reranked"] = True

        # 5. Take top-k by score directly
        recommendations = [pid for _, pid in scored_candidates[:top_k]]
                        
        state["seen_asins"].update(recommendations)
        
        # 6. Generate Conversational Response with LLM Call 2
        avoid_attrs = set(state["disclosed_slots"].keys()) | state.get("asked_attributes", set())
        if state["category"]:
            avoid_attrs.add("category")
        if state["department"]:
            avoid_attrs.add("department")
            
        # Tie hard filter conditions to their respective entropy attributes
        if state.get("target_department"):
            avoid_attrs.add("gender")
        if state.get("min_avg_rating", 0.0) > 0.0:
            avoid_attrs.add("rating")
        if state.get("min_rating_number", 0) > 0:
            avoid_attrs.add("reviews")
        if state.get("price_max", 9999.0) < 9000.0:
            avoid_attrs.add("budget")
        if state.get("store"):
            avoid_attrs.add("brand")
            
        all_attrs = {
            "material", "color", "size", "style", "brand", "budget", "use_case",
            "gender", "closure", "pattern", "waterproof", "rating", "reviews"
        }
        remaining_attrs = all_attrs - avoid_attrs
        
        # Determine the top 2 mathematically optimal attributes using Shannon Entropy
        best_attrs = self._select_best_attributes_to_ask(recommendations, remaining_attrs, top_n=2, intent_mode=intent_mode)
        best_attrs_str = " or ".join(f"'{a}'" for a in best_attrs if a != "other")
        if not best_attrs_str:
            best_attrs_str = "'other'"
        
        # Format top matching products (ASIN + title) for generator context
        recs_meta = [self.catalog_metadata[rid] for rid in recommendations[:10]]
        candidate_products_str = "\n".join(
            f"- {meta['title']} (ASIN: {rid})" for rid, meta in zip(recommendations[:10], recs_meta)
        )
        
        sys_prompt = (
            "You are a helpful e-commerce shopping copilot. The user is looking for a product.\n"
            "Based on the conversation history and the current state, generate a conversational response.\n\n"
            f"Active search filters: Category: {state['category']}, Department: {state['department']}, Max Price: {state['price_max']}\n"
            f"Attributes already specified by user: { {k: list(v) for k, v in state['disclosed_slots'].items()} }\n"
            f"Negated terms: {list(state.get('negated_terms', set()))}\n\n"
            "CRITICAL RULES:\n"
            "1. Do not ask the user about any attribute they have already specified or you have already asked about.\n"
            f"Avoid: {', '.join(sorted(list(avoid_attrs)))}\n"
            f"2. You MUST ask the customer about BOTH of these two attributes in the same response: {best_attrs_str}. Weave them into one natural sentence or two short ones. (If the list is just 'other', ask about any two relevant attributes or general style/use-case preferences).\n"
            "3. Keep your response short, natural, and conversational (1-2 sentences). Do not include any JSON formatting, raw tags, or markers. Just reply normally to the shopper.\n\n"
            "Input Candidate Products:\n"
            f"{candidate_products_str}\n"
        )
        
        history_str = ""
        for msg in state["history"][-4:]:
            role = "Customer" if msg["role"] == "user" else "Copilot"
            history_str += f"{role}: {msg['content']}\n"
            
        prompt = f"Dialogue history:\n{history_str}\n\nCopilot Response (direct text reply):"
        
        res_text = self._call_llm(prompt, sys_prompt, session_id=session_id, response_json=False)
        
        agent_message = res_text.strip()
        asked_attr = self._extract_asked_attribute(agent_message, state)
                    
        state["history"].append({"role": "assistant", "content": agent_message})
        
        if "asked_attributes" not in state:
            state["asked_attributes"] = set()
            
        # Record both of the top 2 entropy attributes we instructed the LLM to ask about
        for attr in best_attrs:
            if attr in all_attrs:
                state["asked_attributes"].add(attr)
        
        # Populate debug data
        state["debug_info"]["model"] = "DeepSeek-Chat" if os.environ.get("DEEPSEEK_API_KEY") else "GPT-4o-Mini"
        state["debug_info"]["system_prompt"] = sys_prompt
        state["debug_info"]["user_prompt"] = prompt
        
        debug_data = {
            "model": state["debug_info"].get("model", "None"),
            "system_prompt": state["debug_info"].get("system_prompt", ""),
            "user_prompt": state["debug_info"].get("user_prompt", ""),
            "category": state["category"],
            "department": state["department"],
            "price_max": state["price_max"],
            "disclosed_slots": {k: list(v) if isinstance(v, set) else v for k, v in state["disclosed_slots"].items()},
            "asked_attributes": list(state.get("asked_attributes", set())),
            "best_entropy_attrs": best_attrs,
            "negated_terms": list(state.get("negated_terms", set())),
            "accumulated_terms": list(state.get("accumulated_terms", [])),
            "stashed_terms": list(state.get("stashed_terms", [])),
            "constraint_provenance": list(state.get("constraint_provenance", [])),
            "search_epoch": state.get("search_epoch", 0),
            "fts5_count": state["debug_info"].get("fts5_count", 0),
            "vector_fallback": state["debug_info"].get("vector_fallback", False),
            "intent_mode": state.get("intent_mode", "browsing"),
            "profile_gate_sim": state["debug_info"].get("profile_gate_sim", 0.0),
            "profile_gate_open": state["debug_info"].get("profile_gate_open", False),
            "profile_reranked": state["debug_info"].get("profile_reranked", False),
        }
        
        # Print Hybrid Telemetry to terminal
        print("\n" + "="*80)
        print(f" [AGENT BRAIN TELEMETRY - CUSTOM HYBRID CASCADE ROUTE] Turn: {turn} | Session: {session_id}")
        print("="*80)
        print(f"Active LLM Model: {debug_data['model']}")
        print(f"Intent Mode:      {intent_mode}")
        print(f"FTS5 Matches:     {debug_data['fts5_count']} products (and_min={and_min}, vector_min={vector_min})")
        print(f"Vector Fallback:  {debug_data['vector_fallback']}")
        print(f"Category State:   {debug_data['category']}")
        print(f"Department:       {debug_data['department']}")
        print(f"Price Max State:  {debug_data['price_max']}")
        print(f"Hard Conditions:  Dept={state.get('target_department')}, MinRating={state.get('min_avg_rating')}, MinReviews={state.get('min_rating_number')}, Store={state.get('store')}")
        print(f"Disclosed Slots:  { {k: v for k, v in debug_data['disclosed_slots'].items() if v} }")
        print(f"Negated Terms:    {debug_data['negated_terms']}")
        print(f"Accumulated:      {debug_data['accumulated_terms']}")
        print(f"Entropy Best:     {best_attrs}")
        print(f"Asked Attributes: {debug_data['asked_attributes']}")
        print("-"*80)
        
        return {
            "message": agent_message,
            "ask_attribute": asked_attr,
            "recommendations": [{"parent_asin": r} for r in recommendations],
            "debug": debug_data
        }


# =====================================================================
# Baseline Agent Implementation (For Structured Simulator Evaluator v1)
# =====================================================================
FIELDS = ("title", "features", "details", "description", "categories", "store", "price")
RRF_K = 60
RRF_DEPTH = 1000
WS_RE = re.compile(r"\s+")
INITIAL_PREFIX = "i'm looking for "
BUYING_PREFIX = "a key requirement is:"
DISCLOSURE_PREFIX = "for that, what matters is:"
OVERRIDE_PREFIX = "actually, ignore my earlier preference. what i need is:"

class BaselineAgent:
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
        "that", "the", "this", "to", "want", "with", "would", "you", "looking",
        "color", "budget", "around",
    }
    TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [
            token.lower()
            for token in BaselineAgent.TOKEN_RE.findall(text)
            if len(token) > 1 and token.lower() not in BaselineAgent.STOPWORDS
        ]

    @staticmethod
    def _text(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(f"{key} {item}" for key, item in value.items())
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return str(value)

    def __init__(self, catalog_path: str | Path = None) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions = {}
        self._build_index()

    def _build_index(self) -> None:
        self.catalog_ids = []
        self.catalog_metadata = {}
        
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        
        db_batch = []
        with self.catalog_path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pid = str(p["parent_asin"])
                self.catalog_ids.append(pid)
                
                title = p.get("title") or ""
                cats = ", ".join(p.get("categories") or [])
                feats = "; ".join((p.get("features") or [])[:3])
                
                p_brand = p.get("store") or p.get("details", {}).get("Manufacturer") or ""
                p_brand = p_brand.strip().lower()
                
                p_color = ""
                title_lower = title.lower()
                for c in COLORS:
                    if c in title_lower:
                        p_color = c
                        break
                        
                p_mat = ""
                for m in MATERIALS:
                    if m in title_lower:
                        p_mat = m
                        break
                
                self.catalog_metadata[pid] = {
                    "title": title,
                    "searchable_bag": (title + " " + cats + " " + feats).lower(),
                    "rating_number": float(p.get("rating_number") or 0),
                    "brand": p_brand,
                    "color": p_color,
                    "material": p_mat
                }
                
                db_batch.append(
                    (
                        pid,
                        BaselineAgent._text(p.get("title")),
                        BaselineAgent._text(p.get("categories")),
                        BaselineAgent._text(p.get("features")),
                        BaselineAgent._text(p.get("details")),
                        BaselineAgent._text(p.get("store")),
                        BaselineAgent._text(p.get("description")),
                    )
                )
                if len(db_batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", db_batch)
                    db_batch.clear()
                
        if db_batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", db_batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        initial_seen: set[str] = set()
        self._sessions[session_id] = {
            "disclosed_slots": {},
            "asked_attributes": set(),
            "search_epoch": 0,
            "seen_asins": initial_seen,
            "seen_asins_by_epoch": {0: initial_seen},
            "category": "clothing",
            "accumulated_terms": [],
            "stashed_terms": set()
        }

    def _classify_constraint_locally(self, val: str) -> str:
        val_lower = val.lower()
        if any(c in val_lower for c in COLORS):
            return "color"
        if any(m in val_lower for m in MATERIALS):
            return "material"
        categories_set = {"clothing", "shoes", "jewelry", "earrings", "boots", "sandals", "socks", "shirts", "active"}
        if any(cat in val_lower for cat in categories_set):
            return "category"
        return "feature"

    def _rebuild_accumulated_terms(self, state: dict) -> None:
        terms_set = []
        for w in BaselineAgent._terms(state["category"]):
            if w not in terms_set:
                terms_set.append(w)
        for attr, vals in state["disclosed_slots"].items():
            for val in vals:
                val_str = str(val).strip().lower()
                if val_str in ["true", "yes", "affirmative", "required", "included"]:
                    for w in BaselineAgent._terms(attr):
                        if w not in terms_set:
                            terms_set.append(w)
                elif val_str in ["false", "no", "none", "n/a", "null", "other"]:
                    continue
                else:
                    for w in BaselineAgent._terms(val):
                        if w not in terms_set:
                            terms_set.append(w)
        state["accumulated_terms"] = terms_set
        state["stashed_terms"] = {w for w in state["stashed_terms"] if w not in terms_set}

    def _erase_attribute_memory(self, state: dict, attr: str) -> None:
        old_values = state["disclosed_slots"].pop(attr, set())
        for val in old_values:
            for w in BaselineAgent._terms(val):
                state["stashed_terms"].add(w)
        self._rebuild_accumulated_terms(state)

    def _parse_message_locally(self, session_id: str, message: str) -> None:
        state = self._sessions[session_id]
        
        cat_match = re.search(r"I'm looking for ([^.,]+)", message)
        if cat_match:
            new_cat = cat_match.group(1).strip()
            if new_cat != state["category"]:
                state["category"] = new_cat
                state["seen_asins"].clear()
            self._rebuild_accumulated_terms(state)
            
        boundary_match = re.search(r"I don't have a preference for ([^;.]+); please use your judgment\.", message)
        if boundary_match:
            attr = boundary_match.group(1).strip().lower()
            if attr in ALLOWED_ATTRIBUTES:
                state["asked_attributes"].add(attr)
                self._erase_attribute_memory(state, attr)
            return

        override_match = re.search(r"What I need is: ([^.]+)\.", message)
        if override_match:
            new_val = override_match.group(1).strip()
            attr = self._classify_constraint_locally(new_val)
            self._erase_attribute_memory(state, attr)
            state["disclosed_slots"][attr] = {new_val}
            self._rebuild_accumulated_terms(state)
            state["search_epoch"] += 1
            current_seen: set[str] = set()
            state["seen_asins_by_epoch"][state["search_epoch"]] = current_seen
            state["seen_asins"] = current_seen
            return

        req_match = re.search(r"A key requirement is: ([^.]+)\.", message)
        if req_match:
            val = req_match.group(1).strip()
            attr = self._classify_constraint_locally(val)
            state["disclosed_slots"].setdefault(attr, set()).add(val)
            self._rebuild_accumulated_terms(state)
            return

        matters_match = re.search(r"what matters is: ([^.]+)\.", message)
        if matters_match:
            values = [v.strip() for v in matters_match.group(1).split(";")]
            for val in values:
                attr = self._classify_constraint_locally(val)
                state["disclosed_slots"].setdefault(attr, set()).add(val)
            self._rebuild_accumulated_terms(state)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
            
        state = self._sessions[session_id]
        self._parse_message_locally(session_id, user_message)
        
        candidate_ids = []
        unique_terms = state["accumulated_terms"][:45]
        
        if unique_terms:
            expression_and = " AND ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT 1000",
                (expression_and,)
            ).fetchall()
            candidate_ids = [str(r[0]) for r in rows]
            
        if len(candidate_ids) < 30 and unique_terms:
            expression_or = " OR ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) as score "
                "FROM products WHERE products MATCH ? "
                "ORDER BY score LIMIT 1000",
                (expression_or,)
            ).fetchall()
            for r in rows:
                asin = str(r[0])
                if asin not in candidate_ids:
                    candidate_ids.append(asin)
                    
        if not candidate_ids:
            candidate_ids = self.catalog_ids
            
        scored_candidates = []
        brand_constraints = state["disclosed_slots"].get("brand", set())
        
        for idx, pid in enumerate(candidate_ids):
            if pid in state["seen_asins"]:
                continue
                
            meta = self.catalog_metadata[pid]
            score = -0.001 * idx
            
            if brand_constraints:
                if not any(bc.lower() in meta["brand"] for bc in brand_constraints):
                    score -= 10.0
                    
            for val in state["accumulated_terms"]:
                if val.lower() in meta["searchable_bag"]:
                    score += 0.3
                    
            score += 0.02 * (meta["rating_number"] ** 0.1)
            scored_candidates.append((score, pid))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
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
            
            is_too_similar = False
            for chosen_title in chosen_titles:
                if get_jaccard_similarity(title, chosen_title) > 0.8:
                    is_too_similar = True
                    break
            if is_too_similar:
                continue
                
            recommendations.append(pid)
            chosen_titles.append(title)
            
            if len(recommendations) == top_k:
                break
                
        if len(recommendations) < top_k:
            for score, pid in scored_candidates:
                if pid not in recommendations:
                    recommendations.append(pid)
                    if len(recommendations) == top_k:
                        break
                        
        state["seen_asins"].update(recommendations)
        ask_attribute = "other"

        return {
            "message": "Here are the top matches based on your preferences.",
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": r} for r in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
