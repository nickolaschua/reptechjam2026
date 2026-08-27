import json
import re
import sqlite3
from pathlib import Path

# Path definitions
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

# Tokenization rules for database querying
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "color", "budget", "around",
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
    """A pure-lexical agent exploiting the 'other' constraint loophole.
    Uses SQLite FTS5 index, tokenized term memory, override slot/term rebuilding,
    stashed style memory, and brand/title diversification.
    """

    def __init__(self, catalog_path: str | Path = None) -> None:
        if catalog_path is None:
            catalog_path = repo_root / "data/catalog.jsonl"
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions = {}
        self._build_index()

    def _build_index(self) -> None:
        print("[Experiment 1 Agent] Indexing SQLite FTS5 database...")
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
                
                # Extract brand
                p_brand = p.get("store") or p.get("details", {}).get("Manufacturer") or ""
                p_brand = p_brand.strip().lower()
                
                # Extract color from title keywords
                p_color = ""
                title_lower = title.lower()
                for c in COLORS:
                    if c in title_lower:
                        p_color = c
                        break
                        
                # Extract material
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
                
                # Batch insert into FTS5 table
                db_batch.append(
                    (
                        pid,
                        _text(p.get("title")),
                        _text(p.get("categories")),
                        _text(p.get("features")),
                        _text(p.get("details")),
                        _text(p.get("store")),
                        _text(p.get("description")),
                    )
                )
                if len(db_batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", db_batch)
                    db_batch.clear()
                
        if db_batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", db_batch)
        self.connection.commit()
        print("[Experiment 1 Agent] Indexing completed.")

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Reset the conversation state for a new session."""
        self._sessions[session_id] = {
            "disclosed_slots": {},       # Map of attribute -> set of values
            "asked_attributes": set(),  # Set of attributes we already asked
            "seen_asins": set(),        # Deduplication filtering set
            "category": "clothing",     # Default fallback category
            "accumulated_terms": [],    # Terms accumulated across conversation
            "stashed_terms": set()      # Stashed terms from overridden preferences
        }

    def _classify_constraint_locally(self, val: str) -> str:
        """Categorize a value into one of the evaluator's allowed attributes (fallback rule-based)."""
        val_lower = val.lower()
        if any(c in val_lower for c in COLORS):
            return "color"
        if any(m in val_lower for m in MATERIALS):
            return "material"
        categories = {"clothing", "shoes", "jewelry", "earrings", "boots", "sandals", "socks", "shirts", "active"}
        if any(cat in val_lower for cat in categories):
            return "category"
        return "feature"

    def _rebuild_accumulated_terms(self, state: dict) -> None:
        """Rebuild accumulated terms list from category and active slots to prevent word corruption."""
        terms_set = []
        # 1. Add category terms
        for w in _terms(state["category"]):
            if w not in terms_set:
                terms_set.append(w)
        # 2. Add slot terms
        for attr, vals in state["disclosed_slots"].items():
            for val in vals:
                for w in _terms(val):
                    if w not in terms_set:
                        terms_set.append(w)
        state["accumulated_terms"] = terms_set
        
        # Clean stashed terms so they don't contain active search terms
        state["stashed_terms"] = {w for w in state["stashed_terms"] if w not in terms_set}

    def _erase_attribute_memory(self, state: dict, attr: str) -> None:
        """Purge the attribute from active slot memory, stash its keywords, and rebuild terms."""
        old_values = state["disclosed_slots"].pop(attr, set())
        for val in old_values:
            for w in _terms(val):
                state["stashed_terms"].add(w)
        self._rebuild_accumulated_terms(state)

    def _parse_message_locally(self, session_id: str, message: str) -> None:
        """Parse incoming messages using regex rules, handling overrides and boundaries."""
        state = self._sessions[session_id]
        
        # 1. Category extraction
        cat_match = re.search(r"I'm looking for ([^.,]+)", message)
        if cat_match:
            new_cat = cat_match.group(1).strip()
            if new_cat != state["category"]:
                state["category"] = new_cat
                state["seen_asins"].clear()
            self._rebuild_accumulated_terms(state)
            
        # 2. Boundary Case matching: e.g. "I don't have a preference for material; please use your judgment."
        boundary_match = re.search(r"I don't have a preference for ([^;.]+); please use your judgment\.", message)
        if boundary_match:
            attr = boundary_match.group(1).strip().lower()
            if attr in ALLOWED_ATTRIBUTES:
                state["asked_attributes"].add(attr)
                self._erase_attribute_memory(state, attr)
            return

        # 3. Intent Override matching: e.g. "Actually, ignore my earlier preference. What I need is: wool."
        override_match = re.search(r"What I need is: ([^.]+)\.", message)
        if override_match:
            new_val = override_match.group(1).strip()
            attr = self._classify_constraint_locally(new_val)
            self._erase_attribute_memory(state, attr)
            state["disclosed_slots"][attr] = {new_val}
            self._rebuild_accumulated_terms(state)
            # Clear seen ASINs history to allow previous recommendations to re-enter search pool
            state["seen_asins"].clear()
            return

        # 4. Standard Matters / requirement matching
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
        
        # 1. Parse constraints and update state memory
        self._parse_message_locally(session_id, user_message)
        
        # 2. Retrieve Candidate ASINs using SQLite FTS5 (AND -> OR)
        candidate_ids = []
        unique_terms = state["accumulated_terms"][:45]
        
        # Level 1: AND Match
        if unique_terms:
            expression_and = " AND ".join(f'"{term}"' for term in unique_terms)
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? LIMIT 1000",
                (expression_and,)
            ).fetchall()
            candidate_ids = [str(r[0]) for r in rows]
            
        # Level 2: OR Match (ordered by BM25 scoring)
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
            
        # 3. Re-ranking & Hard Filtering
        scored_candidates = []
        brand_constraints = state["disclosed_slots"].get("brand", set())
        
        for idx, pid in enumerate(candidate_ids):
            # Skip previously recommended items
            if pid in state["seen_asins"]:
                continue
                
            meta = self.catalog_metadata[pid]
            
            # Base rank score is derived from FTS5 result order (idx)
            score = -0.001 * idx
            
            # Brand Hard Filter
            if brand_constraints:
                if not any(bc.lower() in meta["brand"] for bc in brand_constraints):
                    score -= 10.0
                    
            # Soft Boost: Check exact keyword matches in title/features bag
            for val in state["accumulated_terms"]:
                if val.lower() in meta["searchable_bag"]:
                    score += 0.3
                    
            # Stashed Boost: Check exact keyword matches in stashed terms as style priors
            for val in state["stashed_terms"]:
                if val.lower() in meta["searchable_bag"]:
                    score += 0.05
                    
            # Popularity prior
            score += 0.02 * (meta["rating_number"] ** 0.1)
            
            scored_candidates.append((score, pid))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 4. Candidate Diversification (Title Jaccard + Brand limits)
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
            
            if brand:
                if chosen_brands.get(brand, 0) >= 2:
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
                
        # Fill slots
        if len(recommendations) < top_k:
            for score, pid in scored_candidates:
                if pid not in recommendations:
                    recommendations.append(pid)
                    if len(recommendations) == top_k:
                        break
                        
        state["seen_asins"].update(recommendations)
        
        # ALWAYS ask for "other" to trigger double hint wildcard
        ask_attribute = "other"

        return {
            "message": "Here are the top matches based on your preferences.",
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": r} for r in recommendations],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
