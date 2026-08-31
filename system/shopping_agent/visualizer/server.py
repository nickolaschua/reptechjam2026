"""Yangxu dashboard adapter for the canonical longitudinal shopping agent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import wraps
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any
import urllib.parse
from uuid import uuid4

from ..agent import Agent
from ..config import (
    ALLOW_CATALOG_EMBEDDING,
    CATALOG_PATH,
    MEMORY_STORE_PATH,
    PROJECT_ROOT,
)
from ..memory_store import JsonFileVectorMemoryStore
from ..vector_memory import BuyerMode
from .simulator import (
    call_shopper_llm, coarse_category, load_samples, make_system_prompt,
    materialize_hidden_fields,
)


VISUALIZER_DIR = Path(__file__).resolve().parent
PUBLIC_SET_PATH = PROJECT_ROOT / "techjam-conversational-search" / "data" / "public_set.jsonl"
IMAGE_MAP_PATH = VISUALIZER_DIR / "assets" / "asin_images.json"
MAX_TURNS = 10
TOP_K = 10
CATALOG_PAGE_SIZE = 24
CATALOG_DEPARTMENT_KEYWORDS = {
    "clothing": {
        "clothing", "shirt", "pant", "dress", "jacket", "hoodie", "sweater",
        "coat", "shorts", "top", "blouse", "skirt", "jeans",
    },
    "shoes": {
        "shoes", "shoe", "boot", "sneaker", "sandal", "loafer", "heel",
        "slipper", "clog", "slide", "oxford",
    },
    "jewelry": {
        "jewelry", "jewellery", "necklace", "ring", "bracelet", "earring",
        "pendant", "bangle", "brooch",
    },
    "watches": {"watch", "watches", "wristwatch", "timepiece", "chronograph"},
}


def _synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        lock = getattr(self, "_lifecycle_lock", None)
        if lock is None:
            lock = self._lifecycle_lock = RLock()
        with lock:
            return method(self, *args, **kwargs)
    return wrapped


def _whole_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text.lower()) is not None


def _catalog_family(categories: list[str]) -> str:
    """Assign exactly one dashboard family from leaf-to-root category components."""

    for component in reversed(categories):
        normalized = " ".join(str(component).lower().split())
        root_families = {
            family for family in ("clothing", "shoes", "jewelry")
            if _whole_keyword(normalized, family)
        }
        if len(root_families) > 1:
            continue
        matches = {
            family for family, keywords in CATALOG_DEPARTMENT_KEYWORDS.items()
            if any(_whole_keyword(normalized, keyword) for keyword in keywords)
        }
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            return "other"
    return "other"


def _safe_image_url(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return raw
    if not parsed.scheme and (raw.startswith("/assets/") or raw.startswith("assets/")):
        return raw
    return None


def buyer_mode_for_scenario(scenario: str) -> BuyerMode:
    return BuyerMode.BROWSING if str(scenario).lower() == "browsing" else BuyerMode.BUYING


@dataclass
class BrowserSession:
    sample: dict[str, Any]
    card: dict[str, Any]
    behavior: dict[str, Any]
    session_id: str
    sequence_index: int
    target_asin: str
    turn: int = 0
    committed: bool = False


class BrowserApplication:
    def __init__(
        self,
        *,
        memory_path: str | Path = MEMORY_STORE_PATH,
        allow_catalog_embedding: bool = ALLOW_CATALOG_EMBEDDING,
        agent: Agent | None = None,
        store: JsonFileVectorMemoryStore | None = None,
    ) -> None:
        self.store = store or JsonFileVectorMemoryStore(memory_path)
        self.agent = agent or Agent(
            memory_store=self.store,
            allow_catalog_embedding=allow_catalog_embedding,
        )
        self.samples = {sample["sample_id"]: sample for sample in load_samples(PUBLIC_SET_PATH)}
        self.products = {product["parent_asin"]: product for product in self.agent.catalog_products}
        self.catalog_ids = set(self.products)
        self.images = json.loads(IMAGE_MAP_PATH.read_text(encoding="utf-8")) if IMAGE_MAP_PATH.exists() else {}
        self.catalog_rows = self._build_catalog_rows()
        self.active: dict[str, BrowserSession] = {}
        self.finished_traces: dict[str, dict[str, Any]] = {}
        self._lifecycle_lock = RLock()

    @_synchronized
    def finish(self, sample_id: str, *, reason: str) -> dict[str, Any] | None:
        active = self.active.get(sample_id)
        if active is None or active.committed:
            return self.finished_traces.get(sample_id)
        self.agent.end_session(active.session_id)
        trace = self.agent.get_memory_debug(active.session_id, consume=True)
        trace["commit_reason"] = reason
        trace["ltm_updated_after_session"] = trace["ltm_updated_after_turn"]
        trace["memory_update_text"] = trace["preference_text"]
        active.committed = True
        self.finished_traces[sample_id] = trace
        del self.active[sample_id]
        return trace

    @_synchronized
    def finish_all(self, *, reason: str) -> None:
        for sample_id in list(self.active):
            try:
                self.finish(sample_id, reason=reason)
            except Exception as exc:
                print(f"[Server Warning] Could not commit {sample_id}: {exc}")

    @_synchronized
    def discard(self, sample_id: str) -> None:
        active = self.active.pop(sample_id, None)
        if active is not None and not active.committed:
            self.agent.discard_session(active.session_id)

    @_synchronized
    def start(self, sample_id: str) -> BrowserSession:
        if sample_id not in self.samples:
            raise KeyError("Sample not found")
        # The dashboard presents one active run. Starting another cleanly replaces it.
        self.finish_all(reason="replaced")
        sample = self.samples[sample_id]
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, self.products)
        sequence = self.store.next_sequence_index(sample_id)
        session_id = f"browser-{sample_id}-{sequence}-{uuid4().hex[:8]}"
        self.agent.reset(session_id, sample.get("user_profile", {}), user_id=sample_id, sequence_index=sequence)
        active = BrowserSession(sample, card, behavior, session_id, sequence, target)
        self.active[sample_id] = active
        return active

    @staticmethod
    def _target_meta(product: dict[str, Any]) -> dict[str, Any]:
        details = product.get("details") if isinstance(product.get("details"), dict) else {}
        return {
            "title": product.get("title"),
            "brand": product.get("store") or details.get("Manufacturer") or "Unknown",
            "features": product.get("features", []), "description": product.get("description", ""),
            "details": details, "avg_rating": product.get("average_rating", 0.0),
            "rating_number": product.get("rating_number", 0), "price": product.get("price", 0.0),
            "categories": product.get("categories", []),
        }

    def target_payload(self, active: BrowserSession) -> dict[str, Any]:
        product = self.products[active.target_asin]
        return {
            "session_id": active.sample["sample_id"], "scenario": active.sample["scenario_type"],
            "target_asin": active.target_asin, "target_title": product.get("title"),
            "target_brand": product.get("store") or (product.get("details") or {}).get("Manufacturer") or "Unknown",
            "hard_constraints": active.card.get("hard_constraints", []),
            "soft_preferences": active.card.get("soft_preferences", []),
            "target_meta": self._target_meta(product),
        }

    def sessions_list(self) -> list[dict[str, Any]]:
        result = []
        for sample in self.samples.values():
            target = str(sample["ground_truth"]["parent_asin"])
            if target not in self.products:
                continue
            card, _ = materialize_hidden_fields(sample, self.products)
            product = self.products[target]
            result.append({
                "sample_id": sample["sample_id"], "scenario": sample["scenario_type"],
                "target_title": product.get("title", "Unknown"),
                "target_brand": product.get("store") or (product.get("details") or {}).get("Manufacturer") or "Unknown",
                "target_asin": target, "hard_constraints": card.get("hard_constraints", []),
                "soft_preferences": card.get("soft_preferences", []), "target_meta": self._target_meta(product),
            })
        return result

    @staticmethod
    def _number(value: object, *, integer: bool = False) -> float | int:
        try:
            normalized = str(value or "").replace("$", "").replace(",", "").strip()
            if not normalized:
                return 0
            return int(float(normalized)) if integer else float(normalized)
        except (TypeError, ValueError):
            return 0

    def _build_catalog_rows(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for asin, product in self.products.items():
            details = product.get("details") if isinstance(product.get("details"), dict) else {}
            categories = [str(value) for value in product.get("categories") or []]
            title = str(product.get("title") or "")
            brand = str(product.get("store") or details.get("Manufacturer") or "Unknown")
            rows.append({
                "asin": str(asin),
                "title": title,
                "brand": brand,
                "price": self._number(product.get("price")),
                "avg_rating": self._number(product.get("average_rating")),
                "rating_number": self._number(product.get("rating_number"), integer=True),
                "categories": categories[:3],
                "_family": _catalog_family(categories),
                "_query_text": f"{title} {brand} {' '.join(categories)}".lower(),
            })
        return tuple(rows)

    def catalog_search(
        self,
        *,
        q: str = "",
        department: str = "all",
        max_price: float | None = None,
        min_rating: float = 0.0,
        page: int = 1,
    ) -> dict[str, Any]:
        """Return Yangxu's catalog payload over the complete active catalogue."""

        query = str(q).lower().strip()
        dept = str(department).lower().strip() or "all"
        if dept != "all" and dept not in CATALOG_DEPARTMENT_KEYWORDS:
            dept = "all"
        page_number = max(1, int(page))
        rows = getattr(self, "catalog_rows", None)
        if rows is None:
            rows = self._build_catalog_rows()
            self.catalog_rows = rows

        matches: list[dict[str, Any]] = []
        for row in rows:
            if query and query not in row["_query_text"]:
                continue
            if dept != "all" and row["_family"] != dept:
                continue
            price = float(row["price"])
            if max_price is not None and 0.0 < price > float(max_price):
                continue
            rating = float(row["avg_rating"])
            if float(min_rating) > 0.0 and 0.0 < rating < float(min_rating):
                continue
            matches.append(row)

        if not query:
            matches.sort(key=lambda row: (-int(row["rating_number"]), row["asin"]))
        total = len(matches)
        start = (page_number - 1) * CATALOG_PAGE_SIZE
        products = []
        for row in matches[start:start + CATALOG_PAGE_SIZE]:
            products.append({
                "asin": row["asin"], "title": row["title"], "brand": row["brand"],
                "price": round(float(row["price"]), 2),
                "avg_rating": float(row["avg_rating"]),
                "rating_number": int(row["rating_number"]),
                "categories": list(row["categories"]),
                "image_url": _safe_image_url(self.images.get(row["asin"])),
            })
        return {
            "total": total, "page": page_number,
            "per_page": CATALOG_PAGE_SIZE, "products": products,
        }

    def _product_card(self, asin: str, rank: int, target: str) -> dict[str, Any]:
        product = self.products[asin]
        details = product.get("details") if isinstance(product.get("details"), dict) else {}
        return {
            "rank": rank, "brand": product.get("store") or details.get("Manufacturer") or "Unknown",
            "title": product.get("title"), "asin": asin, "categories": product.get("categories") or [],
            "image_url": _safe_image_url(self.images.get(asin)), "is_target": asin == target,
            "features": product.get("features", []), "details": details,
            "avg_rating": product.get("average_rating", 0.0), "rating_number": product.get("rating_number", 0),
            "price": product.get("price", 0.0),
        }

    @_synchronized
    def step(self, sample_id: str, message: str) -> dict[str, Any]:
        active = self.active.get(sample_id)
        if active is None:
            raise RuntimeError("Session not initialized")
        next_turn = active.turn + 1
        response = self.agent.respond(
            active.session_id, message, next_turn, TOP_K,
            buyer_mode=buyer_mode_for_scenario(active.sample["scenario_type"]), debug=True,
        )
        active.turn = next_turn
        ranked = [str(item.get("parent_asin", "")) for item in response.get("recommendations", []) if str(item.get("parent_asin", "")) in self.catalog_ids][:TOP_K]
        cards = [self._product_card(asin, index, active.target_asin) for index, asin in enumerate(ranked, 1)]
        hit = active.target_asin in ranked
        if hit or active.turn >= MAX_TURNS:
            commit = self.finish(sample_id, reason="target_hit" if hit else "turn_10")
            response["debug"]["memory_trace"].update({
                "ltm_updated_after_session": bool(commit and commit["ltm_updated_after_session"]),
                "memory_update_text": None if commit is None else commit["memory_update_text"],
            })
        return {
            "message": response.get("message"), "ask_attribute": response.get("ask_attribute"),
            "recommendations": cards, "turn": active.turn, "success": hit, "debug": response.get("debug"),
        }

    @_synchronized
    def close(self) -> None:
        try:
            self.finish_all(reason="server_shutdown")
        finally:
            for sample_id in list(self.active):
                self.discard(sample_id)
            self.agent.close()


APPLICATION: BrowserApplication | None = None


class VisualizerHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(VISUALIZER_DIR), **kwargs)

    @property
    def app(self) -> BrowserApplication:
        assert APPLICATION is not None
        return APPLICATION

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/sessions_list": self._json(self.app.sessions_list())
            elif parsed.path == "/manual_start": self._manual_start(parsed.query)
            elif parsed.path == "/manual_step": self._manual_step(parsed.query)
            elif parsed.path == "/stream": self._stream(parsed.query)
            elif parsed.path == "/conversation": self._serve_html("conversation.html")
            elif parsed.path == "/catalog_search": self._catalog_search(parsed.query)
            else: super().do_GET()
        except KeyError as exc:
            self._error(404, str(exc))
        except (ValueError, RuntimeError) as exc:
            self._error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            print(f"[Server Error] {type(exc).__name__}: {exc}")
            try: self._error(500, str(exc))
            except (BrokenPipeError, ConnectionResetError): pass

    def _json(self, data: object, code: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, code: int, message: str) -> None:
        self._json({"error": message}, code)

    def _serve_html(self, filename: str) -> None:
        payload = (VISUALIZER_DIR / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _catalog_search(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        try:
            max_price_text = params.get("max_price", [""])[0].strip()
            max_price = float(max_price_text) if max_price_text else None
        except ValueError:
            max_price = None
        try:
            min_rating = float(params.get("min_rating", ["0"])[0])
        except ValueError:
            min_rating = 0.0
        try:
            page = max(1, int(params.get("page", ["1"])[0]))
        except ValueError:
            page = 1
        self._json(self.app.catalog_search(
            q=params.get("q", [""])[0],
            department=params.get("dept", ["all"])[0],
            max_price=max_price,
            min_rating=min_rating,
            page=page,
        ))

    @staticmethod
    def _sample_id(query: str) -> str:
        return urllib.parse.parse_qs(query).get("sample_id", ["public_0001"])[0]

    def _manual_start(self, query: str) -> None:
        active = self.app.start(self._sample_id(query))
        self._json(self.app.target_payload(active))

    def _manual_step(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        sample_id = params.get("sample_id", ["public_0001"])[0]
        message = params.get("message", [""])[0]
        if not message.strip(): raise ValueError("message must be non-empty")
        self._json(self.app.step(sample_id, message))

    def _send_sse(self, event: str, data: object) -> None:
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _stream(self, query: str) -> None:
        sample_id = self._sample_id(query)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        active: BrowserSession | None = None
        finish_reason = "provider_failure"
        try:
            active = self.app.start(sample_id)
            self._send_sse("target", self.app.target_payload(active))
            product = self.app.products[active.target_asin]
            effective = {**active.sample, "intent_card": active.card, "behavior": active.behavior}
            system_prompt = make_system_prompt(effective, product, coarse_category(product.get("categories") or []))
            previous: dict[str, Any] = {}
            recs: list[dict[str, Any]] = []
            override = active.behavior.get("override", {})
            override_applied = active.sample["scenario_type"] != "intent_override"
            for turn in range(1, MAX_TURNS + 1):
                if turn == 1:
                    prompt = "Start by saying what you are looking for."
                    if not override_applied: prompt += f" Mention your initial preference for '{override.get('old_value', '')}'."
                    elif active.sample["scenario_type"] == "buying" and active.card.get("hard_constraints"):
                        prompt += f" Mention this key requirement: '{active.card['hard_constraints'][0]}'."
                    message = call_shopper_llm(
                        prompt,
                        system_prompt,
                        client=self.app.agent.ollama_client,
                    )
                elif not override_applied and turn == int(override.get("turn", 3)):
                    override_applied = True
                    message = override.get("message", "Actually, ignore my earlier preference.")
                else:
                    listed = "\n".join(f"{i}. {item['title']} (ASIN: {item['asin']})" for i, item in enumerate(recs, 1))
                    prompt = f"Assistant response: {previous.get('message')}\nRecommendations:\n{listed}\nAsked about: {previous.get('ask_attribute')}\nReply briefly in character. If target ASIN {active.target_asin} appears, say you want it."
                    message = call_shopper_llm(
                        prompt,
                        system_prompt,
                        client=self.app.agent.ollama_client,
                    )
                self._send_sse("msg", {"role": "customer", "content": message, "turn": turn})
                time.sleep(0.5)
                previous = self.app.step(sample_id, message)
                recs = previous["recommendations"]
                self._send_sse("msg", {"role": "copilot", "content": previous["message"], "ask": previous["ask_attribute"], "recommendations": recs, "turn": turn, "debug": previous["debug"]})
                if previous["success"]:
                    finish_reason = "target_hit"
                    rank = next(item["rank"] for item in recs if item["is_target"])
                    self._send_sse("status", {"success": True, "turn": turn, "rank": rank})
                    break
                if turn == MAX_TURNS:
                    finish_reason = "turn_10"
                    self._send_sse("status", {"success": False, "turn": turn})
        except (BrokenPipeError, ConnectionResetError):
            finish_reason = "client_disconnect"
            raise
        finally:
            if active is not None:
                if finish_reason in {"target_hit", "turn_10"}:
                    self.app.finish(sample_id, reason=finish_reason)
                else:
                    self.app.discard(sample_id)


def run_server(
    port: int = 8080,
    *,
    allow_catalog_embedding: bool = ALLOW_CATALOG_EMBEDDING,
) -> None:
    global APPLICATION
    print("[Server] Loading the 50,000-row catalogue and local BGE matrix...")
    APPLICATION = BrowserApplication(
        allow_catalog_embedding=allow_catalog_embedding,
    )
    server = ThreadingHTTPServer(("0.0.0.0", int(port)), VisualizerHTTPHandler)
    print(f"[Server] Yangxu dashboard: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Server] Shutting down.")
    finally:
        APPLICATION.close()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    arguments = parser.parse_args()
    run_server(arguments.port)
