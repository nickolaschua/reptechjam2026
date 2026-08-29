from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from memory.embeddings import normalize_rows


PRODUCTS = (
    {
        "parent_asin": "a-blue-shoe",
        "title": "Blue running shoes",
        "features": ["lightweight running"],
        "details": {"color": "blue"},
        "description": "shoes for road running",
        "categories": ["shoes"],
        "store": "Sprint",
        "price": 70.0,
    },
    {
        "parent_asin": "z-red-shoe",
        "title": "Red hiking shoes",
        "features": ["rugged hiking"],
        "details": {"color": "red"},
        "description": "shoes for trail hiking",
        "categories": ["shoes"],
        "store": "Trek",
        "price": 50.0,
    },
    {
        "parent_asin": "b-black-dress",
        "title": "Black formal dress",
        "features": ["formal style"],
        "details": {"color": "black"},
        "description": "dress for work",
        "categories": ["dresses"],
        "store": "Chic",
        "price": 90.0,
    },
    {
        "parent_asin": "c-green-shoe",
        "title": "Green hiking shoes",
        "features": ["outdoor hiking"],
        "details": {"color": "green"},
        "description": "affordable trail shoes",
        "categories": ["shoes"],
        "store": "Acme",
        "price": 40.0,
    },
)


class KeywordEmbeddingProvider:
    dimension = 4
    space_id = "synthetic-keywords-v1"

    WORDS = {
        "red": (2.0, 0.0, 0.0, 0.0),
        "hiking": (1.0, 0.0, 0.0, 0.0),
        "blue": (0.0, 2.0, 0.0, 0.0),
        "running": (0.0, 1.0, 0.0, 0.0),
        "dress": (0.0, 0.0, 2.0, 0.0),
        "dresses": (0.0, 0.0, 2.0, 0.0),
        "black": (0.0, 0.0, 1.0, 0.0),
        "shoes": (0.5, 0.5, 0.0, 0.0),
        "budget": (0.0, 0.0, 0.0, 1.0),
        "green": (0.5, 0.0, 0.0, 0.0),
    }

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        values = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = str(text).lower()
            for word, vector in self.WORDS.items():
                if word in lowered:
                    values[row] += np.asarray(vector, dtype=np.float32)
        return normalize_rows(values)


class FailingEmbeddingProvider:
    dimension = 4
    space_id = "unavailable-model"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise RuntimeError("model unavailable")


def catalog_embeddings(provider: KeywordEmbeddingProvider | None = None) -> np.ndarray:
    encoder = provider or KeywordEmbeddingProvider()
    texts = [
        " ".join((
            str(product["title"]),
            str(product["features"]),
            str(product["details"]),
            str(product["description"]),
            str(product["categories"]),
            str(product["store"]),
        ))
        for product in PRODUCTS
    ]
    return encoder.embed(texts)


class TemporaryCatalog:
    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "catalog.jsonl"
        with self.path.open("w", encoding="utf-8") as handle:
            for product in PRODUCTS:
                handle.write(json.dumps(product) + "\n")

    def close(self) -> None:
        self.directory.cleanup()
