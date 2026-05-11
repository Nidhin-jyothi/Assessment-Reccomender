"""
catalog.py - Load the SHL catalog JSON, build the URL allowlist,
and provide catalog lookup utilities used by the retriever and output guard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from functools import lru_cache
from typing import Optional

#  Test type legend 
TEST_TYPE_LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

KEY_TO_LETTER = {v: k for k, v in TEST_TYPE_LEGEND.items()}

# Priority ordering: when we pick the "primary" test_type for the response
TYPE_PRIORITY = ["A", "K", "S", "P", "B", "C", "E", "D"]


def keys_to_primary_letter(keys: list[str]) -> str:
    """
    Given a list of full key names (e.g. ['Knowledge & Skills', 'Personality & Behavior']),
    return the single primary letter following TYPE_PRIORITY order.
    """
    letters = [KEY_TO_LETTER.get(k, None) for k in keys]
    letters = [l for l in letters if l]
    for pref in TYPE_PRIORITY:
        if pref in letters:
            return pref
    return letters[0] if letters else "K"


def normalize_url(url: str) -> str:
    """Normalize SHL catalog URLs to canonical /products/product-catalog/ form."""
    url = url.replace(
        "/solutions/products/product-catalog/",
        "/products/product-catalog/"
    )
    url = url.rstrip("/") + "/"
    return url


def parse_duration_minutes(duration_str: str) -> Optional[int]:
    """Extract integer minutes from strings like '30 minutes' or 'Approximate...= 30'."""
    if not duration_str:
        return None
    m = re.search(r"(\d+)", duration_str)
    return int(m.group(1)) if m else None


class CatalogItem:
    """Lightweight wrapper around one catalog JSON entry."""

    __slots__ = (
        "entity_id", "name", "url", "description",
        "test_types", "primary_type", "keys",
        "job_levels", "languages",
        "duration_minutes", "remote", "adaptive",
        "corpus",
    )

    def __init__(self, raw: dict):
        self.entity_id   = str(raw.get("entity_id", ""))
        self.name        = raw.get("name", "").strip()
        self.url         = normalize_url(raw.get("link", raw.get("url", "")))
        self.description = raw.get("description", "").strip()
        self.keys        = raw.get("keys", [])
        self.primary_type = keys_to_primary_letter(self.keys)
        self.test_types  = [KEY_TO_LETTER[k] for k in self.keys if k in KEY_TO_LETTER]
        self.job_levels  = raw.get("job_levels", [])
        self.languages   = raw.get("languages", [])
        self.duration_minutes = parse_duration_minutes(raw.get("duration", ""))
        self.remote      = raw.get("remote", "no").lower() == "yes"
        self.adaptive    = raw.get("adaptive", "no").lower() == "yes"
        self.corpus      = self._build_corpus()

    def _build_corpus(self) -> str:
        """Build a rich text blob used for embedding and BM25 indexing."""
        parts = [
            self.name,
            self.description,
            "Test categories: " + ", ".join(self.keys),
            "Job levels: " + ", ".join(self.job_levels),
            "Languages: " + ", ".join(self.languages),
        ]
        if self.duration_minutes:
            parts.append(f"Duration: {self.duration_minutes} minutes")
        if self.remote:
            parts.append("Remote testing available")
        if self.adaptive:
            parts.append("Adaptive/IRT test")
        return " | ".join(p for p in parts if p.strip(" |"))

    def to_recommendation_dict(self) -> dict:
        return {
            "name":      self.name,
            "url":       self.url,
            "test_type": self.primary_type,
        }

    def __repr__(self) -> str:
        return f"CatalogItem(name={self.name!r}, type={self.primary_type})"


class Catalog:
    """
    In-memory catalog store.
    Provides:
      - url_allowlist: set of all valid canonical URLs (used by output guard)
      - lookup by name (case-insensitive)
      - lookup by entity_id
      - all items as list for indexing
    """

    def __init__(self, items: list[CatalogItem]):
        self._items      = items
        self._by_url     = {item.url: item for item in items}
        self._by_id      = {item.entity_id: item for item in items}
        self._by_name    = {item.name.lower(): item for item in items}
        self.url_allowlist: set[str] = set(self._by_url.keys())

    def __len__(self) -> int:
        return len(self._items)

    def all_items(self) -> list[CatalogItem]:
        return self._items

    def get_by_url(self, url: str) -> Optional[CatalogItem]:
        return self._by_url.get(normalize_url(url))

    def get_by_id(self, entity_id: str) -> Optional[CatalogItem]:
        return self._by_id.get(str(entity_id))

    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        return self._by_name.get(name.lower().strip())

    def is_valid_url(self, url: str) -> bool:
        return normalize_url(url) in self.url_allowlist


@lru_cache(maxsize=1)
def load_catalog(catalog_path: str = "shl_catalog.json") -> Catalog:
    """
    Load and cache the catalog JSON.
    Called once at startup; subsequent calls return the cached instance.
    """
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog file not found at {path.resolve()}. "
            "Please place shl_catalog.json in the working directory."
        )
    with path.open("r", encoding="ascii") as f:
        content = f.read()
    raw_items = json.loads(content, strict=False)

    items = [CatalogItem(r) for r in raw_items]
    catalog = Catalog(items)
    print(f"[Catalog] Loaded {len(catalog)} items from {path.resolve()}")
    return catalog