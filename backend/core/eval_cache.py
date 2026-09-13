"""Versioned cache helpers for offline RAG evaluation."""
from __future__ import annotations

import hashlib
import json

CACHE_VERSION = 2


def build_cache_fingerprint(payload: dict) -> str:
    """Build a stable fingerprint from non-secret evaluation inputs."""
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_cache_items(raw: dict | None, fingerprint: str) -> dict:
    """Return cached items only when the version and fingerprint match."""
    if not isinstance(raw, dict):
        return {}
    if raw.get("version") != CACHE_VERSION:
        return {}
    if raw.get("fingerprint") != fingerprint:
        return {}
    items = raw.get("items")
    return items if isinstance(items, dict) else {}


def wrap_cache_items(items: dict, fingerprint: str) -> dict:
    """Wrap generation results with the inputs that produced them."""
    return {
        "version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "items": items,
    }
