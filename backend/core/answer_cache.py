"""Small in-process LRU cache for deterministic repeated RAG requests."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock

_MAX_ENTRIES = 128
_cache: OrderedDict[tuple, dict] = OrderedDict()
_lock = RLock()


def get(key: tuple) -> dict | None:
    with _lock:
        value = _cache.get(key)
        if value is None:
            return None
        _cache.move_to_end(key)
        return value


def put(key: tuple, value: dict) -> None:
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)


def invalidate() -> None:
    """Clear cached answers after the document collection changes."""
    with _lock:
        _cache.clear()
