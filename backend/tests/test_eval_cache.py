from core.eval_cache import build_cache_fingerprint, load_cache_items, wrap_cache_items


def test_eval_cache_requires_matching_fingerprint():
    fingerprint = build_cache_fingerprint({"collection": "v1", "top_k": 4})
    wrapped = wrap_cache_items({"q": {"response": "a", "contexts": ["c"]}}, fingerprint)

    assert load_cache_items(wrapped, fingerprint)["q"]["response"] == "a"
    assert load_cache_items(wrapped, "different") == {}
    assert load_cache_items({"q": {"response": "legacy"}}, fingerprint) == {}


def test_eval_cache_fingerprint_is_order_independent():
    first = build_cache_fingerprint({"top_k": 4, "collection": "v1"})
    second = build_cache_fingerprint({"collection": "v1", "top_k": 4})

    assert first == second
