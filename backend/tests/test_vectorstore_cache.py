from db import milvus


def test_vectorstore_constructor_is_reused_for_same_collection(monkeypatch):
    settings = milvus.get_settings()
    monkeypatch.setattr(settings, "MILVUS_HOST", "localhost")
    monkeypatch.setattr(settings, "MILVUS_PORT", 19530)
    monkeypatch.setattr(settings, "MILVUS_COLLECTION", "cache_test")
    monkeypatch.setattr(settings, "COLLECTION_SCHEMA_VERSION", "v1")
    calls = []
    fake_vectorstore = object()

    def fake_build(**kwargs):
        calls.append(kwargs)
        return fake_vectorstore

    milvus.clear_vectorstore_cache()
    monkeypatch.setattr(milvus, "_build_vectorstore", fake_build)

    assert milvus.get_vectorstore() is fake_vectorstore
    assert milvus.get_vectorstore() is fake_vectorstore
    assert len(calls) == 1

    milvus.clear_vectorstore_cache()
