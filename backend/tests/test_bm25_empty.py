from core import bm25
from langchain_core.documents import Document


def test_keyword_search_returns_empty_for_empty_collection(monkeypatch):
    monkeypatch.setattr(bm25.milvus, "get_all_documents", lambda: [])
    bm25._index_cache.clear()

    assert bm25.keyword_search("任意问题", k=10) == []


def test_keyword_search_reuses_cached_index(monkeypatch):
    docs = [
        Document(
            page_content="Milvus 向量检索",
            metadata={"pk": 1, "upload_time": 1},
        ),
        Document(page_content="BM25 关键词检索", metadata={"pk": 2, "upload_time": 1}),
        Document(page_content="父子块上下文", metadata={"pk": 3, "upload_time": 1}),
    ]
    monkeypatch.setattr(bm25.milvus, "get_all_documents", lambda: docs)
    bm25._index_cache.clear()

    assert bm25.keyword_search("Milvus", k=10)
    assert bm25.keyword_search("Milvus", k=10)
