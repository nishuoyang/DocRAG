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
    monkeypatch.setattr(
        bm25.milvus,
        "get_all_documents",
        lambda: (_ for _ in ()).throw(AssertionError("cache hit should not rescan Milvus")),
    )
    assert bm25.keyword_search("Milvus", k=10)


def test_keyword_search_does_not_mutate_cached_documents(monkeypatch):
    docs = [
        Document(page_content="Milvus 检索", metadata={"pk": 1, "upload_time": 1}),
        Document(page_content="普通文本", metadata={"pk": 2, "upload_time": 1}),
        Document(page_content="其他文本", metadata={"pk": 3, "upload_time": 1}),
    ]
    monkeypatch.setattr(bm25.milvus, "get_all_documents", lambda: docs)
    bm25._index_cache.clear()

    hits = bm25.keyword_search("Milvus", k=10)

    assert hits[0].metadata["bm25_score"] > 0
    assert "bm25_score" not in docs[0].metadata


def test_keyword_search_indexes_raw_text_and_rebuilds_after_invalidation(monkeypatch):
    docs = [
        Document(
            page_content="[章节: 无关章节]\n需要检索的正文",
            metadata={"pk": 1, "raw_text": "唯一关键词", "upload_time": 1},
        ),
        Document(
            page_content="干扰正文一",
            metadata={"pk": 2, "raw_text": "干扰正文一", "upload_time": 1},
        ),
        Document(
            page_content="干扰正文二",
            metadata={"pk": 3, "raw_text": "干扰正文二", "upload_time": 1},
        ),
    ]
    monkeypatch.setattr(bm25.milvus, "get_all_documents", lambda: docs)
    bm25._index_cache.clear()

    assert bm25.keyword_search("唯一关键词", k=10)

    docs.append(
        Document(
            page_content="新增正文",
            metadata={"pk": 4, "raw_text": "新增关键词", "upload_time": 2},
        )
    )
    bm25.invalidate_index()

    assert bm25.keyword_search("新增关键词", k=10)
