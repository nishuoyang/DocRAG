"""BM25 关键词检索：内存索引 + 中文分词，与向量检索组成混合检索。

用 rank_bm25 在内存构建倒排索引，配合 jieba 中文分词。
对当前中小规模语料（千级块以内）重建成本毫秒级；语料超过万级块时再考虑
换 Milvus 原生 BM25 或 ES。
"""
import logging

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from db import milvus

logger = logging.getLogger(__name__)

# 缓存：{collection名: (bm25, all_docs)}，写入/删除后由 ingestion 显式失效
_index_cache: dict[str, tuple] = {}


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def invalidate_index(collection_name: str | None = None) -> None:
    """Drop the cached index after the active collection is mutated."""
    if collection_name:
        _index_cache.pop(collection_name, None)
    else:
        _index_cache.clear()


def _index_text(doc: Document) -> str:
    """Use raw child text so the repeated section prefix does not affect BM25."""
    raw_text = doc.metadata.get("raw_text")
    return str(raw_text) if raw_text else doc.page_content


def _get_index() -> tuple[BM25Okapi | None, list[Document]]:
    """构建（或复用缓存）BM25 索引，返回 (bm25, 全量文档列表)。"""
    name = milvus.get_collection_name()
    cached = _index_cache.get(name)
    if cached:
        return cached
    all_docs = milvus.get_all_documents()
    if not all_docs:
        _index_cache[name] = (None, [])
        return None, []
    tokenized = [_tokenize(_index_text(d)) for d in all_docs]
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 索引重建完成: %d 个块", len(all_docs))
    _index_cache.clear()  # 只保留最新一个库快照
    _index_cache[name] = (bm25, all_docs)
    return bm25, all_docs


def keyword_search(query: str, k: int = 10) -> list[Document]:
    """BM25 关键词检索，返回相关性最高的 k 个块（Document 与向量库同构）。"""
    bm25, all_docs = _get_index()
    if bm25 is None or not all_docs:
        return []
    scores = bm25.get_scores(_tokenize(query))
    # 取分数 > 0 的 top-k（BM25 对无命中词返回 0 分）
    ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
    hits = [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "bm25_score": round(float(score), 4)},
        )
        for score, doc in ranked
        if score > 0
    ][:k]
    return hits
