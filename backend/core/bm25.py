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

# 缓存：{collection名: (bm25, all_docs)}，collection 变化（上传/删除）时块数变化自动失效
_index_cache: dict[str, tuple] = {}


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def _get_index() -> tuple[BM25Okapi, list[Document]]:
    """构建（或复用缓存）BM25 索引，返回 (bm25, 全量文档列表)。"""
    name = milvus.get_collection_name()
    all_docs = milvus.get_all_documents()
    cached = _index_cache.get(name)
    if cached and len(cached[1]) == len(all_docs):
        return cached
    tokenized = [_tokenize(d.page_content) for d in all_docs]
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 索引重建完成: %d 个块", len(all_docs))
    _index_cache.clear()  # 只保留最新一个库快照
    _index_cache[name] = (bm25, all_docs)
    return bm25, all_docs


def keyword_search(query: str, k: int = 10) -> list[Document]:
    """BM25 关键词检索，返回相关性最高的 k 个块（Document 与向量库同构）。"""
    bm25, all_docs = _get_index()
    if not all_docs:
        return []
    scores = bm25.get_scores(_tokenize(query))
    # 取分数 > 0 的 top-k（BM25 对无命中词返回 0 分）
    ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
    hits = [d for s, d in ranked if s > 0][:k]
    for d, s in zip(hits, [x[0] for x in ranked[: len(hits)]]):
        d.metadata["bm25_score"] = round(float(s), 4)
    return hits
