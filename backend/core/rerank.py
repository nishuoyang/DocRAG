"""Rerank 重排序：用交叉编码器对检索结果做相关性重排，提升 Top-K 质量。

向量检索（bi-encoder）先召回候选，这里用 rerank 模型（cross-encoder）对
query 与每个候选块做联合编码，输出相关性分数，按分数重排后返回 Top-N。
"""
from functools import lru_cache

import httpx
from langchain_core.documents import Document

from config import get_settings


@lru_cache(maxsize=1)
def get_http_client() -> httpx.Client:
    """Reuse one keep-alive connection pool for rerank requests."""
    return httpx.Client(timeout=30)


def rerank(query: str, docs: list[Document], top_n: int | None = None) -> list[Document]:
    """对检索结果按相关性重排，返回分数最高的 top_n 条（默认全部）。"""
    if not docs:
        return []
    settings = get_settings()
    if not settings.RERANK_ENABLED or not settings.RERANK_API_KEY:
        return docs  # 开关关闭或未配置 key 时降级为不重排

    n = top_n or len(docs)
    resp = get_http_client().post(
        f"{settings.EMBEDDING_BASE_URL}/rerank",
        headers={"Authorization": f"Bearer {settings.RERANK_API_KEY}"},
        json={
            "model": settings.RERANK_MODEL,
            "query": query,
            "documents": [d.page_content for d in docs],
            "top_n": n,
        },
    )
    resp.raise_for_status()
    # results 按相关性降序，index 指向原列表
    results = resp.json().get("results", [])
    ordered = []
    for item in sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(docs):
            source = docs[idx]
            doc = Document(
                page_content=source.page_content,
                metadata={
                    **source.metadata,
                    "relevance_score": item.get("relevance_score"),
                },
            )
            ordered.append(doc)
    return ordered
