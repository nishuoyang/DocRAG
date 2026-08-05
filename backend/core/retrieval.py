"""检索 + RAG 生成：检索相似块 → 拼上下文 → LLM 生成带来源的回答。"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from config import get_settings
from core import bm25, llm, query_transform, rerank
from db import memory, milvus

SYSTEM_PROMPT = """你是垂直领域的智能问答助手。基于提供的参考资料回答用户问题。
要求：
1. 只依据参考资料回答，资料中没有的内容明确说明"资料中未找到相关答案"，不要编造。
2. 回答要准确、简洁、条理清晰。
3. 涉及数字、事实时以资料原文为准。

参考资料：
{context}
"""


# 最多携带最近 6 轮历史对话，避免上下文过长稀释检索结果
MAX_HISTORY_TURNS = 6


def _format_context(docs: list[Document]) -> str:
    """把检索结果拼成上下文文本。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "未知来源")
        parts.append(f"[{i}] (来源: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_messages(query: str, docs: list[Document], history: list[dict] | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=_format_context(docs))},
    ]
    for turn in history or []:
        messages.append({"role": turn.get("role"), "content": turn.get("content")})
    messages.append({"role": "user", "content": query})
    return messages


def _build_search_queries(query: str) -> list[str]:
    """构造检索查询集：原 query + Query Transformation 改写 + HYDE 假想答案（按开关）。"""
    settings = get_settings()
    queries = [query]
    # 改写与 HYDE 并行调用 LLM（各自独立），失败时内部降级为原 query
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        if settings.QUERY_TRANSFORM:
            futures.append(pool.submit(query_transform.transform_query, query))
        if settings.HYDE:
            futures.append(pool.submit(query_transform.hyde_query, query))
        for f in futures:
            try:
                q = f.result()
            except Exception:
                q = query
            if q and q != query:
                queries.append(q)
    return queries


def _retrieve(query: str, top_k: int | None) -> list[Document]:
    """混合检索：Query 增强（改写+HYDE）→ 向量+BM25 多查询召回 → 合并去重 → rerank。"""
    settings = get_settings()
    n = max(top_k or settings.TOP_K, 10)
    candidates: list[Document] = []
    seen: set[str] = set()
    for q in _build_search_queries(query):
        for doc in milvus.similarity_search(q, k=n) + bm25.keyword_search(q, k=n):
            if doc.page_content not in seen:
                candidates.append(doc)
                seen.add(doc.page_content)
    # rerank 用原始 query（用户原意），保证最终排序贴合意图
    ranked = rerank.rerank(query, candidates, top_n=top_k or settings.TOP_K)
    return ranked or candidates[: top_k or settings.TOP_K]


def _resolve_history(history: list[dict] | None) -> list[dict]:
    """确定对话上下文：优先用前端传入，否则从 SQLite 记忆读。"""
    if history is not None:
        return history[-MAX_HISTORY_TURNS * 2 :]
    return memory.load_messages()[-MAX_HISTORY_TURNS * 2 :]


def rag_query(query: str, top_k: int | None = None, history: list[dict] | None = None) -> dict:
    """单轮 RAG 问答：检索 → 生成 → 返回答案与引用来源。"""
    docs = _retrieve(query, top_k)
    if not docs:
        return {
            "answer": "资料库中尚未检索到相关内容，请先上传文档或调整问题表述。",
            "sources": [],
        }
    chat = llm.get_llm()
    messages = _build_messages(query, docs, _resolve_history(history))
    response = chat.invoke(messages)
    sources = [
        {
            "filename": d.metadata.get("filename", "未知来源"),
            "chunk_index": d.metadata.get("chunk_index"),
            "page": d.metadata.get("page"),
            "content": d.page_content,
        }
        for d in docs
    ]
    return {"answer": response.content, "sources": sources}


async def rag_stream(query: str, top_k: int | None = None, history: list[dict] | None = None):
    """流式 RAG 问答：检索后以 SSE 形式流式输出 LLM 回答。"""
    # _retrieve 内部含同步 embedding/rerank API 调用，放线程池执行避免阻塞事件循环
    docs = await asyncio.to_thread(_retrieve, query, top_k)
    if not docs:
        yield f'data: {json.dumps({"answer": "资料库中尚未检索到相关内容，请先上传文档或调整问题表述。"})}\n\n'
        return
    chat = llm.get_llm()
    messages = _build_messages(query, docs, _resolve_history(history))
    sources = [
        {
            "filename": d.metadata.get("filename", "未知来源"),
            "chunk_index": d.metadata.get("chunk_index"),
            "page": d.metadata.get("page"),
            "content": d.page_content,
        }
        for d in docs
    ]
    answer_parts: list[str] = []
    async for chunk in chat.astream(messages):
        if chunk.content:
            answer_parts.append(chunk.content)
            yield f'data: {json.dumps({"delta": chunk.content})}\n\n'
    yield f'data: {json.dumps({"sources": sources})}\n\n'
    yield "data: [DONE]\n\n"
    # 流结束：把本轮问答写入记忆（前端刷新可恢复）
    memory.add_message("user", query)
    memory.add_message("assistant", "".join(answer_parts))
