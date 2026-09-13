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
        location = []
        section = doc.metadata.get("section")
        if section:
            location.append(f"章节: {section}")
        page_start = doc.metadata.get("page_start", doc.metadata.get("page"))
        page_end = doc.metadata.get("page_end", page_start)
        if page_start:
            page_label = f"第{page_start}页"
            if page_end and page_end != page_start:
                page_label = f"第{page_start}-{page_end}页"
            location.append(page_label)
        suffix = f" | {' | '.join(location)}" if location else ""
        parts.append(f"[{i}] (来源: {source}{suffix})\n{doc.page_content}")
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
    """混合检索 child，再将命中结果物化为完整 parent 上下文。"""
    settings = get_settings()
    final_k = min(top_k or settings.TOP_K, settings.RETRIEVAL_MAX_PARENTS)
    candidate_k = max(settings.RETRIEVAL_CANDIDATE_K, final_k * 4)
    candidates: list[Document] = []
    seen: set[str] = set()
    for q in _build_search_queries(query):
        for doc in milvus.similarity_search(q, k=candidate_k) + bm25.keyword_search(q, k=candidate_k):
            if doc.page_content not in seen:
                candidates.append(doc)
                seen.add(doc.page_content)
    # rerank 用原始 query（用户原意），保证最终排序贴合意图
    ranked = rerank.rerank(query, candidates, top_n=candidate_k)
    return _materialize_parent_contexts(
        ranked or candidates,
        max_parents=final_k,
        max_chars=settings.PARENT_CONTEXT_MAX_CHARS,
    )


def _materialize_parent_contexts(
    ranked: list[Document],
    max_parents: int,
    max_chars: int,
) -> list[Document]:
    """Deduplicate child hits by parent and reconstruct complete parent text."""
    selected: list[tuple[Document, str | None]] = []
    seen_parents: set[str] = set()
    for child in ranked:
        parent_id = child.metadata.get("parent_id")
        if parent_id:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
        selected.append((child, parent_id))
        if len(selected) >= max_parents:
            break

    parent_ids = [parent_id for _, parent_id in selected if parent_id]
    try:
        sibling_map = milvus.get_children_by_parent_ids(parent_ids)
    except Exception:
        sibling_map = {}

    parents: list[Document] = []
    total_chars = 0
    for hit, parent_id in selected:
        siblings = sibling_map.get(parent_id or "", [])
        if not siblings:
            content = hit.page_content
            metadata = dict(hit.metadata)
        else:
            siblings = sorted(siblings, key=lambda doc: doc.metadata.get("child_index") or 0)
            content = "\n\n".join(
                str(sibling.metadata.get("raw_text") or sibling.page_content).strip()
                for sibling in siblings
                if str(sibling.metadata.get("raw_text") or sibling.page_content).strip()
            )
            page_starts = [
                sibling.metadata.get("page_start")
                for sibling in siblings
                if sibling.metadata.get("page_start") is not None
            ]
            page_ends = [
                sibling.metadata.get("page_end")
                for sibling in siblings
                if sibling.metadata.get("page_end") is not None
            ]
            metadata = dict(hit.metadata)
            if page_starts:
                metadata["page_start"] = min(page_starts)
                metadata["page"] = metadata["page_start"]
            if page_ends:
                metadata["page_end"] = max(page_ends)
            metadata["section"] = siblings[0].metadata.get("section") or metadata.get("section")
            metadata["child_count"] = len(siblings)
        metadata["matched_chunk_index"] = hit.metadata.get("chunk_index")
        if total_chars and total_chars + len(content) > max_chars:
            break
        parents.append(Document(page_content=content, metadata=metadata))
        total_chars += len(content)
    return parents


def _source_payload(document: Document, parent_content: str | None = None) -> dict:
    """Build one source payload shared by chat and document agents."""
    metadata = document.metadata
    return {
        "filename": metadata.get("filename", "未知来源"),
        "chunk_index": metadata.get("chunk_index"),
        "matched_chunk_index": metadata.get("matched_chunk_index", metadata.get("chunk_index")),
        "page": metadata.get("page"),
        "page_start": metadata.get("page_start", metadata.get("page")),
        "page_end": metadata.get("page_end", metadata.get("page")),
        "section": metadata.get("section"),
        "content": parent_content if parent_content is not None else document.page_content,
    }


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
    sources = [_source_payload(d) for d in docs]
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
    sources = [_source_payload(d) for d in docs]
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
