"""检索 + RAG 生成：检索相似块 → 拼上下文 → LLM 生成带来源的回答。"""
import asyncio
import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass

from langchain_core.documents import Document
from langsmith import traceable

from config import get_settings
from core import answer_cache, bm25, llm, query_transform, rerank, tracing
from db import memory, milvus

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是垂直领域的智能问答助手。基于提供的参考资料回答用户问题。
要求：
1. 只依据参考资料回答，资料中没有的内容明确说明"资料中未找到相关答案"，不要编造。
2. 回答要准确、简洁、条理清晰。
3. 涉及数字、事实时以资料原文为准。
4. 每个事实性结论必须在句末标注对应资料编号，例如 [1]；只能引用实际使用的资料。

参考资料：
{context}
"""


# 最多携带最近 6 轮历史对话，避免上下文过长稀释检索结果
MAX_HISTORY_TURNS = 6
_EXACT_QUERY_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9_]*[-_][A-Za-z0-9_-]+|"
    r"\b[A-Z]{2,}\d+\b|\b\d{2,}\b|https?://|[\"'“”][^\"'“”]+[\"'“”])"
)
_FOLLOW_UP_RE = re.compile(
    r"(?:它|这个|那个|这些|那些|其中|上述|前面|刚才|该|继续|接着|详细|展开)"
)


@dataclass(frozen=True)
class QueryPlan:
    dense: list[str]
    sparse: list[str]


def _history_fingerprint(history: list[dict] | None) -> str:
    serialized = json.dumps(history or [], ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _answer_cache_key(
    query: str,
    top_k: int | None,
    history: list[dict] | None,
) -> tuple:
    settings = get_settings()
    return (
        "v1",
        milvus.get_collection_name(),
        query.strip().casefold(),
        top_k or settings.TOP_K,
        _history_fingerprint(history),
        settings.LLM_MODEL,
        settings.RAG_ANSWER_TEMPERATURE,
        settings.QUERY_TRANSFORM,
        settings.HYDE,
        settings.RETRIEVAL_MODE,
        settings.RERANK_ENABLED,
        settings.RERANK_MODEL,
    )


def invalidate_answer_cache() -> None:
    answer_cache.invalidate()


@traceable(
    name="Context Assembly",
    run_type="chain",
    tags=["rag", "generation"],
    process_inputs=tracing.process_document_inputs,
    process_outputs=tracing.process_context_output,
)
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


def _dedupe_queries(queries: list[str]) -> list[str]:
    return list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))


def _enhancement_policy(query: str) -> tuple[bool, bool]:
    settings = get_settings()
    mode = (settings.RETRIEVAL_MODE or "balanced").strip().lower()
    if mode == "fast":
        return False, False
    if bool(_EXACT_QUERY_RE.search(query)) and not _FOLLOW_UP_RE.search(query):
        return False, False
    if mode == "quality":
        return settings.QUERY_TRANSFORM, settings.HYDE

    # balanced: only resolve ambiguous follow-ups/short questions.
    needs_rewrite = bool(_FOLLOW_UP_RE.search(query)) or len(query.strip()) <= 12
    return settings.QUERY_TRANSFORM and needs_rewrite, False


def _candidate_k(final_k: int) -> int:
    settings = get_settings()
    mode = (settings.RETRIEVAL_MODE or "balanced").strip().lower()
    configured = max(settings.RETRIEVAL_CANDIDATE_K, final_k * 4)
    if mode == "fast":
        return min(configured, max(8, final_k * 2))
    if mode == "balanced":
        return min(configured, max(12, final_k * 3))
    return configured


@traceable(
    name="Query Planning",
    run_type="chain",
    tags=["rag", "query"],
    process_inputs=tracing.process_query_inputs,
    process_outputs=tracing.process_query_plan_outputs,
)
def _build_query_plan(query: str, history: list[dict] | None = None) -> QueryPlan:
    """Build separate dense and sparse query sets.

    HYDE is only used for dense retrieval because its synthetic text can add
    keywords that do not exist in the source corpus.
    """
    use_rewrite, use_hyde = _enhancement_policy(query)
    if not use_rewrite and not use_hyde:
        return QueryPlan(dense=[query], sparse=[query])

    # 改写与 HYDE 并行调用 LLM（各自独立），失败时内部降级为原 query
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        if use_rewrite:
            futures.append(
                (
                    "rewrite",
                    pool.submit(
                        copy_context().run,
                        query_transform.transform_query,
                        query,
                        history,
                    ),
                )
            )
        if use_hyde:
            futures.append(
                (
                    "hyde",
                    pool.submit(
                        copy_context().run,
                        query_transform.hyde_query,
                        query,
                        history,
                    ),
                )
            )
        results = {}
        for name, future in futures:
            try:
                q = future.result()
            except Exception:
                q = query
            results[name] = q
    rewritten = results.get("rewrite", query)
    hypothetical = results.get("hyde", query)
    return QueryPlan(
        dense=_dedupe_queries([query, rewritten, hypothetical]),
        sparse=_dedupe_queries([query, rewritten]),
    )


def _candidate_key(doc: Document) -> tuple:
    metadata = doc.metadata
    pk = metadata.get("pk")
    if pk is not None and pk != "":
        return ("pk", str(pk))
    parent_id = metadata.get("parent_id")
    child_index = metadata.get("child_index")
    if parent_id and child_index is not None:
        return ("child", str(metadata.get("filename", "")), str(parent_id), str(child_index))
    return ("text", hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest())


@traceable(
    name="RRF Fusion",
    run_type="chain",
    tags=["rag", "retrieval", "fusion"],
    process_inputs=tracing.process_candidate_inputs,
    process_outputs=tracing.process_document_outputs,
)
def _merge_candidates(result_lists: list[list[Document]]) -> list[Document]:
    """Fuse multiple ranked result lists with reciprocal rank fusion."""
    scores: dict[tuple, float] = {}
    docs_by_key: dict[tuple, Document] = {}
    first_seen: dict[tuple, int] = {}
    position = 0
    for docs in result_lists:
        for rank, doc in enumerate(docs, 1):
            key = _candidate_key(doc)
            if key not in docs_by_key:
                docs_by_key[key] = doc
                first_seen[key] = position
                position += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)

    ordered = sorted(
        docs_by_key,
        key=lambda key: (-scores[key], first_seen[key]),
    )
    merged = []
    for key in ordered:
        doc = docs_by_key[key]
        doc.metadata["rrf_score"] = round(scores[key], 6)
        merged.append(doc)
    return merged


@traceable(
    name="Rerank",
    run_type="chain",
    tags=["rag", "retrieval", "rerank"],
    process_inputs=tracing.process_rerank_inputs,
    process_outputs=tracing.process_document_outputs,
)
def _rerank_with_fallback(
    query: str,
    candidates: list[Document],
    top_n: int,
) -> list[Document]:
    if not candidates:
        return []
    if (get_settings().RETRIEVAL_MODE or "").strip().lower() == "fast":
        return candidates[:top_n]
    try:
        ranked = rerank.rerank(query, candidates, top_n=top_n)
    except Exception as exc:
        logger.warning("Rerank failed, falling back to RRF order: %s", exc)
        return candidates[:top_n]
    return ranked or candidates[:top_n]


@traceable(
    name="Answerability Gate",
    run_type="chain",
    tags=["rag", "retrieval", "quality"],
    process_inputs=tracing.process_document_inputs,
    process_outputs=tracing.process_document_outputs,
)
def _filter_answerable(docs: list[Document]) -> list[Document]:
    settings = get_settings()
    if not settings.ANSWERABILITY_GATE_ENABLED or not docs:
        return docs
    scores = [
        float(doc.metadata["relevance_score"])
        for doc in docs
        if isinstance(doc.metadata.get("relevance_score"), (int, float))
    ]
    if scores and max(scores) < settings.RETRIEVAL_MIN_RERANK_SCORE:
        logger.info(
            "Answerability gate rejected retrieval: max_score=%.4f threshold=%.4f",
            max(scores),
            settings.RETRIEVAL_MIN_RERANK_SCORE,
        )
        return []
    return docs


@traceable(
    name="Citation Selection",
    run_type="chain",
    tags=["rag", "generation"],
    process_inputs=tracing.process_citation_inputs,
    process_outputs=tracing.process_citation_outputs,
)
def _select_sources(answer: str, docs: list[Document]) -> list[tuple[int, Document]]:
    """Return only sources explicitly cited as [n], preserving citation order."""
    cited = []
    seen: set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", answer or ""):
        index = int(match.group(1))
        if index < 1 or index > len(docs) or index in seen:
            continue
        seen.add(index)
        cited.append((index, docs[index - 1]))
    if not cited:
        return list(enumerate(docs, 1))
    return cited


@traceable(
    name="Answer Generation",
    run_type="llm",
    tags=["rag", "generation", "llm"],
    process_inputs=tracing.process_message_inputs,
    process_outputs=tracing.process_answer_output,
)
def _generate_answer(messages: list[dict]):
    return llm.get_rag_llm().invoke(messages)


@traceable(
    name="Answer Generation",
    run_type="llm",
    tags=["rag", "generation", "llm", "stream"],
    process_inputs=tracing.process_message_inputs,
    reduce_fn=tracing.join_text_chunks,
)
async def _stream_answer(messages: list[dict]):
    async for chunk in llm.get_rag_llm().astream(messages):
        if chunk.content:
            yield chunk.content


@traceable(
    name="Dense Retrieval",
    run_type="retriever",
    tags=["rag", "retrieval", "dense"],
    process_inputs=tracing.process_search_inputs,
    process_outputs=tracing.process_document_outputs,
)
def _dense_search(query: str, k: int) -> list[Document]:
    return milvus.similarity_search(query, k=k)


@traceable(
    name="Dense Retrieval Batch",
    run_type="retriever",
    tags=["rag", "retrieval", "dense", "batch"],
    process_inputs=tracing.process_batch_search_inputs,
    process_outputs=tracing.process_document_lists_outputs,
)
def _dense_search_many(queries: list[str], k: int) -> list[list[Document]]:
    return milvus.similarity_search_many(queries, k=k)


@traceable(
    name="BM25 Retrieval",
    run_type="retriever",
    tags=["rag", "retrieval", "sparse"],
    process_inputs=tracing.process_search_inputs,
    process_outputs=tracing.process_document_outputs,
)
def _sparse_search(query: str, k: int) -> list[Document]:
    return bm25.keyword_search(query, k=k)


@traceable(
    name="RAG Retrieve",
    run_type="chain",
    tags=["rag", "retrieval"],
    process_inputs=tracing.process_retrieval_inputs,
    process_outputs=tracing.process_document_outputs,
)
def _retrieve(
    query: str,
    top_k: int | None,
    history: list[dict] | None = None,
) -> list[Document]:
    """混合检索 child，再将命中结果物化为完整 parent 上下文。"""
    settings = get_settings()
    final_k = min(top_k or settings.TOP_K, settings.RETRIEVAL_MAX_PARENTS)
    candidate_k = _candidate_k(final_k)
    with ThreadPoolExecutor(max_workers=4) as pool:
        plan_future = pool.submit(
            copy_context().run,
            _build_query_plan,
            query,
            history,
        )
        dense_future = pool.submit(
            copy_context().run,
            _dense_search,
            query,
            candidate_k,
        )
        sparse_future = pool.submit(
            copy_context().run,
            _sparse_search,
            query,
            candidate_k,
        )
        query_plan = plan_future.result()
        result_lists = [dense_future.result(), sparse_future.result()]

        enhanced_dense = [q for q in query_plan.dense if q != query]
        enhanced_sparse = [q for q in query_plan.sparse if q != query]
        dense_batch_future = (
            pool.submit(
                copy_context().run,
                _dense_search_many,
                enhanced_dense,
                candidate_k,
            )
            if enhanced_dense
            else None
        )
        sparse_futures = [
            pool.submit(copy_context().run, _sparse_search, q, candidate_k)
            for q in enhanced_sparse
        ]
        if dense_batch_future is not None:
            result_lists.extend(dense_batch_future.result())
        result_lists.extend(future.result() for future in sparse_futures)
    candidates = _merge_candidates(result_lists)
    # rerank 用原始 query（用户原意），保证最终排序贴合意图
    ranked = _rerank_with_fallback(query, candidates, candidate_k)
    ranked = _filter_answerable(ranked)
    return _materialize_parent_contexts(
        ranked,
        max_parents=final_k,
        max_chars=settings.PARENT_CONTEXT_MAX_CHARS,
    )


@traceable(
    name="Parent Expansion",
    run_type="chain",
    tags=["rag", "retrieval", "parent_child"],
    process_inputs=tracing.process_parent_inputs,
    process_outputs=tracing.process_document_outputs,
)
def _materialize_parent_contexts(
    ranked: list[Document],
    max_parents: int,
    max_chars: int,
) -> list[Document]:
    """Deduplicate child hits by parent and reconstruct complete parent text."""
    settings = get_settings()
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
            hit_child_index = hit.metadata.get("child_index")
            if (
                settings.PARENT_WINDOW_RADIUS >= 0
                and hit_child_index is not None
                and len(siblings) > settings.PARENT_WINDOW_RADIUS * 2 + 1
            ):
                hit_position = next(
                    (
                        index
                        for index, sibling in enumerate(siblings)
                        if sibling.metadata.get("child_index") == hit_child_index
                    ),
                    None,
                )
                if hit_position is not None:
                    start = max(0, hit_position - settings.PARENT_WINDOW_RADIUS)
                    end = min(len(siblings), hit_position + settings.PARENT_WINDOW_RADIUS + 1)
                    siblings = siblings[start:end]
                    metadata_window_start = siblings[0].metadata.get("child_index")
                    metadata_window_end = siblings[-1].metadata.get("child_index")
                else:
                    metadata_window_start = None
                    metadata_window_end = None
            else:
                metadata_window_start = (
                    siblings[0].metadata.get("child_index") if siblings else None
                )
                metadata_window_end = (
                    siblings[-1].metadata.get("child_index") if siblings else None
                )
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
            metadata["parent_window_start"] = metadata_window_start
            metadata["parent_window_end"] = metadata_window_end
        metadata["matched_chunk_index"] = hit.metadata.get("chunk_index")
        if total_chars and total_chars + len(content) > max_chars:
            break
        parents.append(Document(page_content=content, metadata=metadata))
        total_chars += len(content)
    return parents


def _source_payload(
    document: Document,
    parent_content: str | None = None,
    citation_index: int | None = None,
) -> dict:
    """Build one source payload shared by chat and document agents."""
    metadata = document.metadata
    return {
        "citation_index": citation_index,
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


@traceable(
    name="RAG Query",
    run_type="chain",
    tags=["rag", "chat"],
    process_inputs=tracing.process_retrieval_inputs,
    process_outputs=tracing.process_source_outputs,
)
def rag_query(query: str, top_k: int | None = None, history: list[dict] | None = None) -> dict:
    """单轮 RAG 问答：检索 → 生成 → 返回答案与引用来源。"""
    resolved_history = _resolve_history(history)
    cache_key = _answer_cache_key(query, top_k, resolved_history)
    cached = answer_cache.get(cache_key)
    if cached is not None:
        return cached
    docs = _retrieve(query, top_k, resolved_history)
    if not docs:
        return {
            "answer": "资料库中尚未检索到相关内容，请先上传文档或调整问题表述。",
            "sources": [],
        }
    messages = _build_messages(query, docs, resolved_history)
    response = _generate_answer(messages)
    sources = [
        _source_payload(doc, citation_index=index)
        for index, doc in _select_sources(response.content, docs)
    ]
    result = {"answer": response.content, "sources": sources}
    answer_cache.put(cache_key, result)
    return result


@traceable(
    name="RAG Chat Stream",
    run_type="chain",
    tags=["rag", "chat", "stream"],
    process_inputs=tracing.process_retrieval_inputs,
    reduce_fn=tracing.reduce_sse_events,
)
async def rag_stream(query: str, top_k: int | None = None, history: list[dict] | None = None):
    """流式 RAG 问答：检索后以 SSE 形式流式输出 LLM 回答。"""
    resolved_history = _resolve_history(history)
    cache_key = _answer_cache_key(query, top_k, resolved_history)
    cached = answer_cache.get(cache_key)
    if cached is not None:
        answer = cached.get("answer") or ""
        if answer:
            yield f'data: {json.dumps({"delta": answer})}\n\n'
        yield f'data: {json.dumps({"sources": cached.get("sources", [])})}\n\n'
        memory.add_message("user", query)
        memory.add_message("assistant", answer)
        yield "data: [DONE]\n\n"
        return
    # _retrieve 内部含同步 embedding/rerank API 调用，放线程池执行避免阻塞事件循环
    docs = await asyncio.to_thread(_retrieve, query, top_k, resolved_history)
    if not docs:
        yield f'data: {json.dumps({"answer": "资料库中尚未检索到相关内容，请先上传文档或调整问题表述。"})}\n\n'
        return
    messages = _build_messages(query, docs, resolved_history)
    answer_parts: list[str] = []
    async for content in _stream_answer(messages):
        answer_parts.append(content)
        yield f'data: {json.dumps({"delta": content})}\n\n'
    answer = "".join(answer_parts)
    sources = [
        _source_payload(doc, citation_index=index)
        for index, doc in _select_sources(answer, docs)
    ]
    answer_cache.put(cache_key, {"answer": answer, "sources": sources})
    yield f'data: {json.dumps({"sources": sources})}\n\n'
    yield "data: [DONE]\n\n"
    # 流结束：把本轮问答写入记忆（前端刷新可恢复）
    memory.add_message("user", query)
    memory.add_message("assistant", answer)
