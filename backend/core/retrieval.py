"""检索 + RAG 生成：检索相似块 → 拼上下文 → LLM 生成带来源的回答。"""
import json

from langchain_core.documents import Document

from core import llm
from db import milvus

SYSTEM_PROMPT = """你是垂直领域的智能问答助手。基于提供的参考资料回答用户问题。
要求：
1. 只依据参考资料回答，资料中没有的内容明确说明"资料中未找到相关答案"，不要编造。
2. 回答要准确、简洁、条理清晰。
3. 涉及数字、事实时以资料原文为准。

参考资料：
{context}
"""


def _format_context(docs: list[Document]) -> str:
    """把检索结果拼成上下文文本。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "未知来源")
        parts.append(f"[{i}] (来源: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_messages(query: str, docs: list[Document]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=_format_context(docs))},
        {"role": "user", "content": query},
    ]


def rag_query(query: str, top_k: int | None = None) -> dict:
    """单轮 RAG 问答：检索 → 生成 → 返回答案与引用来源。"""
    docs = milvus.similarity_search(query, k=top_k)
    if not docs:
        return {
            "answer": "资料库中尚未检索到相关内容，请先上传文档或调整问题表述。",
            "sources": [],
        }
    chat = llm.get_llm()
    messages = _build_messages(query, docs)
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


async def rag_stream(query: str, top_k: int | None = None):
    """流式 RAG 问答：检索后以 SSE 形式流式输出 LLM 回答。"""
    docs = milvus.similarity_search(query, k=top_k)
    if not docs:
        yield f'data: {json.dumps({"answer": "资料库中尚未检索到相关内容，请先上传文档或调整问题表述。"})}\n\n'
        return
    chat = llm.get_llm()
    messages = _build_messages(query, docs)
    sources = [
        {
            "filename": d.metadata.get("filename", "未知来源"),
            "chunk_index": d.metadata.get("chunk_index"),
            "page": d.metadata.get("page"),
            "content": d.page_content,
        }
        for d in docs
    ]
    async for chunk in chat.astream(messages):
        if chunk.content:
            yield f'data: {json.dumps({"delta": chunk.content})}\n\n'
    yield f'data: {json.dumps({"sources": sources})}\n\n'
    yield "data: [DONE]\n\n"
