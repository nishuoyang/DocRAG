"""5 个成员 agent 的工具实现。每个 agent 是 @tool 异步函数，统一输出 AgentResult dict。"""
import asyncio

from langchain_core.tools import tool

from config import get_settings
from core import llm, retrieval
from core.agents.schemas import AgentResult


async def _documents_rag(question: str) -> AgentResult:
    """文档库 RAG 单轮：检索 → 生成。复用现有 retrieval 链路，sources 与 /chat 一致。"""
    settings = get_settings()
    docs = await asyncio.to_thread(retrieval._retrieve, question, settings.TOP_K)
    if not docs:
        return AgentResult(content="资料库中尚未检索到相关内容，请先上传文档或调整问题表述。")
    chat = llm.get_llm()
    messages = retrieval._build_messages(question, docs, retrieval._resolve_history(None))
    response = await chat.ainvoke(messages)
    sources = [
        {
            "title": d.metadata.get("filename", "未知来源"),
            "file": d.metadata.get("filename", "未知来源"),
            "page": d.metadata.get("page"),
            "content": d.page_content,
        }
        for d in docs
    ]
    return AgentResult(content=response.content, sources=sources)


@tool
async def documents_agent(
    question: str,
    context: str = "",
) -> dict:
    """在本地文档库中检索与 question 相关的资料并给出基于资料的回答。

    适合：基于已上传文档的事实性问题（政策、指标、条款、原文出处）。
    注意：只依据文档库内容回答，库中没有的要明确说明。
    context 是本轮已有的其他 agent 结果摘要，仅作背景参考。
    """
    result = await _documents_rag(question)
    return result.model_dump()
