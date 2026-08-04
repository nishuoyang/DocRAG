"""对话 API：单轮问答 + 流式问答 (SSE) + 记忆持久化。"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core import retrieval
from db import memory
from models.schemas import ChatRequest, ChatResponse, MemoryResponse

router = APIRouter(prefix="/chat", tags=["对话"])


@router.get("/memory", response_model=MemoryResponse, summary="获取历史对话记忆")
async def get_memory():
    """返回全部历史消息（按时间升序），前端刷新后据此恢复对话。"""
    return MemoryResponse(messages=memory.load_messages())


@router.post("", response_model=ChatResponse, summary="单轮 RAG 问答")
async def chat(request: ChatRequest):
    """基于已入库文档回答用户问题，返回答案和引用来源；问答自动写入记忆。"""
    result = retrieval.rag_query(request.query, top_k=request.top_k, history=request.history)
    # 持久化记忆（仅当前 query 与 answer；错误/空库回答不存）
    if result.get("answer"):
        memory.add_message("user", request.query)
        memory.add_message("assistant", result["answer"])
    return ChatResponse(**result)


@router.post("/stream", summary="流式 RAG 问答 (SSE)")
async def chat_stream(request: ChatRequest):
    """以 SSE 流式返回 LLM 回答，末尾附加 sources 事件；问答由 rag_stream 内部写入记忆。"""

    async def event_stream():
        async for event in retrieval.rag_stream(request.query, top_k=request.top_k, history=request.history):
            yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")
