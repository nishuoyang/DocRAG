"""对话 API：单轮问答 + 流式问答 (SSE)。"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core import retrieval
from models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["对话"])


@router.post("", response_model=ChatResponse, summary="单轮 RAG 问答")
async def chat(request: ChatRequest):
    """基于已入库文档回答用户问题，返回答案和引用来源。"""
    history = [m.model_dump() for m in request.history]
    result = retrieval.rag_query(request.query, top_k=request.top_k, history=history)
    return ChatResponse(**result)


@router.post("/stream", summary="流式 RAG 问答 (SSE)")
async def chat_stream(request: ChatRequest):
    """以 SSE 流式返回 LLM 回答，末尾附加 sources 事件。"""
    history = [m.model_dump() for m in request.history]

    async def event_stream():
        async for event in retrieval.rag_stream(request.query, top_k=request.top_k, history=history):
            yield event

    return StreamingResponse(event_stream(), media_type="text/event-stream")
