"""多 agent 研究助理 API：主管调度式流式对话 (SSE)。"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from core.agents import supervisor
from models.schemas import AgentChatRequest

router = APIRouter(prefix="/agent", tags=["多 agent 助理"])


@router.post("/chat/stream", summary="主管 agent 流式对话 (SSE)")
async def agent_chat_stream(request: AgentChatRequest):
    """SSE 事件：activity（agent 活动）/ delta（最终回答增量）/ sources（聚合来源）/ done。"""

    async def event_stream():
        async for event in supervisor.run(request.query, request.history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
