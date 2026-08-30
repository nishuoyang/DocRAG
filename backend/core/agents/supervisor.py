"""主管 agent：LangGraph 决策循环（decide ↔ tools），产出 SSE 事件流。

事件协议（dict，由 API 层序列化为 SSE）：
  {"event": "activity", "agent": <name>, "message": <str>}
  {"event": "delta", "text": <str>}          # 最终回答（打字机逐段）
  {"event": "sources", "sources": [...]}     # 聚合全部 agent 的来源
  {"event": "done"}
"""
import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import get_settings
from core import llm, retrieval
from core.agents.agents import data_agent, documents_agent, multi_hop_agent, search_agent, writer_agent
from db import memory

SUPERVISOR_PROMPT = """你是「研究助理」主管，负责调度成员 agent 完成用户的复合任务，最后自己给出总结回答。

成员 agent：
- documents_agent(question)：在本地文档库检索并回答，适合基于已上传文档的事实性问题。
- search_agent(question)：联网搜索实时信息（新闻/官网/外部资料），给出带 URL 的摘要。
- data_agent(question, data)：对结构化数据（CSV/表格文本）做统计分析与图表；data 要传数据原文，可取自文档检索结果或用户粘贴内容。
- writer_agent(materials, style)：把材料整理成稿（style=report/compare/weekly），适合最后成文。
- multi_hop_agent(question)：把复杂问题拆成子问题逐项查证并汇总，适合交叉引用型问题。

决策准则：
1. 一次提问尽量只调最合适的 1-2 个 agent；复杂任务先拆解再逐个调用。
2. 文档库优先；库外信息用 search_agent；需要成文时最后调 writer_agent。
3. 收到 agent 结果后判断是否足够回答；不足就补调其他 agent，够了就停止调用并直接输出最终总结回答。
4. 最终回答用中文，引用来源时标注（文档名 / 网址）。
5. 所有 agent 都失败时，基于已有信息直接回答并说明局限。"""

TOOLS = [documents_agent, search_agent, data_agent, writer_agent, multi_hop_agent]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int
    queue: asyncio.Queue
    sources: list[dict]
    final: str


async def _emit(state: AgentState, event: str, **payload) -> None:
    await state["queue"].put({"event": event, **payload})


async def decide(state: AgentState) -> dict:
    settings = get_settings()
    turns = state["turns"] + 1
    updated: dict = {"turns": turns}
    chat = llm.get_llm().bind_tools(TOOLS)
    resp: AIMessage = await chat.ainvoke(
        state["messages"] + [HumanMessage(f"[决策轮次 {turns}/{settings.AGENT_MAX_TURNS}]")]
    )
    updated["messages"] = [resp]
    if not resp.tool_calls:
        updated["final"] = resp.content or "（无直接回答，请补充问题）"
    return updated


async def run_tools(state: AgentState) -> dict:
    last: AIMessage = state["messages"][-1]
    tool_msgs: list[ToolMessage] = []
    for call in last.tool_calls:
        name = call["name"]
        await _emit(state, "activity", agent=name, message="执行中…")
        try:
            out = await TOOLS_BY_NAME[name].ainvoke(call["args"])
            if isinstance(out, dict):
                for s in out.get("sources", []):
                    if s not in state["sources"] and not any(x.get("title") == s.get("title") and x.get("url") == s.get("url") for x in state["sources"]):
                        state["sources"].append(s)
            text = out if isinstance(out, str) else str(out)
            await _emit(state, "activity", agent=name, message="完成")
        except Exception as exc:
            text = f"agent 执行失败：{type(exc).__name__}: {exc}"
            await _emit(state, "activity", agent=name, message=f"失败（{type(exc).__name__}）")
        tool_msgs.append(ToolMessage(content=str(text), tool_call_id=call["id"]))
    return {"messages": tool_msgs}


async def force_final(state: AgentState) -> dict:
    """超轮次：不带工具强行收尾。"""
    chat = llm.get_llm()
    resp = await chat.ainvoke(
        state["messages"]
        + [SystemMessage("已达到最大决策轮数。请立即停止调用成员 agent，基于已有结果给出最终总结回答。")]
    )
    return {"messages": [resp], "final": resp.content}


def route(state: AgentState) -> str:
    settings = get_settings()
    if state["turns"] >= settings.AGENT_MAX_TURNS:
        return "force_final"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_supervisor_messages(query: str, history: list[dict] | None) -> list:
    messages: list = [SystemMessage(content=SUPERVISOR_PROMPT)]
    for turn in retrieval._resolve_history(history):
        messages.append(HumanMessage(content=turn["content"]) if turn["role"] == "user" else AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=query))
    return messages


def _make_graph():
    g = StateGraph(AgentState)
    g.add_node("decide", decide)
    g.add_node("tools", run_tools)
    g.add_node("force_final", force_final)
    g.add_edge(START, "decide")
    g.add_conditional_edges("decide", route, {"tools": "tools", "force_final": "force_final", END: END})
    g.add_edge("tools", "decide")
    g.add_edge("force_final", END)
    return g.compile()


GRAPH = _make_graph()


async def run(query: str, history: list[dict] | None = None):
    """主管执行入口：async generator of SSE 事件 dict。"""
    queue: asyncio.Queue = asyncio.Queue()
    state: AgentState = {
        "messages": build_supervisor_messages(query, history),
        "turns": 0,
        "queue": queue,
        "sources": [],
        "final": "",
    }
    await _emit(state, "activity", agent="supervisor", message="正在分析问题…")
    task = asyncio.create_task(GRAPH.ainvoke(state))
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if task.done():
                break
            continue
        yield event
    final_state = await task
    answer = final_state.get("final", "")
    # 打字机逐段输出最终回答（前端真实流式的等价实现，避免图内双重 LLM 调用）
    for i in range(0, len(answer), 8):
        yield {"event": "delta", "text": answer[i : i + 8]}
        await asyncio.sleep(0.01)
    sources = final_state.get("sources", [])
    if sources:
        yield {"event": "sources", "sources": sources}
    yield {"event": "done"}
    # 写入对话记忆（与 rag_stream 惯例一致）
    memory.add_message("user", query)
    memory.add_message("assistant", answer)
