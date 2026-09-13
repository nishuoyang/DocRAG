import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from core.agents import agents, supervisor


class _FakeTool:
    def __init__(self, barrier):
        self.barrier = barrier
        self.started = 0

    async def ainvoke(self, args):
        self.started += 1
        if self.started == 1:
            await asyncio.wait_for(self.barrier.wait(), timeout=0.1)
        else:
            self.barrier.set()
        return {"content": args["value"], "sources": []}


def test_supervisor_runs_independent_tool_calls_concurrently(monkeypatch):
    barrier = asyncio.Event()
    tool = _FakeTool(barrier)
    monkeypatch.setattr(supervisor, "TOOLS_BY_NAME", {"a": tool, "b": tool})
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "a", "args": {"value": "a"}, "id": "call-a", "type": "tool_call"},
                    {"name": "b", "args": {"value": "b"}, "id": "call-b", "type": "tool_call"},
                ],
            )
        ],
        "turns": 1,
        "queue": asyncio.Queue(),
        "sources": [],
        "final": "",
    }

    result = asyncio.run(supervisor.run_tools(state))

    assert len(result["messages"]) == 2
    assert all("agent 执行失败" not in message.content for message in result["messages"])


def test_documents_agent_uses_request_top_k(monkeypatch):
    captured = {}

    def fake_retrieve(question, top_k, history=None):
        captured["top_k"] = top_k
        return [Document(page_content="context", metadata={"filename": "a.md"})]

    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content="answer")

    monkeypatch.setattr(agents.retrieval, "_retrieve", fake_retrieve)
    monkeypatch.setattr(agents.retrieval.llm, "get_rag_llm", lambda: FakeLLM())
    token = agents.set_agent_top_k(3)
    try:
        asyncio.run(agents._documents_rag("question"))
    finally:
        agents.reset_agent_top_k(token)

    assert captured["top_k"] == 3


def test_final_answer_streams_tokens_into_supervisor_queue(monkeypatch):
    class FakeStreamingLLM:
        async def astream(self, messages):
            for text in ("最终", "回答"):
                yield SimpleNamespace(content=text)

    monkeypatch.setattr(supervisor.llm, "get_llm", lambda: FakeStreamingLLM())
    queue = asyncio.Queue()
    state = {
        "messages": [],
        "turns": 1,
        "queue": queue,
        "sources": [],
        "final": "",
        "force_final": False,
    }

    result = asyncio.run(supervisor.final_answer(state))

    assert result["final"] == "最终回答"
    assert [queue.get_nowait()["text"] for _ in range(queue.qsize())] == ["最终", "回答"]


def test_supervisor_tool_timeout_is_reported_without_hanging(monkeypatch):
    class SlowTool:
        async def ainvoke(self, args):
            await asyncio.sleep(0.2)
            return {"content": "late", "sources": []}

    monkeypatch.setattr(supervisor, "TOOLS_BY_NAME", {"slow": SlowTool()})
    monkeypatch.setattr(supervisor.get_settings(), "AGENT_TOOL_TIMEOUT", 0.01)
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "slow", "args": {}, "id": "call-slow", "type": "tool_call"},
                ],
            )
        ],
        "turns": 1,
        "queue": asyncio.Queue(),
        "sources": [],
        "final": "",
        "force_final": False,
    }

    result = asyncio.run(supervisor.run_tools(state))

    assert "agent 执行失败" in result["messages"][0].content
    assert "TimeoutError" in result["messages"][0].content
