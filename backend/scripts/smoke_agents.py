"""逐 agent 冒烟：验证 5 个成员 agent + supervisor 均可执行、失败时不崩溃。

用法：cd backend && ./.venv/Scripts/python.exe -X utf8 scripts/smoke_agents.py [--agent documents|search|data|writer|multi_hop|supervisor]
"""
import argparse
import asyncio
import sys
from pathlib import Path

# 把 backend/ 加入 sys.path，复用项目代码与 .env 配置
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from core.agents.agents import data_agent, documents_agent, multi_hop_agent, search_agent, writer_agent  # noqa: E402
from core.agents.supervisor import run  # noqa: E402

CSV_DATA = "月份,销售额\n1月,120\n2月,150\n3月,100\n4月,180"


async def smoke(name: str) -> None:
    print(f"== {name} ==")
    if name == "documents":
        r = await documents_agent.ainvoke({"question": "系统支持哪些文档格式？"})
    elif name == "search":
        r = await search_agent.ainvoke({"question": "DeepSeek 最新版本"})
    elif name == "data":
        r = await data_agent.ainvoke({"question": "各月销售额平均值", "data": CSV_DATA})
    elif name == "writer":
        r = await writer_agent.ainvoke({"materials": "京：5年社保+首付30%；沪：5年社保+首付35%", "style": "compare"})
    elif name == "multi_hop":
        r = await multi_hop_agent.ainvoke({"question": "系统支持的文档格式有哪些？"})
    elif name == "supervisor":
        events = [e async for e in run("系统支持哪些文档格式？")]
        assert events[0]["event"] == "activity" and events[-1]["event"] == "done", "事件序列异常"
        r = {"content": "".join(e["text"] for e in events if e["event"] == "delta")}
    else:
        raise SystemExit(f"未知 agent: {name}")
    print("OK, content 前 200 字：", (r.get("content") or "")[:200].replace("\n", " "))
    print("sources 数：", len(r.get("sources") or []))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="supervisor", choices=["documents", "search", "data", "writer", "multi_hop", "supervisor"])
    args = parser.parse_args()
    await smoke(args.agent)


if __name__ == "__main__":
    asyncio.run(main())
