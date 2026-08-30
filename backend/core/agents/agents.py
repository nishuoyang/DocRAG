"""5 个成员 agent 的工具实现。每个 agent 是 @tool 异步函数，统一输出 AgentResult dict。"""
import asyncio

import requests

from langchain_core.tools import tool

from config import get_settings
from core import llm, retrieval
from core.agents import data_exec
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


def _search_web(question: str, max_results: int = 5) -> AgentResult:
    """联网搜索（同步函数，内部按 provider 分派；失败降级返回错误说明）。"""
    settings = get_settings()
    if not settings.SEARCH_API_KEY:
        return AgentResult(content="联网搜索未配置（SEARCH_API_KEY 为空），本轮跳过。")
    try:
        if settings.SEARCH_PROVIDER == "bocha":
            resp = requests.post(
                "https://api.bochaai.com/v1/web-search",
                headers={"Authorization": f"Bearer {settings.SEARCH_API_KEY}"},
                json={"query": question, "count": max_results},
                timeout=15,
            )
            resp.raise_for_status()
            items = resp.json().get("data", {}).get("webPages", {}).get("value", [])
            results = [
                {
                    "title": item.get("name") or item.get("title") or item.get("url", ""),
                    "url": item.get("url", ""),
                    "content": item.get("snippet") or item.get("summary", ""),
                }
                for item in items
                if item.get("url")
            ]
        else:  # tavily
            resp = requests.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {settings.SEARCH_API_KEY}"},
                json={"query": question, "max_results": max_results, "search_depth": "basic"},
                timeout=15,
            )
            resp.raise_for_status()
            results = [
                {
                    "title": item.get("title") or item.get("url", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
                for item in resp.json().get("results", [])
                if item.get("url")
            ]
    except Exception as exc:
        return AgentResult(content=f"联网搜索失败：{exc}")
    if not results:
        return AgentResult(content="联网搜索未找到相关结果。")
    summary = build_search_summary(results)
    return AgentResult(content=summary, sources=results)


def build_search_summary(results: list[dict]) -> str:
    """把搜索结果压成一段带序号摘要文本（工具消息返回给 LLM）。"""
    lines = [f"共 {len(results)} 条搜索结果："]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}（{r['url']}）：{r['content'][:200]}")
    return "\n".join(lines)


@tool
async def search_agent(question: str) -> dict:
    """联网搜索实时信息（新闻、官网、竞品、外部资料）。

    适合：文档库之外的最新信息、外部站点内容。
    """
    result = await asyncio.to_thread(_search_web, question)
    return result.model_dump()


DATA_PROMPT = """你是数据分析助手。根据用户问题，编写一段 Python 脚本对数据进行分析。
要求：
1. 只能使用 pandas/numpy/matplotlib；数据在变量 DATA_PATH 指向的 CSV 文件中（用 pd.read_csv(DATA_PATH) 读取）。
2. 图表必须 plt.savefig("chart.png") 且只保存一张；汉字正常显示（matplotlib.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei']，rcParams['axes.unicode_minus']=False）。
3. 最终把结论用 print 输出，每行一个事实；需要表格时 print 出 markdown 表格。
4. 不要读取/写入 DATA_PATH 之外任何文件，不要 import 额外库。

数据预览（前 5 行）：
{preview}

用户问题：{question}
只输出 Python 代码本身，不要注释说明。"""


async def _data_analyze(question: str, data: str) -> AgentResult:
    """生成脚本 → 执行 → 出错回喂 LLM 修正（最多 2 次）→ 汇总输出。"""
    preview = "\n".join(data.splitlines()[:5])
    prompt = DATA_PROMPT.format(preview=preview, question=question)
    last_output: dict = {}
    for attempt in range(3):
        chat = llm.get_llm()
        resp = await chat.ainvoke(prompt if attempt == 0 else prompt + f"\n\n上次脚本错误：{last_output['error']}\n请修正后重新输出代码。")
        script = resp.content
        # 剥掉可能的 ```python 围栏
        script = script.split("```python")[-1].split("```")[0].strip() if "```" in script else script.strip()
        try:
            last_output = await asyncio.to_thread(data_exec.run_script, script, data)
            break
        except data_exec.DataExecError as exc:
            last_output = {"error": str(exc)}
        except Exception as exc:  # 兜底：任何未预期异常也喂回修正
            last_output = {"error": f"{type(exc).__name__}: {exc}"}
    if "error" in last_output:
        return AgentResult(content=f"数据分析失败：{last_output['error']}")
    images = last_output.get("images", [])
    content = last_output["stdout"].strip() or "分析完成（无文本输出）。"
    return AgentResult(content=f"分析结果（data）：\n{content[:2000]}", meta={"images": images} if images else {})


@tool
async def data_agent(question: str, data: str) -> dict:
    """对一段结构化数据（CSV/表格文本）做统计分析与图表。

    输入 data：CSV 或 Markdown 表格原文（可从文档检索结果中提取，或用户直接在问题里粘贴）。
    适合：数据汇总、对比、趋势、画图。
    """
    result = await _data_analyze(question, data)
    return result.model_dump()
