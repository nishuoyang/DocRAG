"""Query Transformation + HYDE：检索前用 LLM 增强查询。

- Query Transformation：把用户问题改写成适合检索的关键词查询（补全指代、扩展表述）
- HYDE（Hypothetical Document Embeddings）：生成假想答案文档，用"答案-文档"空间的
  相似度检索（比"问题-文档"空间更准）

设计：LLM 调用失败/超时一律降级返回原 query，绝不阻塞主流程。
结果按 query 缓存（同一问题追问不重复调 LLM）。
"""
import logging
from functools import lru_cache

from core import llm

logger = logging.getLogger(__name__)

_TRANSFORM_PROMPT = """你是查询改写助手。把用户问题改写成适合文档检索的关键词查询：
- 补全指代（"它"、"这个"等），使其独立可检索
- 提取核心概念，扩展同义表述
- 保留原意，不添加原文没有的信息
- 用中文回答，只输出改写后的查询本身，不要解释、不要引号"""

_HYDE_PROMPT = """你是文档内容生成器。根据用户问题，写一段假想的文档内容（50-100字），
假设这段内容就是文档中与问题直接相关的段落。要求：
- 内容风格像知识文档的正文，不是对话
- 包含问题对应的关键概念和细节
- 只输出内容本身，不要任何前缀或解释"""


@lru_cache(maxsize=256)
def _llm_complete(prompt: str) -> str:
    """调用 LLM 完成 prompt，失败时返回空串（调用方降级）。"""
    try:
        chat = llm.get_llm()
        response = chat.invoke([{"role": "user", "content": prompt}])
        return (response.content or "").strip()
    except Exception as exc:  # 网络错误/限流/超时：降级，不抛出
        logger.warning("LLM 查询增强失败，降级使用原 query: %s", exc)
        return ""


def transform_query(query: str) -> str:
    """LLM 改写 query 为检索友好形式；失败返回原文。"""
    rewritten = _llm_complete(f"{_TRANSFORM_PROMPT}\n\n用户问题：{query}")
    return rewritten if rewritten else query


def hyde_query(query: str) -> str:
    """生成假想答案文档（HYDE）；失败返回原文。"""
    hypo = _llm_complete(f"{_HYDE_PROMPT}\n\n用户问题：{query}")
    return hypo if hypo else query
