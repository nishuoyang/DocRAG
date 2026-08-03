"""LLM 工厂：根据 LLM_PROVIDER 配置返回 OpenAI 兼容的 ChatOpenAI 实例。"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from config import get_settings


@lru_cache
def get_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.llm_base_url,
        temperature=settings.LLM_TEMPERATURE,
        streaming=True,
    )
