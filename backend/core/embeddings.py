"""Embedding 工厂：通过 OpenAI 兼容 API（硅基流动等）获取向量。"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config import get_settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.EMBEDDING_API_KEY,
        base_url=settings.EMBEDDING_BASE_URL,
        check_embedding_ctx_length=False,
    )
