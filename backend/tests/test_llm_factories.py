from config import get_settings
from core import llm


def test_rag_and_query_llms_use_deterministic_temperature(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "LLM_TEMPERATURE", 0.1)
    monkeypatch.setattr(settings, "RAG_ANSWER_TEMPERATURE", 0.0)
    monkeypatch.setattr(settings, "QUERY_ENHANCEMENT_TEMPERATURE", 0.0)
    llm.get_llm.cache_clear()
    llm.get_rag_llm.cache_clear()
    llm.get_query_llm.cache_clear()

    assert llm.get_llm().temperature == 0.1
    assert llm.get_rag_llm().temperature == 0.0
    assert llm.get_query_llm().temperature == 0.0
