from langchain_core.documents import Document

from core import retrieval


def test_rag_query_reuses_cached_answer_for_identical_request(monkeypatch):
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "QUERY_TRANSFORM", False)
    monkeypatch.setattr(settings, "HYDE", False)
    document = Document(
        page_content="GitHub 发布流程",
        metadata={"pk": 1, "filename": "guide.md"},
    )
    calls = {"retrieve": 0, "generate": 0}

    def fake_retrieve(query, top_k, history=None):
        calls["retrieve"] += 1
        return [document]

    def fake_generate(messages):
        calls["generate"] += 1
        return type("Response", (), {"content": "稳定答案 [1]"})()

    monkeypatch.setattr(retrieval, "_retrieve", fake_retrieve)
    monkeypatch.setattr(retrieval, "_generate_answer", fake_generate)
    retrieval.invalidate_answer_cache()

    first = retrieval.rag_query("github发布项目的流程", 3)
    second = retrieval.rag_query("github发布项目的流程", 3)

    assert first == second
    assert calls == {"retrieve": 1, "generate": 1}
