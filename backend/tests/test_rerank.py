from langchain_core.documents import Document

from core import rerank


def test_rerank_scores_do_not_mutate_input_documents(monkeypatch):
    docs = [Document(page_content="candidate", metadata={"pk": 1})]
    settings = rerank.get_settings()
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_API_KEY", "test-key")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    class FakeClient:
        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(rerank, "get_http_client", lambda: FakeClient())

    ranked = rerank.rerank("query", docs, top_n=1)

    assert ranked[0] is not docs[0]
    assert ranked[0].metadata["relevance_score"] == 0.9
    assert "relevance_score" not in docs[0].metadata


def test_rerank_http_client_is_reused():
    rerank.get_http_client.cache_clear()

    assert rerank.get_http_client() is rerank.get_http_client()
