import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document
from langsmith import RunTree, get_current_run_tree, traceable

from config import get_settings
from core import retrieval


class _NoopLangSmithClient:
    otel_exporter = None


def _test_root(monkeypatch) -> RunTree:
    monkeypatch.setattr(RunTree, "patch", lambda self, **kwargs: None)
    monkeypatch.setattr(RunTree, "post", lambda self, **kwargs: None)
    return RunTree(
        name="test-root",
        run_type="chain",
        ls_client=_NoopLangSmithClient(),
    )


def test_retrieve_builds_structured_langsmith_chain(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "QUERY_TRANSFORM", False)
    monkeypatch.setattr(settings, "HYDE", False)
    monkeypatch.setattr(settings, "ANSWERABILITY_GATE_ENABLED", False)

    hit = Document(
        page_content="child",
        metadata={
            "pk": 1,
            "filename": "guide.md",
            "parent_id": "parent-1",
            "child_index": 0,
            "raw_text": "child",
            "relevance_score": 0.9,
        },
    )
    monkeypatch.setattr(retrieval.milvus, "similarity_search", lambda query, k: [hit])
    monkeypatch.setattr(retrieval.bm25, "keyword_search", lambda query, k: [hit])
    monkeypatch.setattr(retrieval.rerank, "rerank", lambda query, docs, top_n: docs)
    monkeypatch.setattr(
        retrieval.milvus,
        "get_children_by_parent_ids",
        lambda parent_ids: {"parent-1": [hit]},
    )
    root = _test_root(monkeypatch)

    docs = retrieval._retrieve(
        "question",
        1,
        langsmith_extra={"run_tree": root},
    )

    assert docs
    assert [child.name for child in root.child_runs] == ["RAG Retrieve"]
    assert [child.name for child in root.child_runs[0].child_runs] == [
        "Query Planning",
        "Dense Retrieval",
        "BM25 Retrieval",
        "RRF Fusion",
        "Rerank",
        "Answerability Gate",
        "Parent Expansion",
    ]


def test_query_enhancement_thread_inherits_langsmith_parent(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "QUERY_TRANSFORM", True)
    monkeypatch.setattr(settings, "HYDE", False)
    recorded = []

    @traceable(name="Fake Query Rewrite", run_type="chain")
    def fake_rewrite(query, history=None):
        run_tree = get_current_run_tree()
        recorded.append((run_tree.name, run_tree.parent_run_id))
        return "rewritten"

    monkeypatch.setattr(retrieval.query_transform, "transform_query", fake_rewrite)
    root = _test_root(monkeypatch)

    plan = retrieval._build_query_plan(
        "question",
        langsmith_extra={"run_tree": root},
    )

    assert plan.dense == ["question", "rewritten"]
    assert recorded
    planning_run = root.child_runs[0]
    assert planning_run.name == "Query Planning"
    assert [child.name for child in planning_run.child_runs] == ["Fake Query Rewrite"]


def test_rag_stream_root_contains_retrieval_generation_and_citations(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "QUERY_TRANSFORM", False)
    monkeypatch.setattr(settings, "HYDE", False)
    monkeypatch.setattr(settings, "ANSWERABILITY_GATE_ENABLED", False)
    hit = Document(
        page_content="answer context",
        metadata={
            "pk": 1,
            "filename": "guide.md",
            "parent_id": "parent-1",
            "chunk_index": 0,
            "raw_text": "answer context",
            "relevance_score": 0.9,
        },
    )
    monkeypatch.setattr(retrieval.milvus, "similarity_search", lambda query, k: [hit])
    monkeypatch.setattr(retrieval.bm25, "keyword_search", lambda query, k: [hit])
    monkeypatch.setattr(retrieval.rerank, "rerank", lambda query, docs, top_n: docs)
    monkeypatch.setattr(
        retrieval.milvus,
        "get_children_by_parent_ids",
        lambda parent_ids: {"parent-1": [hit]},
    )

    class FakeStreamingLLM:
        async def astream(self, messages):
            yield SimpleNamespace(content="answer [1]")

    monkeypatch.setattr(retrieval.llm, "get_rag_llm", lambda: FakeStreamingLLM())
    root = _test_root(monkeypatch)

    async def consume():
        return [
            event
            async for event in retrieval.rag_stream(
                "question",
                1,
                langsmith_extra={"run_tree": root},
            )
        ]

    events = asyncio.run(consume())

    assert any('"delta": "answer [1]"' in event for event in events)
    assert [child.name for child in root.child_runs] == ["RAG Chat Stream"]
    assert [child.name for child in root.child_runs[0].child_runs] == [
        "RAG Retrieve",
        "Context Assembly",
        "Answer Generation",
        "Citation Selection",
    ]
