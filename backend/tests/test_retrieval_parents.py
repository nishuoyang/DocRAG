from langchain_core.documents import Document

from core import retrieval


def test_merge_candidates_uses_stable_ids_instead_of_text():
    first_file = Document(
        page_content="相同正文",
        metadata={"pk": 1, "filename": "a.md", "parent_id": "a-0", "child_index": 0},
    )
    duplicate_pk = Document(
        page_content="相同正文",
        metadata={"pk": 1, "filename": "a.md", "parent_id": "a-0", "child_index": 0},
    )
    second_file = Document(
        page_content="相同正文",
        metadata={"pk": 2, "filename": "b.md", "parent_id": "b-0", "child_index": 0},
    )

    merged = retrieval._merge_candidates([[first_file, duplicate_pk], [second_file]])

    assert [doc.metadata["pk"] for doc in merged] == [1, 2]


def test_merge_candidates_fuses_ranks_with_rrf():
    first = Document(page_content="first", metadata={"pk": 1})
    bridge = Document(page_content="bridge", metadata={"pk": 2})
    second_only = Document(page_content="second", metadata={"pk": 3})

    merged = retrieval._merge_candidates([[first, bridge], [bridge, second_only]])

    assert [doc.metadata["pk"] for doc in merged] == [2, 1, 3]
    assert merged[0].metadata["rrf_score"] > merged[1].metadata["rrf_score"]


def test_retrieve_batches_enhanced_dense_queries(monkeypatch):
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "quality")
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATE_K", 20)
    hit = Document(page_content="hit", metadata={"pk": 1})
    calls = {}
    monkeypatch.setattr(
        retrieval,
        "_build_query_plan",
        lambda query, history=None: retrieval.QueryPlan(
            dense=["原问题", "改写问题", "假想答案"],
            sparse=["原问题", "改写问题"],
        ),
    )
    monkeypatch.setattr(retrieval, "_dense_search", lambda query, k: [hit])
    monkeypatch.setattr(retrieval, "_sparse_search", lambda query, k: [hit])
    monkeypatch.setattr(
        retrieval,
        "_dense_search_many",
        lambda queries, k: calls.setdefault("dense_many", (queries, k)) or [[hit], [hit]],
    )
    monkeypatch.setattr(retrieval, "_merge_candidates", lambda result_lists: [hit])
    monkeypatch.setattr(retrieval, "_rerank_with_fallback", lambda query, docs, top_n: docs)
    monkeypatch.setattr(retrieval, "_filter_answerable", lambda docs: docs)
    monkeypatch.setattr(
        retrieval,
        "_materialize_parent_contexts",
        lambda ranked, max_parents, max_chars: ranked,
    )

    docs = retrieval._retrieve("原问题", 1)

    assert docs == [hit]
    assert calls["dense_many"] == (["改写问题", "假想答案"], 20)


def test_candidate_k_depends_on_retrieval_mode(monkeypatch):
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATE_K", 20)

    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "fast")
    assert retrieval._candidate_k(4) == 8

    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "balanced")
    assert retrieval._candidate_k(4) == 12

    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "quality")
    assert retrieval._candidate_k(4) == 20


def test_rerank_failure_falls_back_to_fused_candidates(monkeypatch):
    candidates = [
        Document(page_content="first", metadata={"pk": 1}),
        Document(page_content="second", metadata={"pk": 2}),
    ]
    monkeypatch.setattr(
        retrieval.rerank,
        "rerank",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rerank unavailable")),
    )

    ranked = retrieval._rerank_with_fallback("query", candidates, top_n=2)

    assert ranked == candidates


def test_fast_mode_skips_remote_rerank(monkeypatch):
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "fast")
    candidates = [
        Document(page_content="first", metadata={"pk": 1}),
        Document(page_content="second", metadata={"pk": 2}),
    ]
    monkeypatch.setattr(
        retrieval.rerank,
        "rerank",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not rerank")),
    )

    ranked = retrieval._rerank_with_fallback("query", candidates, top_n=1)

    assert ranked == candidates[:1]


def test_answerability_gate_rejects_candidates_when_all_scores_are_low(monkeypatch):
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "ANSWERABILITY_GATE_ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_MIN_RERANK_SCORE", 0.2)
    docs = [
        Document(page_content="low", metadata={"relevance_score": 0.1}),
        Document(page_content="lower", metadata={"relevance_score": 0.05}),
    ]

    assert retrieval._filter_answerable(docs) == []


def test_select_sources_only_returns_documents_cited_by_answer():
    docs = [
        Document(page_content="first", metadata={"filename": "a.md"}),
        Document(page_content="second", metadata={"filename": "b.md"}),
    ]

    selected = retrieval._select_sources("结论来自第二份资料 [2]。", docs)
    citation_index, doc = selected[0]
    payload = retrieval._source_payload(doc, citation_index=citation_index)

    assert [item.metadata["filename"] for _, item in selected] == ["b.md"]
    assert payload["citation_index"] == 2


def test_materialize_parent_contexts_deduplicates_and_rebuilds(monkeypatch):
    hit_a = Document(
        page_content="[章节: A]\n命中 A 的 child",
        metadata={
            "filename": "a.md",
            "parent_id": "parent-a",
            "chunk_index": 10,
            "child_index": 1,
            "section": "A",
        },
    )
    hit_a_again = Document(
        page_content="[章节: A]\n命中 A 的另一个 child",
        metadata={**hit_a.metadata, "chunk_index": 11, "child_index": 2},
    )
    hit_b = Document(
        page_content="[章节: B]\n命中 B 的 child",
        metadata={
            "filename": "b.md",
            "parent_id": "parent-b",
            "chunk_index": 20,
            "child_index": 0,
            "section": "B",
        },
    )
    siblings = {
        "parent-a": [
            Document(
                page_content="[章节: A]\n完整 A1",
                metadata={
                    **hit_a.metadata,
                    "raw_text": "完整 A1",
                    "child_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                },
            ),
            Document(
                page_content="[章节: A]\n完整 A2",
                metadata={
                    **hit_a.metadata,
                    "raw_text": "完整 A2",
                    "child_index": 1,
                    "page_start": 1,
                    "page_end": 2,
                },
            ),
        ],
        "parent-b": [
            Document(
                page_content="[章节: B]\n完整 B",
                metadata={
                    **hit_b.metadata,
                    "raw_text": "完整 B",
                    "child_index": 0,
                    "page_start": 3,
                    "page_end": 3,
                },
            )
        ],
    }
    monkeypatch.setattr(retrieval.milvus, "get_children_by_parent_ids", lambda ids: siblings)

    parents = retrieval._materialize_parent_contexts(
        [hit_a, hit_a_again, hit_b],
        max_parents=4,
        max_chars=8000,
    )

    assert [p.metadata["parent_id"] for p in parents] == ["parent-a", "parent-b"]
    assert parents[0].page_content == "完整 A1\n\n完整 A2"
    assert parents[0].metadata["matched_chunk_index"] == 10
    assert parents[0].metadata["page_start"] == 1
    assert parents[0].metadata["page_end"] == 2


def test_materialize_parent_contexts_falls_back_to_child(monkeypatch):
    child = Document(
        page_content="原始 child",
        metadata={"filename": "x.md", "parent_id": "missing", "chunk_index": 1},
    )
    monkeypatch.setattr(retrieval.milvus, "get_children_by_parent_ids", lambda ids: {})

    parents = retrieval._materialize_parent_contexts([child], max_parents=4, max_chars=8000)

    assert len(parents) == 1
    assert parents[0].page_content == child.page_content
    assert parents[0].metadata["parent_id"] == "missing"
    assert parents[0].metadata["matched_chunk_index"] == 1


def test_materialize_parent_contexts_uses_hit_child_neighborhood(monkeypatch):
    hit = Document(
        page_content="hit child",
        metadata={
            "filename": "long.md",
            "parent_id": "parent-long",
            "chunk_index": 5,
            "child_index": 2,
        },
    )
    siblings = {
        "parent-long": [
            Document(
                page_content=f"child {index}",
                metadata={
                    "filename": "long.md",
                    "parent_id": "parent-long",
                    "chunk_index": index,
                    "child_index": index,
                    "raw_text": f"child {index}",
                },
            )
            for index in range(5)
        ]
    }
    monkeypatch.setattr(retrieval.milvus, "get_children_by_parent_ids", lambda ids: siblings)
    settings = retrieval.get_settings()
    monkeypatch.setattr(settings, "PARENT_WINDOW_RADIUS", 1)

    parents = retrieval._materialize_parent_contexts([hit], max_parents=1, max_chars=8000)

    assert parents[0].page_content == "child 1\n\nchild 2\n\nchild 3"
    assert parents[0].metadata["parent_window_start"] == 1
    assert parents[0].metadata["parent_window_end"] == 3


def test_context_format_includes_section_and_page_location():
    document = Document(
        page_content="完整 parent",
        metadata={
            "filename": "guide.pdf",
            "section": "第二章 > 部署",
            "page_start": 3,
            "page_end": 4,
        },
    )

    context = retrieval._format_context([document])

    assert "guide.pdf" in context
    assert "第二章 > 部署" in context
    assert "第3-4页" in context
