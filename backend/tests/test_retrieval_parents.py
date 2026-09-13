from langchain_core.documents import Document

from core import retrieval


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
