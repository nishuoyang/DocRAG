from langchain_core.documents import Document

from core import ingestion


def test_auto_split_uses_parent_child_for_markdown():
    docs = [Document(page_content="# 标题\n\n正文。" * 80)]

    split_docs, mode, parent_count = ingestion._split_documents(
        docs,
        split_mode="auto",
        filename="sample.md",
        file_hash="b" * 64,
        upload_time=123,
    )

    assert mode == "parent_child"
    assert parent_count > 0
    assert split_docs
    assert all(doc.metadata["parent_id"] for doc in split_docs)
    assert all(doc.metadata["chunk_type"] == "parent_child" for doc in split_docs)


def test_explicit_fixed_split_stays_compatible():
    docs = [Document(page_content="固定切分正文。" * 100)]

    split_docs, mode, parent_count = ingestion._split_documents(
        docs,
        split_mode="fixed",
        filename="sample.txt",
        file_hash="c" * 64,
        upload_time=123,
    )
    enriched = ingestion._make_metadata_docs(
        split_docs,
        filename="sample.txt",
        chunk_type=mode,
        file_hash="c" * 64,
    )

    assert mode == "fixed"
    assert parent_count == 0
    assert all(doc.metadata["parent_id"] == "" for doc in enriched)
    assert all(doc.metadata["raw_text"] == doc.page_content for doc in enriched)
