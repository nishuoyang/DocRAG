import os
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from config import get_settings
from db import milvus

pytestmark = pytest.mark.integration


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.0] * 1024 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 1024


def test_milvus_parent_child_schema_round_trip(monkeypatch):
    settings = get_settings()
    test_settings = SimpleNamespace(
        MILVUS_HOST=settings.MILVUS_HOST,
        MILVUS_PORT=settings.MILVUS_PORT,
        MILVUS_COLLECTION=f"codex_parent_child_test_{os.getpid()}",
        EMBEDDING_MODEL="fake",
        COLLECTION_SCHEMA_VERSION="parent_child_v1",
    )
    monkeypatch.setattr(milvus, "get_settings", lambda: test_settings)
    monkeypatch.setattr(milvus, "get_embeddings", _FakeEmbeddings)
    collection_name = milvus.get_collection_name()
    collection = None

    try:
        milvus.add_documents(
            [
                Document(
                    page_content="[章节: A]\nchild one",
                    metadata={
                        "filename": "smoke.md",
                        "chunk_index": 0,
                        "upload_time": 123,
                        "chunk_type": "parent_child",
                        "section": "A",
                        "content_type": "text",
                        "file_hash": "d" * 64,
                        "parent_id": "d" * 16 + "-p00000",
                        "parent_index": 0,
                        "child_index": 0,
                        "raw_text": "child one",
                        "page": None,
                        "page_start": None,
                        "page_end": None,
                    },
                ),
                Document(
                    page_content="[章节: A]\nchild two",
                    metadata={
                        "filename": "smoke.md",
                        "chunk_index": 1,
                        "upload_time": 123,
                        "chunk_type": "parent_child",
                        "section": "A",
                        "content_type": "text",
                        "file_hash": "d" * 64,
                        "parent_id": "d" * 16 + "-p00000",
                        "parent_index": 0,
                        "child_index": 1,
                        "raw_text": "child two",
                        "page": None,
                        "page_start": None,
                        "page_end": None,
                    },
                ),
            ]
        )
        collection = milvus._get_collection(collection_name)
        assert collection is not None
        documents = milvus.get_all_documents()
        assert len(documents) == 2
        parent_id = documents[0].metadata["parent_id"]
        grouped = milvus.get_children_by_parent_ids([parent_id])
        assert [doc.metadata["child_index"] for doc in grouped[parent_id]] == [0, 1]
        assert [doc.metadata["raw_text"] for doc in grouped[parent_id]] == ["child one", "child two"]
    finally:
        collection = collection or milvus._get_collection(collection_name)
        if collection is not None:
            collection.drop()
