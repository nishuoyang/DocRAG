from langchain_core.documents import Document

from db import milvus


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[0.1], [0.2]]


class _FakeVectorstore:
    def __init__(self):
        self.calls = []

    def similarity_search_by_vector(self, embedding, k):
        self.calls.append((embedding, k))
        return [Document(page_content=f"doc-{embedding[0]}")]


def test_similarity_search_many_batches_embeddings(monkeypatch):
    embeddings = _FakeEmbeddings()
    vectorstore = _FakeVectorstore()
    monkeypatch.setattr(milvus, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(milvus, "get_vectorstore", lambda: vectorstore)

    results = milvus.similarity_search_many(["query-a", "query-b"], k=3)

    assert embeddings.calls == [["query-a", "query-b"]]
    assert vectorstore.calls == [([0.1], 3), ([0.2], 3)]
    assert [docs[0].page_content for docs in results] == ["doc-0.1", "doc-0.2"]


def test_similarity_search_many_falls_back_to_individual_queries(monkeypatch):
    class BrokenEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("batch unsupported")

    class FakeVectorstore:
        def similarity_search(self, query, k):
            return [Document(page_content=query)]

    monkeypatch.setattr(milvus, "get_embeddings", lambda: BrokenEmbeddings())
    monkeypatch.setattr(milvus, "get_vectorstore", lambda: FakeVectorstore())

    results = milvus.similarity_search_many(["query-a", "query-b"], k=3)

    assert [docs[0].page_content for docs in results] == ["query-a", "query-b"]
