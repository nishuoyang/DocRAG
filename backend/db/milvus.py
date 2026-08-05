"""Milvus 连接层：封装集合创建、文档写入与相似检索。

同一 Embedding 模型对应一个 Collection（用 EMBEDDING_MODEL 做别名避免维度冲突）。
"""
import re

from langchain_milvus import Milvus
from langchain_core.documents import Document
from pymilvus import DataType

from config import get_settings
from core.embeddings import get_embeddings


def get_collection_name() -> str:
    """以 Embedding 模型命名集合，避免换模型后维度不匹配。"""
    settings = get_settings()
    model_slug = re.sub(r"[^A-Za-z0-9_]", "_", settings.EMBEDDING_MODEL)
    return f"{settings.MILVUS_COLLECTION}_{model_slug}"


def get_vectorstore() -> Milvus:
    """返回已连接的 Milvus vectorstore（自动创建 Collection）。

    显式声明 metadata schema：chunk_type（semantic/fixed）在新 collection 中
    成为正式字段；旧 collection（无此字段）中该元数据会丢失。
    """
    settings = get_settings()
    return Milvus(
        embedding_function=get_embeddings(),
        connection_args={
            "host": settings.MILVUS_HOST,
            "port": settings.MILVUS_PORT,
        },
        collection_name=get_collection_name(),
        auto_id=True,
        drop_old=False,
        metadata_schema={
            "chunk_type": {"dtype": DataType.VARCHAR, "max_length": 32},
        },
    )


def add_documents(docs: list[Document]) -> list[str]:
    """写入文档块，返回分配的向量 ID 列表。"""
    vs = get_vectorstore()
    return vs.add_documents(docs)


def similarity_search(query: str, k: int | None = None) -> list[Document]:
    """按查询文本做向量检索，返回相似文档块。"""
    settings = get_settings()
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k or settings.TOP_K)


def get_all_documents() -> list[Document]:
    """返回集合中全部文档块（用于文档列表接口，按 chunk 去重为文件）。"""
    vs = get_vectorstore()
    # collection 尚未创建（首次上传前）→ 返回空列表
    if vs.col is None:
        return []
    # Milvus 分页取全部数据
    documents: list[Document] = []
    offset = 0
    limit = 100
    while True:
        batch = vs.col.query(
            expr="",
            output_fields=["pk", "text", "filename", "chunk_index", "upload_time", "chunk_type"],
            limit=limit,
            offset=offset,
        )
        if not batch:
            break
        for item in batch:
            documents.append(
                Document(
                    page_content=item.get("text", ""),
                    metadata={
                        "pk": item.get("pk"),
                        "filename": item.get("filename", ""),
                        "chunk_index": item.get("chunk_index"),
                        "upload_time": item.get("upload_time"),
                        "chunk_type": item.get("chunk_type"),
                    },
                )
            )
        if len(batch) < limit:
            break
        offset += limit
    return documents


def delete_document(file_name: str) -> int:
    """按文件名删除文档的全部块，返回删除条数。"""
    vs = get_vectorstore()
    if vs.col is None:
        return 0
    result = vs.col.delete(expr=f'filename == "{file_name}"')
    return result.delete_count if result else 0


def ensure_collection_ready() -> None:
    """应用启动时校验 Milvus 连通性；collection 不存在时首次上传会自动创建。"""
    settings = get_settings()
    vs = get_vectorstore()
    if vs.col is None:
        return
    schema = vs.col.schema
    fields = {f.name: f for f in schema.fields}
    actual_dim = fields["vector"].params.get("dim")
    if actual_dim and actual_dim != settings.EMBEDDING_DIM:
        raise ValueError(
            f"Collection 维度 {actual_dim} 与配置 {settings.EMBEDDING_DIM} 不符，"
            "请确认 EMBEDDING_MODEL / EMBEDDING_DIM 配置一致。"
        )
