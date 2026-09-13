"""Milvus 连接层：封装集合创建、文档写入与相似检索。

同一 Embedding 模型对应一个 Collection（用 EMBEDDING_MODEL 做别名避免维度冲突）。
"""
import json
import re

from langchain_milvus import Milvus
from langchain_core.documents import Document
from pymilvus import Collection, DataType, connections

from config import get_settings
from core.embeddings import get_embeddings


def _base_collection_name() -> str:
    """Legacy collection name derived only from the embedding model."""
    settings = get_settings()
    model_slug = re.sub(r"[^A-Za-z0-9_]", "_", settings.EMBEDDING_MODEL)
    return f"{settings.MILVUS_COLLECTION}_{model_slug}"


def get_legacy_collection_name() -> str:
    """Return the pre-parent-child collection name without touching it."""
    return _base_collection_name()


def get_collection_name() -> str:
    """Return the active schema-versioned collection name."""
    settings = get_settings()
    base = _base_collection_name()
    version = re.sub(r"[^A-Za-z0-9_]", "_", settings.COLLECTION_SCHEMA_VERSION.strip())
    return f"{base}_{version}" if version else base


def _connect_default() -> None:
    """建立默认别名连接（幂等，已连接则跳过）。"""
    settings = get_settings()
    try:
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
        )
    except Exception:
        # 已连接时 reconnect 会抛异常，忽略即可
        pass


def _get_collection(collection_name: str | None = None) -> Collection | None:
    """获取 Collection 对象（不依赖 embedding，用于纯查询场景）。

    返回 None 表示 collection 不存在。
    """
    _connect_default()
    from pymilvus import utility
    name = collection_name or get_collection_name()
    if not utility.has_collection(name):
        return None
    return Collection(name)


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
            "section": {"dtype": DataType.VARCHAR, "max_length": 256},
            "content_type": {"dtype": DataType.VARCHAR, "max_length": 16},
            "file_hash": {"dtype": DataType.VARCHAR, "max_length": 64},
            "parent_id": {"dtype": DataType.VARCHAR, "max_length": 64},
            "parent_index": {"dtype": DataType.INT64},
            "child_index": {"dtype": DataType.INT64},
            "raw_text": {"dtype": DataType.VARCHAR, "max_length": 65535},
            # page 必须显式声明且 nullable：PDF 有页码、DOCX 无页码，
            # 若不声明，langchain-milvus 会按首批 metadata 建成非空字段，
            # 之后上传无页码文档会报 "Insert missed an field `page`"
            "page": {"dtype": DataType.INT64, "nullable": True},
            "page_start": {"dtype": DataType.INT64, "nullable": True},
            "page_end": {"dtype": DataType.INT64, "nullable": True},
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
    """返回集合中全部文档块（用于文档列表接口，按 chunk 去重为文件）。

    直接用 pymilvus Collection 查询，不依赖 embedding 客户端
    （避免 EMBEDDING_API_KEY 未配置时加载文档列表也报错）。
    兼容旧 schema（无 section/content_type/file_hash/page 字段）。
    """
    col = _get_collection()
    if col is None:
        return []

    # 动态检测可用字段，兼容任意 schema 版本
    available_fields = {f.name for f in col.schema.fields}
    all_metadata_fields = [
        "filename", "chunk_index", "upload_time", "chunk_type",
        "section", "content_type", "file_hash", "parent_id", "parent_index",
        "child_index", "raw_text", "page", "page_start", "page_end",
    ]
    query_fields = ["pk", "text"]
    for f in all_metadata_fields:
        if f in available_fields:
            query_fields.append(f)

    documents: list[Document] = []
    offset = 0
    limit = 100
    while True:
        batch = col.query(
            expr="",
            output_fields=query_fields,
            limit=limit,
            offset=offset,
        )
        if not batch:
            break
        for item in batch:
            documents.append(_row_to_document(item, available_fields))
        if len(batch) < limit:
            break
        offset += limit
    return documents


def _row_to_document(item: dict, available_fields: set[str] | None = None) -> Document:
    fields = available_fields or set(item)
    metadata = {"pk": item.get("pk")}
    for field in (
        "filename",
        "chunk_index",
        "upload_time",
        "chunk_type",
        "section",
        "content_type",
        "file_hash",
        "parent_id",
        "parent_index",
        "child_index",
        "raw_text",
        "page",
        "page_start",
        "page_end",
    ):
        if field in fields:
            metadata[field] = item.get(field)
    return Document(page_content=item.get("text", ""), metadata=metadata)


def get_children_by_parent_ids(parent_ids: list[str]) -> dict[str, list[Document]]:
    """Fetch sibling children for virtual parent reconstruction."""
    unique_ids = list(dict.fromkeys(parent_id for parent_id in parent_ids if parent_id))
    if not unique_ids:
        return {}
    col = _get_collection()
    if col is None:
        return {}

    fields = {field.name for field in col.schema.fields}
    if "parent_id" not in fields:
        return {}
    output_fields = [
        field
        for field in (
            "pk",
            "text",
            "filename",
            "chunk_index",
            "upload_time",
            "chunk_type",
            "section",
            "content_type",
            "file_hash",
            "parent_id",
            "parent_index",
            "child_index",
            "raw_text",
            "page",
            "page_start",
            "page_end",
        )
        if field in fields or field in {"pk", "text"}
    ]
    expression = f"parent_id in {json.dumps(unique_ids, ensure_ascii=True)}"
    rows = col.query(expr=expression, output_fields=output_fields, limit=10000)
    grouped: dict[str, list[Document]] = {}
    for row in rows:
        document = _row_to_document(row, fields)
        parent_id = document.metadata.get("parent_id")
        if parent_id:
            grouped.setdefault(parent_id, []).append(document)
    for siblings in grouped.values():
        siblings.sort(key=lambda doc: doc.metadata.get("child_index") or 0)
    return grouped


def delete_document(file_name: str) -> int:
    """按文件名删除文档的全部块，返回删除条数。删除后 flush 保证对后续查询可见。"""
    col = _get_collection()
    if col is None:
        return 0
    result = col.delete(expr=f'filename == "{file_name}"')
    col.flush()
    return result.delete_count if result else 0


def has_file_hash(file_hash: str) -> bool:
    """内容哈希是否已入库（上传去重用）。collection 未创建/旧 schema 无该字段时返回 False。

    查询前 flush，确保能读到最近删除/写入的最新状态。
    """
    col = _get_collection()
    if col is None:
        return False
    field_names = {f.name for f in col.schema.fields}
    if "file_hash" not in field_names:
        return False
    col.flush()
    result = col.query(expr=f'file_hash == "{file_hash}"', output_fields=["pk"], limit=1)
    return bool(result)


def delete_by_hash(file_hash: str) -> int:
    """按内容哈希删除文档的全部块（replace 上传时用），返回删除条数。删除后 flush 保证生效。"""
    col = _get_collection()
    if col is None:
        return 0
    col.flush()
    result = col.delete(expr=f'file_hash == "{file_hash}"')
    col.flush()
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
