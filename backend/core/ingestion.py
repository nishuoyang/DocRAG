"""文档摄入：解析文件 → 分块 → 写入 Milvus。"""
import logging
import os
import tempfile
import time
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from db import milvus

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader}


def _get_loader(file_path: str):
    ext = Path(file_path).suffix.lower()
    loader_cls = SUPPORTED_EXTENSIONS.get(ext)
    if loader_cls is None:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(f"不支持的文件类型 {ext}，支持: {supported}")
    return loader_cls(file_path)


def _make_metadata_docs(docs: list[Document], filename: str) -> list[Document]:
    """为每个文档块补充文件名、上传时间、块序号元数据。"""
    timestamp = int(time.time())
    enriched: list[Document] = []
    for idx, doc in enumerate(docs):
        metadata = {
            "filename": filename,
            "chunk_index": idx,
            "upload_time": timestamp,
        }
        page = doc.metadata.get("page")
        if page is not None:
            metadata["page"] = page
        enriched.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )
    return enriched


def _load_documents(file_path: str) -> list[Document]:
    loader = _get_loader(file_path)
    return loader.load()


def _split_documents(docs: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def ingest_file(filename: str, content: bytes) -> dict:
    """解析上传文件并写入向量库，返回摄入统计。"""
    ext = Path(filename).suffix.lower()
    suffix = ext if ext in (".pdf", ".docx") else ".bin"
    # PyPDFLoader / Docx2txtLoader 需要文件路径，写临时文件
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        raw_docs = _load_documents(tmp_path)
        if not raw_docs:
            raise ValueError("文档内容为空或无法解析")
        split_docs = _split_documents(raw_docs)
        enriched = _make_metadata_docs(split_docs, filename)
        ids = milvus.add_documents(enriched)
        return {
            "ids": ids,
            "chunk_count": len(enriched),
            "filename": filename,
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def list_documents() -> list[dict]:
    """列出全部已入库文档（按文件名去重聚合 chunk）。"""
    docs = milvus.get_all_documents()
    by_name: dict[str, dict] = {}
    for doc in docs:
        name = doc.metadata.get("filename", "unknown")
        if name not in by_name:
            by_name[name] = {
                "filename": name,
                "chunk_count": 0,
                "upload_time": doc.metadata.get("upload_time"),
            }
        by_name[name]["chunk_count"] += 1
    return sorted(by_name.values(), key=lambda d: d.get("upload_time") or 0, reverse=True)


def delete_document(filename: str) -> int:
    return milvus.delete_document(filename)
