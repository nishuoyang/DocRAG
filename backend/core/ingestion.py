"""文档摄入：解析文件 → 分块 → 写入 Milvus。"""
import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pptx import Presentation

from config import get_settings
from db import milvus

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
    ".html": BSHTMLLoader,
}


def _load_xlsx(file_path: str) -> list[Document]:
    """读取 xlsx 全部 sheet，每行文本拼为一个 Document。"""
    xls = pd.read_excel(file_path, sheet_name=None, dtype=str)
    docs: list[Document] = []
    for sheet_name, df in xls.items():
        df = df.fillna("")
        for row in df.itertuples(index=False):
            parts = [f"{col}: {val}" for col, val in zip(df.columns, row) if str(val).strip()]
            if parts:
                docs.append(Document(page_content="\n".join(parts), metadata={"sheet": sheet_name}))
    return docs


def _load_pptx(file_path: str) -> list[Document]:
    """读取 pptx 逐页文本，每页一个 Document。"""
    prs = Presentation(file_path)
    docs: list[Document] = []
    for i, slide in enumerate(prs.slides, 1):
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
        if parts:
            docs.append(Document(page_content="\n".join(parts), metadata={"page": i}))
    return docs


def _get_loader(file_path: str):
    """返回对应扩展名的 loader 工厂；xlsx / pptx 用自写 loader，其余用 langchain loader。"""
    ext = Path(file_path).suffix.lower()
    if ext == ".xlsx":
        return _load_xlsx
    if ext == ".pptx":
        return _load_pptx
    loader_cls = SUPPORTED_EXTENSIONS.get(ext)
    if loader_cls is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"不支持的文件类型 {ext}，支持: {supported}")
    # 文本类 loader 传入 UTF-8 优先的编码探测，避免中文文件被按系统 GBK 解码报错
    kwargs = {}
    if ext in (".txt", ".md"):
        kwargs = {"encoding": "utf-8", "autodetect_encoding": True}
    elif ext == ".csv":
        kwargs = {"encoding": "utf-8-sig", "autodetect_encoding": True}
    elif ext == ".html":
        kwargs = {"open_encoding": "utf-8"}
    return lambda p: loader_cls(p, **kwargs).load()


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
    return _get_loader(file_path)(file_path)


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
    suffix = ext if ext in SUPPORTED_EXTENSIONS or ext in (".xlsx", ".pptx") else ".bin"
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
