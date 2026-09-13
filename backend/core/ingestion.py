"""文档摄入：解析文件 → 分块 → 写入 Milvus。

双引擎：
- v2（默认）：core.parsers 结构化解析（PDF/DOCX 输出 Markdown）→ 噪声清洗
  → core.md_split 结构感知分块（表格保护 + 章节元数据）
- legacy：旧链路（PyPDFLoader/Docx2txtLoader + fixed/semantic），PARSER_ENGINE=legacy 回滚
"""
import hashlib
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
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
from core import answer_cache, bm25
from core.embeddings import get_embeddings
from core.vlm import get_vlm_client
from db import milvus

logger = logging.getLogger(__name__)

# legacy 引擎支持的扩展名（含自写 xlsx/pptx loader）
SUPPORTED_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
    ".html": BSHTMLLoader,
}

class DuplicateFileError(ValueError):
    """上传的文件（按内容哈希）已入库。"""

    def __init__(self, filename: str, file_hash: str):
        self.filename = filename
        self.file_hash = file_hash
        super().__init__(f"文件 {filename} 已入库（内容哈希 {file_hash[:12]}…）")


# ───────────────────────── legacy 解析（回滚链路）─────────────────────────


def _load_xlsx(file_path: str) -> list[Document]:
    """读取 xlsx 全部 sheet，每行文本拼为一个 Document。"""
    import pandas as pd

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


def _legacy_load_documents(file_path: str) -> list[Document]:
    return _get_loader(file_path)(file_path)


# ───────────────────────── 分块（parent_child / fixed / semantic）─────────────────────────


def _make_metadata_docs(
    docs: list[Document],
    filename: str,
    chunk_type: str = "fixed",
    file_hash: str | None = None,
) -> list[Document]:
    """为每个文档块补充文件名、上传时间、块序号、切分类型等元数据。

    markdown 分块产出的块自带 section / content_type，原样保留。
    """
    timestamp = int(time.time())
    enriched: list[Document] = []
    for idx, doc in enumerate(docs):
        metadata = {
            "filename": filename,
            "chunk_index": idx,
            "upload_time": timestamp,
            "chunk_type": chunk_type,
        }
        page = doc.metadata.get("page")
        if page is not None:
            metadata["page"] = page
        section = doc.metadata.get("section")
        if section:
            metadata["section"] = section
        content_type = doc.metadata.get("content_type")
        if content_type:
            metadata["content_type"] = content_type
        metadata["parent_id"] = doc.metadata.get("parent_id", "")
        metadata["parent_index"] = doc.metadata.get("parent_index", 0)
        metadata["child_index"] = doc.metadata.get("child_index", idx)
        metadata["raw_text"] = doc.metadata.get("raw_text", doc.page_content)
        metadata["page_start"] = doc.metadata.get("page_start", page)
        metadata["page_end"] = doc.metadata.get("page_end", page)
        if file_hash:
            metadata["file_hash"] = file_hash
        enriched.append(Document(page_content=doc.page_content, metadata=metadata))
    return enriched


def _resolve_split_mode(docs: list[Document], split_mode: str | None) -> str:
    """确定切分策略；v2 的 auto/markdown 默认使用 parent_child。"""
    settings = get_settings()
    if split_mode in ("markdown", "parent_child"):
        return "parent_child"
    if split_mode in ("semantic", "fixed"):
        return split_mode
    if settings.PARSER_ENGINE == "v2":
        return "parent_child"
    mode = str(settings.SEMANTIC_SPLIT).lower()
    if mode == "auto":
        total_len = sum(len(d.page_content) for d in docs)
        return "semantic" if total_len <= settings.AUTO_SEMANTIC_THRESHOLD else "fixed"
    # 兼容旧布尔配置：true = 全部语义，false = 全部固定
    return "semantic" if mode == "true" else "fixed"


def _split_documents(
    docs: list[Document],
    split_mode: str | None = None,
    filename: str = "",
    file_hash: str = "",
    upload_time: int | None = None,
) -> tuple[list[Document], str, int]:
    """切分文档，返回 (切分结果, 实际策略, parent 数量)。"""
    mode = _resolve_split_mode(docs, split_mode)
    if mode == "parent_child":
        from core.parent_child import build_parent_child_documents

        chunks, parent_count = build_parent_child_documents(
            docs,
            filename=filename,
            file_hash=file_hash,
            upload_time=upload_time or int(time.time()),
        )
        return chunks, "parent_child", parent_count
    if mode == "semantic":
        return _semantic_split_documents(docs), "semantic", 0
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs), "fixed", 0


# 语义切分：按句子 embedding 相似度断块。
# 与固定长度切分不同，断点落在"语义断裂处"（如话题切换），块内内容更连贯。
# 代价：每个文档的所有句子都要调一次 embedding API（计费 + 耗时）。
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|(?<=\n)\s*")
_MIN_SEMANTIC_CHUNK = 40  # 语义块最小长度（字符），低于此的断点合并，防碎块


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点/换行切句，过滤空句。"""
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _semantic_split_documents(docs: list[Document]) -> list[Document]:
    """语义切分：句子 → embedding → 相邻相似度低处断块 → 过长块兜底再切。"""
    settings = get_settings()
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    result: list[Document] = []
    for doc in docs:
        sents = _split_sentences(doc.page_content)
        if len(sents) < 2:
            result.append(doc)
            continue
        # 批量 embedding（分批，防止超长文本一次调用过多）
        vecs: list[list[float]] = []
        batch = 64
        for i in range(0, len(sents), batch):
            vecs.extend(embeddings.embed_documents(sents[i : i + batch]))
        arr = np.array(vecs)
        # 相邻句余弦相似度（归一化后点积）
        norm = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        sims = np.sum(norm[:-1] * norm[1:], axis=1)
        # 断点：相似度低于 均值-1.5σ 的位置（语义断裂处）
        threshold = sims.mean() - 1.5 * sims.std()
        breaks = [i for i, s in enumerate(sims) if s < threshold]
        # 按断点合并句子成块；断点距当前块起点过近则跳过（防碎块）
        chunks = []
        start = 0
        for b in breaks + [len(sents) - 1]:
            text = "".join(sents[start : b + 1])
            if text and len(text) >= _MIN_SEMANTIC_CHUNK:
                chunks.append(text)
                start = b + 1
        if start < len(sents):  # 尾部不足最小长度的句子并入最后一块，不丢弃
            chunks.append("".join(sents[start:]))
        # 过长的语义块用固定长度切分兜底（语义块可能 > CHUNK_SIZE×2）
        for text in chunks:
            if len(text) > settings.CHUNK_SIZE * 2:
                sub_docs = splitter.create_documents([text])
                result.extend(sub_docs)
            else:
                result.append(Document(page_content=text))
    return result


# ───────────────────────── 上传入口 ─────────────────────────


def _uploads_dir() -> Path:
    """原件落盘目录（backend/uploads/），不存在则创建。"""
    settings = get_settings()
    p = Path(settings.UPLOAD_DIR)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / settings.UPLOAD_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_original(filename: str, content: bytes, file_hash: str) -> Path:
    """原件落盘：哈希命名防同名覆盖，保留原扩展名供重解析识别类型。"""
    ext = Path(filename).suffix.lower() or ".bin"
    path = _uploads_dir() / f"{file_hash}{ext}"
    if not path.exists():
        path.write_bytes(content)
    return path


def _check_duplicate(file_hash: str) -> bool:
    """按内容哈希查 Milvus 是否已入库同名内容。"""
    return milvus.has_file_hash(file_hash)


def ingest_file(filename: str, content: bytes, split_mode: str | None = None, replace: bool = False) -> dict:
    """解析上传文件并写入向量库，返回摄入统计。

    split_mode: auto（默认）/ semantic / fixed / markdown。
    replace=True 时内容哈希重复则先删除旧块再重新入库。
    """
    settings = get_settings()
    file_hash = hashlib.sha256(content).hexdigest()
    if _check_duplicate(file_hash):
        if not replace:
            raise DuplicateFileError(filename, file_hash)
        milvus.delete_by_hash(file_hash)
        bm25.invalidate_index()
        answer_cache.invalidate()

    ext = Path(filename).suffix.lower()
    from core.parsers import V2_EXTENSIONS

    use_v2 = settings.PARSER_ENGINE == "v2" and ext in V2_EXTENSIONS

    suffix = ext if ext in SUPPORTED_EXTENSIONS or ext in (".xlsx", ".pptx") else ".bin"
    # PyPDFLoader / Docx2txtLoader 需要文件路径，写临时文件
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    vlm_client = get_vlm_client()
    vlm_token = vlm_client.begin_job()
    try:
        if use_v2:
            from core import cleaning
            from core.parsers import load_documents

            raw_docs = load_documents(tmp_path)
            if ext == ".pdf":
                # 分页分隔线剔除 + 跨页重复行（页眉/页脚/页码）清洗
                raw_docs = [
                    Document(page_content=cleaning.strip_separators(d.page_content), metadata=d.metadata)
                    for d in raw_docs
                ]
                raw_docs = [d for d in raw_docs if d.page_content]
                raw_docs = cleaning.strip_repeated_lines(raw_docs)
        else:
            raw_docs = _legacy_load_documents(tmp_path)
        if not raw_docs:
            raise ValueError("文档内容为空或无法解析")
        upload_time = int(time.time())
        split_docs, used_mode, parent_count = _split_documents(
            raw_docs,
            split_mode,
            filename=filename,
            file_hash=file_hash,
            upload_time=upload_time,
        )
        if used_mode == "parent_child":
            enriched = split_docs
        else:
            enriched = _make_metadata_docs(
                split_docs,
                filename,
                chunk_type=used_mode,
                file_hash=file_hash,
            )
        ids = milvus.add_documents(enriched)
        bm25.invalidate_index()
        answer_cache.invalidate()
        _save_original(filename, content, file_hash)

        # 获取 VLM 处理计数
        vlm_pages = vlm_client.get_processed_count()

        return {
            "ids": ids,
            "chunk_count": len(enriched),
            "child_count": len(enriched),
            "parent_count": parent_count,
            "filename": filename,
            "chunk_type": used_mode,
            "file_hash": file_hash,
            "vlm_pages": vlm_pages,
        }
    finally:
        vlm_client.end_job(vlm_token)
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
                "file_hash": doc.metadata.get("file_hash"),
            }
        by_name[name]["chunk_count"] += 1
    return sorted(by_name.values(), key=lambda d: d.get("upload_time") or 0, reverse=True)


def delete_document(filename: str) -> int:
    deleted = milvus.delete_document(filename)
    if deleted:
        bm25.invalidate_index()
        answer_cache.invalidate()
    return deleted
