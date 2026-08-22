"""文档解析器 v2：按扩展名分发，输出 Markdown 中间格式的 Document 列表。

v2 解析器统一输出 Markdown（表格为 GFM、标题带 # 层级），
下游由 core.md_split 做结构感知分块。
"""
import logging
from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader, CSVLoader, TextLoader
from langchain_core.documents import Document

from core.parsers.docx_parser import parse_docx
from core.parsers.pdf_parser import parse_pdf
from core.parsers.pptx_parser import parse_pptx

logger = logging.getLogger(__name__)

# v2 支持的扩展名（ingestion.SUPPORTED_EXTENSIONS 为其超集：legacy 路径用）
V2_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".xlsx", ".pptx"}


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





def _load_csv(file_path: str) -> list[Document]:
    return CSVLoader(file_path, encoding="utf-8-sig", autodetect_encoding=True).load()


def _load_text(file_path: str) -> list[Document]:
    return TextLoader(file_path, encoding="utf-8", autodetect_encoding=True).load()


def _load_html(file_path: str) -> list[Document]:
    return BSHTMLLoader(file_path, open_encoding="utf-8").load()


def load_documents(file_path: str) -> list[Document]:
    """按扩展名分发解析；不支持的类型抛 ValueError。"""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    if ext == ".docx":
        return parse_docx(file_path)
    if ext in (".txt", ".md"):
        return _load_text(file_path)
    if ext == ".csv":
        return _load_csv(file_path)
    if ext == ".html":
        return _load_html(file_path)
    if ext == ".xlsx":
        return _load_xlsx(file_path)
    if ext == ".pptx":
        return parse_pptx(file_path)
    raise ValueError(f"不支持的文件类型 {ext}，支持: {', '.join(sorted(V2_EXTENSIONS))}")
