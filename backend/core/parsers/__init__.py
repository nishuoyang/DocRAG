"""文档解析器 v2：按扩展名分发，输出 Markdown 中间格式的 Document 列表。

v2 解析器统一输出 Markdown（表格为 GFM、标题带 # 层级），
下游由 core.md_split 做结构感知分块。
"""
import csv
import logging
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from core.parsers.docx_parser import parse_docx
from core.parsers.pdf_parser import parse_pdf
from core.parsers.pptx_parser import parse_pptx

logger = logging.getLogger(__name__)

# v2 支持的扩展名（ingestion.SUPPORTED_EXTENSIONS 为其超集：legacy 路径用）
V2_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".html", ".xlsx", ".pptx"}


def _load_xlsx(file_path: str) -> list[Document]:
    """读取 xlsx 全部 sheet，每个 sheet 规范化为一个 Markdown 表格。"""
    import pandas as pd

    xls = pd.read_excel(file_path, sheet_name=None, dtype=str)
    docs: list[Document] = []
    for sheet_name, df in xls.items():
        df = df.fillna("")
        rows = [list(df.columns)] + df.astype(str).values.tolist()
        docs.append(
            Document(
                page_content=f"# Sheet: {sheet_name}\n\n{_rows_to_markdown(rows)}",
                metadata={"sheet": sheet_name},
            )
        )
    return docs


def _load_csv(file_path: str) -> list[Document]:
    with open(file_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    title = Path(file_path).stem
    return [
        Document(
            page_content=f"# CSV: {title}\n\n{_rows_to_markdown(rows)}",
            metadata={"sheet": title},
        )
    ]


def _load_text(file_path: str) -> list[Document]:
    return TextLoader(file_path, encoding="utf-8", autodetect_encoding=True).load()


def _load_html(file_path: str) -> list[Document]:
    soup = BeautifulSoup(Path(file_path).read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = _html_to_markdown(soup.body or soup).strip()
    return [Document(page_content=text)] if text else []


def _escape_cell(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", "<br>").replace("|", r"\|").strip()


def _rows_to_markdown(rows: list[list[object]]) -> str:
    """把二维数据转成 GFM 表格，保留原始列名。"""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [list(row) + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _html_table_to_markdown(table) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False)
        if cells:
            rows.append([" ".join(cell.stripped_strings) for cell in cells])
    return _rows_to_markdown(rows)


def _html_to_markdown(node) -> str:
    """Convert common semantic HTML elements to Markdown without extra deps."""
    from bs4 import NavigableString, Tag

    if isinstance(node, NavigableString):
        return str(node).strip()
    if not isinstance(node, Tag):
        return ""

    name = node.name.lower()
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return f"{'#' * int(name[1])} {' '.join(node.stripped_strings)}"
    if name == "p":
        return " ".join(node.stripped_strings)
    if name in {"ul", "ol"}:
        lines = []
        for index, item in enumerate(node.find_all("li", recursive=False), 1):
            prefix = "- " if name == "ul" else f"{index}. "
            lines.append(prefix + " ".join(item.stripped_strings))
        return "\n".join(lines)
    if name == "table":
        return _html_table_to_markdown(node)
    if name == "pre":
        return f"```\n{node.get_text().strip()}\n```"
    if name == "br":
        return "\n"

    parts = [_html_to_markdown(child) for child in node.children]
    return "\n\n".join(part for part in parts if part)


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
