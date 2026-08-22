"""Markdown 结构感知分块：按标题层级切节 + 表格保护。

与 fixed/semantic 并存（chunk_type="markdown"）：
- 按 # 标题维护章节栈，内容切成 (章节路径, text|table, 正文) 段
- 节内文本超长用 RecursiveCharacterTextSplitter，每块前置 [章节: 路径] 上下文行
- GFM 表格永不拦腰切断；超大表按行分组、每组重复表头
"""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_MIN_TEXT = 20  # 过短文本段丢弃（解析残留的孤立行）


def _segment(text: str) -> list[tuple[str, str, str]]:
    """把 Markdown 切成 (章节路径, 类型 text|table, 正文) 段列表。"""
    segments: list[tuple[str, str, str]] = []
    heading_stack: list[tuple[int, str]] = []
    cur_lines: list[str] = []
    cur_kind: str | None = None

    def flush() -> None:
        nonlocal cur_lines, cur_kind
        body = "\n".join(cur_lines).strip()
        if body and cur_kind:
            path = " > ".join(title for _, title in heading_stack)
            segments.append((path, cur_kind, body))
        cur_lines, cur_kind = [], None

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue
        kind = "table" if _TABLE_LINE_RE.match(line) else "text"
        if cur_kind is not None and kind != cur_kind:
            flush()
        cur_kind = kind
        cur_lines.append(line)
    flush()
    return segments


def _meta(page_meta: dict, section: str, content_type: str) -> dict:
    meta = {"section": section, "content_type": content_type}
    meta.update(page_meta)
    return meta


def _split_text_segment(text: str, section: str, page_meta: dict, chunk_size: int, overlap: int) -> list[Document]:
    """文本段切块：整节放得下则单块，否则固定切分并前置章节上下文。"""
    if len(text) < _MIN_TEXT:
        return []
    prefix = f"[章节: {section}]\n" if section else ""
    budget = max(chunk_size - len(prefix), 100)
    if len(text) <= budget:
        return [Document(page_content=prefix + text, metadata=_meta(page_meta, section, "text"))]
    splitter = RecursiveCharacterTextSplitter(chunk_size=budget, chunk_overlap=overlap, length_function=len)
    return [
        Document(page_content=prefix + piece, metadata=_meta(page_meta, section, "text"))
        for piece in splitter.split_text(text)
    ]


def _split_table_segment(text: str, section: str, page_meta: dict, chunk_size: int) -> list[Document]:
    """表格段切块：整体放得下（放宽到 2×chunk_size）则单块，否则按行分组并重复表头。"""
    prefix = f"[章节: {section}]\n" if section else ""
    budget = max(chunk_size * 2 - len(prefix), 200)
    if len(text) <= budget:
        return [Document(page_content=prefix + text, metadata=_meta(page_meta, section, "table"))]

    lines = text.splitlines()
    has_sep = len(lines) > 1 and bool(_TABLE_SEP_RE.match(lines[1]))
    head_lines = lines[:2] if has_sep else lines[:1]
    data_rows = lines[len(head_lines):]

    groups: list[list[str]] = []
    cur = list(head_lines)
    cur_len = sum(len(ln) + 1 for ln in cur)
    for row in data_rows:
        if cur_len + len(row) + 1 > budget and len(cur) > len(head_lines):
            groups.append(cur)
            cur = list(head_lines)
            cur_len = sum(len(ln) + 1 for ln in cur)
        cur.append(row)
        cur_len += len(row) + 1
    if len(cur) > len(head_lines):
        groups.append(cur)
    return [
        Document(page_content=prefix + "\n".join(g), metadata=_meta(page_meta, section, "table"))
        for g in groups
    ]


def split_markdown_documents(docs: list[Document]) -> list[Document]:
    """对已解析为 Markdown 的文档块做结构感知分块。

    每个输入 Document 独立处理（章节栈不跨页）；page 元数据原样继承。
    """
    settings = get_settings()
    result: list[Document] = []
    for doc in docs:
        page_meta = {"page": doc.metadata["page"]} if doc.metadata.get("page") is not None else {}
        for section, kind, body in _segment(doc.page_content):
            if kind == "table":
                result.extend(_split_table_segment(body, section, page_meta, settings.CHUNK_SIZE))
            else:
                result.extend(
                    _split_text_segment(body, section, page_meta, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
                )
    return result
