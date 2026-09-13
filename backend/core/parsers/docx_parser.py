"""DOCX 解析（v2）：python-docx 顺序遍历 body，标题层级 + 表格转 Markdown。

P2 增强：
- 内嵌图片提取：提取段落中的图片，调用 VLM 生成描述
"""
import re

from docx import Document as DocxDocument
from docx.document import Document as _DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document

from core.vlm import get_vlm_client

# 兼容中英文 Word 标题样式名："Heading 2" / "标题 2"
_HEADING_STYLE = re.compile(r"(?:Heading|标题)\s*(\d+)", re.IGNORECASE)


def _heading_level(style_name: str | None) -> int:
    """从段落样式名提取标题层级（1-6）；Title 样式视为一级标题；非标题返回 0。"""
    if not style_name:
        return 0
    m = _HEADING_STYLE.search(style_name)
    if m:
        return int(m.group(1))
    return 1 if style_name.strip().lower() in ("title", "标题") else 0


def _iter_block_items(doc: _DocxDocument):
    """按文档真实顺序交替遍历段落与表格。

    python-docx 的 doc.paragraphs / doc.tables 分开提供会丢失混排顺序，
    这里直接遍历 body XML 子元素保序。
    """
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _table_to_markdown(table: Table) -> str:
    """表格转 GFM Markdown；空表返回空串。"""
    if not table.rows:
        return ""
    lines = []
    for i, row in enumerate(table.rows):
        cells = [c.text.replace("\n", " ").strip() for c in row.cells]
        if not any(cells):
            continue
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(lines)


def _extract_images_from_paragraph(paragraph: Paragraph, doc: DocxDocument) -> list[bytes]:
    """从段落中提取内嵌图片的二进制数据。"""
    images: list[bytes] = []
    seen_rids: set[str] = set()
    for blip in paragraph._p.iter(qn("a:blip")):
        rid = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
        if not rid or rid in seen_rids:
            continue
        seen_rids.add(rid)
        try:
            image_part = doc.part.related_parts[rid]
            images.append(image_part.blob)
        except Exception:
            continue
    return images


def parse_docx(file_path: str) -> list[Document]:
    """解析 DOCX 为单个 Markdown 文档块（保留标题层级与表格结构）。

    P2 增强：提取内嵌图片并调用 VLM 生成描述。
    """
    doc = DocxDocument(file_path)
    vlm = get_vlm_client()
    parts: list[str] = []
    image_count = 0

    for block in _iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()

            # 提取并处理内嵌图片
            if vlm.enabled:
                images = _extract_images_from_paragraph(block, doc)
                for img_data in images:
                    image_count += 1
                    description = vlm.process_inline_image_sync(img_data)
                    if description:
                        text += f"\n\n[图片{image_count}: {description}]"

            if not text:
                continue
            level = _heading_level(block.style.name if block.style else None)
            parts.append(f"{'#' * level} {text}" if level else text)
        elif isinstance(block, Table):
            md = _table_to_markdown(block)
            if md:
                parts.append(md)

    if not parts:
        return []
    return [Document(page_content="\n\n".join(parts))]
