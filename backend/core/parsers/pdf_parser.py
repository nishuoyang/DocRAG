"""PDF 解析（v2）：pymupdf4llm 逐页输出 Markdown，表格自动转 GFM。

P2 增强：
- 扫描件检测：页面文本量过低时渲染为图片，调用 VLM 读图转 Markdown
- 内嵌图片提取：提取页面内嵌图片，调用 VLM 描述后替换占位符
"""
import logging
from pathlib import Path

import pymupdf
import pymupdf4llm
from langchain_core.documents import Document

from core.vlm import get_vlm_client, process_page_image_sync, process_inline_image_sync

logger = logging.getLogger(__name__)

# 扫描件判定阈值：页面文本字符数低于此值且有图片时，视为扫描件
SCANNED_PAGE_THRESHOLD = 50


def _is_scanned_page(page: pymupdf.Page, text: str) -> bool:
    """判断页面是否为扫描件（文本量过低且有图片）。"""
    if len(text.strip()) > SCANNED_PAGE_THRESHOLD:
        return False
    # 检查页面是否有图片
    images = page.get_images(full=True)
    return len(images) > 0


def _render_page_to_image(page: pymupdf.Page, dpi: int = 200) -> bytes:
    """将页面渲染为 PNG 图片。"""
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def _extract_page_images(page: pymupdf.Page, doc: pymupdf.Document) -> list[tuple[str, bytes]]:
    """提取页面内嵌图片，返回 (占位符, 图片数据) 列表。"""
    images = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            if base_image:
                img_bytes = base_image["image"]
                placeholder = f"[图片{img_index + 1}]"
                images.append((placeholder, img_bytes))
        except Exception as e:
            logger.warning(f"提取图片失败 (xref={xref}): {e}")
    return images


def parse_pdf(file_path: str) -> list[Document]:
    """逐页解析 PDF 为 Markdown 文档块。

    - 表格自动转为 GFM Markdown 表格（pymupdf4llm 内置表格识别）
    - 标题按字号启发式带 # 层级
    - metadata: page（1 基页码）
    - P2 增强：扫描件页面渲染为图片调用 VLM，内嵌图片调用 VLM 描述
    空页跳过。从字节流打开文档并显式关闭，避免锁定文件句柄
    （Windows 上会导致临时文件 unlink 失败）。
    """
    data = Path(file_path).read_bytes()
    doc = pymupdf.open(stream=data, filetype="pdf")
    vlm = get_vlm_client()
    docs: list[Document] = []

    try:
        for page_idx, page in enumerate(doc):
            page_no = page_idx + 1
            # 先用 pymupdf4llm 尝试解析
            page_md = pymupdf4llm.to_markdown(doc, pages=[page_idx], show_progress=False)
            text = page_md.strip()

            # 检测扫描件页面
            if _is_scanned_page(page, text) and vlm.enabled:
                logger.info(f"检测到扫描件页面 {page_no}，调用 VLM 读图")
                img_bytes = _render_page_to_image(page)
                vlm_text = process_page_image_sync(img_bytes)
                if vlm_text:
                    text = vlm_text
                    logger.info(f"VLM 页面读图成功 (page={page_no})")

            # 提取内嵌图片并调用 VLM 描述
            if vlm.enabled and "[图片" in text or vlm.enabled and page.get_images(full=True):
                images = _extract_page_images(page, doc)
                for placeholder, img_bytes in images:
                    description = process_inline_image_sync(img_bytes)
                    if description:
                        # 替换占位符或追加描述
                        if placeholder in text:
                            text = text.replace(placeholder, f"[图片: {description}]")
                        else:
                            text += f"\n\n[图片: {description}]"

            if text:
                docs.append(Document(page_content=text, metadata={"page": page_no}))

    finally:
        doc.close()

    logger.info("PDF 解析完成: %d 页有内容", len(docs))
    return docs
