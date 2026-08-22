"""解析噪声清洗：跨页重复行（页眉/页脚/页码）剔除。"""
import re
from collections import defaultdict

from langchain_core.documents import Document

# 页眉页脚判定：归一化后出现在超过该比例页面上的行
REPEAT_RATIO = 0.4
# 至少这么多页才做判定（页数太少时误判率高）
MIN_PAGES = 3
# 过长的行不可能是页眉页脚
MAX_LINE_LEN = 80

_NORMALIZE = re.compile(r"\s+")
# 分页分隔线（pymupdf4llm 在每页末尾输出 -----），无信息量
_SEPARATOR_RE = re.compile(r"^\s*[-=_*]{3,}\s*$")


def strip_separators(text: str) -> str:
    """剔除 Markdown 分页分隔线（-----），返回清洗后文本。"""
    return "\n".join(line for line in text.splitlines() if not _SEPARATOR_RE.match(line)).strip()


def _normalize(line: str) -> str:
    """归一化：去空白、去数字（页码）、转小写。"""
    s = _NORMALIZE.sub("", line).lower()
    return re.sub(r"\d+", "", s)


def _is_protected(line: str) -> bool:
    """表格行与标题行不参与页眉页脚判定，避免误删。"""
    stripped = line.strip()
    return stripped.startswith("|") or stripped.startswith("#")


def strip_repeated_lines(docs: list[Document]) -> list[Document]:
    """剔除跨页重复出现的页眉/页脚/页码行，返回新文档列表。

    仅对带 page 元数据且页数 >= MIN_PAGES 的输入生效（PDF 逐页块）；
    其余情况原样返回。
    """
    paged = [d for d in docs if d.metadata.get("page") is not None]
    if len(paged) < MIN_PAGES:
        return docs
    total_pages = len(paged)

    # 统计每页出现的归一化行（同一页内去重计数）
    line_pages: dict[str, set[int]] = defaultdict(set)
    for doc in paged:
        page = doc.metadata["page"]
        seen_on_page: set[str] = set()
        for line in doc.page_content.splitlines():
            if _is_protected(line) or len(line.strip()) > MAX_LINE_LEN:
                continue
            key = _normalize(line)
            if len(key) >= 2 and key not in seen_on_page:
                seen_on_page.add(key)
                line_pages[key].add(page)

    noise = {key for key, pages in line_pages.items() if len(pages) > total_pages * REPEAT_RATIO}
    if not noise:
        return docs

    result: list[Document] = []
    for doc in docs:
        if doc.metadata.get("page") is None:
            result.append(doc)
            continue
        kept = []
        for line in doc.page_content.splitlines():
            if not _is_protected(line) and len(line.strip()) <= MAX_LINE_LEN and _normalize(line) in noise:
                continue
            kept.append(line)
        result.append(Document(page_content="\n".join(kept).strip(), metadata=doc.metadata))
    return result
