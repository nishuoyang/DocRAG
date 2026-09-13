"""Structure-aware parent-child chunking.

Children are optimized for retrieval while parents are reconstructed at query
time from sibling children to provide complete context to the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document

from config import get_settings

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_SENTENCE_END = "。！？；.!?;"
_CLOSING = "\"'”’」』）)]}"


@dataclass
class _Line:
    text: str
    page: int | None


@dataclass
class _Block:
    text: str
    kind: str
    section: str
    page_start: int | None
    page_end: int | None


def _line_page_range(lines: list[_Line]) -> tuple[int | None, int | None]:
    pages = [line.page for line in lines if line.page is not None]
    return (min(pages), max(pages)) if pages else (None, None)


def _iter_lines(docs: list[Document]):
    for doc in docs:
        page = doc.metadata.get("page")
        for line in doc.page_content.splitlines():
            yield _Line(text=line, page=page)


def _heading(line: str) -> tuple[int, str] | None:
    match = _HEADING_RE.match(line)
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def _is_block_start(lines: list[_Line], index: int) -> bool:
    if index >= len(lines):
        return True
    text = lines[index].text
    if not text.strip() or _heading(text) or _FENCE_RE.match(text) or _LIST_RE.match(text):
        return True
    return bool(_TABLE_LINE_RE.match(text))


def _collect_blocks(docs: list[Document]) -> list[_Block]:
    lines = list(_iter_lines(docs))
    blocks: list[_Block] = []
    headings: list[tuple[int, str]] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.text.strip():
            index += 1
            continue

        heading = _heading(line.text)
        if heading:
            level, title = heading
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
            index += 1
            continue

        section = " > ".join(title for _, title in headings)
        fence = _FENCE_RE.match(line.text)
        if fence:
            marker = fence.group(1)
            collected = [line]
            index += 1
            while index < len(lines):
                collected.append(lines[index])
                if lines[index].text.strip().startswith(marker):
                    index += 1
                    break
                index += 1
            page_start, page_end = _line_page_range(collected)
            blocks.append(
                _Block(
                    text="\n".join(item.text for item in collected),
                    kind="code",
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            continue

        if _TABLE_LINE_RE.match(line.text):
            collected = [line]
            index += 1
            while index < len(lines) and _TABLE_LINE_RE.match(lines[index].text):
                collected.append(lines[index])
                index += 1
            page_start, page_end = _line_page_range(collected)
            blocks.append(
                _Block(
                    text="\n".join(item.text for item in collected),
                    kind="table",
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            continue

        if _LIST_RE.match(line.text):
            collected = [line]
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if not candidate.text.strip():
                    break
                if _LIST_RE.match(candidate.text) or candidate.text.startswith((" ", "\t")):
                    collected.append(candidate)
                    index += 1
                    continue
                break
            page_start, page_end = _line_page_range(collected)
            blocks.append(
                _Block(
                    text="\n".join(item.text for item in collected),
                    kind="list",
                    section=section,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            continue

        collected = [line]
        index += 1
        while index < len(lines) and not _is_block_start(lines, index):
            collected.append(lines[index])
            index += 1
        page_start, page_end = _line_page_range(collected)
        blocks.append(
            _Block(
                text="\n".join(item.text for item in collected).strip(),
                kind="text",
                section=section,
                page_start=page_start,
                page_end=page_end,
            )
        )

    return [block for block in blocks if block.text.strip()]


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] in _SENTENCE_END:
            end = index + 1
            while end < len(text) and text[end] in _CLOSING:
                end += 1
            piece = text[start:end].strip()
            if piece:
                sentences.append(piece)
            start = end
            index = end
            continue
        if text[index] == "\n":
            piece = text[start:index].strip()
            if piece:
                sentences.append(piece)
            start = index + 1
        index += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _pack_pieces(pieces: list[str], max_size: int, min_size: int = 0, separator: str = "\n") -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        extra = len(piece) + (len(separator) if current else 0)
        if current and current_len + extra > max_size:
            packed.append(separator.join(current))
            current = []
            current_len = 0
        current.append(piece)
        current_len += len(piece) + (len(separator) if len(current) > 1 else 0)

    if current:
        packed.append(separator.join(current))
    if min_size > 1 and len(packed) > 1 and len(packed[-1]) < min_size:
        packed[-2] = packed[-2] + separator + packed[-1]
        packed.pop()
    return packed


def _hard_split(text: str, max_size: int) -> list[str]:
    """Last-resort split for text without usable punctuation or whitespace."""
    return [text[index : index + max_size] for index in range(0, len(text), max_size)]


def _split_table(text: str, max_size: int) -> list[str]:
    lines = text.splitlines()
    has_separator = len(lines) > 1 and bool(_TABLE_SEP_RE.match(lines[1]))
    header = lines[:2] if has_separator else lines[:1]
    rows = lines[len(header) :]
    if not rows:
        return [text]

    header_text = "\n".join(header)
    header_len = len(header_text) + 1
    groups: list[list[str]] = []
    current_rows: list[str] = []
    current_len = header_len

    for row in rows:
        if current_rows and current_len + len(row) + 1 > max_size:
            groups.append(header + current_rows)
            current_rows = []
            current_len = header_len
        current_rows.append(row)
        current_len += len(row) + 1
    if current_rows:
        groups.append(header + current_rows)
    return ["\n".join(group) for group in groups]


def _split_code(text: str, max_size: int) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 2:
        return [text]
    opening = lines[0]
    closing = lines[-1]
    body: list[str] = []
    overhead = len(opening) + len(closing) + 2
    body_budget = max(50, max_size - overhead)
    for line in lines[1:-1]:
        body.extend(_hard_split(line, body_budget) if len(line) > body_budget else [line])
    groups = _pack_pieces(body, body_budget, separator="\n")
    return [f"{opening}\n{group}\n{closing}" for group in groups]


def _split_block(block: _Block, max_size: int) -> list[_Block]:
    if len(block.text) <= max_size:
        return [block]
    if block.kind == "table":
        pieces = _split_table(block.text, max_size)
    elif block.kind == "code":
        pieces = _split_code(block.text, max_size)
    elif block.kind == "list":
        lines: list[str] = []
        for line in block.text.splitlines():
            lines.extend(_hard_split(line, max_size) if len(line) > max_size else [line])
        pieces = _pack_pieces(lines, max_size)
    else:
        sentences: list[str] = []
        for sentence in _sentences(block.text):
            sentences.extend(_hard_split(sentence, max_size) if len(sentence) > max_size else [sentence])
        pieces = _pack_pieces(sentences, max_size, separator="")
    return [
        _Block(
            text=piece,
            kind=block.kind,
            section=block.section,
            page_start=block.page_start,
            page_end=block.page_end,
        )
        for piece in pieces
        if piece.strip()
    ]


def _group_parents(blocks: list[_Block], max_size: int) -> list[list[_Block]]:
    expanded: list[_Block] = []
    for block in blocks:
        expanded.extend(_split_block(block, max_size))

    groups: list[list[_Block]] = []
    current: list[_Block] = []
    current_size = 0
    current_section: str | None = None
    for block in expanded:
        if block.section != current_section or (current and current_size + len(block.text) > max_size):
            if current:
                groups.append(current)
            current = []
            current_size = 0
            current_section = block.section
        current.append(block)
        current_size += len(block.text)
    if current:
        groups.append(current)
    return groups


def _kind_for_group(group: list[_Block]) -> str:
    kinds = {block.kind for block in group}
    return next(iter(kinds)) if len(kinds) == 1 else "mixed"


def _page_range(group: list[_Block]) -> tuple[int | None, int | None]:
    starts = [block.page_start for block in group if block.page_start is not None]
    ends = [block.page_end for block in group if block.page_end is not None]
    return (min(starts), max(ends)) if starts else (None, None)


def build_parent_child_documents(
    docs: list[Document],
    filename: str,
    file_hash: str,
    upload_time: int,
) -> tuple[list[Document], int]:
    """Build retrieval children and metadata for virtual parent reconstruction."""
    settings = get_settings()
    blocks = _collect_blocks(docs)
    if not blocks:
        return [], 0

    parent_groups = _group_parents(blocks, settings.PARENT_CHUNK_SIZE)
    children: list[Document] = []
    global_child_index = 0

    for parent_index, parent_blocks in enumerate(parent_groups):
        parent_id = f"{file_hash[:16]}-p{parent_index:05d}"
        child_blocks: list[_Block] = []
        for block in parent_blocks:
            child_blocks.extend(_split_block(block, settings.CHILD_CHUNK_SIZE))

        packed: list[list[_Block]] = []
        current: list[_Block] = []
        current_size = 0
        for block in child_blocks:
            if current and current_size + len(block.text) > settings.CHILD_CHUNK_SIZE:
                packed.append(current)
                current = []
                current_size = 0
            current.append(block)
            current_size += len(block.text)
        if current:
            packed.append(current)

        if len(packed) > 1 and sum(len(block.text) for block in packed[-1]) < settings.CHILD_MIN_SIZE:
            packed[-2].extend(packed.pop())

        for child_index, group in enumerate(packed):
            raw_text = "\n\n".join(block.text for block in group)
            section = group[0].section
            page_start, page_end = _page_range(group)
            prefix = f"[章节: {section}]\n" if section else ""
            metadata = {
                "filename": filename,
                "chunk_index": global_child_index,
                "upload_time": upload_time,
                "chunk_type": "parent_child",
                "section": section,
                "content_type": _kind_for_group(group),
                "file_hash": file_hash,
                "parent_id": parent_id,
                "parent_index": parent_index,
                "child_index": child_index,
                "raw_text": raw_text,
                "page": page_start,
                "page_start": page_start,
                "page_end": page_end,
            }
            children.append(Document(page_content=prefix + raw_text, metadata=metadata))
            global_child_index += 1

    return children, len(parent_groups)
