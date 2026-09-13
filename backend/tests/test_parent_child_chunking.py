from collections import defaultdict

from langchain_core.documents import Document

from core.parent_child import build_parent_child_documents


def _build(docs: list[Document], filename: str = "sample.md"):
    return build_parent_child_documents(
        docs,
        filename=filename,
        file_hash="a" * 64,
        upload_time=123,
    )


def test_chinese_text_children_stay_on_sentence_boundaries():
    text = "。".join(f"第{i}句是用于验证中文句子边界的内容" for i in range(40)) + "。"
    docs = [Document(page_content=text)]

    children, _ = _build(docs)

    assert len(children) > 1
    for child in children:
        raw = child.metadata["raw_text"].rstrip()
        assert raw.endswith(("。", "！", "？", "；", "}", "`"))


def test_unbroken_text_uses_character_fallback():
    docs = [Document(page_content="A" * 1200)]

    children, _ = _build(docs)

    assert len(children) >= 3
    assert all(len(d.metadata["raw_text"]) <= 450 for d in children)


def test_large_table_splits_by_rows_and_repeats_header():
    lines = [
        "| 名称 | 数值 | 说明 |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| 项目{i} | {i} | 这是一段足够长的表格说明{i} |" for i in range(80))
    docs = [Document(page_content="# 指标\n\n" + "\n".join(lines))]

    children, _ = _build(docs)
    tables = [d for d in children if d.metadata["content_type"] == "table"]

    assert len(tables) > 1
    for table in tables:
        assert table.metadata["raw_text"].startswith("| 名称 | 数值 | 说明 |")
        assert "| --- | --- | --- |" in table.metadata["raw_text"]


def test_large_code_block_repeats_fence_when_split():
    body = "\n".join(f"print('line {i}')  # 保留代码边界" for i in range(80))
    docs = [Document(page_content="# 示例\n\n```python\n" + body + "\n```")]

    children, _ = _build(docs)
    code = [d for d in children if d.metadata["content_type"] == "code"]

    assert len(code) > 1
    for child in code:
        raw = child.metadata["raw_text"]
        assert raw.startswith("```python")
        assert raw.endswith("```")


def test_section_and_page_range_follow_content_across_pages():
    docs = [
        Document(page_content="# 第一章\n\n第一页介绍。", metadata={"page": 1}),
        Document(page_content="第二页继续同一章节。" * 80, metadata={"page": 2}),
        Document(page_content="第三页补充。" * 60, metadata={"page": 3}),
    ]

    children, _ = _build(docs, filename="sample.pdf")

    assert len(children) >= 3
    assert all(d.metadata["section"] == "第一章" for d in children)
    assert children[0].metadata["page_start"] == 1
    assert children[-1].metadata["page_end"] == 3
    assert all(d.metadata["page"] == d.metadata["page_start"] for d in children)


def test_every_child_has_stable_parent_and_reconstructable_raw_text():
    docs = [
        Document(
            page_content=(
                "# 第一节\n\n"
                + "第一节正文。" * 120
                + "\n\n## 子节\n\n"
                + "子节正文。" * 120
            )
        )
    ]

    children, parent_count = _build(docs)
    by_parent = defaultdict(list)
    for child in children:
        meta = child.metadata
        assert meta["parent_id"].startswith("a" * 16)
        assert meta["child_index"] >= 0
        assert meta["parent_index"] >= 0
        assert meta["raw_text"]
        by_parent[meta["parent_id"]].append(child)

    assert parent_count == len(by_parent)
    for group in by_parent.values():
        ordered = sorted(group, key=lambda d: d.metadata["child_index"])
        rebuilt = "\n\n".join(d.metadata["raw_text"] for d in ordered)
        assert rebuilt.strip()
