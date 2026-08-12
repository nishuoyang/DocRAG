"""查看 Milvus 文档库中的块内容（排查检索问题、检查分块质量用）。

运行（在 backend/ 下，用主环境 venv，无需 ragas 依赖）：
    ./.venv/Scripts/python.exe scripts/inspect_chunks.py                  # 列出前 20 块摘要
    ./.venv/Scripts/python.exe scripts/inspect_chunks.py --limit 50       # 列 50 块
    ./.venv/Scripts/python.exe scripts/inspect_chunks.py --index 3        # 查看第 3 块完整内容
    ./.venv/Scripts/python.exe scripts/inspect_chunks.py --index 3,7,12   # 多个块
    ./.venv/Scripts/python.exe scripts/inspect_chunks.py --file 尚硅谷-02.pdf  # 按文件名过滤
    ./.venv/Scripts/python.exe scripts/inspect_chunks.py --query 模板库    # 按内容关键词搜索
    ./.venv/Scripts/python.exe scripts/inspect_chunks.py --full           # 显示完整内容（默认截断 200 字）

说明：
- 块索引是 get_all_documents() 的返回顺序（Milvus 存储顺序），不是检索排序
- 只看块（不写库、不调 LLM），零成本
"""
import argparse
import sys
from pathlib import Path

# 把 backend/ 加入 sys.path，复用项目代码与 .env 配置
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from db import milvus  # noqa: E402

PREVIEW_LEN = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查看 Milvus 文档库中的块内容（索引/文件/关键词筛选）"
    )
    parser.add_argument("--index", type=str, default="", help="块索引（从 1 开始，逗号分隔，如 3 或 3,7,12）")
    parser.add_argument("--file", type=str, default="", help="按文件名过滤（子串匹配）")
    parser.add_argument("--query", type=str, default="", help="按块内容关键词过滤（子串匹配）")
    parser.add_argument("--limit", type=int, default=20, help="最多显示块数（默认 20，0=全部）")
    parser.add_argument("--full", action="store_true", help="显示完整内容（默认截断 200 字）")
    return parser.parse_args()


def _preview(text: str, full: bool) -> str:
    if full or len(text) <= PREVIEW_LEN:
        return text
    return text[:PREVIEW_LEN] + f"…（共 {len(text)} 字，--full 查看完整）"


def _print_chunk(i: int, doc, full: bool) -> None:
    meta = doc.metadata
    print(f"{'=' * 70}")
    print(f"块 #{i}  来源: {meta.get('filename', '?')}  块序号: {meta.get('chunk_index', '?')}  "
          f"类型: {meta.get('chunk_type', '?')}  page: {meta.get('page', '-')}  字数: {len(doc.page_content)}")
    print("-" * 70)
    print(_preview(doc.page_content, full))
    print()


def main() -> None:
    args = parse_args()
    docs = milvus.get_all_documents()
    if not docs:
        sys.exit("文档库为空，请先上传文档。")
    print(f"文档库共 {len(docs)} 块")

    # 筛选
    selected = list(enumerate(docs, 1))  # (索引, doc)，索引从 1 开始
    if args.index:
        try:
            idxs = {int(i) for i in args.index.split(",") if i.strip()}
        except ValueError:
            sys.exit("--index 格式错误，应为逗号分隔的数字，如 3 或 3,7,12")
        bad = [i for i in idxs if i < 1 or i > len(docs)]
        if bad:
            print(f"  警告: 索引 {bad} 超出范围（1-{len(docs)}），已忽略")
        selected = [(i, d) for i, d in selected if i in idxs]
    if args.file:
        selected = [(i, d) for i, d in selected if args.file in str(d.metadata.get("filename", ""))]
    if args.query:
        selected = [(i, d) for i, d in selected if args.query in d.page_content]

    # 输出
    shown = len(selected) if args.limit == 0 else min(len(selected), args.limit)
    print(f"匹配 {len(selected)} 块" + (f"，显示前 {shown} 块" if shown < len(selected) else ""))
    print()
    for i, d in selected[:shown]:
        _print_chunk(i, d, args.full)


if __name__ == "__main__":
    main()
