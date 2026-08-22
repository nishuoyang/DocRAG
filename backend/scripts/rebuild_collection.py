"""Collection 迁移重建：备份旧数据 → 按新 schema 重建 → 无损迁移存量向量。

背景：Milvus collection 的 metadata schema 在建库时定死，本次升级新增
section / content_type / file_hash 三个字段，并将 page 显式声明为可空，
必须重建 collection。

用法（backend 目录下，主环境 venv）：
    ./.venv/Scripts/python.exe -X utf8 scripts/rebuild_collection.py

流程（无损）：
1. 读取旧 collection 全部块（含向量）
2. 备份到新 collection「<原名>_backup_<时间戳>」（保留原 schema，双保险）
3. 删除旧 collection
4. 用 add_embeddings 按新 schema 重建并回填（保留原向量，不重新 embedding）

说明：
- 本脚本只做 schema 升级，不改写已有块的向量与文本。
- 存量块的 section / content_type / file_hash 为空；它们是在旧解析器下入库的。
  如需让旧文档获得 v2 结构化元数据，请在前端删除后重新上传（走 v2 链路）。
- 新上传的文档自动走 v2 解析 + 新字段。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from db import milvus  # noqa: E402

# 旧 schema 中需要迁移的字段（不含 pk，auto_id 重新生成）
# page 一并迁移，避免升级后丢失 PDF 页码信息
_OLD_FIELDS = ["text", "vector", "filename", "chunk_index", "upload_time", "chunk_type", "page"]


def _read_all_rows(col, fields: list[str]) -> list[dict]:
    """分页读取 collection 全部行。"""
    rows: list[dict] = []
    offset, limit = 0, 100
    while True:
        batch = col.query(expr="", output_fields=fields, limit=limit, offset=offset)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return rows


def backup_old_collection() -> int:
    """把旧 collection 的全部数据复制到备份 collection，返回备份块数。"""
    vs = milvus.get_vectorstore()
    if vs.col is None:
        print("旧 collection 不存在，无需备份")
        return 0
    fields = {f.name for f in vs.col.schema.fields}
    if "vector" not in fields:
        print("旧 collection 无 vector 字段，跳过备份")
        return 0

    backup_name = f"{milvus.get_collection_name()}_backup_{int(time.time())}"
    settings = get_settings()
    from pymilvus import Collection, connections

    connections.connect(alias="backup", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    backup = Collection(backup_name, schema=vs.col.schema, using="backup")

    # insert 时必须跳过 auto_id 主键，否则列数不匹配
    insert_fields = [f.name for f in vs.col.schema.fields if not f.auto_id]
    rows = _read_all_rows(vs.col, insert_fields)
    if rows:
        columns = [[item.get(f) for item in rows] for f in insert_fields]
        backup.insert(columns)
    backup.flush()
    print(f"已备份 {len(rows)} 块 → {backup_name}（字段: {insert_fields}）")
    return len(rows)


def migrate_to_new_schema() -> int:
    """删旧 collection → 用 add_embeddings 按新 schema 重建 → 回填存量向量。

    使用 add_embeddings 而非直接 col.insert，让 langchain-milvus 按当前
    metadata_schema（含新字段）自动创建 collection 并建索引。
    """
    vs = milvus.get_vectorstore()
    if vs.col is None:
        print("无旧 collection，跳过迁移（首次使用将直接创建新 schema）")
        return 0

    old_fields = {f.name for f in vs.col.schema.fields}
    has_new_fields = {"section", "content_type", "file_hash"} <= old_fields
    if has_new_fields:
        print("collection 已含新字段，无需迁移")
        return 0

    # 读取存量数据（只取旧字段，新字段回填为空）
    fields_to_read = [f for f in _OLD_FIELDS if f in old_fields]
    rows = _read_all_rows(vs.col, fields_to_read)
    print(f"读取存量 {len(rows)} 块，准备回填…")

    if not rows:
        print("无存量数据，仅删除旧 collection")
        vs.col.drop()
        vs.col = None
        return 0

    # 删除旧 collection
    vs.col.drop()
    vs.col = None
    print("旧 collection 已删除")

    # 用 add_embeddings 重建：langchain-milvus 会按 metadata_schema 自动建
    # 新 collection（含 section/content_type/file_hash/page），并建向量索引
    new_vs = milvus.get_vectorstore()
    texts = [item.get("text", "") for item in rows]
    embeddings = [item.get("vector", []) for item in rows]
    metadatas = []
    for item in rows:
        meta = {
            "filename": item.get("filename", ""),
            "chunk_index": item.get("chunk_index", 0),
            "upload_time": item.get("upload_time", 0),
            "chunk_type": item.get("chunk_type", ""),
            # 新字段留空
            "section": "",
            "content_type": "",
            "file_hash": "",
            # page 必须显式带键（哪怕是 None），让 langchain-milvus 推断出 INT64 字段
            "page": item.get("page"),
        }
        metadatas.append(meta)

    new_vs.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas)
    new_vs.col.flush()
    new_fields = [f.name for f in new_vs.col.schema.fields]
    print(f"已按新 schema 回填 {len(rows)} 块（新字段留空），字段: {new_fields}")
    return len(rows)


def main() -> None:
    settings = get_settings()
    print(f"目标 collection: {milvus.get_collection_name()}")
    print(f"Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
    backup_old_collection()
    migrate_to_new_schema()
    print("迁移完成")


if __name__ == "__main__":
    main()
