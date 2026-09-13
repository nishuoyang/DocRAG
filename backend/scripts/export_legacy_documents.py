"""Export legacy collection document names without modifying Milvus data."""
import sys
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    pass

from db import milvus  # noqa: E402


def main() -> None:
    collection = milvus._get_collection(milvus.get_legacy_collection_name())
    if collection is None:
        print("旧 collection 不存在。")
        return

    fields = {field.name for field in collection.schema.fields}
    output_fields = [field for field in ("filename", "file_hash", "upload_time") if field in fields]
    rows = collection.query(expr="", output_fields=output_fields, limit=16384)
    grouped: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"chunks": 0, "upload_time": 0}
    )
    for row in rows:
        key = (row.get("filename", ""), row.get("file_hash", ""))
        grouped[key]["chunks"] += 1
        grouped[key]["upload_time"] = max(
            grouped[key]["upload_time"],
            row.get("upload_time") or 0,
        )

    print(f"collection: {milvus.get_legacy_collection_name()}")
    print(f"documents: {len(grouped)}  chunks: {len(rows)}")
    print("filename\tfile_hash\tchunks\tupload_time")
    for (filename, file_hash), info in sorted(grouped.items()):
        print(f"{filename}\t{file_hash}\t{info['chunks']}\t{info['upload_time']}")


if __name__ == "__main__":
    main()
