"""对话记忆持久化：SQLite 单表存储，轻量无依赖。

存储最近 N 轮对话（MAX_MESSAGES=100 条 = 50 轮），供刷新页面/重启后端后恢复。
单连接 + check_same_thread=False：FastAPI 单进程内访问安全。
"""
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "chat_memory.db"
MAX_MESSAGES = 100  # 50 轮 × 2 条

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute(
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at REAL NOT NULL
    )"""
)
_conn.commit()


def load_messages() -> list[dict]:
    """按时间升序返回全部历史消息（旧 → 新）。"""
    rows = _conn.execute(
        "SELECT role, content FROM messages ORDER BY created_at, id"
    ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows]


def add_message(role: str, content: str) -> None:
    """写入一条消息；超上限时删除最旧的。"""
    _conn.execute(
        "INSERT INTO messages (role, content, created_at) VALUES (?, ?, ?)",
        (role, content, time.time()),
    )
    _conn.execute(
        f"DELETE FROM messages WHERE id IN ("
        f"SELECT id FROM messages ORDER BY id LIMIT "
        f"MAX(0, (SELECT COUNT(*) FROM messages) - {MAX_MESSAGES}))"
    )
    _conn.commit()


def clear_messages() -> None:
    """清空全部记忆（调试/重置用）。"""
    _conn.execute("DELETE FROM messages")
    _conn.commit()
