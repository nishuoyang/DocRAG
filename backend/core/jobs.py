"""后台任务管理模块：追踪文档处理任务的状态和进度。"""
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    """任务状态枚举。"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    """任务信息模型。"""
    job_id: str
    filename: str
    status: JobStatus
    progress: int  # 0-100
    message: str
    created_at: str
    updated_at: str
    result: Optional[dict] = None
    error: Optional[str] = None


class JobManager:
    """后台任务管理器。"""
    
    def __init__(self, db_path: str = "jobs.db"):
        """初始化任务管理器。
        
        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = Path(db_path)
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接。"""
        return sqlite3.connect(self.db_path)
    
    def _init_db(self):
        """初始化数据库表。"""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    result TEXT,
                    error TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    def create_job(self, filename: str) -> str:
        """创建新任务。
        
        Args:
            filename: 文件名
            
        Returns:
            job_id: 任务 ID
        """
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO jobs (job_id, filename, status, progress, message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, filename, JobStatus.PENDING, 0, "任务已创建", now, now)
            )
            conn.commit()
        finally:
            conn.close()
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[JobInfo]:
        """获取任务信息。
        
        Args:
            job_id: 任务 ID
            
        Returns:
            JobInfo 或 None
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                SELECT job_id, filename, status, progress, message, 
                       created_at, updated_at, result, error
                FROM jobs WHERE job_id = ?
                """,
                (job_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            result = json.loads(row[7]) if row[7] else None
            
            return JobInfo(
                job_id=row[0],
                filename=row[1],
                status=JobStatus(row[2]),
                progress=row[3],
                message=row[4],
                created_at=row[5],
                updated_at=row[6],
                result=result,
                error=row[8]
            )
        finally:
            conn.close()
    
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[dict] = None,
        error: Optional[str] = None
    ):
        """更新任务信息。
        
        Args:
            job_id: 任务 ID
            status: 新状态
            progress: 新进度（0-100）
            message: 新消息
            result: 任务结果
            error: 错误信息
        """
        now = datetime.now().isoformat()
        updates = ["updated_at = ?"]
        params = [now]
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        
        if message is not None:
            updates.append("message = ?")
            params.append(message)
        
        if result is not None:
            updates.append("result = ?")
            params.append(json.dumps(result))
        
        if error is not None:
            updates.append("error = ?")
            params.append(error)
        
        params.append(job_id)
        
        conn = self._get_conn()
        try:
            conn.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?",
                params
            )
            conn.commit()
        finally:
            conn.close()
    
    def list_jobs(self, limit: int = 50) -> list[JobInfo]:
        """列出最近的任务。
        
        Args:
            limit: 最大返回数量
            
        Returns:
            JobInfo 列表
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """
                SELECT job_id, filename, status, progress, message, 
                       created_at, updated_at, result, error
                FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,)
            )
            
            jobs = []
            for row in cursor:
                result = json.loads(row[7]) if row[7] else None
                jobs.append(
                    JobInfo(
                        job_id=row[0],
                        filename=row[1],
                        status=JobStatus(row[2]),
                        progress=row[3],
                        message=row[4],
                        created_at=row[5],
                        updated_at=row[6],
                        result=result,
                        error=row[8]
                    )
                )
            
            return jobs
        finally:
            conn.close()
    
    def delete_old_jobs(self, max_age_hours: int = 24):
        """删除旧任务。
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        
        conn = self._get_conn()
        try:
            conn.execute(
                "DELETE FROM jobs WHERE created_at < ?",
                (cutoff,)
            )
            conn.commit()
        finally:
            conn.close()


# 全局实例
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """获取全局任务管理器实例。"""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
