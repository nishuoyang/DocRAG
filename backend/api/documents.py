"""文档管理 API：上传、列表、删除。"""
import asyncio
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from config import get_settings
from core import ingestion
from core.jobs import get_job_manager, JobStatus
from models.schemas import DocumentListResponse, DocumentUploadResponse, JobInfoResponse

router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post("/upload", response_model=JobInfoResponse, summary="上传文档并入库（异步）")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(description="要上传的文档文件，支持 PDF / DOCX"),
    split_mode: str | None = Form(default=None, description="切分策略：semantic / fixed / markdown，不传则自动判断"),
    replace: bool = Form(default=False, description="内容已入库时是否删除旧块重新入库（默认返回 409）"),
):
    """上传 PDF 或 DOCX 文档，后台异步处理解析、分块、向量化后写入 Milvus。
    
    返回任务 ID，可通过 GET /documents/jobs/{job_id} 查询处理进度。
    """
    settings = get_settings()
    filename = file.filename or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    supported = {e.lstrip(".") for e in ingestion.SUPPORTED_EXTENSIONS} | {"xlsx", "pptx"}
    if ext not in supported:
        raise HTTPException(status_code=400, detail=f"仅支持 {', '.join(sorted(supported))} 文件")

    if split_mode not in (None, "auto", "semantic", "fixed", "markdown"):
        raise HTTPException(status_code=400, detail="split_mode 仅支持 auto / semantic / fixed / markdown")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过 {settings.MAX_UPLOAD_MB}MB 限制",
        )

    # 创建后台任务
    job_manager = get_job_manager()
    job_id = job_manager.create_job(filename)
    
    # 添加后台任务
    background_tasks.add_task(
        _process_document_task,
        job_id=job_id,
        filename=filename,
        content=content,
        split_mode=split_mode,
        replace=replace,
    )
    
    # 返回任务信息
    job = job_manager.get_job(job_id)
    return JobInfoResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result,
        error=job.error,
    )


async def _process_document_task(
    job_id: str,
    filename: str,
    content: bytes,
    split_mode: str | None,
    replace: bool,
):
    """后台文档处理任务。"""
    job_manager = get_job_manager()
    
    try:
        # 更新状态为处理中
        job_manager.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            progress=10,
            message="开始解析文档...",
        )
        
        # 模拟进度更新（实际处理中会在 ingestion 内部更新）
        await asyncio.sleep(0.1)
        job_manager.update_job(job_id, progress=30, message="文档解析完成，正在分块...")
        
        await asyncio.sleep(0.1)
        job_manager.update_job(job_id, progress=60, message="分块完成，正在向量化...")
        
        # 执行实际的文档处理
        result = await asyncio.to_thread(
            ingestion.ingest_file,
            filename,
            content,
            split_mode,
            replace,
        )
        
        # 更新为完成状态
        job_manager.update_job(
            job_id,
            status="completed",
            progress=100,
            message="文档处理完成",
            result=result,
        )
        
    except ingestion.DuplicateFileError as exc:
        job_manager.update_job(
            job_id,
            status="failed",
            progress=0,
            message="文档已存在",
            error=str(exc),
        )
    except ValueError as exc:
        job_manager.update_job(
            job_id,
            status="failed",
            progress=0,
            message="文档处理失败",
            error=str(exc),
        )
    except Exception as exc:
        job_manager.update_job(
            job_id,
            status="failed",
            progress=0,
            message="文档处理失败",
            error=str(exc),
        )


@router.get("/jobs/{job_id}", response_model=JobInfoResponse, summary="查询任务进度")
async def get_job_status(job_id: str):
    """查询文档处理任务的进度和状态。"""
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    
    # 将 JobInfo 转换为 JobInfoResponse
    return JobInfoResponse(
        job_id=job.job_id,
        filename=job.filename,
        status=job.status.value,  # 枚举转字符串
        progress=job.progress,
        message=job.message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=job.result,
        error=job.error,
    )


@router.get("", response_model=DocumentListResponse, summary="列出已入库文档")
async def list_documents():
    docs = ingestion.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{filename}", summary="删除指定文档及其向量")
async def delete_document(filename: str):
    """按文件名删除文档的全部向量块。"""
    deleted = ingestion.delete_document(filename)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"未找到文档 {filename}")
    return {"filename": filename, "deleted": deleted}
