"""文档管理 API：上传、列表、删除。"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from config import get_settings
from core import ingestion
from models.schemas import DocumentListResponse, DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["文档管理"])


@router.post("/upload", response_model=DocumentUploadResponse, summary="上传文档并入库")
async def upload_document(file: UploadFile = File(description="要上传的文档文件，支持 PDF / DOCX")):
    """上传 PDF 或 DOCX 文档，自动解析、分块、向量化后写入 Milvus。"""
    settings = get_settings()
    filename = file.filename or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="仅支持 PDF / DOCX 文件")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过 {settings.MAX_UPLOAD_MB}MB 限制",
        )

    try:
        result = ingestion.ingest_file(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {exc}") from exc

    return DocumentUploadResponse(
        filename=result["filename"],
        chunk_count=result["chunk_count"],
        ids=result["ids"],
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
