"""FastAPI 应用入口。"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import agents, chat, documents
from db import milvus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        milvus.ensure_collection_ready()
        logger.info("Milvus 连接成功，Collection 就绪")
    except Exception as exc:
        logger.warning("Milvus 未就绪（请确认已执行 docker compose up -d）: %s", exc)
    yield


app = FastAPI(
    title="垂直领域智能文档问答系统",
    description="基于 LangChain + Milvus 的 RAG 问答后端。上传 PDF/DOCX 文档后即可基于文档内容提问。",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(agents.router)


@app.get("/health", tags=["系统"])
async def health():
    """健康检查。"""
    return {"status": "ok"}


# 生产模式：托管前端构建产物（frontend/dist 存在时生效）。
# 注意：必须放在所有 API 路由之后，否则会抢占 /health 等路径。
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
