# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

基于 LangChain 的垂直领域智能文档问答系统（RAG）：上传 PDF/DOCX → 分块向量化存入 Milvus → 检索相似块 → LLM 生成带来源的回答。前后端分离：`backend/`（FastAPI）+ `frontend/`（Vue 3）。LLM 和 Embedding 均用 OpenAI 兼容的云 API（默认硅基流动 SiliconFlow），不运行本地模型。

## 常用命令

工作目录通常是 `backend/`（后端）或 `frontend/`（前端）。

```bash
# 基础设施：Milvus 向量库（etcd + MinIO，Docker Compose）
docker compose up -d

# 后端（依赖由 Poetry 管理，venv 在 backend/.venv）
cd backend && ./.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# 后端 lint
cd backend && ./.venv/Scripts/ruff.exe check .

# 前端（Vite dev server，代理 /health /documents /chat 到 8000）
cd frontend && npm run dev      # http://localhost:5173

# 前端生产构建（产物 frontend/dist/，由 FastAPI 静态托管）
cd frontend && npm run build

# 测试：无测试套件。手工验证链路：curl /health → POST /documents/upload → POST /chat/stream
```

环境配置从 `.env` 读取（参考 `backend/.env.example`，含 Milvus 地址、LLM/Embedding 的 API Key、模型名、分块参数）。API Key 不提交进 git。

## 架构

### 数据流

```
上传 → ingestion.py（解析 → 分块 → 补元数据）→ milvus.py（写入向量）
提问 → retrieval.py（向量检索 Top-K → 拼系统 Prompt）→ llm.py（ChatOpenAI）→ SSE 流式回答 + 来源
```

### 后端 `backend/`

- `main.py` — FastAPI 入口。CORS 全开、无认证；启动时校验 Milvus 维度。**静态托管前端 dist 的 mount 必须放在所有 API 路由之后**，否则会抢占 `/health` 等路径
- `api/documents.py` / `api/chat.py` — 路由层。6 个 endpoint：`GET /health`、`POST /documents/upload`、`GET /documents`、`DELETE /documents/{filename}`、`POST /chat`、`POST /chat/stream`（SSE）
- `core/retrieval.py` — 检索 + RAG 生成。`_build_messages` 拼 system prompt（含参考资料）和用户问题；多轮记忆靠前端携带 `history`（最多保留最近 6 轮，`MAX_HISTORY_TURNS`）
- `core/ingestion.py` — 文件解析（PyPDFLoader / Docx2txtLoader）、分块（RecursiveCharacterTextSplitter，`CHUNK_SIZE=500`/`CHUNK_OVERLAP=50`）、写入临时文件后删除
- `db/milvus.py` — Milvus 封装。Collection 以 Embedding 模型名命名（换模型避免维度冲突）；`vs.col` 是底层 Collection 对象；删除按 `filename` 字段匹配；`ensure_collection_ready` 校验向量维度与配置一致
- `models/schemas.py` — Pydantic 模型（请求/响应，含 Swagger 描述）
- `config.py` — `Settings`（pydantic-settings）+ `get_settings()` 缓存

### 前端 `frontend/`

- Vue 3 `<script setup>` + Vite + Tailwind CSS v4（`@tailwindcss/vite` 插件，无 tailwind.config.js）
- 无 Vue Router / Pinia / TypeScript — 单页操作台
- `src/App.vue` — 左侧导航（聊天 / 文档管理）切换右侧面板
- `src/api.js` — fetch 封装。`chatStream` 解析 SSE：`data: {json}\n\n` 事件流，支持 `delta`（流式文本）、`sources`（引用）、`answer`（空库兜底）、`[DONE]`
- 组件在 `src/components/`：`ChatPanel.vue`（流式对话 + Top-K 滑块）、`DocManager.vue`（拖拽上传 + 列表 + 删除）、`SourceCard.vue`（来源展开卡片）

## 关键约束与已知问题

- **pymilvus 必须锁定 `>=2.5.5,<2.6.0`** — 2.6.x 与 langchain-milvus 不兼容，升上去会崩
- Milvus v3.0-beta 是官方 Docker 镜像；langchain_milvus 内部字段名是 `pk`/`text`/`col` 属性（非 `collection`）
- PDF 的 `page` 元数据是 int，docx 无 page（`None`）；schemas.py 里 `page: int | None`
- 上传同一文件两次会重复入库（无去重），删除按文件名精确匹配
- 对话历史存在前端内存，刷新即失（后端无状态）
- 上传失败无回滚（临时文件已清理，但部分写入的向量残留）
