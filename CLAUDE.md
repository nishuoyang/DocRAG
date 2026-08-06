# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

基于 LangChain 的垂直领域智能文档问答系统（RAG）：上传 PDF/DOCX 等 8 种格式 → 分块向量化存入 Milvus → 混合检索（向量+BM25+Query增强+rerank）→ LLM 生成带来源的回答。前后端分离：`backend/`（FastAPI）+ `frontend/`（Vue 3）。LLM/Embedding/Rerank 均用 OpenAI 兼容的云 API（Embedding/Rerank 走硅基流动 SiliconFlow，LLM 走 DeepSeek 官方），不运行本地模型。

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
cd frontend && npm run dev      # http://localhost:5173（注意用 localhost 而非 127.0.0.1，Vite 默认只监听 IPv6）

# 前端生产构建（产物 frontend/dist/，由 FastAPI 静态托管）
cd frontend && npm run build

# 测试：无测试套件。手工验证链路：curl /health → POST /documents/upload → POST /chat/stream
# 清空对话记忆：删 backend/chat_memory.db（或在 Python 里调 db.memory.clear_messages()）

# RAGAS 评估（独立 venv，不污染主环境；生成结果缓存在 backend/.ragas_cache.json）
cd backend && ./.venv-ragas/Scripts/python.exe -X utf8 scripts/eval_ragas.py
#   RAGAS_TESTSET=docs/test2.py   换测试集（默认 docs/test.py）
#   RAGAS_NO_CACHE=1              换文档库/检索参数后必须加（强制重新走检索链路）
#   RAGAS_CACHE=<path>            改缓存路径
```

环境配置从 `.env` 读取（参考 `backend/.env.example`，含 Milvus 地址、LLM/Embedding 的 API Key、模型名、分块参数）。API Key 不提交进 git。

## 架构

### 检索数据流

```
上传 → ingestion.py（解析 → 分块[fixed/semantic] → 补元数据含 chunk_type）→ milvus.py（写入向量）
提问 → query_transform.py（LLM 改写 + HYDE 假想答案，并行）→ 混合检索：
       milvus 向量召回 + bm25 关键词召回（多查询合并去重）→ rerank.py 精排
     → llm.py（ChatOpenAI）→ SSE 流式回答 + 来源；问答写入 memory.py（SQLite）
```

### 后端 `backend/`

- `main.py` — FastAPI 入口。CORS 全开、无认证；启动时校验 Milvus 维度。**静态托管前端 dist 的 mount 必须放在所有 API 路由之后**，否则会抢占 `/health` 等路径
- `api/documents.py` / `api/chat.py` — 路由层。7 个 endpoint：`GET /health`、`POST /documents/upload`（可选 `split_mode` form 参数）、`GET /documents`、`DELETE /documents/{filename}`、`POST /chat`、`POST /chat/stream`（SSE）、`GET /chat/memory`
- `core/retrieval.py` — 检索 + RAG 生成。`_retrieve` 是唯一检索入口：多查询（原 query + 改写 + HYDE）→ 向量+BM25 合并去重 → rerank（**用原始 query 精排**）→ top_k。`rag_stream` 里 `_retrieve` 放 `asyncio.to_thread` 防阻塞事件循环；流结束把问答写入记忆
- `core/query_transform.py` — `transform_query`（LLM 改写补指代）+ `hyde_query`（LLM 生成假想答案）。**任何异常降级返回原 query，绝不抛出**；`lru_cache` 按 prompt 缓存
- `core/bm25.py` — 内存 BM25 索引（rank_bm25 + jieba 中文分词）。缓存键含块数，上传/删除自动重建；语料到万级块需换 Milvus 原生 BM25
- `core/rerank.py` — SiliconFlow `/rerank` API，`RERANK_ENABLED=false` 或未配 key 时跳过
- `core/ingestion.py` — 解析（PyPDFLoader/Docx2txtLoader/TextLoader/CSVLoader/BSHTMLLoader + 自写 xlsx/pptx loader）→ 切分（fixed: RecursiveCharacterTextSplitter；semantic: 句子 embedding 相似度断块，均值-1.5σ 阈值，碎块<40字符合并，过长块兜底再切）→ `_make_metadata_docs` 补 filename/chunk_index/upload_time/chunk_type
- `core/embeddings.py` / `core/llm.py` — `@lru_cache` 工厂，OpenAI 兼容
- `db/milvus.py` — Collection 以 Embedding 模型名命名（换模型避免维度冲突）；`vs.col` 是底层 Collection 对象；**`metadata_schema` 显式声明 chunk_type 字段**（VARCHAR 32，langchain-milvus 默认按首批 metadata 定 schema，新字段会丢）；`get_all_documents`/`delete_document` 有 `vs.col is None` 防护（collection 未创建时返回空/0）
- `db/memory.py` — SQLite 单表持久化对话（最近 100 条 = 50 轮，自动截断），单连接 `check_same_thread=False`
- `models/schemas.py` — Pydantic 模型（请求/响应，含 Swagger 描述）
- `config.py` — `Settings`（pydantic-settings）+ `get_settings()` 缓存。**改 .env 后需重启进程，lru_cache 不自动刷新**

### 评估 `backend/scripts/`

- `eval_ragas.py` — RAGAS 评估（faithfulness / answer_relevancy / context_precision / context_recall）。逐条走真实检索链路 `retrieval._retrieve` 生成回答 → 判分；测试集（questions/ground_truths 列表）在 `docs/test.py`。**用独立 venv `.venv-ragas/` 运行**（ragas 0.4.3 要求 langchain-core 0.3.x，与主环境 1.5.3 冲突，绝不能装进主 venv）
- `.ragas_cache.json` — 生成结果缓存（question → response + contexts），重跑复用、只重新判分；**改测试集题目/重新上传文档/调检索参数后必须删掉或加 RAGAS_NO_CACHE=1**，否则结果是旧库的
- **`.venv-ragas/` 版本锁定**：pymilvus 2.5.18、langchain-milvus 0.1.10、langchain-core 1.5.3（ragas 装时会把 core 降到 0.3.86，必须 --force-reinstall 回 1.5.3）。评估脚本在模块级 `connections.connect(alias="default", uri=...)`（**必须用 uri 形式**，只传 host/port 会报 ConnLackConf）；指标用 `ragas.metrics` 单例（collections 里的类是 `SimpleBaseMetric` 体系，`evaluate` 的 `isinstance(m, Metric)` 校验不过）；判分 LLM 用 `ChatOpenAI` + 项目 `get_embeddings()`（llm_factory/embedding_factory 的现代实现不兼容旧指标）

### 前端 `frontend/`

- Vue 3 `<script setup>` + Vite + Tailwind CSS v4（`@tailwindcss/vite` 插件，无 tailwind.config.js）
- 无 Vue Router / Pinia / TypeScript — 单页操作台
- `src/App.vue` — 左侧导航（聊天 / 文档管理）切换右侧面板
- `src/api.js` — fetch 封装。`chatStream` 解析 SSE：`data: {json}\n\n` 事件流，支持 `delta`（流式文本）、`sources`（引用）、`answer`（空库兜底）、`[DONE]`
- `src/components/`：`ChatPanel.vue`（流式对话 + Top-K 滑块 + 打字机渲染 + 首 token 前思考动画）、`DocManager.vue`（拖拽上传 + 切分策略下拉 + 列表 + 删除）、`SourceCard.vue`（来源展开卡片）

## 关键约束与已知问题

- **pymilvus 必须锁定 `>=2.5.5,<2.6.0`** — 2.6.x 与 langchain-milvus 不兼容，升上去会崩
- Milvus v3.0-beta 是官方 Docker 镜像；langchain_milvus 内部字段名是 `pk`/`text`/`col` 属性（非 `collection`）
- PDF 的 `page` 元数据是 int，docx 无 page（`None`）；schemas.py 里 `page: int | None`
- 上传同一文件两次会重复入库（无去重），删除按文件名精确匹配
- 上传失败无回滚（临时文件已清理，但部分写入的向量残留）
- **Collection schema 在建库时定死**：加新 metadata 字段必须 `metadata_schema` 声明 + 重建 collection（现有数据丢失）
- **依赖版本已漂移**：langchain-core 实际 1.5.3（装 langchain-experimental 时被升），pyproject 锁 ^0.3.0。当前 Milvus/LLM 链路实测正常，但升依赖前需验证
- `get_settings()` 有 lru_cache：改 .env 不热生效，重启进程
- 对话历史在 SQLite（`backend/chat_memory.db`，gitignore 忽略），刷新页面自动恢复
- 语义切分对每句调一次 embedding API（计费 + 耗时），auto 模式只对 <3000 字符短文档启用
