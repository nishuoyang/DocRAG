# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

基于 LangChain 的垂直领域智能文档问答系统（RAG）：上传 PDF/DOCX 等 8 种格式 → 结构化解析（v2 引擎）→ 父子块分块并向量化存入 Milvus → Query Planning + 向量/BM25 混合召回 → RRF 融合 → Rerank/可答性门控 → Parent 回填 → LLM 生成带引用回答。**上层叠加多 Agent 研究助理工作台**（LangGraph supervisor 并发调度 文档/联网/数据/写作/多跳 5 个成员 agent，`/agent/chat/stream` SSE 事件流）。前后端分离：`backend/`（FastAPI）+ `frontend/`（Vue 3）。LLM/Embedding/Rerank 均用 OpenAI 兼容的云 API，LangSmith tracing 可选。

**v2 升级亮点**：
- 结构化解析：PDF/DOCX 表格转 Markdown、标题层级识别
- Markdown 结构感知分块：按章节切分、表格保护
- VLM 多模态：扫描件 OCR、内嵌图片描述（硅基流动 Qwen2.5-VL）
- 异步上传：后台任务处理 + 进度查询
- 内容哈希去重：避免重复入库
- 检索分级：`fast` 纯原 query / `balanced` 仅短问题或指代追问改写 / `quality` 开启 Rewrite+HYDE
- RRF 融合向量与 BM25 排名，Rerank 失败自动回退；低分候选由可答性门控拦截
- Parent 回填以命中 child 为窗口中心，控制上下文长度并保留章节/页码元数据
- 进程内答案缓存、LangSmith trace、带指纹的 RAGAS 生成缓存

## 常用命令

工作目录通常是 `backend/`（后端）或 `frontend/`（前端）。

```bash
# 基础设施：Milvus 向量库（etcd + MinIO，Docker Compose）
docker compose up -d

# 后端（依赖由 Poetry 管理，venv 在 backend/.venv）
cd backend && ./.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001

# 后端 lint
cd backend && ./.venv/Scripts/ruff.exe check .

# 前端（Vite dev server，代理 /health /documents /chat /agent 到 8001）
cd frontend && npm run dev      # http://localhost:5173（注意用 localhost 而非 127.0.0.1，Vite 默认只监听 IPv6）

# 前端生产构建（产物 frontend/dist/，由 FastAPI 静态托管）
cd frontend && npm run build

# 后端测试：默认排除 integration，需 Milvus 的用 -m integration
cd backend && ./.venv/Scripts/python.exe -m pytest

# 前端 Markdown/XSS 冒烟（需前端 dev server + Chrome）
cd frontend && node scripts/verify-markdown.mjs

# 清空对话记忆：删 backend/chat_memory.db（或在 Python 里调 db.memory.clear_messages()）

# RAGAS 评估（独立 venv，不污染主环境；生成结果缓存在 backend/.ragas_cache.json）
cd backend && ./.venv-ragas/Scripts/python.exe -X utf8 scripts/eval_ragas.py
#   --testset docs/test.py     换测试集（默认 docs/test.py；RAGAS_TESTSET 同效）
#   --top-k 5                  检索返回块数（默认读 .env TOP_K）
#   --no-transform/--no-hyde/--no-rerank   关闭对应检索开关（运行时覆盖，不改 .env）
#   --no-cache                 忽略缓存强制重新生成（换文档库/检索参数后必须）
#   --report-dir reports/      报告目录（默认 reports/，自动创建；产出 md + json 带时间戳）

# RAGAS 测试集自动生成（从 Milvus 文档库合成 questions/ground_truths）
cd backend && ./.venv-ragas/Scripts/python.exe -X utf8 scripts/generate_testset.py
#   --size 20                  题目数（默认 20）
#   --strategy single-hop      生成策略：single-hop（默认，最快）/ multi-hop / full
#   --lang zh                  测试集语言（默认 zh 中文；ragas 默认英文会让 context_recall 偏低）
#   --max-chunks 100           抽样块数上限（大库建议 100-200，抽取耗时与块数成正比）
#   --workers 32               并发 LLM 调用数（默认 32；遇 API 限流调低）
#   --style clean              排除 POOR_GRAMMAR/MISSPELLED 错别字风格（仅保留规范问题）
#   --no-cache                 忽略知识图谱/personas 缓存强制重新抽取

# 多 agent 工作台冒烟（逐 agent 验证：documents/search/data/writer/multi_hop/supervisor）
cd backend && ./.venv/Scripts/python.exe -X utf8 scripts/smoke_agents.py --agent supervisor

# 查看 Milvus 文档库块内容（主环境 venv 即可，零成本）
cd backend && ./.venv/Scripts/python.exe scripts/inspect_chunks.py
#   --index 3,7,12  按块索引查看   --file 关键词   --query 内容关键词   --full 完整内容
```

环境配置从 `.env` 读取（参考 `backend/.env.example`，含 Milvus 地址、LLM/Embedding 的 API Key、模型名、分块参数）。API Key 不提交进 git。`.env` 另有 RAGAS 测试集生成专用变量（见下）。

## 架构

### 检索数据流

```
上传 → parsers/（v2 结构化解析：PDF/DOCX 表格转 Markdown、标题层级）
     → cleaning.py → parent_child.py（child 向量召回 + parent/邻域回填）
     → milvus.py（写入 child 向量，schema 含 section/content_type/file_hash/page）
提问 → retrieval.py Query Planning（按 RETRIEVAL_MODE 决定 Rewrite/HYDE）
     → milvus 向量召回 + bm25 关键词召回（dense/sparse 查询集分开）
     → RRF 融合 → rerank.py（原始 query）→ Answerability Gate
     → Parent 窗口回填 → llm.py → SSE 流式回答 + [n] 引用
     → answer_cache.py；问答写入 memory.py（SQLite）
```

### 后端 `backend/`

- `main.py` — FastAPI 入口。CORS 全开、无认证；启动时校验 Milvus 维度。**静态托管前端 dist 的 mount 必须放在所有 API 路由之后**，否则会抢占 `/health` 等路径
- `api/documents.py` / `api/chat.py` — 路由层。8 个 endpoint：`GET /health`、`POST /documents/upload`（异步，返回 job_id）、`GET /documents/jobs/{job_id}`（进度查询）、`GET /documents`、`DELETE /documents/{filename}`、`POST /chat`、`POST /chat/stream`（SSE）、`GET /chat/memory`
- `api/agents.py` — 多 agent 主管路由：`POST /agent/chat/stream`（SSE 事件协议 activity/delta/sources/done，与 /chat/stream 的 delta/sources/[DONE] 协议不同）
- `core/parsers/` — v2 结构化解析器：`pdf_parser.py`（pymupdf4llm，表格转 GFM）、`docx_parser.py`（python-docx，标题层级+表格+段落图片）、`pptx_parser.py`（备注页+表格+图片）
- `core/cleaning.py` — 噪声清洗：跨页重复行（页眉/页脚/页码）剔除
- `core/md_split.py` — Markdown 结构感知分块：按 `#` 标题切节、节内超长 Recursive 切分并前置章节上下文、GFM 表格永不切断、超大表按行分组重复表头
- `core/vlm.py` — VLM 多模态客户端：页面读图（扫描件 OCR）+ 图片描述，硅基流动 Qwen2.5-VL；每个上传任务独立预算（默认 30 次调用）并缓存相同图片
- `core/jobs.py` — 后台任务管理：SQLite 持久化任务状态（pending/processing/completed/failed），支持进度百分比
- `core/retrieval.py` — 检索 + RAG 生成。`_retrieve` 是唯一检索入口：Query Planning → dense batch + sparse 检索 → RRF（稳定块 ID）→ rerank（**原始 query**）→ 可答性门控 → parent 窗口回填。`RETRIEVAL_MODE` 控制增强/召回规模和 fast 模式是否跳过 Rerank；`rag_stream` 把 `_retrieve` 放 `asyncio.to_thread`，流结束后按 `[n]` 选择实际引用来源并写记忆
- `core/query_transform.py` — 基于最近历史做 `transform_query`（LLM 改写补指代）和 `hyde_query`（LLM 生成假想答案）。**任何异常降级返回原 query，绝不抛出**；`lru_cache` 按 prompt 缓存，使用零温度 Query LLM
- `core/bm25.py` — 内存 BM25 索引（rank_bm25 + jieba 中文分词），索引原始 `raw_text` 而非带章节前缀的 child 文本。上传/删除/替换后由 ingestion 显式 `invalidate_index()`；语料到万级块需换 Milvus 原生全文检索
- `core/answer_cache.py` / `core/eval_cache.py` — RAG 回答进程内 LRU（集合/query/history/model/检索参数组成 key）；RAGAS 生成缓存带版本和语料/参数指纹
- `core/tracing.py` — LangSmith `@traceable` 的 compact input/output serializer，避免把完整上下文和密钥写入 trace
- `core/rerank.py` — SiliconFlow `/rerank` API，`RERANK_ENABLED=false` 或未配 key 时跳过
- `core/agents/` — 多 agent 研究助理工作台：`supervisor.py`（LangGraph decide↔tools，独立工具调用并发执行，超轮次强制收尾，最终回答真实 token 流）→ 5 个成员 agent（`agents.py`：documents 复用 `retrieval._retrieve` 并透传 top_k、search 走 Tavily/Bocha HTTP、data 生成 pandas 脚本经 `data_exec.py` 白名单子进程执行、writer 模板成稿、multi_hop 拆子问题逐轮查证）；`api/agents.py` 提供 `POST /agent/chat/stream`（SSE：activity/delta/sources/done）
- `core/ingestion.py` — 文档摄入：v2 解析（parsers/）→ 噪声清洗（cleaning.py）→ 结构分块（md_split.py）→ 补充元数据（含 section/content_type/file_hash）→ SHA256 去重 → 写入 Milvus；原件落盘 `backend/uploads/` 支持重解析
- `core/embeddings.py` / `core/llm.py` — `@lru_cache` 工厂，OpenAI 兼容；`get_llm` 通用、`get_rag_llm` 低温度回答、`get_query_llm` 零温度查询增强
- `db/milvus.py` — Collection 以 Embedding 模型名命名（换模型避免维度冲突）；**新 schema 显式声明 section/content_type/file_hash/page 字段**（page 为 nullable INT64）；缓存 vectorstore，提供 dense 批量查询；`_get_collection()` 直接用 pymilvus 连接，`get_all_documents()`/`delete_document()` 兼容旧 schema
- `db/memory.py` — SQLite 单表持久化对话（最近 100 条 = 50 轮，自动截断），单连接 `check_same_thread=False`
- `models/schemas.py` — Pydantic 模型（请求/响应，含 Swagger 描述）；`Source` 带 `citation_index`，Agent/Chat 请求都支持 `top_k`
- `config.py` — `Settings`（pydantic-settings）+ `get_settings()` 缓存。**改 .env 后需重启进程，lru_cache 不自动刷新**；关键新增项包括 `RAG_ANSWER_TEMPERATURE`、`QUERY_ENHANCEMENT_TEMPERATURE`、`RETRIEVAL_MODE`、`ANSWERABILITY_GATE_ENABLED`、`RETRIEVAL_MIN_RERANK_SCORE`、`PARENT_WINDOW_RADIUS`、`AGENT_TOOL_TIMEOUT`。API key 字段使用 `repr=False`

### 评估 `backend/scripts/`

- `eval_ragas.py` — RAGAS 评估（faithfulness / answer_relevancy / context_precision / context_recall）。逐条走真实检索链路 `retrieval._retrieve` 生成回答 → 判分；报告包含四指标、context/来源数和检索/生成延迟 P50/P95。参数化 CLI：`--testset` / `--top-k` / `--no-transform` / `--no-hyde` / `--no-rerank` / `--no-cache` / `--report-dir`。**用独立 venv `.venv-ragas/` 运行**（ragas 0.4.3 要求 langchain-core 0.3.x，与主环境冲突，绝不能装进主 venv）
- `generate_testset.py` — 用 ragas `TestsetGenerator` 从 Milvus 文档库自动合成测试集（questions + 参考答案）。transforms 构建知识图谱（Summary/NER 抽取）→ 合成问题。图谱缓存 `.ragas_kg_cache.json` + personas 缓存 `.ragas_personas.json`（重跑复用，跳过抽取阶段）。中文支持：覆盖合成器 prompt 的 instruction 类属性（不覆盖则生成英文问题，中文库下 context_recall 偏低）。`--style clean` 通过子类化合成器排除错别字风格。**坑**：脚本入口必须 `nest_asyncio.apply()`（ragas 多次 `asyncio.run()` 关闭循环导致 openai client 报 `Event loop is closed`，含 pymilvus 导入时更易触发）；`load_documents()` 过滤空块（ragas 抽取器对空块 `IndexError`）；`default_query_distribution` 传自定义分布时必须带 knowledge_graph 过滤多跳合成器
- `inspect_chunks.py` — 查看 Milvus 块内容（索引/文件名/关键词筛选），排查检索问题、检查分块质量用。**主环境 venv 即可**（只依赖 milvus 读取）
- `rebuild_collection.py` — Collection 迁移重建：备份旧数据 → 按新 schema 重建 → 用 `add_embeddings` 回填存量向量（保留原向量，不重新 embedding）
- `.ragas_cache.json` — v2 生成缓存（response + contexts + run metrics），带 collection/文档块指纹/Embedding/Rerank/Query/检索模式/Top-K 指纹。指纹不匹配自动忽略旧结果；`--no-cache` 强制重新生成
- **`.venv-ragas/` 版本锁定**：pymilvus 2.5.18、langchain-milvus 0.1.10、langchain-core 1.5.3（ragas 装时会把 core 降到 0.3.86，必须 --force-reinstall 回 1.5.3）。另需 `pip install rapidfuzz`（0.4.3 的 `OverlapScoreBuilder` 依赖，默认没装）。评估脚本在模块级 `connections.connect(alias="default", uri=...)`（**必须用 uri 形式**，只传 host/port 会报 ConnLackConf）；指标用 `ragas.metrics` 单例（collections 里的类是 `SimpleBaseMetric` 体系，`evaluate` 的 `isinstance(m, Metric)` 校验不过）；判分 LLM 用 `ChatOpenAI` + 项目 `get_embeddings()`（llm_factory/embedding_factory 的现代实现不兼容旧指标）

### RAGAS 环境变量（.env）

| 变量 | 作用 | 未设置时 |
|---|---|---|
| `RAGAS_FAST_LLM_MODEL` | transforms（Summary/NER 抽取）用的快模型 | 回落 `LLM_MODEL` |
| `RAGAS_FAST_LLM_BASE_URL` / `RAGAS_FAST_LLM_API_KEY` | 快模型的独立端点/凭据（如硅基流动） | 回落主 LLM 配置 |
| `RAGAS_LLM_MODEL` / `RAGAS_LLM_BASE_URL` / `RAGAS_LLM_API_KEY` | 评估/合成判分 LLM 覆盖 | 回落 .env 主配置 |
| `RAGAS_TESTSET` / `RAGAS_CACHE` / `RAGAS_NO_CACHE` | eval 脚本兼容变量 | — |
| `RAGAS_KG_CACHE` / `RAGAS_PERSONAS_CACHE` | 图谱/personas 缓存路径 | `backend/.ragas_*.json` |

### 前端 `frontend/`

- Vue 3 `<script setup>` + Vite + Tailwind CSS v4（`@tailwindcss/vite` 插件，无 tailwind.config.js）
- 无 Vue Router / Pinia / TypeScript — 单页操作台
- `src/App.vue` — 左侧导航（聊天 / 文档管理）切换右侧面板
- `src/api.js` — fetch 封装。`chatStream` 解析 SSE：`data: {json}\n\n` 事件流，支持 `delta`（流式文本）、`sources`（引用）、`answer`（空库兜底）、`[DONE]`；新增 `getJobStatus(jobId)` 查询上传进度
- `src/components/`：`ChatPanel.vue`（流式对话 + Top-K 滑块 + 打字机渲染）、`AgentPanel.vue`（多 agent 工作台 + 活动日志 + 文档/网页双型来源）、`DocManager.vue`（拖拽上传 + 切分策略下拉 + 进度条 + 列表 + 删除）、`SourceCard.vue`（`[n]` 引用展开卡片）、`MarkdownContent.vue`（marked + DOMPurify 安全 GFM 渲染）
- `scripts/verify-markdown.mjs` — Playwright 验证两个面板的 Markdown 表格/代码块渲染和 XSS 过滤

## 关键约束与已知问题

- **pymilvus 必须锁定 `>=2.5.5,<2.6.0`** — 2.6.x 与 langchain-milvus 不兼容，升上去会崩
- Milvus v3.0-beta 是官方 Docker 镜像；langchain_milvus 内部字段名是 `pk`/`text`/`col` 属性（非 `collection`）
- PDF 的 `page` 元数据是 int，docx 无 page（`None`）；**新 schema 中 page 显式声明为 nullable INT64**
- **内容哈希去重**：同文件重复上传返回 409（可 `replace=true` 覆盖）
- 上传失败无回滚（临时文件已清理，但部分写入的向量残留）
- **父块窗口**：`PARENT_WINDOW_RADIUS=1` 默认只回填命中 child 前后各 1 个兄弟块；设为 `-1` 回填完整 parent
- **答案缓存**：单进程内存 LRU，上传/删除/替换时失效；重启或多 worker 不共享
- **可答性门控**：仅在拿到 `relevance_score` 且最高分低于 `RETRIEVAL_MIN_RERANK_SCORE` 时拦截；Rerank 失败会回退 RRF 顺序
- **Collection schema 在建库时定死**：加新 metadata 字段必须 `metadata_schema` 声明 + 重建 collection（现有数据丢失）
- **依赖版本已漂移**：langchain-core 实际 1.6.1（装 langgraph 1.x 时被升；# 记载原 1.5.3，实际安装时还是 0.3.86——两次过程都有漂移），pyproject 锁 ^0.3.0。当前 Milvus/LLM/multi-agent 链路实测正常，但升依赖前需验证
- **langgraph 1.x 经 pip 装入主 venv**（poetry 因 core 漂移解析冲突，未走 poetry add）；pyproject 已手工登记 `langgraph = ">=1.0"`、`matplotlib = ">=3.10"`
- `get_settings()` 有 lru_cache：改 .env 不热生效，重启进程
- 对话历史在 SQLite（`backend/chat_memory.db`，gitignore 忽略），刷新页面自动恢复
- 语义切分对每句调一次 embedding API（计费 + 耗时），auto 模式只对 <3000 字符短文档启用
- 根 `.gitignore` 忽略 `reports/`（评估报告）与 `docs/` 下的测试集文件（`docs/test.py` 等已 tracked 的除外）
- ragas TestsetGenerator 0.4.3 的 API 限制与踩坑详见 [docs/ragas-testset-issues.md](docs/ragas-testset-issues.md)
- `backend/main.py` 和 `start.ps1` 当前含 LangSmith key 的调试代码；**公开提交前必须移除并轮换 key**，正式配置只走环境变量
