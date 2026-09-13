# 📚 DocRAG · 多 Agent 文档研究工作台

> 上传你的 PDF / DOCX / PPT / Excel…，问它任何问题——单问（RAG 流式问答），或把任务交给**主管 agent**：自动调度文档检索、联网搜索、数据分析、写作汇总、多跳查证，最后成稿。
> 一套完整的 RAG 工程实践：**结构化解析 → 父子块分块 → Query Planning → 向量/BM25 混合召回 → RRF 融合 → Rerank → Parent 上下文回填 → 带引用流式问答**，上层再叠一层 **LangGraph 多 agent 编排**。

---

## ✨ 特色

### 多 Agent 研究助理

| | |
|---|---|
| 🛰️ **主管调度** | LangGraph decide↔tools 决策循环：拆解任务 → 并发调成员 → 流式汇总；对话流实时展示各 agent 活动 |
| 📚 **文档 agent** | 复用本地 RAG 链路，只依据文档库回答，引用带来源 |
| 🌐 **联网 agent** | Tavily / Bocha 可选，带 URL 摘要；未配 key 自动降级 |
| 📊 **数据 agent** | LLM 生成 pandas 脚本 → 白名单沙箱子进程执行（超时/凭据隔离）→ 表格 + 图表 |
| ✍️ **写作 agent** | 报告 / 对比 / 周报三种模板成稿 |
| 🔀 **多跳查证** | 复杂问题拆子问题逐项查证、去重汇总，交叉引用型问题 |

### RAG 核心链路

| | |
|---|---|
| 🔀 **混合检索** | 向量（语义）+ BM25（关键词）双路召回，RRF 按排名融合，专有名词/代码片段不漏 |
| 🔍 **自适应 Query 增强** | `fast / balanced / quality` 三档策略：短追问才改写，HYDE 只进 dense 召回，失败自动降级 |
| 🎯 **Rerank 精排** | bge-reranker-v2-m3 交叉编码，用**原始 query** 贴合用户意图 |
| 🚦 **可答性门控** | Rerank 最高分低于阈值时返回“资料中未找到”，减少低相关上下文触发幻觉 |
| 🧾 **引用编号** | 回答要求使用 `[1] [2]` 标注事实来源，后端仅返回实际引用的 source card |
| 📄 **结构化解析** | PDF/DOCX 表格转 Markdown、标题层级识别、扫描件 OCR |
| 🧩 **父子块分块** | child 精准召回 + 命中 child 邻域回填，中文句子/表格/代码块边界保护 |
| 🖼️ **多模态理解** | VLM 描述内嵌图片、扫描件页面读图（硅基流动 Qwen2.5-VL） |
| ⚡ **异步上传** | 后台任务处理 + 实时进度查询 |
| 💬 **流式问答** | SSE 逐字输出 + Markdown/GFM 渲染 + 引用来源卡片 |
| 🧠 **对话记忆** | SQLite 持久化，刷新页面/重启服务对话自动续上 |
| ⚡ **请求缓存** | 相同问题、历史、模型和检索配置复用进程内 LRU 结果；文档变更自动失效 |
| 🛰️ **可观测性** | 可选 LangSmith tracing：检索、RRF、Rerank、生成、引用选择逐层可查 |
| 📊 **离线评估** | RAGAS 自动化：缓存带语料/参数指纹 + 4 项指标量化 + 延迟/上下文统计 |

---

## 🧱 技术栈

```
Frontend: Vue 3 + Vite + Tailwind CSS v4（marked + DOMPurify 安全 Markdown 渲染）
Backend : FastAPI + LangChain + LangGraph（langchain-milvus / ChatOpenAI）
VectorDB: Milvus（Docker Compose，etcd + MinIO）
Parsing : pymupdf4llm（PDF 表格）+ python-docx（DOCX 结构）+ VLM（多模态）
Retrieval: Query Planning → 向量 + BM25（rank_bm25 + jieba）→ RRF → Rerank → 可答性门控
LLM/Embedding/Rerank: DeepSeek 官方 + 硅基流动 SiliconFlow（OpenAI 兼容云 API）
Search  : Tavily / Bocha（联网搜索，SEARCH_API_KEY 可选）
Eval    : RAGAS（faithfulness / answer_relevancy / context_precision / context_recall）
Tests   : pytest（后端单元/集成标记）+ Vite build + Playwright Markdown 冒烟
Tracing : LangSmith（可选，通过环境变量启用）
```

---

## 快速开始

### 0. 前置条件

- Docker（Milvus 向量库）
- Python 3.11+（后端依赖装在 `backend/.venv`，终端直接入仓库即可，无需 Poetry 环境）
- Node.js 18+（前端）
- 推荐直接使用启动脚本：`./start.sh`（Linux/macOS）或 `.\start.ps1`（Windows PowerShell）

### 1. 启动基础设施

```bash
docker compose up -d        # Milvus + etcd + MinIO
```

### 2. 配置环境变量

```bash
cp backend/.env.example backend/.env
# 必填：LLM_API_KEY（DeepSeek）、EMBEDDING_API_KEY / RERANK_API_KEY（硅基流动）
# 可选：VLM_API_KEY（硅基流动，扫描件 OCR）、SEARCH_API_KEY（联网搜索）
# 检索默认 RETRIEVAL_MODE=balanced；需要全量改写+HYDE时改为 quality
# LangSmith 通过进程环境变量启用，不要把 key 写进代码或提交到仓库
```

### 3. 启动后端

```bash
cd backend
./.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173（注意用 localhost 而非 127.0.0.1）
```

### 5. 开始使用

1. **文档管理** → 拖拽上传（支持 PDF / DOCX / TXT / MD / CSV / XLSX / PPTX / HTML，最大 20MB）
2. 选择切分策略：**自动父子块**（推荐）/ 父子块 / 语义 / 固定
3. **智能问答** → 单问，看流式回答 + 引用来源；拖 Top-K 滑块控制检索数量
4. **研究助理** → 直接把复合任务交给主管 agent，例如：
   - 「北京和上海的首套房首付比例分别是多少？」（单跳文档问答）
   - 「文档里的评分标准在哪几个文档出现过？」（多跳查证）
   - 「对比京沪政策，联网查一下最新调整，写份对比报告」（文档 + 联网 + 写作全链路）

### 6. 可选：启用 LangSmith tracing

PowerShell 示例：

```powershell
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_API_KEY = "<your-langsmith-key>"
$env:LANGSMITH_PROJECT = "rag-agent"
cd backend
./.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

未设置 `LANGSMITH_TRACING=true` 时 tracing 自动关闭。`backend/main.py` 与 `start.ps1` 当前仍含调试阶段的硬编码/打印逻辑，提交公开版本前必须移除并轮换已暴露的 key。

### 检索与 Agent 关键配置

| 变量 | 默认值 | 作用 |
|---|---|---|
| `RETRIEVAL_MODE` | `balanced` | `fast` / `balanced` / `quality` 三档检索策略 |
| `RETRIEVAL_CANDIDATE_K` | `20` | 融合前每路候选数量上限 |
| `ANSWERABILITY_GATE_ENABLED` | `true` | 是否启用 Rerank 低分门控 |
| `RETRIEVAL_MIN_RERANK_SCORE` | `0.1` | 触发“资料中未找到”的最高分阈值 |
| `PARENT_WINDOW_RADIUS` | `1` | 命中 child 前后回填窗口；`-1` 表示完整 parent |
| `RAG_ANSWER_TEMPERATURE` | `0.0` | RAG 事实回答温度 |
| `QUERY_ENHANCEMENT_TEMPERATURE` | `0.0` | Query Rewrite/HYDE 温度 |
| `AGENT_TOOL_TIMEOUT` | `60` | 单个成员 agent 超时（秒） |

---

## 🧠 架构

### 单链 RAG（智能问答）

```
用户提问
   │
   ▼
Query Planning（fast / balanced / quality）
   ├─ fast     : 只用原 query，跳过远程 Rerank
   ├─ balanced : 短问题/指代追问才 Rewrite，不做 HYDE
   └─ quality  : Rewrite + HYDE（HYDE 只用于 dense 召回）
                │
                ▼
   向量检索 + BM25 关键词检索（多查询结果分别保留排名）
                │
                ▼
        RRF 融合（按稳定块 ID，不按文本内容去重）
                │
                ▼
   Rerank（原始 query）→ 可答性门控
                │
                ▼
   命中 child 邻域 → Parent 上下文回填
                │
                ▼
   system(带 [n] 编号资料) + history(最近 6 轮) + query → LLM 流式生成
                │
                ▼
   引用选择 + SSE / SQLite 记忆 + 进程内 LRU 缓存
```

### 多 Agent 编排（研究助理）

```
用户提问 + top_k → Supervisor（LangGraph：decide ↔ tools 决策循环，超轮次强制收尾）
             ├─ documents_agent ── 本地 RAG 检索（复用单链）
             ├─ search_agent ───── 联网搜索（Tavily / Bocha）
             ├─ data_agent ─────── pandas 脚本 → 沙箱子进程
             ├─ writer_agent ───── 报告 / 对比 / 周报
             └─ multi_hop_agent ── 拆子问题逐项查证
   → 独立工具并发执行 + 单工具超时；最终回答由 LLM 真实 token 流推送
   → SSE 事件流：activity → delta → sources → done
```

---

## 📁 项目结构

```
backend/                  FastAPI 后端
├── api/                  路由层（documents / chat / agents）
│   └── agents.py         /agent/chat/stream（SSE 事件协议 activity/delta/sources/done）
├── core/                 业务层
│   ├── agents/           多 agent 工作台
│   │   ├── supervisor.py LangGraph 决策循环
│   │   ├── agents.py     5 个成员 agent（documents/search/data/writer/multi_hop）
│   │   ├── data_exec.py  数据分析沙箱（AST 白名单 + 子进程隔离）
│   │   └── schemas.py    AgentTask / AgentResult
│   ├── parsers/          v2 结构化解析器（pdf/docx/pptx）
│   ├── cleaning.py       噪声清洗（页眉页脚剔除）
│   ├── parent_child.py   原子块、child 召回块与 parent 分组
│   ├── vlm.py            VLM 多模态客户端
│   ├── jobs.py           后台任务管理
│   ├── ingestion.py      文档摄入（解析→清洗→分块→去重→入库）
│   ├── retrieval.py      查询规划、混合检索、RRF、门控、Parent 回填、RAG 生成
│   ├── query_transform.py 基于近期历史的改写+HYDE
│   ├── bm25.py           BM25 关键词检索
│   ├── rerank.py         Rerank 精排
│   ├── answer_cache.py   进程内 RAG 答案 LRU 缓存
│   ├── tracing.py        LangSmith 输入/输出精简器
│   ├── llm.py            LLM 工厂（通用/RAG/Query 三套温度）
│   └── embeddings.py     Embedding 工厂
├── db/                   milvus.py（向量库）/ memory.py（SQLite 对话记忆）
├── scripts/              smoke_agents.py / eval_ragas.py / generate_testset.py /
│                         inspect_chunks.py / rebuild_collection.py
├── tests/                pytest 单元测试 + Milvus integration 标记
├── uploads/              上传原件落盘（支持重解析）
└── config.py             pydantic-settings 配置（.env）
frontend/                 Vue 3 前端
├── components/           ChatPanel（智能问答）/ AgentPanel（研究助理）/
│                         DocManager（文档管理）/ SourceCard（来源）/
│                         MarkdownContent（GFM 安全渲染）
├── scripts/              verify-markdown.mjs（Playwright Markdown/XSS 冒烟）
└── src/App.vue           左侧导航切换三个面板
docs/                     工作流程详解、测试集、踩坑记录
docker-compose.yml        Milvus + etcd + MinIO
```

---

## ✅ 开发与测试

```powershell
# 后端单元测试（默认排除 integration）
cd backend
./.venv/Scripts/python.exe -m pytest

# 需要本地 Milvus 的集成测试
./.venv/Scripts/python.exe -m pytest -m integration

# 后端 lint
./.venv/Scripts/ruff.exe check .

# 前端生产构建
cd ../frontend
npm run build
```

---

## 📊 RAGAS 评估

### 多 agent 冒烟

```bash
cd backend
./.venv/Scripts/python.exe -X utf8 scripts/smoke_agents.py --agent supervisor
# --agent documents|search|data|writer|multi_hop|supervisor
```

### 评估

```bash
cd backend
./.venv-ragas/Scripts/python.exe -X utf8 scripts/eval_ragas.py
# --testset docs/test.py   换测试集（默认 docs/test.py）
# --top-k 5                检索返回块数
# --no-transform / --no-hyde / --no-rerank   关闭对应检索开关（不改 .env）
# --no-cache               忽略缓存强制重新生成
# --report-dir reports/    报告目录（自动创建，产出带时间戳的 Markdown + JSON）
```

- 独立 venv `.venv-ragas/`（ragas 0.4.3 与主环境依赖冲突，绝不装进主 venv）
- 缓存文件记录版本、collection、文档块指纹、Embedding/Rerank/Query/检索模式/Top-K 等参数；指纹不匹配会自动忽略旧结果。
- 报告包含四项 RAGAS 指标、运行时上下文长度/来源数，以及检索和生成延迟的 mean/P50/P95。

### 自动生成测试集

```bash
cd backend
./.venv-ragas/Scripts/python.exe -X utf8 scripts/generate_testset.py --size 20 --max-chunks 100
# --strategy single-hop   生成策略（默认 single-hop；multi-hop 需图谱建边更慢）
# --lang zh               测试集语言（默认 zh 中文，匹配中文文档库）
# --style clean           排除错别字风格问题（POOR_GRAMMAR / MISSPELLED）
# --workers 32            并发 LLM 调用数（遇 API 限流调低）
```

---

## 📚 文档

- [docs/chat-and-documents-workflow.md](docs/chat-and-documents-workflow.md) — 接口内部执行流程（Query Planning、RRF、门控、引用、SSE、排查表）
- [docs/ragas-testset-issues.md](docs/ragas-testset-issues.md) — ragas TestsetGenerator 接入排坑记录（事件循环修复、空块过滤）
- [CLAUDE.md](CLAUDE.md) — 给 Claude Code 的项目指南（架构 + 已知问题 + 常用命令）

---

## ⚠️ 已知问题

- **pymilvus 锁定 `>=2.5.5,<2.6.0`**：2.6.x 与 langchain-milvus 不兼容
- **Collection schema 建库时定死**：加 metadata 字段需重建（数据丢失），换 Embedding 模型同理
- 上传失败无事务回滚（可能残留部分向量）
- 改 `.env` 不热生效（`get_settings()` lru_cache），需重启后端
- VLM 每次上传任务限制调用次数（默认 30 次），超大文档可能跳过部分页面
- 研究助理与智能问答共享同一份对话记忆（SQLite），切换面板时历史会互相看到
- RAG 答案缓存是单进程内存 LRU；重启或启用多 worker 后不共享
- BM25 是内存索引，首次检索 lazy build；万级块以上建议迁移到 Milvus 原生全文检索/ES
- 服务 CORS 全开且无认证，只能作为本地/受信网络部署
- 当前工作区的 `backend/main.py`、`start.ps1` 含 LangSmith 调试代码和疑似真实 key；公开分发前必须清理并轮换

---

## 🗂️ License

MIT
