# 📚 DocRAG · 垂直领域智能文档问答系统

> 上传你的 PDF / DOCX / PPT / Excel… 问它任何问题，回答带引用来源。
> 一套完整的 RAG 工程实践：**结构化解析 → Markdown 分块 → 混合检索 → Query 增强 → Rerank → 流式问答**。

---

## ✨ 特色

| | |
|---|---|
|  **混合检索** | 向量（语义）+ BM25（关键词）双路召回，专有名词/代码片段不漏 |
| 🔍 **Query 增强** | LLM 改写补指代 + HYDE 假想答案，并行调用，失败自动降级 |
| 🎯 **Rerank 精排** | bge-reranker-v2-m3 交叉编码，用**原始 query** 贴合用户意图 |
| 📄 **结构化解析** | PDF/DOCX 表格转 Markdown、标题层级识别、扫描件 OCR |
|  **Markdown 分块** | 按章节切分、表格保护、每块带章节上下文 |
| 🖼️ **多模态理解** | VLM 描述内嵌图片、扫描件页面读图（硅基流动 Qwen2.5-VL） |
| ⚡ **异步上传** | 后台任务处理 + 实时进度查询 |
| 💬 **流式问答** | SSE 逐字输出 + 打字机渲染 + 思考动画 + 引用来源卡片 |
| 🧠 **对话记忆** | SQLite 持久化，刷新页面/重启服务对话自动续上 |
| 📊 **离线评估** | RAGAS 自动化：测试集自动生成（中文）+ 4 项指标量化 + Markdown/JSON 报告 |

---

## ️ 技术栈

```
Frontend: Vue 3 + Vite + Tailwind CSS v4（打字机式 SSE 渲染）
Backend : FastAPI + LangChain（langchain-milvus / ChatOpenAI）
VectorDB: Milvus（Docker Compose，etcd + MinIO）
Parsing : pymupdf4llm（PDF 表格）+ python-docx（DOCX 结构）+ VLM（多模态）
Retrieval: 向量 + BM25（rank_bm25 + jieba）→ 多查询合并去重 → Rerank
LLM/Embedding/Rerank: DeepSeek 官方 + 硅基流动 SiliconFlow（OpenAI 兼容云 API）
Eval    : RAGAS（faithfulness / answer_relevancy / context_precision / context_recall）
```

---

##  快速开始

### 0. 前置条件

- Docker（Milvus 向量库）
- Python 3.12 + Poetry（后端依赖）
- Node.js 18+（前端）

### 1. 启动基础设施

```bash
docker compose up -d        # Milvus + etcd + MinIO
```

### 2. 配置环境变量

```bash
cp backend/.env.example backend/.env
# 填入 LLM_API_KEY（DeepSeek）、EMBEDDING_API_KEY / RERANK_API_KEY（硅基流动）
# 可选：VLM_API_KEY（硅基流动，用于扫描件 OCR 和图片描述）
```

### 3. 启动后端

```bash
cd backend
poetry install
./.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173（用 localhost 而非 127.0.0.1）
```

> 或者直接用仓库自带的脚本：`./start.sh`（Linux/macOS）或 `.\start.ps1`（Windows PowerShell）。

### 5. 开始使用

1. **文档管理** → 拖拽上传（支持 PDF / DOCX / TXT / MD / CSV / XLSX / PPTX / HTML，最大 20MB）
2. 选择切分策略：**自动**（PDF/DOCX 结构分块、短文档语义、长文档固定）/ 结构分块 / 语义 / 固定
3. **聊天** → 提问，看流式回答 + 引用来源；拖 Top-K 滑块控制检索数量

---

## 🧠 RAG 检索链路

```
用户提问
   │
   ▼
┌──────────────┐   ┌──────────────
│ Query 改写    │   │ HYDE 假想答案 │   ← 并行 LLM 调用，失败降级原 query
└─────────────┘   └─────────────┘
       └─────────────────┘
                ▼
    多查询分别检索（向量 Top-10 + BM25 Top-10）
                │  按文本内容合并去重
                ▼
        Rerank 精排（原始 query，bge-reranker-v2-m3）
                │
                ▼
   system(参考资料) + history(最近 6 轮) + query → LLM 流式生成
                │
                ▼
   SSE 逐字输出 + 来源引用 → 写入 SQLite 记忆
```

**为什么这样做？**

| 设计 | 解决什么问题 |
|---|---|
| Query 改写 + HYDE | 口语/指代不清、短 query 语义模糊 → 扩大召回 |
| 向量 + BM25 混合 | 语义相似但无关键词、或关键词精确但语义跑偏 → 互补 |
| Rerank 用原始 query | 改写/假文档只用于召回，最终排序贴合用户本意 |
| 记忆存 50 轮取 6 轮 | 刷新恢复对话，又不稀释检索上下文 |
| Markdown 结构分块 | 按章节切分更精准，表格不切断，每块带章节上下文 |

---

## 📊 RAGAS 评估

### 评估

```bash
cd backend
./.venv-ragas/Scripts/python.exe -X utf8 scripts/eval_ragas.py
# --testset docs/test.py   换测试集（默认 docs/test.py）
# --top-k 5                检索返回块数
# --no-transform / --no-hyde / --no-rerank   关闭对应检索开关（不改 .env）
# --no-cache               换文档库/检索参数后强制重新生成
# --report-dir reports/    报告目录（自动创建，产出带时间戳的 Markdown + JSON）
```

- 测试集：`docs/test.py`（questions + ground_truths 两列表）
- 指标：faithfulness（忠实度）/ answer_relevancy（相关性）/ context_precision（上下文精度）/ context_recall（上下文召回）
- 真实链路：逐条走 `_retrieve`（与线上 /chat 完全一致）→ 生成 → 判分
- 独立 venv `.venv-ragas/`（ragas 0.4.3 与主环境依赖冲突，绝不装进主 venv）

### 自动生成测试集

```bash
cd backend
./.venv-ragas/Scripts/python.exe -X utf8 scripts/generate_testset.py --size 20 --max-chunks 100
# --strategy single-hop   生成策略（默认 single-hop；multi-hop 需图谱建边更慢）
# --lang zh               测试集语言（默认 zh 中文，匹配中文文档库）
# --style clean           排除错别字风格问题（POOR_GRAMMAR / MISSPELLED）
# --workers 32            并发 LLM 调用数（遇 API 限流调低）
```

- 从 Milvus 文档库自动合成 questions + 参考答案，输出到 `docs/test_generated.py`（与 test.py 同格式，可直接给评估脚本用）
- 知识图谱与 personas 缓存（`.ragas_kg_cache.json` / `.ragas_personas.json`），重跑复用跳过抽取阶段
- 中文问题支持：覆盖合成器 prompt 指令（ragas 默认英文，中文库下 context_recall 会偏低）
- `.env` 可配 `RAGAS_FAST_LLM_MODEL`（+ 独立 BASE_URL / API_KEY）给抽取阶段指定快模型

### 查看块内容

```bash
cd backend
./.venv/Scripts/python.exe scripts/inspect_chunks.py --index 3,7,12 --full
```

- 排查检索问题 / 检查分块质量用；主环境 venv 即可

**最近一次评估结果（20 题，自动生成测试集）**：

| 指标 | 平均分 | 中位数 |
|---|---|---|
| Faithfulness 忠实度 | 0.915 | 1.00 |
| Answer Relevancy 相关性 | 0.785 | 0.81 |
| Context Precision 精度 | 0.830 | 0.92 |
| Context Recall 召回 | 0.950 | 1.00 |

---

## 📁 项目结构

```
backend/                  FastAPI 后端
├── api/                  路由层（documents / chat，8 个 endpoint）
├── core/                 业务层
│   ├── parsers/          v2 结构化解析器（pdf/docx/pptx）
│   ├── cleaning.py       噪声清洗（页眉页脚剔除）
│   ├── md_split.py       Markdown 结构感知分块
│   ├── vlm.py            VLM 多模态客户端
│   ├── jobs.py           后台任务管理
│   ├── ingestion.py      文档摄入（解析→清洗→分块→去重→入库）
│   ├── retrieval.py      检索+RAG 生成
│   ├── query_transform.py 改写+HYDE
│   ├── bm25.py           BM25 关键词检索
│   ├── rerank.py         Rerank 精排
│   ├── llm.py            LLM 工厂
│   └── embeddings.py     Embedding 工厂
├── db/                   milvus.py（向量库）/ memory.py（SQLite 对话记忆）
── scripts/              eval_ragas.py / generate_testset.py / inspect_chunks.py / rebuild_collection.py
├── uploads/              上传原件落盘（支持重解析）
└── config.py             pydantic-settings 配置（.env）
frontend/                 Vue 3 前端（ChatPanel 流式对话 / DocManager 文档管理 / SourceCard 来源）
docs/                     工作流程详解、测试集、踩坑记录
reports/                  评估报告（Markdown + JSON，gitignore 忽略）
docker-compose.yml        Milvus + etcd + MinIO
```

---

## 📚 文档

- [docs/chat-and-documents-workflow.md](docs/chat-and-documents-workflow.md) — 每个接口的内部执行流程（12 步检索链路、SSE 事件流、排查表）
- [docs/ragas-testset-issues.md](docs/ragas-testset-issues.md) — ragas TestsetGenerator 接入排坑记录（事件循环修复、空块过滤）
- [CLAUDE.md](CLAUDE.md) — 给 Claude Code 的项目指南（架构 + 已知问题 + 常用命令）

---

## ⚠️ 已知问题

- **pymilvus 锁定 `>=2.5.5,<2.6.0`**：2.6.x 与 langchain-milvus 不兼容
- **Collection schema 建库时定死**：加 metadata 字段需重建（数据丢失），换 Embedding 模型同理
- 上传失败无事务回滚（可能残留部分向量）
- 改 `.env` 不热生效（`get_settings()` lru_cache），需重启后端
- VLM 处理页数有限制（默认 30 页），超大文档可能跳过部分页面

---

## 🗂️ License

MIT
