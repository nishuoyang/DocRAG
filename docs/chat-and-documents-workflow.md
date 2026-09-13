# Chat 与 Documents 工作流程详解

> 适用版本：v2（结构化解析 + 父子块 + RRF 混合检索 + 可答性门控 + 引用 + VLM 多模态 + 异步上传）。
> 阅读本文前建议先看根目录 `CLAUDE.md` 了解整体架构；本文深入每个接口的内部执行流程。

---

## 1. 总览

系统由两大模块组成：

- **Documents（文档管理）**：上传 → 结构化解析（v2）→ 噪声清洗 → Markdown 分块 → 向量化 → 入库；以及列表、删除、进度查询。
- **Chat（智能问答）**：Query Planning（Rewrite/HYDE 分档）→ 向量+BM25 混合召回 → RRF 融合 → Rerank + 可答性门控 → Parent 窗口回填 → LLM 生成 → `[n]` 引用；支持 SQLite 持久化记忆、进程内答案缓存与 SSE 流式输出。

```
┌─────────┐   HTTP   ┌──────────────┐   调用    ┌────────────────────────
│ 前端 Vue │ ───────▶ │  FastAPI 路由 │ ───────▶ │  core 业务层            │
─────────┘          │  api/*.py    │          │  parsers/ cleaning.py   │
                     └──────────────┘          │  md_split.py vlm.py     │
                                              │  ingestion / retrieval / │
                                              │  query_transform /      │
                                              │  bm25 / rerank / llm    │
                                              ────────┬───────────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │ db/milvus.py     │
                                              │  (pymilvus 直连)  │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │ Milvus 向量库     │
                                              └──────────────────┘
```

| 功能 | 路由 | 调用链 |
|---|---|---|
| 上传文档 | `POST /documents/upload` | `api/documents.py` → `core/jobs.py`（创建任务）→ 后台 `core/ingestion.py` → `core/parsers/` → `core/cleaning.py` → `core/md_split.py` → `db/milvus.py` → Milvus |
| 查询进度 | `GET /documents/jobs/{job_id}` | `api/documents.py` → `core/jobs.py`（SQLite） |
| 文档列表 | `GET /documents` | `api/documents.py` → `core/ingestion.py` → `db/milvus.py`（pymilvus 直连）→ Milvus |
| 删除文档 | `DELETE /documents/{filename}` | `api/documents.py` → `core/ingestion.py` → `db/milvus.py` → Milvus |
| 历史记忆 | `GET /chat/memory` | `api/chat.py` → `db/memory.py`（SQLite） |
| 单轮问答 | `POST /chat` | `api/chat.py` → `core/retrieval.py` → query_transform/bm25/rerank/milvus → `core/llm.py` |
| 流式问答 | `POST /chat/stream` | 同上，SSE 输出 |

> **重要：所有路由本身不含业务逻辑**，只做参数校验、请求体解析和异常 → HTTP 状态码的转换；真正的流程在 `core/` 和 `db/` 层。这样设计的好处是业务逻辑可以脱离 HTTP 单独测试/复用。

---

## 2. Documents 模块工作流程

### 2.1 上传文档：`POST /documents/upload`（异步）

**请求**：`multipart/form-data`，字段 `file`（支持 PDF/DOCX/TXT/MD/CSV/XLSX/PPTX/HTML，最大 20MB）+ 可选 `split_mode`（`auto` | `markdown` | `semantic` | `fixed`，默认 `auto`）+ 可选 `replace`（`true` 覆盖已入库文件）。

**执行步骤**：

1. **格式校验**（[api/documents.py](../backend/api/documents.py)）
   从 `file.filename` 提取扩展名，白名单 = `ingestion.SUPPORTED_EXTENSIONS` + xlsx/pptx（单一事实源）。

2. **split_mode 校验**（[api/documents.py](../backend/api/documents.py)）
   仅接受 `auto` / `markdown` / `semantic` / `fixed`，其他返回 400。

3. **大小校验**（[api/documents.py](../backend/api/documents.py)）
   一次性读入全部字节（`await file.read()`），按 `MAX_UPLOAD_MB`（默认 20MB）判断，超限返回 413。

4. **创建后台任务**（[api/documents.py](../backend/api/documents.py)）
   - `job_manager.create_job(filename)` 创建任务记录（SQLite），返回 `job_id`
   - `background_tasks.add_task(_process_document_task, ...)` 添加后台任务
   - 立即返回 `JobInfoResponse`（job_id、filename、status=pending、progress=0）

5. **后台任务处理** `_process_document_task`（[api/documents.py](../backend/api/documents.py)）
   - 更新状态为 `processing`，progress=10
   - 调用 `ingestion.ingest_file()`（[core/ingestion.py](../backend/core/ingestion.py)），内部流程：
     - **SHA256 去重**：计算文件哈希，若已入库且 `replace=false` 则抛 `DuplicateFileError`
     - **写临时文件**：`NamedTemporaryFile(suffix=ext)`。因为 loader 只接受文件路径，不接受字节流。
     - **v2 解析** `_load_documents`（[core/parsers/__init__.py](../backend/core/parsers/__init__.py)）：
       - PDF → `pymupdf4llm` 逐页输出 Markdown（表格转 GFM、标题带 `#` 层级）
       - DOCX → `python-docx` 顺序遍历 body（标题 Heading 1-9 → `#` 层级、表格转 GFM）
       - PPTX → `python-pptx` 逐页提取文本 + 表格
       - XLSX → `pandas` 按 sheet 分行
       - CSV/HTML/TXT/MD → langchain loader
     - **噪声清洗**（[core/cleaning.py](../backend/core/cleaning.py)）：PDF 逐页剔除跨页重复行（页眉/页脚/页码）
     - **确定切分策略** `_resolve_split_mode`（[core/ingestion.py](../backend/core/ingestion.py)）：
       - 手动 `split_mode` 优先
       - `auto` 时 PDF/DOCX 走 `markdown`，其他按 `SEMANTIC_SPLIT` 配置（<3000 字符语义、否则固定）
     - **分块** `_split_documents`（[core/ingestion.py](../backend/core/ingestion.py)），返回 `(结果，实际策略)`：
       - **markdown**（[core/md_split.py](../backend/core/md_split.py)）：按 `#` 标题切节 → 节内文本超长 Recursive 切分（每块前置 `[章节：path]`）→ GFM 表格永不切断、超大表按行分组重复表头 → `chunk_type="markdown"`
       - **fixed**：`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
       - **semantic**：按句切 → 批量 embedding → 相邻句余弦相似度 → 断点 = 低于 `均值 -1.5σ` → 按断点合并成块
     - **补充元数据** `_make_metadata_docs`（[core/ingestion.py](../backend/core/ingestion.py)）：为每个块添加 `filename`、`chunk_index`、`upload_time`、`chunk_type`、`section`（章节路径）、`content_type`（text/table）、`file_hash`（SHA256）；PDF/pptx 额外保留 `page`
     - **写入向量库** `milvus.add_documents(enriched)`（[db/milvus.py](../backend/db/milvus.py)）
     - **原件落盘** `_save_original`（[core/ingestion.py](../backend/core/ingestion.py)）：哈希命名存入 `backend/uploads/`，支持后续重解析
   - 更新状态为 `completed`，progress=100，保存 result（chunk_count、chunk_type、file_hash、vlm_pages）
   - 异常时更新为 `failed`，保存 error 信息

**异常处理**：
- `DuplicateFileError` → 409（可 `replace=true` 覆盖）
- `ValueError`（空文档/无法解析）→ 400，前端展示错误信息
- 其他异常 → 500 `文档处理失败：<原因>`

**响应示例**（立即返回）：
```json
{
  "job_id": "78a2d67e-01d3-47f0-9228-7c45d74737bb",
  "filename": "test.txt",
  "status": "pending",
  "progress": 0,
  "message": "任务已创建",
  "created_at": "2026-08-20T15:36:40.324568",
  "updated_at": "2026-08-20T15:36:40.324568",
  "result": null,
  "error": null
}
```

### 2.2 查询进度：`GET /documents/jobs/{job_id}`

**执行步骤**：

1. `job_manager.get_job(job_id)`（[core/jobs.py](../backend/core/jobs.py)）：
   - 从 SQLite 查询任务记录
   - 不存在 → 返回 404

2. 返回 `JobInfoResponse`（job_id、filename、status、progress、message、created_at、updated_at、result、error）

**响应示例**（处理中）：
```json
{
  "job_id": "78a2d67e-01d3-47f0-9228-7c45d74737bb",
  "filename": "test.txt",
  "status": "processing",
  "progress": 60,
  "message": "分块完成，正在向量化...",
  "created_at": "2026-08-20T15:36:40.324568",
  "updated_at": "2026-08-20T15:36:42.807795",
  "result": null,
  "error": null
}
```

**响应示例**（已完成）：
```json
{
  "job_id": "78a2d67e-01d3-47f0-9228-7c45d74737bb",
  "filename": "test.txt",
  "status": "completed",
  "progress": 100,
  "message": "文档处理完成",
  "result": {
    "chunk_count": 1,
    "chunk_type": "semantic",
    "file_hash": "57cb5792017c8d0dca82652da1199f726d5e221690c8c81baebefa1add48bc6b",
    "vlm_pages": 0
  },
  "error": null
}
```

### 2.3 文档列表：`GET /documents`

**执行步骤**：

1. `milvus.get_all_documents()`（[db/milvus.py](../backend/db/milvus.py)）：
   - **直接用 pymilvus Collection 查询**（不依赖 embedding 配置，避免 `EMBEDDING_API_KEY` 为空时报错）
   - **动态检测可用字段**，兼容旧 schema（无 section/content_type/file_hash/page 字段也能正常读取）
   - 用底层 `col.query` 分页拉取全部数据（每批 100 条，`offset` 递增直到取完）
   - 每条数据重建为 `Document`，元数据含 `pk`/`filename`/`chunk_index`/`upload_time`/`chunk_type`/`section`/`content_type`/`file_hash`/`page`

2. `ingestion.list_documents()`（[core/ingestion.py](../backend/core/ingestion.py)）：
   - 按 `filename` 聚合：同文件所有 chunk 合并为一条记录，`chunk_count` 累计块数
   - 按 `upload_time` 降序排序（新上传的在前）

3. 返回 `{documents: [...], total: N}`

### 2.4 删除文档：`DELETE /documents/{filename}`

**执行步骤**：

1. `ingestion.delete_document(filename)` → `milvus.delete_document(filename)`（[db/milvus.py](../backend/db/milvus.py)）：
   - collection 不存在 → 返回 0
   - 构造表达式 `filename == "<filename>"`，用底层 `col.delete` 删除**该文件名对应的全部向量块**
   - `col.flush()` 保证删除对后续查询可见
   - 返回 `delete_count`

2. 删除 0 条 → `404 未找到文档`；否则返回 `{filename, deleted}`

**注意**：按文件名精确匹配（同名文件一起删）；删除的是 Milvus 向量，不是磁盘文件；不可撤销。

---

## 3. Chat 模块工作流程

### 3.1 统一入口：`POST /chat/stream`（SSE 流式，前端主用）与 `POST /chat`（一次性返回）

两个接口共享相同的检索与提示词构建逻辑，区别仅在 LLM 输出方式：

- `/chat/stream`：SSE（`text/event-stream`），LLM 每个 token 生成后立即推送，前端逐字显示。
- `/chat`：等待 LLM 全部生成完毕，一次性返回 JSON。

**请求体**（`ChatRequest`）：
```json
{
  "query": "用户的提问",
  "top_k": 5,                          // 可选，1-20，默认取配置 TOP_K=5
  "history": [                          // 可选，兼容旧前端；不传则后端从 SQLite 记忆读
    { "role": "user", "content": "上一轮问题" },
    { "role": "assistant", "content": "上一轮回答" }
  ]
}
```

### 3.2 完整流程（以 `/chat/stream` 为例）

```
前端发送 query
        │
        ▼
① api/chat.py 转发到 retrieval.rag_stream()
        │
        ▼
 rag_stream: _retrieve 放 asyncio.to_thread 执行（不阻塞事件循环）
        │
        ├─ ② _resolve_history: 前端 history 优先，否则 SQLite 最近 6 轮
        ├─ ③ 查询 answer_cache: 相同问题/历史/模型/检索配置直接返回
        ├─ ④ Query Planning: 按 RETRIEVAL_MODE 决定 Rewrite/HYDE
        │      ├─ fast: 只用原 query
        │      ├─ balanced: 仅短问题/指代追问 Rewrite，不做 HYDE
        │      └─ quality: Rewrite + HYDE；HYDE 只进 dense 查询集
        │
        ├─ ⑤ 混合召回: 原 query 向量+BM25；增强 query 分别召回
        │      └─ dense 多查询批量 embedding；sparse 不接收 HYDE
        │
        ├─ ⑥ RRF 融合: 按稳定块 ID 汇总各结果列表排名
        ├─ ⑦ Rerank: 原始 query 精排；fast 模式跳过；失败回退 RRF
        ├─ ⑧ Answerability Gate: 最高 rerank 分低于阈值时返回未找到
        ├─ ⑨ Parent Expansion: 命中 child 前后 PARENT_WINDOW_RADIUS 个兄弟块
        ├─ ⑩ _build_messages: system(带 [n] 编号资料) + history + query
        ├─ ⑪ RAG LLM astream → 每个 chunk yield {"delta": "..."}
        ├─ ⑫ 解析回答中的 [n] → 只返回实际引用 sources → [DONE]
        └─ ⑬ 写答案缓存 + SQLite 记忆
```

**逐步说明**：

**① 请求预处理**（[api/chat.py](../backend/api/chat.py)）
`chat_stream` 返回 `StreamingResponse(event_stream(), media_type="text/event-stream")`。`event_stream` 是异步生成器，每次 `yield` 立即推给前端——SSE 逐字效果的原理：**整个请求生命周期内只建立一个 HTTP 连接，连接不断，数据按事件持续推送**。

**② 历史解析**（[core/retrieval.py](../backend/core/retrieval.py)）
`_resolve_history`：请求显式传 history 时优先使用；否则从 `memory.load_messages()` 读 SQLite 最近 `MAX_HISTORY_TURNS=6` 轮。SQLite 保留最近 50 轮供页面恢复，请求层只携带 6 轮，避免生成上下文过长。

**③ 答案缓存**
`answer_cache` 的 key 包含 collection、规范化 query、Top-K、history 指纹、LLM/温度、Query/HYDE、检索模式、Rerank 开关和模型。命中后仍按 SSE 协议发送 `delta + sources + [DONE]` 并写对话记忆。上传、删除、替换文档时调用 `invalidate()` 清空。

**④ Query Planning**（[core/retrieval.py](../backend/core/retrieval.py) → [core/query_transform.py](../backend/core/query_transform.py)）
- `RETRIEVAL_MODE=fast`：只检索原 query，不做远程 Rerank。
- `RETRIEVAL_MODE=balanced`（默认）：只有指代/承接类追问或长度 ≤12 的问题才调用 Rewrite；不做 HYDE。
- `RETRIEVAL_MODE=quality`：按 `QUERY_TRANSFORM` / `HYDE` 开关开启增强；带精确标识符的问题仍跳过增强。
- `transform_query` 使用最近 6 轮历史补全指代；`hyde_query` 生成假想答案。
- 两个 LLM 调用可并行；`lru_cache` 按 prompt 缓存；任何异常都降级为原 query，不中断主链路。
- HYDE 只进入 dense 查询集，不进入 BM25，避免假想文本中的关键词污染稀疏召回。

**⑤ 混合召回**
- 原 query 的向量和 BM25 检索与其他增强任务并行启动。
- Dense：`milvus.similarity_search_many()` 先批量 embedding，再逐查询走 `similarity_search_by_vector`；批量失败时逐 query 降级。
- Sparse：`bm25.keyword_search()` 使用 jieba + rank_bm25，索引 child 的 `raw_text`，不带重复的章节前缀。
- 候选数按模式调整：fast 最小，balanced 中等，quality 使用 `RETRIEVAL_CANDIDATE_K`。

**⑥ RRF 融合**
每路结果保留自己的排名，用 `1 / (60 + rank)` 累加 Reciprocal Rank Fusion 分数。候选身份优先用 Milvus `pk`，其次 `parent_id + child_index`，最后才用文本哈希，因此相同文本但不同来源不会被误删。融合分数写入 `metadata.rrf_score`。

**⑦ Rerank 精排**（[core/rerank.py](../backend/core/rerank.py)）
`rerank(query, candidates, top_n)` 调 SiliconFlow `/rerank`（bge-reranker-v2-m3）：
- query 与每个候选块**联合编码**（cross-encoder），输出相关性分数
- 按分数降序取 top_n，分数写入 `metadata.relevance_score`
- **注意：用原始 query 做相关性判断**（用户原意），不用改写/假文档——保证最终排序贴合用户意图
- `fast` 模式跳过远程 Rerank，直接取 RRF 前 Top-K
- Rerank 请求失败时记录 warning，并回退 RRF 顺序

**⑧ 可答性门控**
若已拿到 `relevance_score`，且候选最高分低于 `RETRIEVAL_MIN_RERANK_SCORE`，则判定当前资料库不足以回答，返回空结果和“资料中未找到相关答案”的兜底文案。门控关闭或无分数时不拦截。

**⑨ Parent 上下文回填**
命中的是 child，但送给 LLM 的内容会从 `parent_child.py` 原文重建。默认 `PARENT_WINDOW_RADIUS=1`，只取命中 child 前后各一个兄弟块；设为 `-1` 回填完整 parent。重建结果保留 section、page_start/page_end、matched_chunk_index、parent_window_start/end。

**⑩ 构建 messages**（[core/retrieval.py](../backend/core/retrieval.py)）
```
1. system: 系统提示词 + 检索到的参考资料（按 [i] 编号 + 来源文件名）
2. user/assistant 交替：历史对话（最近 6 轮）
3. user: 当前问题
```
关键设计：`_format_context` 要求 LLM 只依据资料回答、不编造、数字以原文为准，并要求事实性结论使用 `[n]` 引用。历史可用于 Query Rewrite，但候选排序和 Rerank 始终以当前问题为主。

**⑪ 流式生成**
`get_rag_llm().astream(messages)` 使用低温度 RAG LLM；每个非空 chunk 包装为 `data: {"delta": "..."}\n\n` 推送。完整回答同时累积，供引用选择、缓存和记忆持久化。

**⑫ 来源事件**
流结束后解析回答中的 `[n]`，只保留实际引用的文档，并按引用出现顺序返回 `data: {"sources": [...]}`。每条 source 含 `citation_index` / `filename` / chunk/page 元数据 / `content`。如果模型未输出引用，为保证可追溯性返回全部候选来源。

**⑬ 缓存与记忆**
完整回答和引用写入 `answer_cache`；`memory.add_message()` 写入 SQLite，自动截断到最近 100 条（50 轮）。**空库/门控兜底场景不写答案缓存**。

**SSE 事件流示例**：
```
data: {"delta": "根据"}
data: {"delta": "文档"}
data: {"delta": "内容"}
data: {"sources": [{"citation_index": 1, "filename": "notes.md", "chunk_index": 2, "page": null, "content": "..."}]}
data: [DONE]
```

### 3.3 单轮问答 `POST /chat` 的差异

与流式流程完全一致（同样的缓存、Query Planning、混合召回、RRF、rerank、门控、Parent 回填），仅输出方式不同：
- 调用 `_generate_answer()` 同步等待完整回答，返回 `ChatResponse {answer, sources}`
- 记忆写入在 api 层（[api/chat.py](../backend/api/chat.py)），空库回答不存

### 3.4 历史记忆：`GET /chat/memory`

`memory.load_messages()` 按时间升序返回全部消息（最多 100 条）。前端 `onMounted` 调用，把返回的 messages 填入本地数组恢复对话——**刷新页面/重启后端后对话自动续上**。

### 3.5 前端如何消费 SSE

[frontend/src/api.js](../frontend/src/api.js) 的 `chatStream()`：
1. `fetch` 发起 `POST /chat/stream`，不 `await` 响应体，直接拿 `response.body` 的 ReadableStream。
2. `TextDecoder` 解码二进制流 → 按 `\n\n` 切分事件 → 解析每行 `data: {...}`。
3. 事件分发：`delta` 追加到当前气泡（逐字效果）、`sources` 存起来、`answer` 显示兜底文案、`[DONE]` 结束。

[ChatPanel.vue](../frontend/src/components/ChatPanel.vue) 渲染层：
- **打字机**：SSE 文本先入 `pending` 队列，`setInterval(16ms)` 逐字刷出；真实流式 chunk 到达 ~25ms 直接透传，API 聚合的大段文本被平滑逐字渲染
- **思考动画**：首 token 前（LLM 生成中）显示三个弹跳点
- **Markdown**：assistant 内容交给 `MarkdownContent.vue`，marked 解析 GFM 后用 DOMPurify 清洗，支持表格、代码块、列表和安全链接
- **响应式陷阱**：`messages.value.push()` 后必须**取回数组存储的代理引用**再改 `content`（直接操作原对象不走 Vue setter，不触发渲染）

---

## 4. 前后端交互时序

```
前端                        FastAPI                         Milvus              LLM/Embedding/Rerank
 │                              │                              │                      │
 │  POST /documents/upload ────▶│                              │                      │
 │  (multipart + split_mode)    │ 校验格式/split_mode/大小       │                      │
 │                              │ 创建 job → 返回 job_id        │                      │
 │ ◀── 200 {job_id, status=pending}                            │                      │
 │                              │                              │                      │
 │                              │ [后台任务]                    │                      │
 │                              │ 解析 → 清洗 → 分块            │                      │
 │                              │ ──▶ add_documents ──────────▶│                      │
 │                              │                              │ Embedding 调用 ──────▶│
 │                              │                              │── 1024 维向量 ───────│
 │                              │◀─ 写入完成 ──────────────────│                      │
 │                              │ 更新 job status=completed    │                      │
 │                              │                              │                      │
 │  GET /documents/jobs/{id} ──▶│◀─ 查询 SQLite ──────────────│                      │
 │ ── 200 {status, progress}  │                              │                      │
 │                              │                              │                      │
 │  POST /chat/stream ────────▶│                              │                      │
 │  {query}                     │ Query Planning（按模式增强）─▶────────────────────▶│
 │                              │ 向量+BM25 多路检索 ──────────▶│ Embedding 调用 ──────▶│
 │                              │◀─ RRF 融合候选池 ────────────│                      │
 │                              │ rerank + 可答性门控 ─────────▶────────────────────▶│
 │                              │ Parent 窗口回填 + [n] 引用     │                      │
 │                              │ rag LLM astream ─────────────▶────────────────────▶│
 │ ◀── data: {"delta":"..."} ──│─ token 流 ──────────────────│◀── token 流 ─────────│
 │ ◀── data: {"sources":[...]} │                              │                      │
 │ ◀── data: [DONE]            │                              │                      │
 │                              │ 写记忆 → SQLite              │                      │
 │  GET /chat/memory ─────────▶│─ 读取 SQLite ───────────────│                      │
 │ ◀── {messages: [...]}       │                              │                      │
```

---

## 5. 关键设计决策与注意事项

| 设计 | 说明 | 影响 |
|---|---|---|
| 路由层无业务逻辑 | 校验 + 转发，业务全在 `core/` | 易于测试、复用 |
| 异步上传 + 进度查询 | 后台任务处理，前端轮询进度 | 大文件上传体验好，不阻塞 HTTP |
| 结构化解析（v2） | PDF/DOCX 表格转 Markdown、标题层级识别 | 表格类问答质量提升 |
| Markdown 分块 | 按章节切分、表格保护、每块带章节上下文 | 检索更精准，引用定位更准 |
| VLM 多模态 | 扫描件 OCR、内嵌图片描述 | 图文混排文档信息不丢失 |
| 内容哈希去重 | SHA256 哈希，避免重复入库 | 同文件重复上传返回 409（可 replace 覆盖） |
| 原件落盘 | `backend/uploads/` 哈希命名 | 支持后续重解析/迁移重建 |
| 检索模式分档 | fast 低延迟；balanced 默认兼顾延迟与追问；quality 完整 Rewrite+HYDE | 不再每轮固定付两次 LLM 增强成本 |
| 混合召回 | 向量（语义）+ BM25（关键词）互补 | 专有名词/代码片段靠 BM25 补漏 |
| RRF 融合 | 保留各路排名，按稳定块 ID 融合 | 避免大库中相同文本被错误去重 |
| Rerank 用原始 query | 改写/假文档只用于召回，不参与精排 | 最终排序贴合用户原意 |
| Rerank 降级 | API 异常回退 RRF；fast 模式完全跳过 | Rerank 故障不拖垮问答 |
| 可答性门控 | rerank 最高分低于阈值时不生成事实回答 | 降低低相关上下文导致的幻觉 |
| `[n]` 引用 | Prompt 要求句末标注，后端反查实际引用 | 来源卡片与回答一一对应 |
| Answer Cache | 相同 query/history/model/检索配置复用 | 重复问题降延迟；集合变化自动失效 |
| 记忆 SQLite 持久化 | 存 50 轮、请求层取 6 轮 | 刷新恢复对话，上下文不膨胀 |
| Collection 按模型命名 | `doc_collection_BAAI_bge_m3` | 换 Embedding 模型不冲突，但旧库作废需重传 |
| Collection schema 定死 | `metadata_schema` 显式声明新字段 | 加字段必须重建 collection（数据丢失） |
| pymilvus 直连 | `get_all_documents()` 等不依赖 embedding | `EMBEDDING_API_KEY` 为空时文档列表仍正常 |

---

## 6. 常见排查点

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| `/documents` 空列表 | collection 尚未创建（首次上传前） | 先上传任意文档；或看后端日志 |
| 回答与资料无关 | Embedding 模型换了但旧库未重建 | 换模型后需重新上传全部文档 |
| 流式卡住无输出 | LLM API 不可用/超时 | 看后端日志 httpx 请求状态；直接 curl 测 LLM 接口 |
| 首 token 很慢 | quality 模式叠加 Rewrite/HYDE/Rerank | 改用 `RETRIEVAL_MODE=balanced` 或 `fast`；检查 LangSmith |
| 改 .env 不生效 | `get_settings()` lru_cache | 重启后端进程 |
| 上传报错 `文档已入库` | 内容哈希重复 | 前端会提示是否覆盖；或 API 传 `replace=true` |
| 上传报错 `文档处理失败` | 文件损坏/加密 PDF | 后端日志看具体异常栈 |
| 删除后列表没变 | 文件名含特殊字符导致表达式匹配失败 | 用 `GET /documents` 确认准确文件名 |
| VLM 处理页数为 0 | 未配置 `VLM_API_KEY` 或 `VLM_ENABLED=false` | 检查 `.env` 配置；同图缓存命中也计 0 |
| 上传的 chunk_type 是 None | 旧 collection（无该字段） | 重建 collection 后重新上传 |
| 重复问题仍重新调用 LLM | history 或检索配置不同，answer cache 未命中 | 查看 trace/key 组成；确认没有隐式 history |

---

## 7. RAGAS 评估（离线效果评价）

> 适用：对整套 RAG 链路（检索 + 生成）做量化评价，输出 faithfulness / answer_relevancy / context_precision / context_recall 四项指标。

### 7.1 测试集

`docs/test.py`（或 `RAGAS_TESTSET` 指定的文件）包含两个等长列表：

- `questions` — 与资料库内容对应的问题
- `ground_truths` — 每题的标准答案（列表包字符串）

### 7.2 运行

```bash
cd backend && ./.venv-ragas/Scripts/python.exe -X utf8 scripts/eval_ragas.py
```

**必须用独立 venv `.venv-ragas/`**（ragas 0.4.3 会降级 langchain-core 到 0.3.x，与主环境 1.5.3 冲突，装进主 venv 会崩）。版本锁定：pymilvus 2.5.18、langchain-milvus 0.1.10、langchain-core 1.5.3。

### 7.3 执行流程

```
questions/ground_truths（docs/test.py）
        │
        ▼
① 逐条走真实检索链路 retrieval._retrieve(query, TOP_K=5)
        │   （Query Planning → 向量+BM25 → RRF → rerank/门控，与 /chat 完全一致）
        ▼
② LLM 生成回答（同一套 SYSTEM_PROMPT）
        │
        ▼
③ 组 ragas 数据集：user_input / response / reference / retrieved_contexts
        │
        ▼
④ 4 指标判分（Judge LLM = .env 的 DeepSeek，Embedding = SiliconFlow bge-m3）
        │   faithfulness / answer_relevancy / context_precision / context_recall
        ▼
⑤ 输出逐条明细 + RAGAS 指标 + 检索/生成延迟 + context/来源数统计
```

**每步细节**：

- **生成缓存**：①-② 结果按 question 存 `.ragas_cache.json`（response + contexts + run metrics）。文件带 `version + fingerprint`，指纹覆盖 collection、文档块/上传时间、Embedding、Rerank、Query、HYDE、RETRIEVAL_MODE、Top-K、分块与 parent 参数。
- **缓存失效**：指纹不匹配时旧缓存自动忽略，无需手工删除；`--no-cache` 或 `RAGAS_NO_CACHE=1` 强制重新生成。
- **判分**：指标用 `ragas.metrics` 单例（`faithfulness` 等）；`ragas.metrics.collections` 里的类是 `SimpleBaseMetric` 体系，过不了 `evaluate` 的 `isinstance(m, Metric)` 校验，不能用。判分 LLM 用 `ChatOpenAI` + 项目 `get_embeddings()`（`llm_factory`/`embedding_factory` 的现代实现不兼容旧指标）。
- **连接**：脚本模块级 `connections.connect(alias="default", uri="http://host:port")`（**必须 uri 形式**，只传 host/port 走环境变量分支报 ConnLackConf）。

### 7.4 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAGAS_TESTSET` | `docs/test.py` | 测试集路径（相对项目根目录） |
| `RAGAS_NO_CACHE=1` | 关 | 忽略指纹缓存，强制重新生成 |
| `RAGAS_CACHE` | `backend/.ragas_cache.json` | 缓存文件路径 |
| `RAGAS_LLM_MODEL` / `RAGAS_LLM_BASE_URL` / `RAGAS_LLM_API_KEY` | 复用 .env LLM | 单独指定判分模型 |

### 7.5 已知局限

- 判分 LLM 偶发输出截断（`IncompleteOutputException`）→ 该题指标为 NaN，重跑即可
- 检索失败（返回默认"未检索到"文案）时 answer_relevancy=0，属真实信号（如 #3 api_base 题）
- 50 题全量约 12 分钟（生成 7 分钟 + 判分 5 分钟），命中缓存后仅判分 ~6 分钟
