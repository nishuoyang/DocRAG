# Chat 与 Documents 工作流程详解

> 适用版本：v2（结构化解析 + Markdown 分块 + VLM 多模态 + 异步上传）。
> 阅读本文前建议先看根目录 `CLAUDE.md` 了解整体架构；本文深入每个接口的内部执行流程。

---

## 1. 总览

系统由两大模块组成：

- **Documents（文档管理）**：上传 → 结构化解析（v2）→ 噪声清洗 → Markdown 分块 → 向量化 → 入库；以及列表、删除、进度查询。
- **Chat（智能问答）**：Query 增强（改写+HYDE）→ 混合检索（向量+BM25）→ Rerank 精排 → LLM 生成 → 答案+来源；支持 SQLite 持久化记忆与 SSE 流式输出。

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

1. **格式校验**（[api/documents.py:27-29](backend/api/documents.py#L27-L29)）
   从 `file.filename` 提取扩展名，白名单 = `ingestion.SUPPORTED_EXTENSIONS` + xlsx/pptx（单一事实源）。

2. **split_mode 校验**（[api/documents.py:31-32](backend/api/documents.py#L31-L32)）
   仅接受 `auto` / `markdown` / `semantic` / `fixed`，其他返回 400。

3. **大小校验**（[api/documents.py:34-40](backend/api/documents.py#L34-L40)）
   一次性读入全部字节（`await file.read()`），按 `MAX_UPLOAD_MB`（默认 20MB）判断，超限返回 413。

4. **创建后台任务**（[api/documents.py:42-54](backend/api/documents.py#L42-L54)）
   - `job_manager.create_job(filename)` 创建任务记录（SQLite），返回 `job_id`
   - `background_tasks.add_task(_process_document_task, ...)` 添加后台任务
   - 立即返回 `JobInfoResponse`（job_id、filename、status=pending、progress=0）

5. **后台任务处理** `_process_document_task`（[api/documents.py:71-138](backend/api/documents.py#L71-L138)）
   - 更新状态为 `processing`，progress=10
   - 调用 `ingestion.ingest_file()`（[core/ingestion.py:284](backend/core/ingestion.py#L284)），内部流程：
     - **SHA256 去重**：计算文件哈希，若已入库且 `replace=false` 则抛 `DuplicateFileError`
     - **写临时文件**：`NamedTemporaryFile(suffix=ext)`。因为 loader 只接受文件路径，不接受字节流。
     - **v2 解析** `_load_documents`（[core/parsers/__init__.py](backend/core/parsers/__init__.py)）：
       - PDF → `pymupdf4llm` 逐页输出 Markdown（表格转 GFM、标题带 `#` 层级）
       - DOCX → `python-docx` 顺序遍历 body（标题 Heading 1-9 → `#` 层级、表格转 GFM）
       - PPTX → `python-pptx` 逐页提取文本 + 表格
       - XLSX → `pandas` 按 sheet 分行
       - CSV/HTML/TXT/MD → langchain loader
     - **噪声清洗**（[core/cleaning.py](backend/core/cleaning.py)）：PDF 逐页剔除跨页重复行（页眉/页脚/页码）
     - **确定切分策略** `_resolve_split_mode`（[core/ingestion.py:221](backend/core/ingestion.py#L221)）：
       - 手动 `split_mode` 优先
       - `auto` 时 PDF/DOCX 走 `markdown`，其他按 `SEMANTIC_SPLIT` 配置（<3000 字符语义、否则固定）
     - **分块** `_split_documents`（[core/ingestion.py:234](backend/core/ingestion.py#L234)），返回 `(结果，实际策略)`：
       - **markdown**（[core/md_split.py](backend/core/md_split.py)）：按 `#` 标题切节 → 节内文本超长 Recursive 切分（每块前置 `[章节：path]`）→ GFM 表格永不切断、超大表按行分组重复表头 → `chunk_type="markdown"`
       - **fixed**：`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
       - **semantic**：按句切 → 批量 embedding → 相邻句余弦相似度 → 断点 = 低于 `均值 -1.5σ` → 按断点合并成块
     - **补充元数据** `_make_metadata_docs`（[core/ingestion.py:194](backend/core/ingestion.py#L194)）：为每个块添加 `filename`、`chunk_index`、`upload_time`、`chunk_type`、`section`（章节路径）、`content_type`（text/table）、`file_hash`（SHA256）；PDF/pptx 额外保留 `page`
     - **写入向量库** `milvus.add_documents(enriched)`（[db/milvus.py:51](backend/db/milvus.py#L51)）
     - **原件落盘** `_save_original`（[core/ingestion.py:260](backend/core/ingestion.py#L260)）：哈希命名存入 `backend/uploads/`，支持后续重解析
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

1. `job_manager.get_job(job_id)`（[core/jobs.py:87](backend/core/jobs.py#L87)）：
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

1. `milvus.get_all_documents()`（[db/milvus.py:64](backend/db/milvus.py#L64)）：
   - **直接用 pymilvus Collection 查询**（不依赖 embedding 配置，避免 `EMBEDDING_API_KEY` 为空时报错）
   - **动态检测可用字段**，兼容旧 schema（无 section/content_type/file_hash/page 字段也能正常读取）
   - 用底层 `col.query` 分页拉取全部数据（每批 100 条，`offset` 递增直到取完）
   - 每条数据重建为 `Document`，元数据含 `pk`/`filename`/`chunk_index`/`upload_time`/`chunk_type`/`section`/`content_type`/`file_hash`/`page`

2. `ingestion.list_documents()`（[core/ingestion.py:348](backend/core/ingestion.py#L348)）：
   - 按 `filename` 聚合：同文件所有 chunk 合并为一条记录，`chunk_count` 累计块数
   - 按 `upload_time` 降序排序（新上传的在前）

3. 返回 `{documents: [...], total: N}`

### 2.4 删除文档：`DELETE /documents/{filename}`

**执行步骤**：

1. `ingestion.delete_document(filename)` → `milvus.delete_document(filename)`（[db/milvus.py:115](backend/db/milvus.py#L115)）：
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
        ├─ ③ _build_search_queries: 原 query + 改写 + HYDE 并行（ThreadPoolExecutor）
        │      └─ LLM 失败 → 降级原 query（绝不抛出）
        │
        ├─ ④ 每个查询分别做：向量检索 (10 条) + BM25 检索 (10 条)
        │      └─ 全部合并去重（按文本内容）→ 候选池（最多 ~30 条）
        │
        ├─ ⑤ rerank.rerank(原始 query, 候选，top_n=top_k)
        │      └─ SiliconFlow bge-reranker-v2-m3 交叉编码 → 相关性分数排序
        │      └─ 未配置/关闭 → 跳过，取候选前 top_k
        │
        ├─ ⑥ 无结果？ → yield {"answer": "资料库中尚未检索到相关内容..."} → 结束
        │
        ├─ ⑦ _resolve_history: 前端 history 优先，否则 SQLite 记忆取最近 6 轮
        │
        ├─ ⑧ _build_messages: system(提示词 + 参考资料) + history + query
        │
        ├─ ⑨ chat.astream(messages) 流式调用 LLM
        │      └─ 每个 chunk → yield {"delta": "..."}   ← 逐字推送
        │
        ├─ ⑩ 结束后 yield {"sources": [...]}            ← 引用来源
        │
        ├─ ⑪ yield "data: [DONE]"                      ← 流结束标记
        │
        └─ ⑫ 写记忆：user=query, assistant=完整回答 → SQLite
```

**逐步说明**：

**① 请求预处理**（[api/chat.py:33-37](backend/api/chat.py#L33-L37)）
`chat_stream` 返回 `StreamingResponse(event_stream(), media_type="text/event-stream")`。`event_stream` 是异步生成器，每次 `yield` 立即推给前端——SSE 逐字效果的原理：**整个请求生命周期内只建立一个 HTTP 连接，连接不断，数据按事件持续推送**。

**③ 查询集构造**（[core/retrieval.py:46](backend/core/retrieval.py#L46) → [core/query_transform.py](backend/core/query_transform.py)）
- `transform_query`：LLM 把口语/指代不清的问题改写成检索友好查询（补全"它/这个"，提取关键词，扩展同义表述）
- `hyde_query`：LLM 生成 50-100 字假想答案文档（假设这段就是文档相关段落）
- 两者 `ThreadPoolExecutor(max_workers=2)` **并行**调用（总延迟 = max 而非相加）；`lru_cache` 缓存（同 query 不重复调 LLM）；**任何异常降级返回原 query**，主流程绝不中断

**④ 混合召回**（[core/retrieval.py:64](backend/core/retrieval.py#L64)）
每个查询分别做：
- 向量检索 `milvus.similarity_search(q, k=10)`：query → 1024 维向量 → Milvus 相似度检索
- BM25 检索 `bm25.keyword_search(q, k=10)`（[core/bm25.py](backend/core/bm25.py)）：jieba 中文分词 → 内存倒排索引 → 关键词得分排序。**缓存键含库内块数，上传/删除自动重建**
- 全部结果按文本内容合并去重

**⑤ Rerank 精排**（[core/rerank.py](backend/core/rerank.py)）
`rerank(query, candidates, top_n)` 调 SiliconFlow `/rerank`（bge-reranker-v2-m3）：
- query 与每个候选块**联合编码**（cross-encoder），输出相关性分数
- 按分数降序取 top_n，分数写入 `metadata.relevance_score`
- **注意：用原始 query 做相关性判断**（用户原意），不用改写/假文档——保证最终排序贴合用户意图

**⑦ 记忆解析**（[core/retrieval.py:61](backend/core/retrieval.py#L61)）
`_resolve_history`：前端传 history 用前端的；否则 `memory.load_messages()`（SQLite 全部）取最近 `MAX_HISTORY_TURNS=6` 轮（12 条）——**存储层保留 50 轮供"续聊"，请求层只取 6 轮防上下文过长稀释检索**。

**⑧ 构建 messages**（[core/retrieval.py:36](backend/core/retrieval.py#L36)）
```
1. system: 系统提示词 + 检索到的参考资料（按 [i] 编号 + 来源文件名）
2. user/assistant 交替：历史对话（最近 6 轮）
3. user: 当前问题
```
关键设计：`_format_context` 要求 LLM 只依据资料回答、不编造、数字以原文为准；检索与生成解耦——每次提问重新检索，历史不参与检索，保证回答基于最新入库文档。

**⑨ 流式生成**（[core/retrieval.py:126](backend/core/retrieval.py#L126)）
`chat.astream(messages)` 异步迭代 LLM 输出，每个非空 chunk 包装为 `data: {"delta": "..."}\n\n` 推送；`answer_parts` 同时累积完整回答（供记忆持久化）。**全程只 await 不阻塞**。

**⑩ 来源事件**
`data: {"sources": [...]}`，每条含 `filename` / `chunk_index` / `page` / `content`（完整块文本）。前端据此展示「引用来源」折叠卡片。

**⑫ 记忆持久化**（[core/retrieval.py:132](backend/core/retrieval.py#L132)）
流结束后 `memory.add_message("user", query)` + `memory.add_message("assistant", 完整回答)`。SQLite 自动截断到最近 100 条（50 轮）。**空库兜底场景不存**（无 LLM 回答）。`/chat` 单轮在 [api/chat.py:23-25](backend/api/chat.py#L23-L25) 存。

**SSE 事件流示例**：
```
data: {"delta": "根据"}
data: {"delta": "文档"}
data: {"delta": "内容"}
data: {"sources": [{"filename": "notes.md", "chunk_index": 2, "page": null, "content": "..."}]}
data: [DONE]
```

### 3.3 单轮问答 `POST /chat` 的差异

与流式流程完全一致（同样的查询集、混合召回、rerank、消息构建），仅两处不同：
- 调用 `chat.invoke(messages)` 同步等待完整回答，返回 `ChatResponse {answer, sources}`
- 记忆写入在 api 层（[api/chat.py:23-25](backend/api/chat.py#L23-L25)），空库回答不存

### 3.4 历史记忆：`GET /chat/memory`

`memory.load_messages()` 按时间升序返回全部消息（最多 100 条）。前端 `onMounted` 调用，把返回的 messages 填入本地数组恢复对话——**刷新页面/重启后端后对话自动续上**。

### 3.5 前端如何消费 SSE

[frontend/src/api.js](frontend/src/api.js) 的 `chatStream()`：
1. `fetch` 发起 `POST /chat/stream`，不 `await` 响应体，直接拿 `response.body` 的 ReadableStream。
2. `TextDecoder` 解码二进制流 → 按 `\n\n` 切分事件 → 解析每行 `data: {...}`。
3. 事件分发：`delta` 追加到当前气泡（逐字效果）、`sources` 存起来、`answer` 显示兜底文案、`[DONE]` 结束。

[ChatPanel.vue](frontend/src/components/ChatPanel.vue) 渲染层：
- **打字机**：SSE 文本先入 `pending` 队列，`setInterval(16ms)` 逐字刷出；真实流式 chunk 到达 ~25ms 直接透传，API 聚合的大段文本被平滑逐字渲染
- **思考动画**：首 token 前（LLM 生成中）显示三个弹跳点
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
 │  {query}                     │ 改写+HYDE 并行（LLM 2 次）────▶────────────────────▶│
 │                              │ 每个查询：向量+BM25 检索 ────▶│ Embedding 调用 ──────▶│
 │                              │◀─ 候选池（去重） ─────────────│                      │
 │                              │ rerank 精排（原始 query）─────▶────────────────────▶│
 │                              │ 拼 system+history+query       │                      │
 │                              │ chat.astream ────────────────▶────────────────────▶│
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
| Query 增强（改写+HYDE） | LLM 补全指代/生成假想答案，扩大召回 | 每轮问答 +2 次 LLM 调用（并行，~3.5s） |
| 混合召回 | 向量（语义）+ BM25（关键词）互补 | 专有名词/代码片段靠 BM25 补漏 |
| Rerank 用原始 query | 改写/假文档只用于召回，不参与精排 | 最终排序贴合用户原意 |
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
| 首 token 很慢（~7s） | 改写+HYDE+rerank 三次 API 调用 | `.env` 关 `QUERY_TRANSFORM`/`HYDE`/`RERANK_ENABLED` 可加速 |
| 改 .env 不生效 | `get_settings()` lru_cache | 重启后端进程 |
| 上传报错 `文档已入库` | 内容哈希重复 | 前端会提示是否覆盖；或 API 传 `replace=true` |
| 上传报错 `文档处理失败` | 文件损坏/加密 PDF | 后端日志看具体异常栈 |
| 删除后列表没变 | 文件名含特殊字符导致表达式匹配失败 | 用 `GET /documents` 确认准确文件名 |
| VLM 处理页数为 0 | 未配置 `VLM_API_KEY` 或 `VLM_ENABLED=false` | 检查 `.env` 配置 |
| 上传的 chunk_type 是 None | 旧 collection（无该字段） | 重建 collection 后重新上传 |

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
        │   （改写+HYDE → 向量+BM25 → rerank，与 /chat 完全一致）
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
⑤ 输出逐条明细 + 聚合统计（mean/min/分位数）
```

**每步细节**：

- **生成缓存**：①-② 结果按 question 存 `.ragas_cache.json`（response + contexts），重跑命中缓存、只重新判分（省 6-7 分钟）。**缓存键是 question 字符串**：新题自动生成并添加，同题覆盖。
- **缓存失效**：改了测试集题目、重新上传/删除文档、或调了检索参数（TOP_K/分块/rerank）后，**必须 `RAGAS_NO_CACHE=1` 强制重新生成**，否则评估结果是旧库的。
- **判分**：指标用 `ragas.metrics` 单例（`faithfulness` 等）；`ragas.metrics.collections` 里的类是 `SimpleBaseMetric` 体系，过不了 `evaluate` 的 `isinstance(m, Metric)` 校验，不能用。判分 LLM 用 `ChatOpenAI` + 项目 `get_embeddings()`（`llm_factory`/`embedding_factory` 的现代实现不兼容旧指标）。
- **连接**：脚本模块级 `connections.connect(alias="default", uri="http://host:port")`（**必须 uri 形式**，只传 host/port 走环境变量分支报 ConnLackConf）。

### 7.4 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAGAS_TESTSET` | `docs/test.py` | 测试集路径（相对项目根目录） |
| `RAGAS_NO_CACHE=1` | 关 | 忽略缓存强制重新生成（换库/调参后必须） |
| `RAGAS_CACHE` | `backend/.ragas_cache.json` | 缓存文件路径 |
| `RAGAS_LLM_MODEL` / `RAGAS_LLM_BASE_URL` / `RAGAS_LLM_API_KEY` | 复用 .env LLM | 单独指定判分模型 |

### 7.5 已知局限

- 判分 LLM 偶发输出截断（`IncompleteOutputException`）→ 该题指标为 NaN，重跑即可
- 检索失败（返回默认"未检索到"文案）时 answer_relevancy=0，属真实信号（如 #3 api_base 题）
- 50 题全量约 12 分钟（生成 7 分钟 + 判分 5 分钟），命中缓存后仅判分 ~6 分钟
