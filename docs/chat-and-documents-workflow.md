# Chat 与 Documents 工作流程详解

# 本文档的输出提示词：“给我一份文档，对现在项目chat和document各部分的具体工作流程进行详细的解释和说明”

> 适用版本：当前 main 分支（后端 FastAPI + 前端 Vue 3）。
> 阅读本文前建议先看根目录 `CLAUDE.md` 了解整体架构；本文深入每个接口的内部执行流程。

---

## 1. 总览

系统由两大模块组成：

- **Documents（文档管理）**：上传 → 解析 → 分块 → 向量化 → 入库；以及列表、删除。
- **Chat（智能问答）**：向量检索 → 拼上下文 → LLM 生成 → 返回答案 + 来源；支持多轮记忆与 SSE 流式输出。

```
┌─────────┐   HTTP   ┌──────────────┐   调用    ┌──────────────────┐
│ 前端 Vue │ ───────▶ │  FastAPI 路由 │ ───────▶ │  core 业务层      │
└─────────┘          │  api/*.py    │          │  retrieval /      │
                     └──────────────┘          │  ingestion / llm  │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │ db/milvus.py     │
                                              │  (langchain_milvus)│
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │ Milvus 向量库     │
                                              └──────────────────┘
```

每个模块的完整调用链：

| 功能 | 路由 | 调用链 |
|---|---|---|
| 上传文档 | `POST /documents/upload` | `api/documents.py` → `core/ingestion.py` → `db/milvus.py` → Milvus |
| 文档列表 | `GET /documents` | `api/documents.py` → `core/ingestion.py` → `db/milvus.py` → Milvus |
| 删除文档 | `DELETE /documents/{filename}` | `api/documents.py` → `core/ingestion.py` → `db/milvus.py` → Milvus |
| 单轮问答 | `POST /chat` | `api/chat.py` → `core/retrieval.py` → `db/milvus.py` → `core/llm.py` |
| 流式问答 | `POST /chat/stream` | `api/chat.py` → `core/retrieval.py` → `db/milvus.py` → `core/llm.py` |

> **重要：所有路由本身不含业务逻辑**，只做参数校验、请求体解析和异常 → HTTP 状态码的转换；真正的流程在 `core/` 和 `db/` 层。这样设计的好处是业务逻辑可以脱离 HTTP 单独测试/复用。

---

## 2. Documents 模块工作流程

### 2.1 上传文档：`POST /documents/upload`

**请求**：`multipart/form-data`，字段 `file`（支持 PDF / DOCX，最大 20MB）。

**执行步骤**：

1. **格式校验**（[api/documents.py:17-18](backend/api/documents.py#L17-L18)）
   从 `file.filename` 提取扩展名，仅接受 `pdf` / `docx`，否则返回 `400 仅支持 PDF / DOCX 文件`。

2. **大小校验**（[api/documents.py:20-26](backend/api/documents.py#L20-L26)）
   一次性读入全部字节（`await file.read()`），按 `MAX_UPLOAD_MB`（默认 20MB）判断，超限返回 `413`。

3. **摄入处理** `ingestion.ingest_file(filename, content)`（[core/ingestion.py:66](backend/core/ingestion.py#L66)），内部 4 步：
   - **写临时文件**：`NamedTemporaryFile(suffix=ext)`。因为 `PyPDFLoader` / `Docx2txtLoader` 只接受文件路径，不接受字节流。
   - **解析** `_load_documents`：按扩展名选 loader（`SUPPORTED_EXTENSIONS` 映射）→ `loader.load()` 得到原始文档对象（PDF 每页一个 Document，DOCX 整体一个）。
   - **分块** `_split_documents`：`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`。按递归分隔符（段落 → 句子 → 字符）切分，相邻块重叠 50 字符，防止跨块语义被切断。**注意：分块粒度由 `CHUNK_SIZE` / `CHUNK_OVERLAP` 配置决定，是影响检索质量的关键参数。**
   - **补充元数据** `_make_metadata_docs`：为每个块添加 `filename`、`chunk_index`（块序号）、`upload_time`（unix 秒），PDF 额外保留 `page`（页码）；DOCX 无页码，`page` 为 `None`。

4. **写入向量库** `milvus.add_documents(enriched)`（[db/milvus.py:36](backend/db/milvus.py#L36)）：
   - 通过 `get_vectorstore()` 建立 langchain_milvus 连接（内部调用 `core/embeddings.py` 的 OpenAI 兼容 Embedding API 把文本转成 1024 维向量）。
   - `vs.add_documents(docs)`：文本 → 向量 → 连同元数据一起写入 Milvus Collection。
   - **Collection 不存在时自动创建**，命名规则见 `get_collection_name()`（`doc_collection_` + Embedding 模型名 slug）。

5. **清理与返回**：`finally` 中删除临时文件（无论成败）；返回 `{filename, chunk_count, ids}`。

**异常处理**：
- `ValueError`（空文档/无法解析）→ `400`，前端展示错误信息。
- 其他异常 → `500 文档处理失败: <原因>`。

**响应示例**：
```json
{ "filename": "test_doc.docx", "chunk_count": 1, "ids": [12345] }
```

### 2.2 文档列表：`GET /documents`

**执行步骤**：

1. `milvus.get_all_documents()`（[db/milvus.py:49](backend/db/milvus.py#L49)）：
   - 用底层 `vs.col.query` 分页拉取 Collection 全部数据（每批 100 条，`offset` 递增直到取完）。
   - 每条数据重建为 `Document`，元数据含 `pk`、`filename`、`chunk_index`、`upload_time`。

2. `ingestion.list_documents()`（[core/ingestion.py:91](backend/core/ingestion.py#L91)）：
   - 按 `filename` 聚合：同文件的所有 chunk 合并为一条记录，`chunk_count` 累计块数。
   - 按 `upload_time` 降序排序（新上传的在前）。

3. 返回 `{documents: [...], total: N}`。

**响应示例**：
```json
{
  "documents": [
    { "filename": "test_doc.docx", "chunk_count": 1, "upload_time": 1785771513 }
  ],
  "total": 1
}
```

### 2.3 删除文档：`DELETE /documents/{filename}`

**执行步骤**：

1. `ingestion.delete_document(filename)` → `milvus.delete_document(filename)`（[db/milvus.py:80](backend/db/milvus.py#L80)）：
   - 构造表达式 `filename == "<filename>"`，用底层 `vs.col.delete` 删除**该文件名对应的全部向量块**。
   - 返回 `delete_count`（删除条数）。

2. 删除 0 条 → `404 未找到文档`；否则返回 `{filename, deleted}`。

**注意**：
- **按文件名精确匹配**，同名文件会被一起删除。
- 该操作不可撤销，删除的是 Milvus 中的向量，不是磁盘文件。

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
  "history": [                          // 可选，多轮记忆（不含当前问题）
    { "role": "user", "content": "上一轮问题" },
    { "role": "assistant", "content": "上一轮回答" }
  ]
}
```

### 3.2 完整流程（以 `/chat/stream` 为例）

```
前端发送 query + history
        │
        ▼
① api/chat.py 把 history 转成 dict 列表
        │
        ▼
② retrieval.rag_stream() 开始生成 SSE 事件
        │
        ├─ ③ milvus.similarity_search(query, k=top_k)
        │      └─ 文本 → Embedding 向量 → Milvus 向量相似度检索 → Top-K 文档块
        │
        ├─ ④ 无结果？ → yield {"answer": "资料库中尚未检索到相关内容..."} → 结束
        │
        ├─ ⑤ _build_messages() 拼出 messages（system + history + query）
        │
        ├─ ⑥ chat.astream(messages) 流式调用 LLM
        │      └─ 每个 chunk → yield {"delta": "..."}   ← 逐字推送
        │
        ├─ ⑦ 结束后 yield {"sources": [...]}            ← 引用来源
        │
        └─ ⑧ yield "data: [DONE]"                      ← 流结束标记
```

**逐步说明**：

**① 请求预处理**（[api/chat.py:22](backend/api/chat.py#L22)）
`request.history`（Pydantic 对象列表）转为普通 dict 列表，供 LLM 消息构建使用。Pydantic 校验 `query` 非空、`top_k` 在 1-20。

**② 流式生成器**（[api/chat.py:24-28](backend/api/chat.py#L24-L28)）
`chat_stream` 返回 `StreamingResponse(event_stream(), media_type="text/event-stream")`。`event_stream` 是异步生成器，每次 `yield` 立即推给前端——这就是 SSE 逐字效果的原理：**整个请求生命周期内只建立一个 HTTP 连接，连接不断，数据按事件持续推送**。

**③ 向量检索**（[core/retrieval.py:68](backend/core/retrieval.py#L68) → [db/milvus.py:42](backend/db/milvus.py#L42)）
`milvus.similarity_search(query, k=top_k)`：
1. 用户 query 经 `core/embeddings.py` 转成 1024 维向量（与入库时同一模型 `BAAI/bge-m3`，保证向量空间一致）。
2. 在 Milvus 中做相似度检索（默认欧氏/余弦），返回相似度最高的 `k` 个文档块（`k` 默认 5）。
3. 返回 `list[Document]`，每个 Document 含 `page_content`（块文本）和 `metadata`（filename / chunk_index / page）。

**④ 空库兜底**（[core/retrieval.py:69-71](backend/core/retrieval.py#L69-L71)）
检索结果为空（库空 / 无相似内容）时，**不调用 LLM**，直接返回固定提示，避免浪费 API 调用且避免模型瞎编。

**⑤ 构建 messages**（[core/retrieval.py:33](backend/core/retrieval.py#L33)）

`_build_messages(query, docs, history)` 生成发送给 LLM 的消息列表：
```
1. system: 系统提示词 + 检索到的参考资料（按 [i] 编号 + 来源文件名）
2. user/assistant 交替: 历史对话（最近 MAX_HISTORY_TURNS=6 轮）
3. user: 当前问题
```

关键设计：
- **参考资料拼接**：`_format_context` 把 Top-K 块拼成 `[1] (来源: xxx.docx)\n块内容\n\n[2] ...`，要求 LLM 只依据资料回答、不编造、数字以原文为准。
- **多轮记忆**：`history` 由前端携带（后端无状态）。截取 `history[-12:]`（6 轮 × 2 条消息）防止上下文过长稀释检索结果——检索只针对**当前问题**，历史仅作为对话上下文。
- 检索与 LLM 生成解耦：每次提问都重新检索，历史不参与检索，**保证回答永远基于最新入库的文档**。

**⑥ 流式生成**（[core/retrieval.py:83](backend/core/retrieval.py#L83)）
`chat.astream(messages)` 异步迭代 LLM 输出，每个非空 chunk 包装为 SSE 事件 `data: {"delta": "..."}\n\n` 推送。**全程只 await 不阻塞**，事件循环可同时服务其他请求（这是 asyncio 在该项目中的实际应用）。

**⑦ 来源事件**（[core/retrieval.py:86](backend/core/retrieval.py#L86)）
LLM 输出结束后推送 `data: {"sources": [...]}`，每条含 `filename` / `chunk_index` / `page` / `content`（完整块文本）。前端据此展示「引用来源」折叠卡片。

**⑧ 结束标记**（[core/retrieval.py:87](backend/core/retrieval.py#L87)）
`data: [DONE]` 通知前端流已完整结束，可停掉加载状态。

**SSE 事件流示例**：
```
data: {"delta": "根据"}
data: {"delta": "文档"}
data: {"delta": "内容"}
data: {"sources": [{"filename": "test_doc.docx", "chunk_index": 0, "page": null, "content": "..."}]}
data: [DONE]
```

### 3.3 单轮问答 `POST /chat` 的差异

与流式流程完全一致（同样的检索、同样的消息构建），仅两处不同：
- 调用 `chat.invoke(messages)` 同步等待完整回答，返回 `ChatResponse {answer, sources}`。
- 空库时返回同样兜底文案 + 空 sources。

### 3.4 前端如何消费 SSE

[frontend/src/api.js](frontend/src/api.js) 的 `chatStream()`：
1. `fetch` 发起 `POST /chat/stream`，不 `await` 响应体，直接拿 `response.body` 的 ReadableStream。
2. `TextDecoder` 解码二进制流 → 按 `\n\n` 切分事件 → 解析每行 `data: {...}`。
3. 事件分发：`delta` 追加到当前气泡（逐字效果）、`sources` 存起来、`answer` 显示兜底文案、`[DONE]` 结束。
4. 每次提问前，前端把全部历史消息（role + content）组装成 `history` 数组随请求发送。

---

## 4. 前后端交互时序

```
前端                        FastAPI                         Milvus              LLM/Embedding API
 │                              │                              │                      │
 │  POST /documents/upload ────▶│                              │                      │
 │  (multipart file)            │ 校验格式/大小                  │                      │
 │                              │ 写临时文件 → 解析 → 分块       │                      │
 │                              │ 补元数据 ──▶ add_documents ──▶│                      │
 │                              │                              │ Embedding 调用 ──────▶│
 │                              │                              │◀── 1024 维向量 ───────│
 │                              │◀─ 写入完成 ──────────────────│                      │
 │ ◀── 200 {filename,chunk_count,ids}                          │                      │
 │                              │                              │                      │
 │  POST /chat/stream ────────▶│                              │                      │
 │  {query, history}            │ 检索 query 向量化 ──────────▶│ Embedding 调用 ──────▶│
 │                              │                              │◀── 1024 维向量 ───────│
 │                              │◀─ Top-K 文档块 ──────────────│                      │
 │                              │ 拼 system+history+query      │                      │
 │                              │ chat.astream ────────────────▶─────────────────────▶│
 │ ◀── data: {"delta":"..."} ──│◀─ token 流 ──────────────────│◀── token 流 ──────────│
 │ ◀── data: {"sources":[...]} │                              │                      │
 │ ◀── data: [DONE]            │                              │                      │
```

---

## 5. 关键设计决策与注意事项

| 设计 | 说明 | 影响 |
|---|---|---|
| 路由层无业务逻辑 | 校验 + 转发，业务全在 `core/` | 易于测试、复用 |
| 分块参数可配置 | `CHUNK_SIZE=500` / `CHUNK_OVERLAP=50` | 直接影响检索质量，调优入口 |
| Collection 按模型命名 | `doc_collection_BAAI_bge_m3` | 换 Embedding 模型不冲突，但旧库作废需重传 |
| 历史由前端携带 | 后端无状态 | 刷新页面即失忆；无跨设备同步 |
| 检索只针对当前问题 | 历史不参与向量检索 | 多轮追问时可能丢失上下文（如"它呢？"） |
| 无去重 | 同文件重复上传会重复入库 | 删除时按文件名精确匹配，会一并删除 |
| 上传无事务 | 部分块写入失败无回滚 | 可能残留部分向量 |
| SSE 单连接长连接 | 流式期间占用一个连接 | 与普通请求并存，事件循环不阻塞 |

---

## 6. 常见排查点

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| `/documents` 500 `NoneType has no attribute query` | Collection 尚未创建 | 先上传任意文档（首次上传自动建库）；或启动时看日志 |
| 回答与资料无关 | Embedding 模型换了但旧库未重建 | 换模型后需重新上传全部文档 |
| 流式卡住无输出 | LLM API 不可用/超时 | 看后端日志的 httpx 请求状态；直接 curl 测试 LLM 接口 |
| 删除后列表没变 | 文件名含特殊字符导致表达式匹配失败 | 用 `GET /documents` 确认准确文件名 |
| 上传报错 `文档处理失败` | 文件损坏/加密 PDF | 后端日志看具体异常栈 |
