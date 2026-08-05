# Chat 与 Documents 工作流程详解

> 适用版本：当前 main 分支（后端 FastAPI + 前端 Vue 3）。
> 阅读本文前建议先看根目录 `CLAUDE.md` 了解整体架构；本文深入每个接口的内部执行流程。

---

## 1. 总览

系统由两大模块组成：

- **Documents（文档管理）**：上传 → 解析 → 分块（固定/语义）→ 向量化 → 入库；以及列表、删除。
- **Chat（智能问答）**：Query 增强（改写+HYDE）→ 混合检索（向量+BM25）→ Rerank 精排 → LLM 生成 → 答案+来源；支持 SQLite 持久化记忆与 SSE 流式输出。

```
┌─────────┐   HTTP   ┌──────────────┐   调用    ┌────────────────────────┐
│ 前端 Vue │ ───────▶ │  FastAPI 路由 │ ───────▶ │  core 业务层            │
└─────────┘          │  api/*.py    │          │  ingestion / retrieval / │
                     └──────────────┘          │  query_transform /      │
                                              │  bm25 / rerank / llm    │
                                              └────────┬───────────────┘
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

| 功能 | 路由 | 调用链 |
|---|---|---|
| 上传文档 | `POST /documents/upload` | `api/documents.py` → `core/ingestion.py` → `db/milvus.py` → Milvus |
| 文档列表 | `GET /documents` | `api/documents.py` → `core/ingestion.py` → `db/milvus.py` → Milvus |
| 删除文档 | `DELETE /documents/{filename}` | `api/documents.py` → `core/ingestion.py` → `db/milvus.py` → Milvus |
| 历史记忆 | `GET /chat/memory` | `api/chat.py` → `db/memory.py`（SQLite） |
| 单轮问答 | `POST /chat` | `api/chat.py` → `core/retrieval.py` → query_transform/bm25/rerank/milvus → `core/llm.py` |
| 流式问答 | `POST /chat/stream` | 同上，SSE 输出 |

> **重要：所有路由本身不含业务逻辑**，只做参数校验、请求体解析和异常 → HTTP 状态码的转换；真正的流程在 `core/` 和 `db/` 层。这样设计的好处是业务逻辑可以脱离 HTTP 单独测试/复用。

---

## 2. Documents 模块工作流程

### 2.1 上传文档：`POST /documents/upload`

**请求**：`multipart/form-data`，字段 `file`（支持 PDF/DOCX/TXT/MD/CSV/XLSX/PPTX/HTML，最大 20MB）+ 可选 `split_mode`（`semantic` | `fixed` | 不传自动）。

**执行步骤**：

1. **格式校验**（[api/documents.py:17-20](backend/api/documents.py#L17-L20)）
   从 `file.filename` 提取扩展名，白名单 = `ingestion.SUPPORTED_EXTENSIONS` + xlsx/pptx（单一事实源）。

2. **split_mode 校验**（[api/documents.py:23-24](backend/api/documents.py#L23-L24)）
   仅接受 `semantic` / `fixed`，其他返回 400。

3. **大小校验**（[api/documents.py:28-32](backend/api/documents.py#L28-L32)）
   一次性读入全部字节（`await file.read()`），按 `MAX_UPLOAD_MB`（默认 20MB）判断，超限返回 413。

4. **摄入处理** `ingestion.ingest_file(filename, content, split_mode)`（[core/ingestion.py:203](backend/core/ingestion.py#L203)），内部 5 步：
   - **写临时文件**：`NamedTemporaryFile(suffix=ext)`。因为 loader 只接受文件路径，不接受字节流。
   - **解析** `_load_documents`：按扩展名选 loader（`SUPPORTED_EXTENSIONS` 映射 + 自写 `_load_xlsx` / `_load_pptx`）→ `loader.load()` 得到原始文档对象（PDF 每页一个 Document，DOCX 整体一个，pptx 每页一个）。
   - **确定切分策略** `_resolve_split_mode`（[core/ingestion.py:121](backend/core/ingestion.py#L121)）：
     - 手动 `split_mode` 优先
     - 否则按 `SEMANTIC_SPLIT` 配置：`auto` 时总字符 ≤ `AUTO_SEMANTIC_THRESHOLD`（3000）用 semantic，否则 fixed；`true` 全 semantic；`false` 全 fixed（兼容旧布尔值）
   - **分块** `_split_documents`（[core/ingestion.py:134](backend/core/ingestion.py#L134)），返回 `(结果, 实际策略)`：
     - **fixed**：`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`，递归分隔符（段落→句子→字符），相邻块重叠防语义切断
     - **semantic**：`_semantic_split_documents`（[core/ingestion.py:161](backend/core/ingestion.py#L161)）——按句切（`[。！？!?；;]` + 换行）→ 批量 embedding（每批 64 句）→ 相邻句余弦相似度 → 断点 = 低于 `均值-1.5σ` → 按断点合并成块；碎块（<40 字符）合并防碎、过长块（>1000 字符）Recursive 兜底再切、尾部不足 40 字符并入前块不丢内容。**代价：每句一次 embedding API（计费 + 耗时）**
   - **补充元数据** `_make_metadata_docs`（[core/ingestion.py:98](backend/core/ingestion.py#L98)）：为每个块添加 `filename`、`chunk_index`、`upload_time`（unix 秒）、`chunk_type`（semantic/fixed）；PDF/pptx 额外保留 `page`；xlsx 保留 `sheet`。

5. **写入向量库** `milvus.add_documents(enriched)`（[db/milvus.py:44](backend/db/milvus.py#L44)）：
   - `get_vectorstore()` 建立 langchain_milvus 连接（内部调用 `core/embeddings.py` 把文本转成 1024 维向量）。
   - `vs.add_documents(docs)`：文本 → 向量 → 连同元数据写入 Milvus Collection。
   - **Collection 不存在时自动创建**，命名 `doc_collection_` + Embedding 模型名 slug；**`metadata_schema` 显式声明 chunk_type 字段**（VARCHAR 32）——langchain-milvus 默认按首批 metadata 定 schema，新字段会静默丢失。

6. **清理与返回**：`finally` 中删除临时文件（无论成败）；返回 `{filename, chunk_count, ids, chunk_type}`。

**异常处理**：
- `ValueError`（空文档/无法解析）→ 400，前端展示错误信息。
- 其他异常 → 500 `文档处理失败: <原因>`。

**响应示例**：
```json
{ "filename": "notes.md", "chunk_count": 3, "ids": [12345], "chunk_type": "semantic" }
```

### 2.2 文档列表：`GET /documents`

**执行步骤**：

1. `milvus.get_all_documents()`（[db/milvus.py:57](backend/db/milvus.py#L57)）：
   - **collection 尚未创建（首次上传前）→ 直接返回空列表**（`vs.col is None` 防护）。
   - 用底层 `vs.col.query` 分页拉取全部数据（每批 100 条，`offset` 递增直到取完），output_fields 含 `chunk_type`。
   - 每条数据重建为 `Document`，元数据含 `pk`/`filename`/`chunk_index`/`upload_time`/`chunk_type`。

2. `ingestion.list_documents()`（[core/ingestion.py:230](backend/core/ingestion.py#L230)）：
   - 按 `filename` 聚合：同文件所有 chunk 合并为一条记录，`chunk_count` 累计块数。
   - 按 `upload_time` 降序排序（新上传的在前）。

3. 返回 `{documents: [...], total: N}`。

### 2.3 删除文档：`DELETE /documents/{filename}`

**执行步骤**：

1. `ingestion.delete_document(filename)` → `milvus.delete_document(filename)`（[db/milvus.py:95](backend/db/milvus.py#L95)）：
   - collection 不存在 → 返回 0。
   - 构造表达式 `filename == "<filename>"`，用底层 `vs.col.delete` 删除**该文件名对应的全部向量块**。
   - 返回 `delete_count`。

2. 删除 0 条 → `404 未找到文档`；否则返回 `{filename, deleted}`。

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
② rag_stream: _retrieve 放 asyncio.to_thread 执行（不阻塞事件循环）
        │
        ├─ ③ _build_search_queries: 原 query + 改写 + HYDE 并行（ThreadPoolExecutor）
        │      └─ LLM 失败 → 降级原 query（绝不抛出）
        │
        ├─ ④ 每个查询分别做: 向量检索(10条) + BM25 检索(10条)
        │      └─ 全部合并去重（按文本内容）→ 候选池（最多 ~30 条）
        │
        ├─ ⑤ rerank.rerank(原始query, 候选, top_n=top_k)
        │      └─ SiliconFlow bge-reranker-v2-m3 交叉编码 → 相关性分数排序
        │      └─ 未配置/关闭 → 跳过，取候选前 top_k
        │
        ├─ ⑥ 无结果？ → yield {"answer": "资料库中尚未检索到相关内容..."} → 结束
        │
        ├─ ⑦ _resolve_history: 前端 history 优先，否则 SQLite 记忆取最近 6 轮
        │
        ├─ ⑧ _build_messages: system(提示词+参考资料) + history + query
        │
        ├─ ⑨ chat.astream(messages) 流式调用 LLM
        │      └─ 每个 chunk → yield {"delta": "..."}   ← 逐字推送
        │
        ├─ ⑩ 结束后 yield {"sources": [...]}            ← 引用来源
        │
        ├─ ⑪ yield "data: [DONE]"                      ← 流结束标记
        │
        └─ ⑫ 写记忆: user=query, assistant=完整回答 → SQLite
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
2. user/assistant 交替: 历史对话（最近 6 轮）
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
 │                              │ 写临时文件 → 解析 → 分块        │                      │
 │                              │ (fixed 或 semantic) → 补元数据 │                      │
 │                              │ ──▶ add_documents ──────────▶│                      │
 │                              │                              │ Embedding 调用 ──────▶│
 │                              │                              │◀── 1024 维向量 ───────│
 │                              │◀─ 写入完成 ──────────────────│                      │
 │ ◀── 200 {filename,chunk_count,ids,chunk_type}               │                      │
 │                              │                              │                      │
 │  POST /chat/stream ────────▶│                              │                      │
 │  {query}                     │ 改写+HYDE 并行（LLM 2 次）────▶────────────────────▶│
 │                              │ 每个查询: 向量+BM25 检索 ────▶│ Embedding 调用 ──────▶│
 │                              │◀─ 候选池（去重） ─────────────│                      │
 │                              │ rerank 精排（原始 query）─────▶────────────────────▶│
 │                              │ 拼 system+history+query       │                      │
 │                              │ chat.astream ────────────────▶────────────────────▶│
 │ ◀── data: {"delta":"..."} ──│◀─ token 流 ──────────────────│◀── token 流 ─────────│
 │ ◀── data: {"sources":[...]} │                              │                      │
 │ ◀── data: [DONE]            │                              │                      │
 │                              │ 写记忆 → SQLite              │                      │
 │  GET /chat/memory ─────────▶│◀─ 读取 SQLite ───────────────│                      │
 │ ◀── {messages: [...]}       │                              │                      │
```

---

## 5. 关键设计决策与注意事项

| 设计 | 说明 | 影响 |
|---|---|---|
| 路由层无业务逻辑 | 校验 + 转发，业务全在 `core/` | 易于测试、复用 |
| Query 增强（改写+HYDE） | LLM 补全指代/生成假想答案，扩大召回 | 每轮问答 +2 次 LLM 调用（并行，~3.5s） |
| 混合召回 | 向量（语义）+ BM25（关键词）互补 | 专有名词/代码片段靠 BM25 补漏 |
| Rerank 用原始 query | 改写/假文档只用于召回，不参与精排 | 最终排序贴合用户原意 |
| 记忆 SQLite 持久化 | 存 50 轮、请求层取 6 轮 | 刷新恢复对话，上下文不膨胀 |
| 切分策略按文档长度 | auto: <3000 字符语义、否则固定 | 语义切分成本花在便宜处；可手动覆盖 |
| Collection 按模型命名 | `doc_collection_BAAI_bge_m3` | 换 Embedding 模型不冲突，但旧库作废需重传 |
| Collection schema 定死 | `metadata_schema` 显式声明新字段 | 加字段必须重建 collection（数据丢失） |
| 无去重 | 同文件重复上传重复入库 | 删除按文件名精确匹配，一并删除 |
| 上传无事务 | 部分块写入失败无回滚 | 可能残留部分向量 |

---

## 6. 常见排查点

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| `/documents` 空列表 | collection 尚未创建（首次上传前） | 先上传任意文档；或看后端日志 |
| 回答与资料无关 | Embedding 模型换了但旧库未重建 | 换模型后需重新上传全部文档 |
| 流式卡住无输出 | LLM API 不可用/超时 | 看后端日志 httpx 请求状态；直接 curl 测 LLM 接口 |
| 首 token 很慢（~7s） | 改写+HYDE+rerank 三次 API 调用 | `.env` 关 `QUERY_TRANSFORM`/`HYDE`/`RERANK_ENABLED` 可加速 |
| 改 .env 不生效 | `get_settings()` lru_cache | 重启后端进程 |
| 上传的 chunk_type 是 None | 旧 collection（无该字段） | 重建 collection 后重新上传 |
| 删除后列表没变 | 文件名含特殊字符导致表达式匹配失败 | 用 `GET /documents` 确认准确文件名 |
| 上传报错 `文档处理失败` | 文件损坏/加密 PDF | 后端日志看具体异常栈 |
