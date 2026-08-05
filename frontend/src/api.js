// 后端 API 封装。开发时经 Vite proxy 转发到 8000，生产时同源。

async function request(path, options = {}) {
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      /* 非 JSON 响应，保留状态码 */
    }
    throw new Error(detail)
  }
  return res.json()
}

export function uploadDocument(file, splitMode = 'auto') {
  const form = new FormData()
  form.append('file', file)
  if (splitMode && splitMode !== 'auto') form.append('split_mode', splitMode)
  return request('/documents/upload', { method: 'POST', body: form })
}

export function listDocuments() {
  return request('/documents')
}

export function deleteDocument(filename) {
  return request(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' })
}

// 获取历史对话记忆（后端 SQLite 持久化，刷新页面后恢复）
export function getMemory() {
  return request('/chat/memory')
}

export function chat(query, topK, history = []) {
  return request('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK, history }),
  })
}

// 流式问答：返回 { answer, sources }，answer 逐段追加到 onDelta。
export async function chatStream(query, topK, onDelta, history = []) {
  const res = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK, history }),
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      detail = (await res.json()).detail || detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  if (!res.body) throw new Error('浏览器不支持流式响应')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let answer = ''
  let sources = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE 事件以空行分隔
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    for (const raw of events) {
      for (const line of raw.split('\n')) {
        if (!line.startsWith('data: ')) continue
        const payload = line.slice(6)
        if (payload === '[DONE]') break
        try {
          const data = JSON.parse(payload)
          if (data.delta) {
            answer += data.delta
            onDelta(data.delta)
          }
          if (data.sources) sources = data.sources
          if (data.answer) {
            // 空资料库场景：一次性返回完整答案
            answer = data.answer
            onDelta(data.answer)
          }
        } catch {
          /* 跳过无法解析的行 */
        }
      }
    }
  }
  return { answer, sources }
}
