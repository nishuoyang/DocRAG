<script setup>
import { ref, nextTick, onMounted } from 'vue'
import { chatStream, getMemory } from '../api'
import SourceCard from './SourceCard.vue'

const messages = ref([])
const input = ref('')
const sending = ref(false)
const topK = ref(5)
const scrollBox = ref(null)

// 打字机渲染间隔（毫秒/字）。真实流式时 chunk 到达 ~25ms，16ms/字不会形成瓶颈；
// API 聚合推来的大段文本则被逐字刷出，保持流式观感。
const TYPING_MS = 16

async function send() {
  const query = input.value.trim()
  if (!query || sending.value) return

  // 历史由后端从 SQLite 记忆读取，前端不再随请求携带
  const userMsg = { role: 'user', content: query }
  messages.value.push(
    userMsg,
    { role: 'assistant', content: '', sources: [], streaming: true }
  )
  // 关键：push 后取回数组存储的代理引用——直接引用原对象改 content 不触发 Vue 响应式
  const aiMsg = messages.value[messages.value.length - 1]
  input.value = ''
  sending.value = true

  // 打字机队列：SSE 到达的文本先入 pending，定时器逐字刷出
  let pending = ''
  let typingTimer = null

  const flush = () => {
    if (pending) {
      aiMsg.content += pending
      pending = ''
    }
    scrollToBottom()
  }

  const tick = () => {
    if (!pending) return
    aiMsg.content += pending[0]
    pending = pending.slice(1)
    scrollToBottom()
  }

  try {
    const onDelta = (text) => {
      pending += text
      if (!typingTimer) typingTimer = setInterval(tick, TYPING_MS)
    }
    const { answer, sources } = await chatStream(query, topK.value, onDelta)
    if (typingTimer) clearInterval(typingTimer)
    typingTimer = null
    flush()
    aiMsg.content = answer || aiMsg.content
    aiMsg.sources = sources
    aiMsg.streaming = false
  } catch (e) {
    if (typingTimer) clearInterval(typingTimer)
    typingTimer = null
    aiMsg.content = `出错了：${e.message}`
    aiMsg.streaming = false
  } finally {
    sending.value = false
    scrollToBottom()
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (scrollBox.value) scrollBox.value.scrollTop = scrollBox.value.scrollHeight
  })
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

function clearChat() {
  messages.value = []
}

// 页面加载时从后端恢复历史对话（SQLite 持久化）
onMounted(async () => {
  try {
    const { messages: saved } = await getMemory()
    if (saved?.length) {
      messages.value = saved.map((m) => ({ ...m, sources: [], streaming: false }))
      scrollToBottom()
    }
  } catch {
    /* 后端不可达时静默，保持空对话 */
  }
})
</script>

<template>
  <div class="h-full flex flex-col">
    <!-- 顶部栏 -->
    <div class="px-6 py-4 border-b border-gray-200 bg-white flex items-center justify-between">
      <div>
        <h2 class="text-lg font-bold text-gray-800">智能问答</h2>
        <p class="text-xs text-gray-400 mt-0.5">基于已上传文档检索回答，回答内容会标注来源</p>
      </div>
      <button
        v-if="messages.length"
        class="text-xs text-gray-400 hover:text-gray-600"
        @click="clearChat"
      >
        清空对话
      </button>
    </div>

    <!-- 消息区 -->
    <div ref="scrollBox" class="flex-1 overflow-y-auto px-6 py-6 space-y-5">
      <div v-if="messages.length === 0" class="h-full flex flex-col items-center justify-center text-center text-gray-400">
        <div class="text-5xl mb-4">💬</div>
        <p class="text-sm">上传文档后，就可以向我提问了</p>
        <p class="text-xs mt-2 text-gray-300">例如：「系统支持哪些文档格式？」</p>
      </div>

      <div v-for="(m, i) in messages" :key="i" class="flex" :class="m.role === 'user' ? 'justify-end' : 'justify-start'">
        <div class="max-w-[75%]" :class="m.role === 'user' ? 'order-first' : ''">
          <div
            class="px-4 py-3 rounded-2xl text-sm whitespace-pre-wrap leading-relaxed"
            :class="m.role === 'user'
              ? 'bg-blue-500 text-white rounded-br-md'
              : 'bg-white border border-gray-200 text-gray-800 rounded-bl-md shadow-sm'"
          >
            {{ m.content }}
            <!-- 流式生成中 -->
            <span v-if="m.streaming" class="inline-flex items-center gap-0.5 ml-1 align-middle">
              <!-- 有内容时显示呼吸光标 -->
              <span
                v-if="m.content"
                class="w-1.5 h-4 bg-blue-400 animate-pulse inline-block"
              />
              <!-- 无内容时显示思考中动画（首 token 前） -->
              <span v-else class="inline-flex gap-1">
                <span
                  v-for="n in 3"
                  :key="n"
                  class="w-1.5 h-1.5 rounded-full bg-blue-400"
                  :style="{ animation: `bounce 1.2s ${(n - 1) * 0.2}s infinite` }"
                />
              </span>
            </span>
          </div>
          <div v-if="m.role === 'assistant' && m.sources?.length" class="mt-2 space-y-2">
            <p class="text-xs text-gray-400">📎 引用来源（{{ m.sources.length }} 条）</p>
            <SourceCard v-for="(s, si) in m.sources" :key="si" :source="s" />
          </div>
        </div>
      </div>
    </div>

    <!-- 输入区 -->
    <div class="px-6 py-4 border-t border-gray-200 bg-white">
      <div class="flex items-center gap-4 mb-3">
        <span class="text-xs text-gray-500">检索数量 Top-K</span>
        <input
          v-model.number="topK"
          type="range"
          min="1"
          max="20"
          class="flex-1 max-w-xs accent-blue-500"
        />
        <span class="text-xs text-gray-700 font-medium w-8">{{ topK }}</span>
      </div>
      <div class="flex items-end gap-3">
        <textarea
          v-model="input"
          rows="2"
          class="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          :placeholder="sending ? '正在生成回答...' : '输入你的问题，Enter 发送，Shift+Enter 换行'"
          :disabled="sending"
          @keydown="onKeydown"
        />
        <button
          class="shrink-0 px-6 py-3 rounded-xl bg-blue-500 text-white text-sm font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="sending || !input.trim()"
          @click="send"
        >
          {{ sending ? '生成中...' : '发送' }}
        </button>
      </div>
    </div>
  </div>
</template>
