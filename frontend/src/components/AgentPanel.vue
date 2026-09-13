<script setup>
import { ref, nextTick, onMounted } from 'vue'
import { agentStream, getMemory } from '../api'
import MarkdownContent from './MarkdownContent.vue'

const messages = ref([])
const input = ref('')
const sending = ref(false)
const scrollBox = ref(null)

const TYPING_MS = 16

async function send() {
  const query = input.value.trim()
  if (!query || sending.value) return
  const userMsg = { role: 'user', content: query }
  messages.value.push(
    userMsg,
    { role: 'assistant', content: '', activities: [], sources: [], streaming: true }
  )
  const aiMsg = messages.value[messages.value.length - 1]
  input.value = ''
  sending.value = true

  let pending = ''
  let typingTimer = null
  const flush = () => {
    if (pending) { aiMsg.content += pending; pending = '' }
    scrollToBottom()
  }
  const tick = () => {
    if (!pending) return
    aiMsg.content += pending[0]
    pending = pending.slice(1)
    scrollToBottom()
  }

  try {
    const onEvent = (e) => {
      if (e.type === 'activity') {
        aiMsg.activities.push({ agent: e.agent, message: e.message })
        scrollToBottom()
      } else if (e.type === 'delta') {
        pending += e.text
        if (!typingTimer) typingTimer = setInterval(tick, TYPING_MS)
      }
    }
    const { answer, sources } = await agentStream(query, onEvent)
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
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
}

function clearChat() {
  messages.value = []
}

onMounted(async () => {
  try {
    const { messages: saved } = await getMemory()
    if (saved?.length) {
      messages.value = saved.map((m) => ({ ...m, activities: [], sources: [], streaming: false }))
      scrollToBottom()
    }
  } catch { /* 后端不可达时静默 */ }
})
</script>

<template>
  <div class="h-full flex flex-col">
    <div class="px-6 py-4 border-b border-gray-200 bg-white flex items-center justify-between">
      <div>
        <h2 class="text-lg font-bold text-gray-800">研究助理</h2>
        <p class="text-xs text-gray-400 mt-0.5">主管 agent 自动调度：文档检索 / 联网搜索 / 数据分析 / 写作汇总 / 多跳查证</p>
      </div>
      <button v-if="messages.length" class="text-xs text-gray-400 hover:text-gray-600" @click="clearChat">清空对话</button>
    </div>

    <div ref="scrollBox" class="flex-1 overflow-y-auto px-6 py-6 space-y-5">
      <div v-if="messages.length === 0" class="h-full flex flex-col items-center justify-center text-center text-gray-400">
        <div class="text-5xl mb-4">🛰️</div>
        <p class="text-sm">向主管提问，它会自动调度各成员 agent 完成任务</p>
        <p class="text-xs mt-2 text-gray-300">例如：「上传的政策里，北京和上海的购房门槛有什么差异？」</p>
      </div>

      <div v-for="(m, i) in messages" :key="i" class="flex" :class="m.role === 'user' ? 'justify-end' : 'justify-start'">
        <div class="max-w-[75%]" :class="m.role === 'user' ? 'order-first' : ''">
          <div
            class="px-4 py-3 rounded-2xl text-sm leading-relaxed"
            :class="m.role === 'user'
              ? 'bg-blue-500 text-white rounded-br-md whitespace-pre-wrap'
              : 'bg-white border border-gray-200 text-gray-800 rounded-bl-md shadow-sm'"
          >
            <MarkdownContent
              v-if="m.role === 'assistant' && m.content"
              :content="m.content"
            />
            <template v-else>{{ m.content }}</template>
            <span v-if="m.streaming" class="inline-flex items-center gap-0.5 ml-1 align-middle">
              <span v-if="m.content" class="w-1.5 h-4 bg-blue-400 animate-pulse inline-block" />
              <span v-else class="inline-flex gap-1">
                <span v-for="n in 3" :key="n" class="w-1.5 h-1.5 rounded-full bg-blue-400"
                  :style="{ animation: `bounce 1.2s ${(n - 1) * 0.2}s infinite` }" />
              </span>
            </span>
          </div>

          <!-- agent 活动日志 -->
          <div v-if="m.role === 'assistant' && m.activities?.length" class="mt-2 space-y-1">
            <div v-for="(a, ai) in m.activities" :key="ai" class="text-xs text-gray-400 flex items-center gap-1.5">
              <span class="px-1.5 py-0.5 rounded bg-slate-100 text-slate-500 font-medium">{{ a.agent }}</span>
              <span>{{ a.message }}</span>
            </div>
          </div>

          <!-- 来源：文档型与网页型 -->
          <div v-if="m.role === 'assistant' && m.sources?.length" class="mt-2 space-y-2">
            <p class="text-xs text-gray-400">📎 引用来源（{{ m.sources.length }} 条）</p>
            <div v-for="(s, si) in m.sources" :key="si"
              class="text-xs bg-gray-50 border border-gray-200 rounded-lg px-3 py-2">
              <template v-if="s.url">
                <a :href="s.url" target="_blank" rel="noopener" class="text-blue-600 hover:underline">{{ s.title || s.url }}</a>
              </template>
              <template v-else>
                <span class="text-gray-700 font-medium">{{ s.title }}</span>
                <span v-if="s.section" class="text-gray-400 ml-1">{{ s.section }}</span>
                <span v-if="s.page_start || s.page" class="text-gray-400 ml-1">
                  p.{{ s.page_start || s.page }}<template v-if="s.page_end && s.page_end !== (s.page_start || s.page)">-{{ s.page_end }}</template>
                </span>
              </template>
              <p class="text-gray-500 mt-0.5 line-clamp-2">{{ s.content }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="px-6 py-4 border-t border-gray-200 bg-white">
      <div class="flex items-end gap-3">
        <textarea
          v-model="input"
          rows="2"
          class="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          :placeholder="sending ? '主管正在调度成员 agent...' : '输入你的问题，Enter 发送，Shift+Enter 换行'"
          :disabled="sending"
          @keydown="onKeydown"
        />
        <button
          class="shrink-0 px-6 py-3 rounded-xl bg-blue-500 text-white text-sm font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="sending || !input.trim()"
          @click="send"
        >
          {{ sending ? '调度中...' : '发送' }}
        </button>
      </div>
    </div>
  </div>
</template>
