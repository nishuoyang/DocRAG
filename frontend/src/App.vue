<script setup>
import { ref } from 'vue'
import DocManager from './components/DocManager.vue'
import ChatPanel from './components/ChatPanel.vue'

const tabs = [
  { key: 'chat', label: '智能问答', icon: '💬' },
  { key: 'docs', label: '文档管理', icon: '📚' },
]
const activeTab = ref('chat')
</script>

<template>
  <div class="flex h-screen bg-gray-100">
    <!-- 左侧导航 -->
    <aside class="w-64 shrink-0 bg-slate-900 text-slate-200 flex flex-col">
      <div class="px-5 py-6">
        <h1 class="text-lg font-bold text-white">智能文档问答</h1>
        <p class="mt-1 text-xs text-slate-400">LangChain + Milvus</p>
      </div>
      <nav class="flex-1 px-3 space-y-1">
        <button
          v-for="t in tabs"
          :key="t.key"
          class="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left text-sm transition-colors"
          :class="activeTab === t.key ? 'bg-slate-700 text-white' : 'hover:bg-slate-800'"
          @click="activeTab = t.key"
        >
          <span>{{ t.icon }}</span>
          {{ t.label }}
        </button>
      </nav>
      <div class="px-5 py-4 text-xs text-slate-500 border-t border-slate-800">
        v0.1.0 · RAG Demo
      </div>
    </aside>

    <!-- 右侧内容区 -->
    <main class="flex-1 min-w-0">
      <ChatPanel v-if="activeTab === 'chat'" />
      <DocManager v-else />
    </main>
  </div>
</template>
