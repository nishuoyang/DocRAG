<script setup>
import { ref, onMounted } from 'vue'
import { listDocuments, uploadDocument, deleteDocument } from '../api'

const docs = ref([])
const loading = ref(false)
const uploading = ref(false)
const message = ref('')
const dragOver = ref(false)
const fileInput = ref(null)

const ALLOWED = ['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx', 'pptx']

async function refresh() {
  loading.value = true
  try {
    const res = await listDocuments()
    docs.value = res.documents
  } catch (e) {
    message.value = `加载失败: ${e.message}`
  } finally {
    loading.value = false
  }
}

function formatTime(ts) {
  if (!ts) return '—'
  return new Date(ts * 1000).toLocaleString('zh-CN')
}

async function doUpload(file) {
  const ext = file.name.toLowerCase().split('.').pop()
  if (!ALLOWED.includes(ext)) {
    message.value = `仅支持 ${ALLOWED.join(' / ').toUpperCase()} 文件`
    return
  }
  uploading.value = true
  message.value = ''
  try {
    const res = await uploadDocument(file)
    message.value = `上传成功：${res.filename}（${res.chunk_count} 个分块）`
    await refresh()
  } catch (e) {
    message.value = `上传失败: ${e.message}`
  } finally {
    uploading.value = false
  }
}

function onFileChange(e) {
  const file = e.target.files?.[0]
  if (file) doUpload(file)
  e.target.value = ''
}

function onDrop(e) {
  dragOver.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file) doUpload(file)
}

async function remove(filename) {
  if (!confirm(`确定删除「${filename}」及其全部向量吗？`)) return
  try {
    await deleteDocument(filename)
    message.value = `已删除 ${filename}`
    await refresh()
  } catch (e) {
    message.value = `删除失败: ${e.message}`
  }
}

onMounted(refresh)
</script>

<template>
  <div class="h-full overflow-y-auto p-8">
    <h2 class="text-xl font-bold text-gray-800 mb-6">文档管理</h2>

    <div v-if="message" class="mb-4 px-4 py-3 rounded-lg bg-blue-50 text-blue-700 text-sm">
      {{ message }}
    </div>

    <!-- 上传区 -->
    <div
      class="border-2 border-dashed rounded-xl p-10 text-center transition-colors cursor-pointer"
      :class="dragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'"
      @dragover.prevent="dragOver = true"
      @dragleave="dragOver = false"
      @drop.prevent="onDrop"
      @click="fileInput.click()"
    >
      <input ref="fileInput" type="file" accept=".pdf,.docx,.txt,.md,.csv,.xlsx,.pptx" class="hidden" @change="onFileChange" />
      <div class="text-4xl mb-2">{{ uploading ? '⏳' : '📤' }}</div>
      <p class="text-gray-600">{{ uploading ? '正在解析并入库...' : '点击或拖拽文件到此处上传' }}</p>
      <p class="mt-1 text-xs text-gray-400">支持 PDF / DOCX / TXT / MD / CSV / XLSX / PPTX，单文件不超过 20MB</p>
    </div>

    <!-- 文档列表 -->
    <div class="mt-8 bg-white rounded-xl shadow-sm overflow-hidden">
      <div class="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
        <h3 class="font-semibold text-gray-800">已入库文档</h3>
        <span class="text-xs text-gray-400">{{ docs.length }} 个文件</span>
      </div>

      <div v-if="loading" class="p-8 text-center text-gray-400 text-sm">加载中...</div>
      <div v-else-if="docs.length === 0" class="p-8 text-center text-gray-400 text-sm">
        暂无文档，先上传一个试试
      </div>
      <table v-else class="w-full text-sm">
        <thead>
          <tr class="text-left text-gray-400 border-b border-gray-100">
            <th class="px-5 py-3 font-medium">文件名</th>
            <th class="px-5 py-3 font-medium w-28">分块数</th>
            <th class="px-5 py-3 font-medium w-48">上传时间</th>
            <th class="px-5 py-3 font-medium w-24">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="d in docs" :key="d.filename" class="border-b border-gray-50 hover:bg-gray-50">
            <td class="px-5 py-3 text-gray-800 truncate max-w-xs">{{ d.filename }}</td>
            <td class="px-5 py-3 text-gray-600">{{ d.chunk_count }}</td>
            <td class="px-5 py-3 text-gray-500">{{ formatTime(d.upload_time) }}</td>
            <td class="px-5 py-3">
              <button
                class="text-red-500 hover:text-red-700 text-xs font-medium"
                @click="remove(d.filename)"
              >
                删除
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
