<script setup>
import { ref, onMounted } from 'vue'
import { listDocuments, uploadDocument, deleteDocument, getJobStatus } from '../api'

const docs = ref([])
const loading = ref(false)
const uploading = ref(false)
const message = ref('')
const dragOver = ref(false)
const fileInput = ref(null)
const progress = ref(0)
const progressMessage = ref('')

const ALLOWED = ['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx', 'pptx', 'html']
// auto 默认父子块；semantic / fixed 保留为显式兼容策略。
const splitMode = ref('auto')
const SPLIT_OPTIONS = [
  { value: 'auto', label: '自动父子块（推荐）' },
  { value: 'parent_child', label: '父子块（child 召回 + parent 上下文）' },
  { value: 'semantic', label: '语义切分（按话题断块）' },
  { value: 'fixed', label: '固定长度切分（500 字符）' },
]

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

async function pollJobStatus(jobId) {
  const maxAttempts = 300 // 最多轮询 5 分钟（300 * 1秒）
  let attempts = 0
  
  while (attempts < maxAttempts) {
    try {
      const job = await getJobStatus(jobId)
      progress.value = job.progress
      progressMessage.value = job.message
      
      if (job.status === 'completed') {
        return job
      } else if (job.status === 'failed') {
        throw new Error(job.error || job.message)
      }
      
      // 等待 1 秒后继续轮询
      await new Promise(resolve => setTimeout(resolve, 1000))
      attempts++
    } catch (e) {
      throw e
    }
  }
  
  throw new Error('任务超时')
}

async function doUpload(file, replace = false) {
  const ext = file.name.toLowerCase().split('.').pop()
  if (!ALLOWED.includes(ext)) {
    message.value = `仅支持 ${ALLOWED.join(' / ').toUpperCase()} 文件`
    return
  }
  uploading.value = true
  message.value = ''
  progress.value = 0
  progressMessage.value = '正在上传...'
  
  try {
    // 1. 提交上传任务
    const job = await uploadDocument(file, splitMode.value, replace)
    const jobId = job.job_id
    
    // 2. 轮询任务状态
    const result = await pollJobStatus(jobId)
    
    // 3. 显示成功消息
    const chunkType = result.result?.chunk_type
    const label = SPLIT_OPTIONS.find((o) => o.value === chunkType)?.label ?? chunkType
    message.value = `上传成功：${file.name}（${result.result?.chunk_count || 0} 个分块，${label}）`
    
    // 4. 刷新文档列表
    await refresh()
  } catch (e) {
    // 检测到重复文件时，提示用户是否覆盖
    if (e.message.includes('已入库') && !replace) {
      if (confirm(`文件「${file.name}」已入库。是否覆盖旧版本重新上传？`)) {
        uploading.value = false
        await doUpload(file, true)
        return
      }
    }
    message.value = `上传失败: ${e.message}`
  } finally {
    uploading.value = false
    progress.value = 0
    progressMessage.value = ''
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
      class="border-2 border-dashed rounded-xl p-10 text-center transition-colors"
      :class="[
        dragOver ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400',
        uploading ? 'cursor-not-allowed' : 'cursor-pointer'
      ]"
      @dragover.prevent="dragOver = true"
      @dragleave="dragOver = false"
      @drop.prevent="onDrop"
      @click="!uploading && fileInput.click()"
    >
      <input ref="fileInput" type="file" accept=".pdf,.docx,.txt,.md,.csv,.xlsx,.pptx" class="hidden" @change="onFileChange" :disabled="uploading" />
      
      <template v-if="uploading">
        <div class="text-4xl mb-2">⏳</div>
        <p class="text-gray-600 mb-3">{{ progressMessage }}</p>
        <div class="w-full max-w-md mx-auto">
          <div class="bg-gray-200 rounded-full h-2 overflow-hidden">
            <div 
              class="bg-blue-500 h-full transition-all duration-300"
              :style="{ width: `${progress}%` }"
            ></div>
          </div>
          <p class="text-xs text-gray-500 mt-2">{{ progress }}%</p>
        </div>
      </template>
      
      <template v-else>
        <div class="text-4xl mb-2">📤</div>
        <p class="text-gray-600">点击或拖拽文件到此处上传</p>
        <p class="mt-1 text-xs text-gray-400">支持 PDF / DOCX / TXT / MD / CSV / XLSX / PPTX，单文件不超过 20MB</p>
        <div class="mt-3 inline-flex items-center gap-2">
          <span class="text-xs text-gray-500">切分策略</span>
          <select
            v-model="splitMode"
            class="text-xs border border-gray-300 rounded-lg px-2 py-1 bg-white text-gray-700 focus:outline-none focus:border-blue-400"
            @click.stop
          >
            <option v-for="o in SPLIT_OPTIONS" :key="o.value" :value="o.value">{{ o.label }}</option>
          </select>
        </div>
      </template>
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
