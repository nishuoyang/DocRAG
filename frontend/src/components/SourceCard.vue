<script setup>
import { ref } from 'vue'

defineProps({
  source: { type: Object, required: true },
})

const expanded = ref(false)
</script>

<template>
  <div class="mt-2 border border-gray-200 rounded-lg bg-gray-50">
    <button
      class="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-gray-600 hover:bg-gray-100 rounded-lg"
      @click="toggle"
    >
      <span class="text-gray-400">{{ expanded ? '▾' : '▸' }}</span>
      <span class="truncate">{{ source.filename }}</span>
      <span class="ml-auto text-gray-400 shrink-0">
        {{ source.page ? `第${source.page}页` : '' }}
        {{ source.chunk_index != null ? `块${source.chunk_index}` : '' }}
      </span>
    </button>
    <p v-if="expanded" class="px-3 pb-3 text-xs text-gray-500 whitespace-pre-wrap max-h-32 overflow-y-auto">
      {{ source.content }}
    </p>
  </div>
</template>
