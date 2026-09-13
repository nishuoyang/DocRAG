<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  source: { type: Object, required: true },
})

const expanded = ref(false)
const pageLabel = computed(() => {
  const start = props.source.page_start ?? props.source.page
  const end = props.source.page_end ?? props.source.page
  if (!start) return ''
  return end && end !== start ? `第${start}-${end}页` : `第${start}页`
})
</script>

<template>
  <div class="mt-2 border border-gray-200 rounded-lg bg-gray-50">
    <button
      class="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-gray-600 hover:bg-gray-100 rounded-lg"
      @click="expanded = !expanded"
    >
      <span class="text-gray-400">{{ expanded ? '▾' : '▸' }}</span>
      <span
        v-if="source.citation_index != null"
        class="shrink-0 font-mono text-gray-500"
      >
        [{{ source.citation_index }}]
      </span>
      <span class="truncate">{{ source.filename }}</span>
      <span class="ml-auto text-gray-400 shrink-0">
        {{ pageLabel }}
        {{ source.chunk_index != null ? `块${source.chunk_index}` : '' }}
      </span>
    </button>
    <div v-if="expanded" class="px-3 pb-3">
      <p v-if="source.section" class="text-[11px] text-gray-400 mb-1">{{ source.section }}</p>
      <p class="text-xs text-gray-500 whitespace-pre-wrap max-h-32 overflow-y-auto">
        {{ source.content }}
      </p>
    </div>
  </div>
</template>
