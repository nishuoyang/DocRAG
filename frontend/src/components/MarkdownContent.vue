<script setup>
import { computed } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'

const props = defineProps({
  content: { type: String, default: '' },
})

marked.setOptions({
  gfm: true,
  breaks: true,
})

const rendered = computed(() => {
  if (!props.content) return ''
  const html = marked.parse(props.content)
  return DOMPurify.sanitize(html)
})
</script>

<template>
  <div class="markdown-body" v-html="rendered" />
</template>

<style scoped>
.markdown-body {
  overflow-wrap: anywhere;
}

.markdown-body :deep(> :first-child) {
  margin-top: 0;
}

.markdown-body :deep(> :last-child) {
  margin-bottom: 0;
}

.markdown-body :deep(p) {
  margin: 0.55rem 0;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  margin: 1rem 0 0.45rem;
  color: #0f172a;
  font-weight: 700;
  line-height: 1.35;
}

.markdown-body :deep(h1) {
  font-size: 1.2rem;
}

.markdown-body :deep(h2) {
  font-size: 1.08rem;
}

.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  font-size: 0.98rem;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin: 0.55rem 0;
  padding-left: 1.35rem;
}

.markdown-body :deep(ul) {
  list-style: disc;
}

.markdown-body :deep(ol) {
  list-style: decimal;
}

.markdown-body :deep(li) {
  margin: 0.2rem 0;
}

.markdown-body :deep(li > p) {
  margin: 0.15rem 0;
}

.markdown-body :deep(blockquote) {
  margin: 0.7rem 0;
  border-left: 3px solid #93c5fd;
  background: #eff6ff;
  padding: 0.55rem 0.75rem;
  color: #475569;
}

.markdown-body :deep(table) {
  display: block;
  width: max-content;
  min-width: 100%;
  max-width: 100%;
  margin: 0.75rem 0;
  overflow-x: auto;
  border-collapse: collapse;
  font-size: 0.78rem;
  line-height: 1.5;
}

.markdown-body :deep(thead) {
  background: #f1f5f9;
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  min-width: 6rem;
  border: 1px solid #dbe3ec;
  padding: 0.5rem 0.65rem;
  text-align: left;
  vertical-align: top;
}

.markdown-body :deep(th) {
  color: #334155;
  font-weight: 700;
  white-space: nowrap;
}

.markdown-body :deep(tbody tr:nth-child(even)) {
  background: #f8fafc;
}

.markdown-body :deep(code) {
  border-radius: 0.3rem;
  background: #eef2f7;
  padding: 0.12rem 0.3rem;
  color: #9f1239;
  font-size: 0.85em;
}

.markdown-body :deep(pre) {
  margin: 0.75rem 0;
  overflow-x: auto;
  border: 1px solid #dbe3ec;
  border-radius: 0.5rem;
  background: #0f172a;
  padding: 0.75rem;
  color: #e2e8f0;
}

.markdown-body :deep(pre code) {
  background: transparent;
  padding: 0;
  color: inherit;
  font-size: 0.78rem;
}

.markdown-body :deep(a) {
  color: #2563eb;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.markdown-body :deep(hr) {
  margin: 0.9rem 0;
  border: 0;
  border-top: 1px solid #e2e8f0;
}
</style>
