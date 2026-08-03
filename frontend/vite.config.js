import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/chat': 'http://localhost:8000',
    },
  },
})
