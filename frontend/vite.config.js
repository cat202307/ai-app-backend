import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// 后端地址：本地开发默认代理到 8001（本机 8000 被 multi-agent-platform 容器占用）。
// 如需改回 8000 或指向其他后端，可设置环境变量 VITE_API_BASE。
const apiTarget = process.env.VITE_API_BASE || 'http://localhost:8001'
const proxy = Object.fromEntries(
  ['/agent', '/files', '/upload', '/history', '/metrics', '/static'].map((p) => [
    p,
    { target: apiTarget, changeOrigin: true },
  ])
)

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3000,
    proxy,
  },
})
