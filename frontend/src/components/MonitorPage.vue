<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// 状态
const metrics = ref({})
const isLoading = ref(false)
let timer = null

// 加载系统指标
async function loadMetrics() {
  isLoading.value = true
  try {
    const response = await fetch('/metrics')
    const result = await response.json()
    metrics.value = result.data || {}
  } catch (error) {
    console.error('加载指标失败：', error)
  } finally {
    isLoading.value = false
  }
}

// 格式化运行时间
function formatUptime(seconds) {
  if (!seconds) return '-'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  if (hours > 0) return `${hours}小时 ${minutes}分`
  if (minutes > 0) return `${minutes}分 ${secs}秒`
  return `${secs}秒`
}

// 刷新
function refresh() {
  loadMetrics()
}

onMounted(() => {
  loadMetrics()
  // 每 10 秒自动刷新
  timer = setInterval(loadMetrics, 10000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div class="monitor-page">
    <div class="page-header">
      <h2>📊 系统监控</h2>
      <button class="refresh-btn" @click="refresh()">🔄 刷新</button>
    </div>

    <div v-if="isLoading && !metrics.uptime_seconds" class="loading">加载中...</div>

    <div v-else class="metrics-grid">
      <!-- 系统信息 -->
      <div class="metric-card">
        <h3>系统信息</h3>
        <div class="metric-item">
          <span class="label">运行时间</span>
          <span class="value">{{ formatUptime(metrics.uptime_seconds) }}</span>
        </div>
        <div class="metric-item">
          <span class="label">总请求数</span>
          <span class="value">{{ metrics.total_requests || 0 }}</span>
        </div>
        <div class="metric-item">
          <span class="label">每秒请求数</span>
          <span class="value">{{ metrics.avg_request_per_second || 0 }}</span>
        </div>
      </div>

      <!-- 请求统计 -->
      <div class="metric-card">
        <h3>请求统计</h3>
        <div class="metric-item">
          <span class="label">成功请求</span>
          <span class="value success">{{ metrics.success_requests || 0 }}</span>
        </div>
        <div class="metric-item">
          <span class="label">失败请求</span>
          <span class="value error">{{ metrics.error_requests || 0 }}</span>
        </div>
        <div class="metric-item">
          <span class="label">成功率</span>
          <span class="value">
            {{ metrics.total_requests ? ((metrics.success_requests / metrics.total_requests) * 100).toFixed(1) : 0 }}%
          </span>
        </div>
      </div>

      <!-- 最慢接口 Top 5 -->
      <div class="metric-card full-width">
        <h3>🐢 最慢接口 Top 5</h3>
        <table v-if="metrics.top_slow_paths && metrics.top_slow_paths.length > 0">
          <thead>
            <tr>
              <th>路径</th>
              <th>调用次数</th>
              <th>平均耗时</th>
              <th>最大耗时</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(path, index) in metrics.top_slow_paths.slice(0, 5)" :key="index">
              <td>{{ path.path }}</td>
              <td>{{ path.count }}</td>
              <td>{{ path.avg_ms }} ms</td>
              <td>{{ path.max_ms }} ms</td>
            </tr>
          </tbody>
        </table>
        <p v-else class="empty-hint">暂无数据</p>
      </div>
    </div>

    <div class="auto-refresh-hint">
      💡 每 10 秒自动刷新一次
    </div>
  </div>
</template>

<style scoped>
.monitor-page {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.page-header h2 {
  font-size: 20px;
  font-weight: 600;
}

.refresh-btn {
  padding: 10px 16px;
  background: #f5f5f5;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
}

.loading {
  text-align: center;
  padding: 60px;
  color: #999;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
}

.metric-card {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.metric-card.full-width {
  grid-column: span 2;
}

.metric-card h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.metric-item {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  font-size: 14px;
}

.metric-item .label {
  color: #666;
}

.metric-item .value {
  font-weight: 500;
  color: #333;
}

.metric-item .value.success {
  color: #10b981;
}

.metric-item .value.error {
  color: #ff4d4f;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
  font-size: 13px;
}

th {
  background: #fafafa;
  font-weight: 600;
  color: #666;
}

.empty-hint {
  text-align: center;
  color: #999;
  padding: 20px;
}

.auto-refresh-hint {
  margin-top: 20px;
  text-align: center;
  font-size: 12px;
  color: #999;
}
</style>
