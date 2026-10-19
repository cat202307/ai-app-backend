<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { apiFetch } from '../api.js'

const metrics = ref({})
const isLoading = ref(false)
const lastUpdated = ref('')
const autoRefresh = ref(true)
let timer = null

async function loadMetrics() {
  isLoading.value = true
  try {
    const result = await apiFetch('/metrics')
    metrics.value = result.data || {}
    lastUpdated.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  } catch (error) {
    console.error('加载指标失败：', error)
  } finally {
    isLoading.value = false
  }
}

function refresh() {
  loadMetrics()
}

function toggleAuto() {
  autoRefresh.value = !autoRefresh.value
  if (autoRefresh.value) {
    timer = setInterval(loadMetrics, 10000)
  } else if (timer) {
    clearInterval(timer)
    timer = null
  }
}

function formatUptime(seconds) {
  if (!seconds) return '-'
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  if (days > 0) return `${days}天 ${hours}小时`
  if (hours > 0) return `${hours}小时 ${minutes}分`
  if (minutes > 0) return `${minutes}分 ${secs}秒`
  return `${secs}秒`
}

function num(v) {
  return (v || 0).toLocaleString('en-US')
}

function successRate() {
  const m = metrics.value
  if (!m.total_requests) return '0.0'
  return ((m.success_requests / m.total_requests) * 100).toFixed(1)
}

function cost(c) {
  const v = Number(c || 0)
  if (v === 0) return '¥0.00'
  return '¥' + v.toFixed(4)
}

// ---- 检索质量 ----
function retrieval() {
  return metrics.value.retrieval || {}
}
function matchPct() {
  const r = retrieval()
  return ((r.match_rate || 0) * 100).toFixed(1)
}
function modeLabel(m) {
  return ({
    advanced_rag: '高级问答',
    agent_retrieval: 'Agent 检索',
    overview_opening: '概述取文',
  })[m] || m || '-'
}
function recentRuns() {
  return (retrieval().recent_runs || []).slice().reverse()
}

// ---- 缓存 ----
function cachesList() {
  const c = metrics.value.caches || {}
  return Object.keys(c).map((k) => ({ name: k, ...c[k] }))
}
function cacheLabel(name) {
  return ({ embedding: '向量 Embedding', rewrite: '查询改写' })[name] || name
}
function cacheHitRate(c) {
  const total = (c.hits || 0) + (c.misses || 0)
  return total ? ((c.hits / total) * 100).toFixed(1) : '0.0'
}

onMounted(() => {
  loadMetrics()
  timer = setInterval(loadMetrics, 10000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div class="monitor-page">
    <header class="page-header">
      <div class="page-title">
        <svg class="title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 3v18h18"></path>
          <path d="M7 14l4-4 4 4 5-6"></path>
        </svg>
        系统监控
        <span class="page-sub" v-if="lastUpdated">更新于 {{ lastUpdated }}</span>
      </div>
      <div class="header-actions">
        <button
          class="btn"
          :class="autoRefresh ? 'btn-primary' : 'btn-ghost'"
          @click="toggleAuto"
        >
          <span class="dot" :class="{ on: autoRefresh }"></span>
          {{ autoRefresh ? '自动刷新中' : '已暂停' }}
        </button>
        <button class="btn btn-ghost" @click="refresh" :disabled="isLoading">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" :class="{ spin: isLoading }">
            <polyline points="23 4 23 10 17 10"></polyline>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
          </svg>
          刷新
        </button>
      </div>
    </header>

    <div v-if="isLoading && !metrics.uptime_seconds" class="state-card">
      <div class="spinner"></div>
      <div class="state-text">加载中…</div>
    </div>

    <div v-else>
      <!-- 顶部 KPI -->
      <div class="kpi-grid">
        <div class="card kpi">
          <div class="kpi-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="9"></circle>
              <polyline points="12 7 12 12 15 14"></polyline>
            </svg>
          </div>
          <div class="kpi-body">
            <div class="kpi-label">运行时间</div>
            <div class="kpi-value">{{ formatUptime(metrics.uptime_seconds) }}</div>
          </div>
        </div>

        <div class="card kpi">
          <div class="kpi-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
            </svg>
          </div>
          <div class="kpi-body">
            <div class="kpi-label">总请求数</div>
            <div class="kpi-value">{{ num(metrics.total_requests) }}</div>
          </div>
        </div>

        <div class="card kpi">
          <div class="kpi-icon ok">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
          </div>
          <div class="kpi-body">
            <div class="kpi-label">成功率</div>
            <div class="kpi-value">{{ successRate() }}%</div>
            <div class="kpi-foot">
              <span class="ok">{{ num(metrics.success_requests) }} 成功</span>
              <span class="err" v-if="metrics.error_requests">· {{ num(metrics.error_requests) }} 失败</span>
            </div>
          </div>
        </div>

        <div class="card kpi">
          <div class="kpi-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M13 2L3 14h7l-1 8 10-12h-7z"></path>
            </svg>
          </div>
          <div class="kpi-body">
            <div class="kpi-label">每秒请求数</div>
            <div class="kpi-value">{{ metrics.avg_request_per_second || 0 }}</div>
          </div>
        </div>
      </div>

      <!-- LLM 用量 -->
      <div class="card section">
        <div class="section-head">
          <h3>LLM 用量</h3>
          <span class="badge badge-neutral">自服务启动累计</span>
        </div>
        <div class="llm-grid">
          <div class="llm-item">
            <div class="llm-label">调用次数</div>
            <div class="llm-value">{{ num(metrics.llm_usage && metrics.llm_usage.calls) }}</div>
          </div>
          <div class="llm-item">
            <div class="llm-label">Prompt Tokens</div>
            <div class="llm-value">{{ num(metrics.llm_usage && metrics.llm_usage.prompt_tokens) }}</div>
          </div>
          <div class="llm-item">
            <div class="llm-label">Completion Tokens</div>
            <div class="llm-value">{{ num(metrics.llm_usage && metrics.llm_usage.completion_tokens) }}</div>
          </div>
          <div class="llm-item">
            <div class="llm-label">总 Tokens</div>
            <div class="llm-value">{{ num(metrics.llm_usage && metrics.llm_usage.total_tokens) }}</div>
          </div>
          <div class="llm-item highlight">
            <div class="llm-label">估算成本</div>
            <div class="llm-value cost">{{ cost(metrics.llm_usage && metrics.llm_usage.estimated_cost) }}</div>
          </div>
        </div>
      </div>

      <!-- 检索质量（Advanced RAG） -->
      <div class="card section">
        <div class="section-head">
          <h3>检索质量</h3>
          <span class="badge badge-neutral">Advanced RAG · 自启动累计</span>
        </div>
        <div class="ret-grid">
          <div class="ret-item">
            <div class="llm-label">检索次数</div>
            <div class="llm-value">{{ num(retrieval().runs) }}</div>
          </div>
          <div class="ret-item highlight">
            <div class="llm-label">命中率</div>
            <div class="llm-value">{{ matchPct() }}%</div>
          </div>
          <div class="ret-item">
            <div class="llm-label">命中 / 未命中</div>
            <div class="llm-value dual">
              <span class="ok">{{ num(retrieval().matched) }}</span>
              <span class="sep">/</span>
              <span class="err">{{ num(retrieval().not_matched) }}</span>
            </div>
          </div>
          <div class="ret-item">
            <div class="llm-label">平均查询数</div>
            <div class="llm-value">{{ retrieval().avg_query_count || 0 }}</div>
          </div>
          <div class="ret-item">
            <div class="llm-label">平均耗时</div>
            <div class="llm-value">{{ retrieval().avg_latency_ms || 0 }} <small>ms</small></div>
          </div>
          <div class="ret-item">
            <div class="llm-label">最大耗时</div>
            <div class="llm-value">{{ retrieval().max_latency_ms || 0 }} <small>ms</small></div>
          </div>
          <div class="ret-item">
            <div class="llm-label">top1 相关分（最小/平均/最大）</div>
            <div class="llm-value score-line">
              <span>{{ retrieval().rerank_score_min ?? '—' }}</span>
              <span class="sep">/</span>
              <span>{{ retrieval().rerank_score_avg ?? '—' }}</span>
              <span class="sep">/</span>
              <span>{{ retrieval().rerank_score_max ?? '—' }}</span>
            </div>
          </div>
        </div>

        <table v-if="recentRuns().length" class="recent-table">
          <thead>
            <tr><th>模式</th><th>状态</th><th class="num">查询数</th><th class="num">片段</th><th class="num">top1 分</th><th class="num">耗时</th></tr>
          </thead>
          <tbody>
            <tr v-for="(r, i) in recentRuns().slice(0, 6)" :key="i">
              <td>{{ modeLabel(r.mode) }}</td>
              <td>
                <span class="pill" :class="r.status === 'matched' ? 'pill-ok' : 'pill-warn'">
                  {{ r.status === 'matched' ? '命中' : '未命中' }}
                </span>
              </td>
              <td class="num">{{ r.query_count }}</td>
              <td class="num">{{ r.result_count }}</td>
              <td class="num">{{ r.best_rerank_score ?? '—' }}</td>
              <td class="num">{{ Math.round(r.latency_ms) }} ms</td>
            </tr>
          </tbody>
        </table>
        <div v-else class="empty-hint">暂无检索数据</div>
      </div>

      <!-- 缓存命中 -->
      <div class="card section">
        <div class="section-head">
          <h3>缓存命中</h3>
          <span class="badge badge-neutral">LRU + TTL · 有界</span>
        </div>
        <div class="cache-grid">
          <div class="cache-item" v-for="c in cachesList()" :key="c.name">
            <div class="cache-top">
              <span class="cache-name">{{ cacheLabel(c.name) }}</span>
              <span class="cache-rate">{{ cacheHitRate(c) }}%</span>
            </div>
            <div class="cache-bar">
              <div class="cache-bar-fill" :style="{ width: cacheHitRate(c) + '%' }"></div>
            </div>
            <div class="cache-meta">
              <span>命中 {{ num(c.hits) }} · 未命中 {{ num(c.misses) }}</span>
              <span>淘汰 {{ num(c.evictions) }} · 过期 {{ num(c.expired) }}</span>
              <span>占用 {{ c.size }}/{{ c.maxsize }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="two-col">
        <!-- 检索阶段耗时 -->
        <div class="card section">
          <div class="section-head"><h3>检索阶段耗时</h3></div>
          <table v-if="metrics.retrieval_stages && metrics.retrieval_stages.length">
            <thead>
              <tr><th>阶段</th><th>次数</th><th>平均</th><th>最大</th></tr>
            </thead>
            <tbody>
              <tr v-for="(s, i) in metrics.retrieval_stages" :key="i">
                <td>{{ s.stage }}</td>
                <td class="num">{{ s.count }}</td>
                <td class="num">{{ s.avg_ms }} ms</td>
                <td class="num">{{ s.max_ms }} ms</td>
              </tr>
            </tbody>
          </table>
          <div v-else class="empty-hint">暂无检索数据</div>
        </div>

        <!-- 最慢接口 -->
        <div class="card section">
          <div class="section-head"><h3>最慢接口 Top 5</h3></div>
          <table v-if="metrics.top_slow_paths && metrics.top_slow_paths.length">
            <thead>
              <tr><th>路径</th><th>调用</th><th>平均</th><th>最大</th></tr>
            </thead>
            <tbody>
              <tr v-for="(p, i) in metrics.top_slow_paths.slice(0, 5)" :key="i">
                <td class="path" :title="p.path">{{ p.path }}</td>
                <td class="num">{{ p.count }}</td>
                <td class="num">{{ p.avg_ms }} ms</td>
                <td class="num">{{ p.max_ms }} ms</td>
              </tr>
            </tbody>
          </table>
          <div v-else class="empty-hint">暂无数据</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.monitor-page {
  padding: 26px 30px;
  height: 100%;
  overflow-y: auto;
  animation: fadeIn 0.2s ease;
}
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 22px;
  gap: 16px;
  flex-wrap: wrap;
}
.title-icon {
  width: 34px;
  height: 34px;
  padding: 7px;
  border-radius: 10px;
  background: linear-gradient(145deg, var(--primary-tint), var(--primary-tint-2));
  color: var(--primary);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}
.header-actions { display: flex; gap: 10px; }
.dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--text-3); display: inline-block;
}
.dot.on { background: var(--success); box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.18); }

.state-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 70px 20px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.spinner {
  width: 28px; height: 28px;
  border: 3px solid var(--border);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-bottom: 12px;
}
.state-text { color: var(--text-3); font-size: 13px; }

/* KPI */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 18px;
}
.kpi {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 20px;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-card-hover);
}
.kpi-icon {
  width: 44px; height: 44px;
  border-radius: 12px;
  background: linear-gradient(145deg, var(--primary-tint), var(--primary-tint-2));
  color: var(--primary);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}
.kpi-icon svg { width: 22px; height: 22px; }
.kpi-icon.ok {
  background: linear-gradient(145deg, var(--success-tint), rgba(16, 185, 129, 0.08));
  color: var(--success);
}
.kpi-label { font-size: 12.5px; color: var(--text-2); margin-bottom: 3px; }
.kpi-value {
  font-size: 23px;
  font-weight: 700;
  color: var(--text-1);
  line-height: 1.2;
  letter-spacing: -0.3px;
  font-variant-numeric: tabular-nums;
}
.kpi-foot { font-size: 12px; margin-top: 3px; color: var(--text-3); }
.kpi-foot .ok { color: var(--success); }
.kpi-foot .err { color: var(--danger); }

/* 区块 */
.section {
  padding: 22px 24px;
  margin-bottom: 18px;
}
.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
.section-head h3 {
  font-size: 15.5px;
  font-weight: 700;
  letter-spacing: 0.1px;
  color: var(--text-1);
}
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

/* LLM */
.llm-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
}
.llm-item {
  background: linear-gradient(180deg, var(--surface-2), var(--bg-soft));
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 14px 16px;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.llm-item:hover {
  border-color: var(--border-strong);
  box-shadow: var(--shadow-xs);
}
.llm-item.highlight {
  background: linear-gradient(145deg, rgba(124, 193, 247, 0.22), rgba(47, 143, 224, 0.14));
  border-color: rgba(124, 193, 247, 0.42);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6),
              0 10px 22px -14px rgba(47, 143, 224, 0.55);
}
.llm-label { font-size: 12px; color: var(--text-2); margin-bottom: 6px; }
.llm-value {
  font-size: 19px;
  font-weight: 700;
  color: var(--text-1);
  font-variant-numeric: tabular-nums;
}
.llm-value.cost {
  background: linear-gradient(118deg, #5aa5f4, #2f8fe0);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

/* 检索质量 */
.ret-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}
.ret-item {
  background: linear-gradient(180deg, var(--surface-2), var(--bg-soft));
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 14px 16px;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.ret-item:hover {
  border-color: var(--border-strong);
  box-shadow: var(--shadow-xs);
}
.ret-item.highlight {
  background: linear-gradient(145deg, rgba(124, 193, 247, 0.22), rgba(47, 143, 224, 0.14));
  border-color: rgba(124, 193, 247, 0.42);
  box-shadow: inset 0 1px 0 rgba(255, 255, 0.6),
              0 10px 22px -14px rgba(47, 143, 224, 0.55);
}
.ret-item .llm-value { font-size: 18px; }
.ret-item .llm-value small { font-size: 12px; font-weight: 500; color: var(--text-3); }
.llm-value.dual .ok { color: var(--success); }
.llm-value.dual .err { color: var(--danger); }
.llm-value.dual .sep,
.score-line .sep { color: var(--text-3); margin: 0 2px; }
.score-line { font-size: 16px; }

.recent-table { margin-top: 4px; }
.pill {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
}
.pill-ok { background: var(--success-tint); color: var(--success); }
.pill-warn { background: var(--danger-tint); color: var(--danger); }

/* 缓存 */
.cache-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
}
.cache-item {
  background: linear-gradient(180deg, var(--surface-2), var(--bg-soft));
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 16px 18px;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.cache-item:hover {
  border-color: var(--border-strong);
  box-shadow: var(--shadow-xs);
}
.cache-top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}
.cache-name { font-size: 13.5px; font-weight: 650; color: var(--text-1); }
.cache-rate {
  font-size: 21px;
  font-weight: 700;
  letter-spacing: -0.3px;
  color: var(--primary);
  font-variant-numeric: tabular-nums;
}
.cache-bar {
  height: 6px;
  border-radius: 999px;
  background: var(--surface-3);
  overflow: hidden;
  margin-bottom: 10px;
}
.cache-bar-fill {
  height: 100%;
  background: var(--primary-gradient);
  border-radius: 999px;
  box-shadow: 0 0 10px -2px rgba(124, 193, 247, 0.7);
  transition: width 0.4s ease;
}
.cache-meta {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--text-3);
  font-variant-numeric: tabular-nums;
}

/* 表格 */
table { width: 100%; border-collapse: collapse; }
th, td {
  padding: 10px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
}
tbody tr:last-child td { border-bottom: none; }
th {
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
td { color: var(--text-1); }
td.num { text-align: right; font-variant-numeric: tabular-nums; color: var(--text-2); }
th.num { text-align: right; }
td.path {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-2);
}
tbody tr { transition: background 0.14s ease; }
tbody tr:hover { background: var(--surface-2); }
.empty-hint { text-align: center; color: var(--text-3); padding: 24px; font-size: 13px; }

@media (max-width: 1100px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .llm-grid { grid-template-columns: repeat(2, 1fr); }
  .llm-item.highlight { grid-column: span 2; }
  .ret-grid { grid-template-columns: repeat(2, 1fr); }
  .ret-item.highlight { grid-column: span 2; }
  .cache-grid { grid-template-columns: 1fr; }
  .two-col { grid-template-columns: 1fr; }
}
</style>
