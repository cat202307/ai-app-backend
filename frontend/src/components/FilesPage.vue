<script setup>
import { ref, reactive, onMounted } from 'vue'
import { apiFetch, apiUpload } from '../api.js'

// 单文件大小上限（与后端 UPLOAD_MAX_MB 对齐，前端先拦截）
const MAX_MB = Number(import.meta.env.VITE_UPLOAD_MAX_MB || 50)

// 状态
const files = ref([])
const isLoading = ref(false)
const isUploading = ref(false)
const isRebuilding = ref(false)
const fileInput = ref(null)

// 轻量 Toast
const toast = reactive({ visible: false, type: 'info', message: '' })
let toastTimer = null
function showToast(type, message) {
  toast.type = type
  toast.message = message
  toast.visible = true
  if (toastTimer) clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.visible = false }, 2600)
}

// 确认弹窗（delete / rebuild）
const confirm = reactive({
  visible: false,
  title: '',
  text: '',
  danger: false,
  confirmText: '确认',
  action: null,
})
function openConfirm(opts) {
  Object.assign(confirm, { visible: true, ...opts })
}
function closeConfirm() {
  confirm.visible = false
  confirm.action = null
}

// 重建结果
const rebuildResult = reactive({
  visible: false,
  success: 0,
  failed: 0,
  failures: [],
})

async function loadFiles() {
  isLoading.value = true
  try {
    const result = await apiFetch('/upload/files')
    files.value = result.data || []
  } catch (error) {
    showToast('danger', '加载文件失败：' + error.message)
  } finally {
    isLoading.value = false
  }
}

function triggerUpload() {
  fileInput.value && fileInput.value.click()
}

async function uploadFile(event) {
  const file = event.target.files[0]
  if (!file) return

  // 前端大小预校验
  if (file.size > MAX_MB * 1024 * 1024) {
    showToast('danger', `文件超过 ${MAX_MB}MB，无法上传`)
    event.target.value = ''
    return
  }

  isUploading.value = true
  const formData = new FormData()
  formData.append('file', file)

  try {
    const res = await apiUpload('/upload', formData)
    const data = res.data || {}
    if (data.index_status === 'success') {
      showToast('success', `已上传并建立索引（${data.indexed_chunks || 0} 个片段）`)
    } else {
      showToast('warning', `已上传，但索引未就绪，可点「重建索引」`)
    }
    loadFiles()
  } catch (error) {
    showToast('danger', error.message)
  } finally {
    isUploading.value = false
    event.target.value = ''
  }
}

function askDelete(file) {
  openConfirm({
    title: '删除该文件？',
    text: `「${file.filename}」及其向量索引将被删除，且无法恢复。`,
    danger: true,
    confirmText: '删除',
    action: () => doDelete(file.id),
  })
}

async function doDelete(fileId) {
  closeConfirm()
  try {
    await apiFetch(`/upload/${fileId}`, { method: 'DELETE' })
    showToast('success', '文件已删除')
    loadFiles()
  } catch (error) {
    showToast('danger', '删除失败：' + error.message)
  }
}

function askRebuild() {
  openConfirm({
    title: '重建全部索引？',
    text: `将为当前 ${files.value.length} 个文件重新解析、切分并建立向量索引，过程中可能需要一些时间。`,
    danger: false,
    confirmText: '开始重建',
    action: doRebuild,
  })
}

async function doRebuild() {
  closeConfirm()
  isRebuilding.value = true
  try {
    const res = await apiFetch('/upload/rebuild-index', { method: 'POST' })
    const results = (res.data && res.data.results) || []
    const success = results.filter(r => r.status === 'success').length
    const failed = results.filter(r => r.status === 'failed')
    rebuildResult.success = success
    rebuildResult.failed = failed.length
    rebuildResult.failures = failed
    rebuildResult.visible = true
    loadFiles()
  } catch (error) {
    showToast('danger', '重建失败：' + error.message)
  } finally {
    isRebuilding.value = false
  }
}

function formatTime(timeStr) {
  if (!timeStr) return '-'
  const d = new Date(timeStr)
  if (isNaN(d.getTime())) return '-'
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}/${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function formatSize(bytes) {
  if (bytes === null || bytes === undefined) return '-'
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / 1024 / 1024).toFixed(2) + ' MB'
}

function typeMeta(type) {
  const map = {
    '.txt': { label: 'TXT', cls: 't-txt' },
    '.md': { label: 'MD', cls: 't-md' },
    '.pdf': { label: 'PDF', cls: 't-pdf' },
    '.docx': { label: 'DOC', cls: 't-doc' },
  }
  return map[type] || { label: (type || 'FILE').replace('.', '').toUpperCase(), cls: 't-doc' }
}

onMounted(() => {
  loadFiles()
})
</script>

<template>
  <div class="files-page">
    <header class="page-header">
      <div>
        <div class="page-title">
          <svg class="title-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          </svg>
          文件管理
          <span class="page-sub" v-if="files.length">共 {{ files.length }} 个文档</span>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn btn-ghost" @click="loadFiles" :disabled="isLoading">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" :class="{ spin: isLoading }">
            <polyline points="23 4 23 10 17 10"></polyline>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
          </svg>
          刷新
        </button>
        <button class="btn btn-ghost" @click="askRebuild" :disabled="isRebuilding || files.length === 0">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" :class="{ spin: isRebuilding }">
            <path d="M21 12a9 9 0 1 1-2.64-6.36"></path>
            <polyline points="21 3 21 9 15 9"></polyline>
          </svg>
          {{ isRebuilding ? '重建中…' : '重建索引' }}
        </button>
        <button class="btn btn-primary" @click="triggerUpload" :disabled="isUploading">
          <svg v-if="!isUploading" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
          <svg v-else class="spin" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
            <path d="M21 12a9 9 0 1 1-6.219-8.56" />
          </svg>
          {{ isUploading ? '上传中…' : '上传文件' }}
        </button>
        <input
          ref="fileInput"
          type="file"
          accept=".txt,.md,.pdf,.docx"
          style="display: none"
          @change="uploadFile"
        />
      </div>
    </header>

    <div v-if="isLoading && files.length === 0" class="state-card">
      <div class="spinner"></div>
      <div class="state-text">加载中…</div>
    </div>

    <div v-else-if="files.length === 0" class="state-card empty">
      <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
      </svg>
      <div class="state-title">暂无文档</div>
      <div class="state-text">上传 TXT / Markdown / PDF / DOCX（≤ {{ MAX_MB }}MB），系统将自动解析并建立向量索引</div>
      <button class="btn btn-primary" @click="triggerUpload" style="margin-top: 16px">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        上传第一个文档
      </button>
    </div>

    <div v-else class="card table-card card-accent">
      <table>
        <thead>
          <tr>
            <th class="col-file">文件</th>
            <th class="col-type">类型</th>
            <th class="col-size">大小</th>
            <th class="col-time">上传时间</th>
            <th class="col-actions">操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="file in files" :key="file.id">
            <td>
              <div class="file-cell">
                <span class="file-ico" :class="typeMeta(file.file_type).cls">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                  </svg>
                </span>
                <span class="file-name" :title="file.filename">{{ file.filename }}</span>
              </div>
            </td>
            <td><span class="badge type-badge" :class="typeMeta(file.file_type).cls">{{ typeMeta(file.file_type).label }}</span></td>
            <td class="muted col-size">{{ formatSize(file.file_size) }}</td>
            <td class="muted col-time">{{ formatTime(file.created_at) }}</td>
            <td class="col-actions">
              <button class="icon-btn danger" title="删除" @click="askDelete(file)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  <line x1="10" y1="11" x2="10" y2="17"></line>
                  <line x1="14" y1="11" x2="14" y2="17"></line>
                </svg>
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 确认弹窗 -->
    <Teleport to="body">
      <div v-if="confirm.visible" class="modal-overlay" @click.self="closeConfirm">
        <div class="modal" role="dialog" aria-modal="true">
          <div class="modal-icon" :class="{ danger: confirm.danger }">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
              <line x1="12" y1="9" x2="12" y2="13"></line>
              <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>
          </div>
          <div class="modal-title">{{ confirm.title }}</div>
          <div class="modal-text">{{ confirm.text }}</div>
          <div class="modal-actions">
            <button class="btn btn-ghost" @click="closeConfirm">取消</button>
            <button class="btn" :class="confirm.danger ? 'btn-danger' : 'btn-primary'" @click="confirm.action">
              {{ confirm.confirmText }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- 重建结果 -->
    <Teleport to="body">
      <div v-if="rebuildResult.visible" class="modal-overlay" @click.self="rebuildResult.visible = false">
        <div class="modal" role="dialog" aria-modal="true">
          <div class="modal-icon success">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="20 6 9 17 4 12"></polyline>
            </svg>
          </div>
          <div class="modal-title">索引重建完成</div>
          <div class="modal-text">
            成功 {{ rebuildResult.success }} 个<span v-if="rebuildResult.failed">，失败 {{ rebuildResult.failed }} 个</span>
          </div>
          <div v-if="rebuildResult.failures.length" class="rebuild-failures">
            <div v-for="(f, i) in rebuildResult.failures" :key="i" class="fail-item">
              <span class="fail-name">{{ f.filename }}</span>
              <span class="fail-msg">{{ f.message }}</span>
            </div>
          </div>
          <div class="modal-actions">
            <button class="btn btn-primary" @click="rebuildResult.visible = false">知道了</button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Toast -->
    <Teleport to="body">
      <div v-if="toast.visible" class="toast" :class="toast.type">
        <svg v-if="toast.type === 'success'" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
        <svg v-else-if="toast.type === 'danger'" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
        <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        <span>{{ toast.message }}</span>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.files-page {
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
.header-actions {
  display: flex;
  gap: 10px;
}

/* 状态 / 空态 */
.state-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  box-shadow: var(--shadow-xs);
  padding: 70px 20px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.empty-icon {
  width: 52px;
  height: 52px;
  color: var(--text-3);
  margin-bottom: 14px;
  opacity: 0.7;
}
.state-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-1);
  margin-bottom: 6px;
}
.state-text {
  font-size: 13px;
  color: var(--text-3);
  max-width: 420px;
}
.spinner {
  width: 28px;
  height: 28px;
  border: 3px solid var(--border);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-bottom: 12px;
}

/* 表格 */
.table-card {
  overflow: hidden;
  padding-top: 3px;
}
table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}
th,
td {
  padding: 13px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
tbody tr:last-child td {
  border-bottom: none;
}
tbody tr {
  transition: background 0.14s ease;
}
tbody tr:hover {
  background: var(--primary-tint);
}
th {
  background: var(--surface-2);
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-3);
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
td {
  font-size: 14px;
  color: var(--text-1);
  white-space: nowrap;
}
.col-file { width: 42%; }
.col-type { width: 11%; }
.col-size { width: 13%; }
.col-time { width: 22%; }
.col-actions { width: 12%; text-align: center !important; }
.muted {
  color: var(--text-2);
  font-variant-numeric: tabular-nums;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-cell {
  display: flex;
  align-items: center;
  gap: 11px;
  min-width: 0;
}
.file-ico {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55);
}
.file-ico svg {
  width: 18px;
  height: 18px;
}
.file-ico.t-txt { background: linear-gradient(145deg, #e9f1fe, #dae7fb); color: #2563eb; }
.file-ico.t-md  { background: linear-gradient(145deg, #fdf4e3, #fbe9cc); color: #d97706; }
.file-ico.t-pdf { background: linear-gradient(145deg, #fdeef0, #fbdde1); color: #e11d48; }
.file-ico.t-doc { background: linear-gradient(145deg, #e7f7f0, #d8f2e6); color: #059669; }
.file-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 550;
}

.type-badge {
  font-weight: 600;
  letter-spacing: 0.4px;
}
.type-badge.t-txt { background: #e5edfd; color: #2563eb; }
.type-badge.t-md  { background: #fdf0d9; color: #d97706; }
.type-badge.t-pdf { background: #fde4e7; color: #e11d48; }
.type-badge.t-doc { background: #dcf4ea; color: #059669; }

.icon-btn {
  width: 32px;
  height: 32px;
  border: 1px solid transparent;
  background: transparent;
  border-radius: 8px;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--text-3);
  transition: all 0.15s ease;
}
.icon-btn:hover.danger {
  background: var(--danger-tint);
  color: var(--danger);
}

/* 弹窗 */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(17, 24, 39, 0.42);
  backdrop-filter: blur(1px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  animation: fadeIn 0.14s ease;
}
.modal {
  width: 380px;
  max-width: calc(100vw - 40px);
  background: var(--surface);
  border-radius: 16px;
  padding: 26px 26px 22px;
  box-shadow: var(--shadow-md);
  text-align: center;
}
.modal-icon {
  width: 50px;
  height: 50px;
  margin: 0 auto 14px;
  border-radius: 50%;
  background: var(--primary-tint);
  color: var(--primary);
  display: flex;
  align-items: center;
  justify-content: center;
}
.modal-icon.danger { background: var(--danger-tint); color: var(--danger); }
.modal-icon.success { background: var(--success-tint); color: var(--success); }
.modal-title {
  font-size: 16.5px;
  font-weight: 650;
  color: var(--text-1);
  margin-bottom: 8px;
}
.modal-text {
  font-size: 13.5px;
  color: var(--text-2);
  line-height: 1.65;
  margin-bottom: 20px;
}
.modal-actions {
  display: flex;
  gap: 10px;
  justify-content: center;
}
.modal-actions .btn {
  flex: 1;
  height: 38px;
}
.rebuild-failures {
  text-align: left;
  max-height: 150px;
  overflow-y: auto;
  margin-bottom: 16px;
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  padding: 8px 10px;
}
.fail-item {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 5px 0;
  font-size: 12.5px;
  border-bottom: 1px solid var(--border);
}
.fail-item:last-child { border-bottom: none; }
.fail-name { color: var(--text-1); font-weight: 500; }
.fail-msg { color: var(--danger); text-align: right; }

/* Toast */
.toast {
  position: fixed;
  bottom: 34px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 8px;
  background: #1f2737;
  color: #fff;
  padding: 11px 18px;
  border-radius: var(--r-pill);
  font-size: 13.5px;
  box-shadow: var(--shadow-md);
  z-index: 2100;
  animation: toastIn 0.2s ease;
  max-width: 80vw;
}
.toast.success { background: #0f7a5c; }
.toast.danger { background: #d23b40; }
.toast.warning { background: #b45309; }
@keyframes toastIn {
  from { opacity: 0; transform: translateX(-50%) translateY(8px); }
  to { opacity: 1; transform: translateX(-50%) translateY(0); }
}
</style>
