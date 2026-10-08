<script setup>
import { ref, onMounted } from 'vue'

// 状态
const files = ref([])
const isLoading = ref(false)
const isUploading = ref(false)

// 加载文件列表
async function loadFiles() {
  isLoading.value = true
  try {
    const response = await fetch('/upload/files')
    const result = await response.json()
    files.value = result.data || []
  } catch (error) {
    alert('加载文件失败：' + error.message)
  } finally {
    isLoading.value = false
  }
}

// 上传文件
async function uploadFile(event) {
  const file = event.target.files[0]
  if (!file) return

  isUploading.value = true

  const formData = new FormData()
  formData.append('file', file)

  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    })
    const data = await response.json()
    alert('上传成功！')
    loadFiles()
  } catch (error) {
    alert('上传失败：' + error.message)
  } finally {
    isUploading.value = false
    event.target.value = ''
  }
}

// 删除文件
async function deleteFile(fileId) {
  if (!confirm('确定要删除这个文件吗？')) return

  try {
    await fetch(`/upload/${fileId}`, { method: 'DELETE' })
    alert('删除成功！')
    loadFiles()
  } catch (error) {
    alert('删除失败：' + error.message)
  }
}

// 重建索引
async function rebuildIndex(fileId) {
  if (!confirm('确定要重建所有文件的索引吗？')) return
  try {
    const res = await fetch('/upload/rebuild-index', { method: 'POST' })
    const data = await res.json()
    alert('重建完成！\n成功: ' + data.data.results.filter(r => r.status === 'success').length + ' 个\n失败: ' + data.data.results.filter(r => r.status === 'failed').length + ' 个')
  } catch (e) {
    alert('重建失败: ' + e.message)
  }
}

// 格式化时间
function formatTime(timeStr) {
  if (!timeStr) return '-'
  return new Date(timeStr).toLocaleString('zh-CN')
}

// 格式化文件大小
function formatSize(bytes) {
  if (!bytes) return '-'
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / 1024 / 1024).toFixed(1) + ' MB'
}

onMounted(() => {
  loadFiles()
})
</script>

<template>
  <div class="files-page">
    <div class="page-header">
      <h2>📁 文件管理</h2>
      <div class="header-actions">
        <label class="upload-btn">
          📤 上传文件
          <input type="file" @change="uploadFile" style="display: none" />
        </label>
        <button class="refresh-btn" @click="loadFiles()">🔄 刷新</button>
      </div>
    </div>

    <div v-if="isLoading" class="loading">加载中...</div>

    <div v-else-if="files.length === 0" class="empty">
      <p>暂无文件，点击上方"上传文件"上传第一个文档吧</p>
    </div>

    <div v-else class="files-table">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>文件名</th>
            <th>类型</th>
            <th>大小</th>
            <th>上传时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="file in files" :key="file.id">
            <td>{{ file.id }}</td>
            <td>{{ file.filename }}</td>
            <td>{{ file.file_type }}</td>
            <td>{{ formatSize(file.file_size) }}</td>
            <td>{{ formatTime(file.created_at) }}</td>
            <td>
              <button class="action-btn rebuild" @click="rebuildIndex(file.id)">🔄 重建索引</button>
              <button class="action-btn delete" @click="deleteFile(file.id)">🗑️ 删除</button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.files-page {
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

.header-actions {
  display: flex;
  gap: 12px;
}

.upload-btn {
  padding: 10px 16px;
  background: #409eff;
  color: #fff;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
}

.refresh-btn {
  padding: 10px 16px;
  background: #f5f5f5;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
}

.loading, .empty {
  text-align: center;
  padding: 60px;
  color: #999;
}

.files-table {
  background: #fff;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 14px 16px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
}

th {
  background: #fafafa;
  font-weight: 600;
  font-size: 13px;
  color: #666;
}

td {
  font-size: 14px;
}

.action-btn {
  padding: 4px 10px;
  border: 1px solid #e0e0e0;
  background: #fff;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  margin-right: 8px;
}

.action-btn.rebuild:hover {
  background: #ecf5ff;
  border-color: #409eff;
  color: #409eff;
}

.action-btn.delete:hover {
  background: #fff1f0;
  border-color: #ff4d4f;
  color: #ff4d4f;
}
</style>
