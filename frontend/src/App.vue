<script setup>
import { ref, reactive, onMounted, nextTick } from 'vue'
import { marked } from 'marked'
import FilesPage from './components/FilesPage.vue'
import MonitorPage from './components/MonitorPage.vue'

// API 认证：生产环境通过 VITE_API_KEY 注入，开发环境为空（后端放行）
const API_KEY = import.meta.env.VITE_API_KEY || ''
function withAuth(extra = {}) {
  return { ...extra, 'X-API-Key': API_KEY }
}

// 状态
const activePage = ref('chat')
const messages = reactive([])
const inputText = ref('')
const isLoading = ref(false)
const statusText = ref('就绪')
const currentAbortController = ref(null)
const conversations = ref([])
const currentSessionId = ref(null)
const showScrollDown = ref(false)
const showScrollbar = ref(false)

// 删除确认弹窗
const showDeleteConfirm = ref(false)
const pendingDeleteSessionId = ref(null)
const historyCollapsed = ref(false)
let scrollTimer = null

// DOM 引用
const chatContainer = ref(null)
const textareaRef = ref(null)

// 滚动到底部
function scrollToBottom() {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    showScrollDown.value = false
  }
}

// 监听滚动
function handleScroll() {
  if (chatContainer.value) {
    const { scrollTop, scrollHeight, clientHeight } = chatContainer.value
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100
    showScrollDown.value = !isAtBottom
    
    // 显示滚动条，停止滚动后隐藏
    showScrollbar.value = true
    if (scrollTimer) clearTimeout(scrollTimer)
    scrollTimer = setTimeout(() => {
      showScrollbar.value = false
    }, 1000)
  }
}

// 输入框自动调整高度
function autoResize() {
  const textarea = textareaRef.value
  if (!textarea) return
  textarea.style.height = 'auto'
  textarea.style.height = textarea.scrollHeight + 'px'
}

// 初始化
onMounted(async () => {
  loadChatHistory()
  await loadConversations()
  // 如果有对话，自动加载最近的一段
  if (conversations.value.length > 0 && !currentSessionId.value) {
    loadConversation(conversations.value[0].session_id)
  }
})

// 加载对话列表
async function loadConversations() {
  try {
    const response = await fetch('/history?size=100', { headers: withAuth() })
    const result = await response.json()
    const data = result.data || []
    
    // 先按时间倒序排列所有消息
    data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    
    // 按 session_id 分组，取第一个就是最新的
    const sessionMap = {}
    data.forEach(item => {
      const sid = item.session_id || 'default'
      if (!sessionMap[sid]) {
        sessionMap[sid] = {
          session_id: sid,
          title: item.user_message.slice(0, 20) + (item.user_message.length > 20 ? '...' : ''),
          created_at: item.created_at
        }
      }
    })
    
    conversations.value = Object.values(sessionMap).sort((a, b) => 
      new Date(b.created_at) - new Date(a.created_at)
    )
  } catch (error) {
    console.error('加载对话列表失败:', error)
  }
}


// 切换页面
async function switchPage(page) {
  activePage.value = page
  if (page === 'chat') {
    await loadConversations()
    // 如果没有当前会话，自动加载最近的一段
    if (conversations.value.length > 0 && !currentSessionId.value) {
      loadConversation(conversations.value[0].session_id)
    }
  }
}

// 加载聊天历史
function loadChatHistory() {
  const saved = localStorage.getItem('chatHistory')
  if (saved) {
    try {
      const history = JSON.parse(saved)
      if (history.length > 0) {
        messages.length = 0
        messages.push(...history)
        nextTick(() => {
          chatContainer.value.scrollTop = chatContainer.value.scrollHeight
        })
      }
    } catch (e) {}
  }
}

// 保存聊天历史
function saveChatHistory() {
  const history = messages.map(msg => ({
    role: msg.role,
    text: msg.text,
    html: msg.html,
    toolName: msg.toolName,
    toolCount: msg.toolCount,
    duration: msg.duration,
    time: msg.time || '',
    citations: msg.citations || []
  }))
  localStorage.setItem('chatHistory', JSON.stringify(history.slice(-50)))
}

// 清空对话
function clearChat() {
  messages.length = 0
  currentSessionId.value = 'sess_' + Date.now()
  localStorage.removeItem('chatHistory')
}

// 导出对话
function exportChat() {
  let markdown = '# AI 知识库助手对话记录\n\n'
  markdown += '导出时间：' + new Date().toLocaleString('zh-CN') + '\n\n---\n\n'
  messages.forEach(msg => {
    const role = msg.role === 'user' ? '🧑 我' : '🤖 AI'
    markdown += `## ${role}\n\n${msg.text}\n\n`
  })
  const blob = new Blob([markdown], { type: 'text/markdown' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `chat_${Date.now()}.md`
  a.click()
  URL.revokeObjectURL(url)
}

// 格式化时间：
// 今天 → 「今天 HH:MM」；昨天 → 「昨天 HH:MM」；
// 今年更早 → 「M月d日 HH:MM」；跨年 → 「yyyy/M/d HH:MM」
function formatTime(d) {
  const hhmm = d.getHours() + ':' + String(d.getMinutes()).padStart(2, '0')
  const now = new Date()
  const startOfDay = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate())
  const dayDiff = Math.round((startOfDay(now) - startOfDay(d)) / 86400000)
  if (dayDiff === 0) return '今天 ' + hhmm
  if (dayDiff === 1) return '昨天 ' + hhmm
  if (d.getFullYear() === now.getFullYear()) {
    return `${d.getMonth() + 1}月${d.getDate()}日 ` + hhmm
  }
  return `${d.getFullYear()}/${d.getMonth() + 1}/${d.getDate()} ` + hhmm
}

// 当前时间（今天 HH:MM）
function nowTime() {
  return formatTime(new Date())
}

// 调用后端流式接口，把结果实时写入指定的 AI 消息
async function streamAnswer({ message, history, aiMsg }) {
  currentAbortController.value = new AbortController()
  const response = await fetch('/agent/graph-run-stream', {
    method: 'POST',
    headers: withAuth({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      message: message,
      history: history,
      session_id: currentSessionId.value
    }),
    signal: currentAbortController.value.signal
  })

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let answerText = ''
  let updateTimer = null
  let buffer = ''

  const handleEvent = (data) => {
    if (data.type === 'step') {
      const toolName = data.step?.tool_name || data.tool_name || '未知工具'
      aiMsg.toolName = toolName
      aiMsg.toolCount++
      statusText.value = '调用工具：' + toolName
    } else if (data.type === 'answer_chunk') {
      answerText += data.content || data.chunk || ''
      aiMsg.text = answerText
      // 节流：每80ms解析一次Markdown，避免长文本卡顿
      if (!updateTimer) {
        updateTimer = setTimeout(() => {
          aiMsg.html = marked.parse(answerText)
          updateTimer = null
        }, 80)
      }
    } else if (data.type === 'citations') {
      aiMsg.citations = data.citations || []
    } else if (data.type === 'done') {
      if (updateTimer) { clearTimeout(updateTimer); updateTimer = null }
      aiMsg.html = marked.parse(answerText)
      aiMsg.toolCount = data.tool_call_count || aiMsg.toolCount
      aiMsg.duration = (data.total_duration_ms || 0) / 1000
      aiMsg.time = nowTime()
      statusText.value = '就绪'
    } else if (data.type === 'error') {
      aiMsg.html = '<span style="color:#ff4d4f">出错了：' + data.message + '</span>'
      statusText.value = '错误'
    }
  }

  const processEventText = (raw) => {
    const line = raw.split('\n').find(l => l.startsWith('data: '))
    if (!line) return
    try { handleEvent(JSON.parse(line.slice(6))) } catch (e) {}
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx
    // 按完整事件分隔符 \n\n 切分，跨 chunk 也不丢数据
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      processEventText(rawEvent)
    }
    nextTick(() => {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight
    })
  }
  // 流关闭后兜底处理残留事件
  if (buffer.trim()) processEventText(buffer)
}

// 发送消息
async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || isLoading.value) return

  // 添加用户消息
  messages.push({
    role: 'user',
    text: text,
    html: marked.parse(text),
    time: nowTime()
  })
  inputText.value = ''

  // 添加 AI 空消息
  let aiMsg = {
    role: 'assistant',
    text: '',
    html: '',
    toolCount: 0,
    toolName: '',
    duration: 0,
    time: '',
    citations: []
  }
  messages.push(aiMsg)
  // 取回 reactive proxy，后续修改才能触发视图更新
  aiMsg = messages[messages.length - 1]

  isLoading.value = true
  statusText.value = '思考中...'

  nextTick(() => {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  })

  // 收集历史对话
  const history = []
  messages.slice(0, -1).forEach(msg => {
    history.push({ role: msg.role, content: msg.text })
  })

  try {
    await streamAnswer({
      message: text,
      history: history.slice(-20),
      aiMsg: aiMsg
    })
  } catch (error) {
    if (error.name === 'AbortError') {
      statusText.value = '已停止'
    } else {
      aiMsg.html = '<span style="color:#ff4d4f">请求失败：' + error.message + '</span>'
      statusText.value = '错误'
    }
  } finally {
    isLoading.value = false
    saveChatHistory()
    loadConversations()
  }
}

// 重新生成指定的 AI 回答
async function regenerate(index) {
  if (isLoading.value) return
  const aiMsg = messages[index]
  if (!aiMsg || aiMsg.role !== 'assistant') return

  // 定位该回答对应的用户问题
  let userIdx = index - 1
  while (userIdx >= 0 && messages[userIdx].role !== 'user') userIdx--
  if (userIdx < 0) return
  const question = messages[userIdx].text

  // 丢弃该回答之后的内容（产生新分支）
  messages.splice(index + 1)

  // 重置该 AI 消息后重新流式生成
  aiMsg.text = ''
  aiMsg.html = ''
  aiMsg.toolCount = 0
  aiMsg.toolName = ''
  aiMsg.duration = 0
  aiMsg.time = ''
  aiMsg.citations = []

  isLoading.value = true
  statusText.value = '重新生成中...'

  nextTick(() => {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight
  })

  // 该回答之前的内容作为历史
  const history = []
  messages.slice(0, index).forEach(msg => {
    history.push({ role: msg.role, content: msg.text })
  })

  try {
    await streamAnswer({
      message: question,
      history: history.slice(-20),
      aiMsg: aiMsg
    })
  } catch (error) {
    if (error.name === 'AbortError') {
      statusText.value = '已停止'
    } else {
      aiMsg.html = '<span style="color:#ff4d4f">请求失败：' + error.message + '</span>'
      statusText.value = '错误'
    }
  } finally {
    isLoading.value = false
    saveChatHistory()
    loadConversations()
  }
}

// 处理发送按钮点击
function handleSendClick() {
  if (isLoading.value) {
    if (currentAbortController.value) {
      currentAbortController.value.abort()
      isLoading.value = false
      statusText.value = '已停止'
    }
  } else {
    sendMessage()
  }
}

// 输入框快捷键
function handleInputKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

// 快捷问题
function quickAsk(question) {
  inputText.value = question
  sendMessage()
}

// 复制消息
function copyMessage(msg) {
  const text = msg.text
  if (navigator.clipboard) {
    navigator.clipboard.writeText(msg.text)
      .then(() => {
        msg.copied = true
        setTimeout(() => msg.copied = false, 1500)
      })
  }
}

// 加载对话
async function loadConversation(sessionId) {
  currentSessionId.value = sessionId
  activePage.value = 'chat'
  
  try {
    const response = await fetch(`/history/session/${sessionId}`, { headers: withAuth() })
    const result = await response.json()
    const data = result.data || []
    
    // 转换为消息格式
    messages.length = 0
    data.forEach(item => {
      const createdTime = new Date(item.created_at)
      const timeStr = formatTime(createdTime)
      
      messages.push({
        role: 'user',
        text: item.user_message,
        html: marked.parse(item.user_message),
        time: timeStr
      })
      messages.push({
        role: 'assistant',
        text: item.ai_answer,
        html: marked.parse(item.ai_answer),
        time: timeStr,
        duration: (item.duration_ms || 0) / 1000,
        toolCount: item.tool_call_count || 0,
        toolName: item.tool_name || '',
        citations: item.citations || []
      })
    })
    
    nextTick(() => {
      setTimeout(() => {
        if (chatContainer.value) {
          chatContainer.value.scrollTop = chatContainer.value.scrollHeight
        }
      }, 100)
    })
  } catch (error) {
    console.error('加载对话失败:', error)
  }
}

// 点击删除：打开自定义确认弹窗（不立即删除）
function deleteConversation(sessionId) {
  pendingDeleteSessionId.value = sessionId
  showDeleteConfirm.value = true
}

// 取消删除
function cancelDelete() {
  showDeleteConfirm.value = false
  pendingDeleteSessionId.value = null
}

// 确认删除：执行真正的删除
async function confirmDelete() {
  const sessionId = pendingDeleteSessionId.value

  // 先关闭弹窗
  showDeleteConfirm.value = false
  pendingDeleteSessionId.value = null

  if (sessionId === null) {
    return
  }

  try {
    // 一次调用删除整个会话
    await fetch(
      `/history/session/${encodeURIComponent(sessionId)}`,
      { method: 'DELETE', headers: withAuth() }
    )

    // 删除的是当前正在查看的会话：重置为新对话
    if (currentSessionId.value === sessionId) {
      messages.length = 0
      currentSessionId.value = 'sess_' + Date.now()
      localStorage.removeItem('chatHistory')
    }

    // 刷新侧边栏
    await loadConversations()
  } catch (error) {
    console.error('删除对话失败:', error)
  }
}
</script>

<template>
  <div class="app">
    <!-- 左侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-logo">🤖 AI 知识库</div>
      <button class="sidebar-new-btn" @click="clearChat()">+ 新对话</button>
      
      <div class="sidebar-item" :class="{ active: activePage === 'chat' }" @click="switchPage('chat')">💬 对话</div>
      <div class="sidebar-item" :class="{ active: activePage === 'files' }" @click="switchPage('files')">📁 文件管理</div>
      <div class="sidebar-item" :class="{ active: activePage === 'monitor' }" @click="switchPage('monitor')">📊 系统监控</div>
      <div class="sidebar-item" @click="exportChat()">📥 导出对话</div>

      <div class="sidebar-section collapsible" @click="historyCollapsed = !historyCollapsed">
        <span>最近</span>
        <span class="collapse-arrow" :class="{ collapsed: historyCollapsed }">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
        </span>
      </div>
      <div class="sidebar-history" v-show="!historyCollapsed">
        <div
          v-for="conv in conversations"
          :key="conv.session_id"
          class="sidebar-item conversation-item"
          :class="{ active: currentSessionId === conv.session_id }"
          @click="loadConversation(conv.session_id)"
        >
          <span class="conv-title">💬 {{ conv.title }}</span>
          <button class="conv-delete" title="删除对话" @click.stop="deleteConversation(conv.session_id)">×</button>
        </div>
      </div>
    </aside>

    <!-- 主内容区 -->
    <main class="main">
      <!-- 聊天页面 -->
      <div v-if="activePage === 'chat'" class="chat-page" style="position: relative;">
        <div class="chat-container" :class="{ 'show-scrollbar': showScrollbar }" ref="chatContainer" @scroll="handleScroll">
          <!-- 欢迎消息 -->
          <div v-if="messages.length === 0" class="message assistant">
            <div class="avatar">AI</div>
            <div class="bubble">
              请问有什么可以帮您？
              <br><br>
              <button @click="quickAsk('系统里有哪些文件？')" class="quick-btn">📋 列出所有文件</button>
              <button @click="quickAsk('test02.txt 的主要内容是什么？')" class="quick-btn">📖 查看文档内容</button>
            </div>
          </div>

          <!-- 消息列表 -->
          <div v-for="(msg, index) in messages" :key="index" class="message" :class="msg.role">
            <div class="avatar">{{ msg.role === 'user' ? '我' : 'AI' }}</div>
            
            <!-- AI 消息：内容 + 操作栏都在气泡里 -->
            <div v-if="msg.role === 'assistant'" class="bubble">
              <div v-if="msg.toolName" class="tool-tag">🔧 {{ msg.toolName }}</div>
              <div v-if="!msg.html" class="loading-dots"><span></span><span></span><span></span></div>
              <div v-html="msg.html" v-else></div>
              <div v-if="msg.citations && msg.citations.length" class="citations">
                <div class="citations-title">📚 参考来源 · {{ msg.citations.length }}</div>
                <div v-for="cit in msg.citations" :key="cit.citation_id" class="citation-item">
                  <div class="citation-head">
                    <span class="citation-num">[{{ cit.citation_id }}]</span>
                    <span class="citation-file">📄 {{ cit.filename }}<template v-if="cit.chunk_id !== null && cit.chunk_id !== undefined"> · 块 {{ cit.chunk_id }}</template></span>
                  </div>
                  <div class="citation-text">{{ cit.content }}</div>
                </div>
              </div>
              <div v-if="msg.html" class="chat-meta">
                <span v-if="msg.duration">耗时 {{ msg.duration.toFixed(1) }}s · </span>
                <span v-if="msg.toolCount">工具调用 {{ msg.toolCount }} 次 · </span>
                <span v-if="msg.time">{{ msg.time }} · </span>
                <button @click="regenerate(index)" class="meta-icon-btn" title="重新生成">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <polyline points="1 20 1 14 7 14"></polyline>
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                  </svg>
                </button>
                <button @click="copyMessage(msg, $event)" class="meta-icon-btn" title="复制">
                  <svg v-if="!msg.copied" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <svg v-else width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                </button>
              </div>
            </div>

            <!-- 用户消息：内容在气泡里，操作栏在气泡外 -->
            <div v-else class="user-content">
              <div class="bubble">{{ msg.text }}</div>
              <div class="user-meta">
                <span class="meta-time">{{ msg.time || '今天 ' + new Date().getHours() + ':' + String(new Date().getMinutes()).padStart(2, '0') }}</span>
                <button @click="copyMessage(msg, $event)" class="meta-icon-btn" title="复制">
                  <svg v-if="!msg.copied" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <svg v-else width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- 输入区 -->
        <div class="input-area">
          <div class="input-box">
            <div class="input-top">
              <button class="plus-btn">+</button>
              <textarea
                v-model="inputText"
                placeholder="输入你的问题..."
                rows="1"
                ref="textareaRef"
                @keydown="handleInputKeydown"
                @input="autoResize"
              ></textarea>
              <button class="send-btn" :class="{ loading: isLoading }" @click="handleSendClick">
                {{ isLoading ? '停止' : '↑' }}
              </button>
            </div>
          </div>
        </div>

        <!-- 回到底部按钮 -->
        <button 
          v-if="showScrollDown" 
          class="scroll-down-btn" 
          @click="scrollToBottom"
        >
          ↓
        </button>

      </div>

      <!-- 文件管理页面 -->
      <FilesPage v-else-if="activePage === 'files'" />

      <!-- 系统监控页面 -->
      <MonitorPage v-else-if="activePage === 'monitor'" />
    </main>

    <!-- 删除确认弹窗（传送到 body，遮罩覆盖全屏） -->
    <Teleport to="body">
      <div
        v-if="showDeleteConfirm"
        class="confirm-overlay"
        @click.self="cancelDelete"
      >
        <div
          class="confirm-dialog"
          role="dialog"
          aria-modal="true"
          aria-labelledby="confirm-title"
        >
          <div class="confirm-icon">
            <svg
              width="26"
              height="26"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#e5484d"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M3 6h18" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
              <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              <line x1="10" y1="11" x2="10" y2="17" />
              <line x1="14" y1="11" x2="14" y2="17" />
            </svg>
          </div>
          <div id="confirm-title" class="confirm-title">
            删除这段对话？
          </div>
          <div class="confirm-text">
            删除后该对话将无法恢复，请确认是否继续。
          </div>
          <div class="confirm-actions">
            <button class="confirm-btn cancel" @click="cancelDelete">
              取消
            </button>
            <button class="confirm-btn danger" @click="confirmDelete">
              删除
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style>
/* ============ 删除确认弹窗 ============ */
.confirm-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  animation: confirmFade 0.15s ease;
}

.confirm-dialog {
  width: 360px;
  max-width: calc(100vw - 40px);
  background: #ffffff;
  border-radius: 14px;
  padding: 28px 26px 22px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.18);
  text-align: center;
  animation: confirmPop 0.18s ease;
}

.confirm-icon {
  width: 52px;
  height: 52px;
  margin: 0 auto 14px;
  border-radius: 50%;
  background: #fdecec;
  display: flex;
  align-items: center;
  justify-content: center;
}

.confirm-title {
  font-size: 17px;
  font-weight: 600;
  color: #1f2329;
  margin-bottom: 8px;
}

.confirm-text {
  font-size: 14px;
  color: #8a8f99;
  line-height: 1.6;
  margin-bottom: 22px;
}

.confirm-actions {
  display: flex;
  gap: 12px;
}

.confirm-btn {
  flex: 1;
  height: 40px;
  border-radius: 10px;
  font-size: 14px;
  cursor: pointer;
  border: 1px solid #e0e0e0;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.confirm-btn.cancel {
  background: #ffffff;
  color: #4e5969;
}

.confirm-btn.cancel:hover {
  background: #f5f6f7;
}

.confirm-btn.danger {
  background: #e5484d;
  border-color: #e5484d;
  color: #ffffff;
}

.confirm-btn.danger:hover {
  background: #d23b40;
  border-color: #d23b40;
}

@keyframes confirmFade {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes confirmPop {
  from { opacity: 0; transform: scale(0.94); }
  to { opacity: 1; transform: scale(1); }
}
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #f5f5f5;
}

.app {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* 侧边栏 */
.sidebar {
  width: 240px;
  background: #fff;
  border-right: 1px solid #e0e0e0;
  padding: 16px;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}

/* 最近对话列表：占满剩余高度、独立滚动，
   顶部导航始终固定可见。滚动条平时很淡、
   滚动 / 悬停时才明显 */
.sidebar-history {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 0, 0, 0.18) transparent;
}
.sidebar-history::-webkit-scrollbar {
  width: 6px;
}
.sidebar-history::-webkit-scrollbar-track {
  background: transparent;
}
.sidebar-history::-webkit-scrollbar-thumb {
  background: rgba(0, 0, 0, 0.15);
  border-radius: 3px;
}
.sidebar-history::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 0, 0, 0.32);
}

.sidebar-logo {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  padding: 8px;
}

.sidebar-new-btn {
  padding: 10px 16px;
  background: #409eff;
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 20px;
  font-size: 14px;
}

.sidebar-section {
  font-size: 12px;
  color: #999;
  margin: 12px 0 8px;
  padding: 0 8px;
}

/* 可折叠的“最近对话”标题 */
.sidebar-section.collapsible {
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: pointer;
  user-select: none;
}
.sidebar-section.collapsible:hover {
  color: #666;
}
.collapse-arrow {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #b0b0b0;
  line-height: 1;
  transition: transform 0.2s ease;
}
.sidebar-section.collapsible:hover .collapse-arrow {
  color: #888;
}
.collapse-arrow.collapsed {
  transform: rotate(-90deg);
}

.sidebar-item {
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  margin-bottom: 4px;
}

.sidebar-item:hover {
  background: #f5f5f5;
}

.sidebar-item.active {
  background: #ecf5ff;
  color: #409eff;
}

/* 对话列表项 */
.conversation-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-right: 8px;
}

.conv-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: left;
}

.conv-delete {
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 16px;
  color: #999;
  padding: 2px 6px;
  border-radius: 4px;
  opacity: 0;
  transition: opacity 0.2s;
}

.conversation-item:hover .conv-delete {
  opacity: 1;
}

.conv-delete:hover {
  background: #fde2e2;
  color: #e5484d;
}

/* 主内容区 */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* 聊天页面 */
.chat-page {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.chat-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  min-height: 0;
}

/* 回到底部按钮 */
.scroll-down-btn {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  bottom: 92px;
  width: 38px;
  height: 38px;
  border-radius: 50%;
  background: #fff;
  border: none;
  box-shadow: 0 2px 8px rgba(0,0,0,0.12);
  cursor: pointer;
  font-size: 17px;
  color: #666;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  opacity: 0.92;
  transition: all 0.2s;
}

.scroll-down-btn:hover {
  background: #f5f5f5;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
  transform: translateX(-50%) translateY(-2px);
  opacity: 1;
}

/* 消息样式 */
.message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
}

.message.user {
  flex-direction: row-reverse;
  justify-content: flex-start;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 600;
  flex-shrink: 0;
  background: #10b981;
  color: #fff;
}

.message.user .avatar {
  background: #409eff;
}

.bubble {
  max-width: 75%;
  padding: 14px 18px;
  border-radius: 16px;
  line-height: 1.7;
  font-size: 14px;
  background: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
  text-align: left;
}

/* Markdown 渲染样式 */
.bubble p {
  margin: 8px 0;
}
.bubble h1, .bubble h2, .bubble h3 {
  margin: 16px 0 8px 0;
  font-weight: 600;
}
.bubble h1 {
  font-size: 1.3em;
}
.bubble h2 {
  font-size: 1.2em;
}
.bubble h3 {
  font-size: 1.1em;
}
.bubble ul, .bubble ol {
  margin: 8px 0;
  padding-left: 24px;
}
.bubble li {
  margin: 4px 0;
}
.bubble code {
  background: #f5f5f5;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
}
.bubble pre {
  background: #f5f5f5;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 8px 0;
}
.bubble pre code {
  background: none;
  padding: 0;
}
.bubble table {
  width: 100%;
  border-collapse: collapse;
  margin: 8px 0;
}
.bubble th, .bubble td {
  border: 1px solid #e0e0e0;
  padding: 8px;
  text-align: left;
}
.bubble strong {
  font-weight: 600;
}

.message.user .bubble {
  background: #f0f0f0;
  color: #333;
  border-top-right-radius: 6px;
  max-width: 70%;
  flex-shrink: 1;
  word-break: break-word;
}
.tool-tag {
  display: inline-block;
  padding: 4px 10px;
  background: #f0f5ff;
  color: #409eff;
  border-radius: 12px;
  font-size: 12px;
  margin-bottom: 8px;
}

.loading-dots { display: flex; gap: 5px; padding: 4px 0; }
.loading-dots span {
  width: 8px; height: 8px; border-radius: 50%; background: #888;
  animation: ld 1.2s infinite ease-in-out both;
}
.loading-dots span:nth-child(1) { animation-delay: -0.24s; }
.loading-dots span:nth-child(2) { animation-delay: -0.12s; }
@keyframes ld {
  0%, 80%, 100% { transform: translateY(0); opacity: .4; }
  40% { transform: translateY(-5px); opacity: 1; }
}
.chat-meta {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid #f0f0f0;
  font-size: 12px;
  color: #999;
  display: flex;
  align-items: center;
  gap: 4px;
}

.meta-icon-btn {
  padding: 4px;
  border: none;
  background: transparent;
  border-radius: 6px;
  cursor: pointer;
  color: #999;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.meta-icon-btn:hover {
  background: #f0f0f0;
  color: #666;
}

.meta-time {
  margin-left: auto;
  font-size: 12px;
  color: #999;
}

/* 用户消息内容容器 */
.user-content {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  flex: 1;
  min-width: 0;
}

/* 用户消息操作栏（hover 才显示，在气泡外面） */
.user-meta {
  margin-top: 6px;
  font-size: 12px;
  color: #999;
  opacity: 0;
  transition: opacity 0.2s;
  display: flex;
  align-items: center;
  gap: 8px;
}

.message.user:hover .user-meta {
  opacity: 1;
}

.meta-icon-btn {
  padding: 2px 6px;
  border: none;
  background: transparent;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.meta-icon-btn:hover {
  background: #e0e0e0;
}

.meta-btn {
  padding: 2px 8px;
  border: 1px solid #e0e0e0;
  background: #fff;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
}

.quick-btn {
  padding: 6px 12px;
  border: 1px solid #e0e0e0;
  background: #fff;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  margin: 4px;
}

/* 输入区 */
.input-area {
  padding: 16px 20px;
  background: #f5f5f5;
}

.input-box {
  background: #fff;
  border-radius: 16px;
  padding: 12px 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.input-top {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.plus-btn {
  width: 32px;
  height: 32px;
  border: 1px solid #e0e0e0;
  background: #fff;
  border-radius: 50%;
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 4px;
}

.input-top textarea {
  flex: 1;
  border: none;
  outline: none;
  font-size: 14px;
  resize: none;
  font-family: inherit;
  line-height: 1.5;
  padding: 6px 0;
}

.input-bottom {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  padding-top: 8px;
}

.input-left-bar {
  display: flex;
  gap: 8px;
  align-items: center;
}

.tag {
  padding: 4px 10px;
  background: #f5f5f5;
  border-radius: 12px;
  font-size: 12px;
  color: #666;
}

.input-right-bar {
  display: flex;
  align-items: center;
  gap: 12px;
}

.version-tag {
  font-size: 12px;
  color: #999;
}

.send-btn {
  width: 36px;
  height: 36px;
  background: #409eff;
  color: #fff;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-top: 4px;
}

.send-btn.loading {
  background: #ff4d4f;
}

.status-bar {
  padding: 8px 20px;
  font-size: 12px;
  color: #999;
  background: #fff;
}

/* Toast 动画 */
@keyframes fadeInOut {
  0% { opacity: 0; }
  20% { opacity: 1; }
  80% { opacity: 1; }
  100% { opacity: 0; }
}

/* 引用来源 */
.citations {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 1px dashed #e5e7eb;
}
.citations-title {
  font-size: 12px;
  color: #8a8f99;
  margin-bottom: 8px;
  font-weight: 500;
}
.citation-item {
  background: #f8f9fb;
  border: 1px solid #eef0f3;
  border-radius: 8px;
  padding: 8px 10px;
  margin-bottom: 6px;
}
.citation-head {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  margin-bottom: 4px;
}
.citation-num {
  color: #3b82f6;
  font-weight: 600;
}
.citation-file {
  color: #8a8f99;
}
.citation-text {
  font-size: 12.5px;
  color: #6b7280;
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.citation-item:hover .citation-text {
  -webkit-line-clamp: unset;
}
</style>
