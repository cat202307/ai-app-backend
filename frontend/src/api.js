// 统一的后端请求封装：
// 1. 所有请求自动带上 X-API-Key（生产环境通过 VITE_API_KEY 注入）；
// 2. 统一解析 JSON、处理非 2xx 与网络错误，抛出带中文信息的 Error。

export const API_KEY = import.meta.env.VITE_API_KEY || ''

export function authHeaders(extra = {}) {
  return { ...extra, 'X-API-Key': API_KEY }
}

/**
 * 统一 fetch 封装。
 * @returns {Promise<any>} 解析后的 JSON
 */
export async function apiFetch(url, options = {}) {
  const headers = authHeaders(options.headers || {})

  let response
  try {
    response = await fetch(url, { ...options, headers })
  } catch (networkError) {
    throw new Error('网络异常，请检查后端服务是否已启动')
  }

  // 尝试解析 JSON（即使状态码非 200，FastAPI 通常也返回 detail）
  let payload = null
  try {
    payload = await response.json()
  } catch (e) {
    payload = null
  }

  if (!response.ok) {
    const detail =
      payload && (payload.detail || payload.message)
        ? payload.detail || payload.message
        : `请求失败（HTTP ${response.status}）`
    throw new Error(detail)
  }

  return payload
}

/**
 * 上传文件（multipart/form-data），并在出错时带上后端 detail。
 */
export async function apiUpload(url, formData) {
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
    headers: authHeaders(),
  })

  let payload = null
  try {
    payload = await response.json()
  } catch (e) {
    payload = null
  }

  if (!response.ok) {
    const detail =
      payload && (payload.detail || payload.message)
        ? payload.detail || payload.message
        : `上传失败（HTTP ${response.status}）`
    throw new Error(detail)
  }

  return payload
}
