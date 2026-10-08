"""
统一安全中间件：API Key 认证 + 基于 IP 的内存限流。

- 公开路径（/health、/docs、静态资源、前端 SPA）放行。
- API_KEY 未配置时不启用认证（开发模式）。
- LLM 相关接口（/agent、/chat、/rag）按 IP 限流。
- 限流为单进程内存实现；多 worker 部署需换 Redis 等共享存储。
"""

import json
import time
from collections import defaultdict, deque

from core.config import settings


# 不需要认证的公开路径前缀
_PUBLIC_PATH_PREFIXES = (
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/static",
    "/assets",
    "/favicon.ico",
)

# 需要限流的路径前缀（LLM 相关接口）
_RATE_LIMITED_PREFIXES = (
    "/agent",
    "/chat",
    "/rag",
)


class _RateLimiter:
    """内存滑动窗口限流（单进程有效）。"""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        dq = self._requests[key]
        while dq and now - dq[0] > self.window:
            dq.popleft()
        if len(dq) >= self.max_requests:
            return False
        dq.append(now)
        return True


_rate_limiter = _RateLimiter(
    max_requests=settings.rate_limit_per_minute,
    window_seconds=60,
)


def _is_public(path: str) -> bool:
    if path in ("/", "/chat"):
        return True
    return any(
        path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES
    )


def _is_rate_limited(path: str) -> bool:
    return any(
        path.startswith(prefix) for prefix in _RATE_LIMITED_PREFIXES
    )


def _get_client_ip(scope: dict) -> str:
    client = scope.get("client")
    if client:
        return client[0]
    headers = dict(scope.get("headers", []))
    forwarded = headers.get(b"x-forwarded-for", b"").decode()
    if forwarded:
        return forwarded.split(",")[0].strip()
    return "unknown"


async def _send_json_response(send, status_code: int, body: dict):
    body_bytes = json.dumps(body, ensure_ascii=False).encode()
    await send({
        "type": "http.response.start",
        "status": status_code,
        "headers": [
            (b"content-type", b"application/json"),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body_bytes,
    })


class SecurityMiddleware:
    """
    统一安全中间件：API Key 认证 + 限流。

    执行顺序：公开路径放行 → 限流检查 → 认证检查 → 放行。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # 公开路径直接放行
        if _is_public(path):
            await self.app(scope, receive, send)
            return

        # 限流（仅 LLM 相关接口）
        if _is_rate_limited(path):
            client_ip = _get_client_ip(scope)
            if not _rate_limiter.is_allowed(client_ip):
                await _send_json_response(send, 429, {
                    "code": 429,
                    "message": "error",
                    "error": {
                        "type": "rate_limit",
                        "message": "请求过于频繁，请稍后再试。",
                    },
                })
                return

        # 认证（仅在配置了 API_KEY 时启用）
        if settings.api_key:
            headers = dict(scope.get("headers", []))
            api_key = headers.get(b"x-api-key", b"").decode()
            if api_key != settings.api_key:
                await _send_json_response(send, 401, {
                    "code": 401,
                    "message": "error",
                    "error": {
                        "type": "unauthorized",
                        "message": "缺少或无效的 API Key。",
                    },
                })
                return

        await self.app(scope, receive, send)
