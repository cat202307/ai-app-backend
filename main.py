import os
import time
from contextlib import asynccontextmanager

# 必须在导入其他库之前设置离线环境变量
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.exceptions import AppError
from core.logging_config import setup_logging, get_logger
from core.metrics import metrics
from database import init_db
from routers import chat, history, rag, upload, agent


# 初始化日志
setup_logging()
logger = get_logger(__name__)

logger.info("Starting AI App Backend...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    启动 / 关闭生命周期管理。

    启动时先执行 Alembic 迁移（生产 schema 演进入口），
    再预热本地 Embedding / Cross-Encoder，
    把第一次推理的冷启动开销移到启动阶段。
    """
    # 应用启动时统一执行数据库迁移；新库建表、老库增量升级
    init_db()
    logger.info("Database schema up to date")

    try:
        from services.vector_service import embedding_model
        from services.advanced_rag_service import reranker_model

        embedding_model.encode(["知识库预热"])
        reranker_model.predict([("预热问题", "预热内容")])
        logger.info("Local models warmed up")
    except Exception as exc:
        # 预热失败不影响服务启动
        logger.warning(f"Model warmup skipped: {exc}")

    yield


app = FastAPI(
    title="AI App Backend",
    description="AI 应用开发后端服务",
    version="1.2.0",
    lifespan=lifespan,
)


# CORS：来源从配置读取（CORS_ORIGINS 环境变量），
# 默认允许 Vue 开发服务器(3000)直连后端，绕过 Vite 代理对 SSE 的缓冲
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 请求中间件：记录耗时、IP、状态码
# 注意：必须用纯 ASGI 中间件。BaseHTTPMiddleware(@app.middleware http)
# 会缓冲整个响应体，导致 SSE 流式输出退化为“一次性返回”。
# ============================================================

class LogAndMetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            # 直接透传，不读取/缓冲 body，SSE 保持流式
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.time() - start_time) * 1000
            client = scope.get("client")
            client_ip = client[0] if client else "unknown"
            path = scope.get("path", "")
            metrics.record_request(
                method=scope.get("method", ""),
                path=path,
                duration_ms=duration_ms,
                status_code=status_code,
            )
            logger.info(
                f"{client_ip} | {scope.get('method','')} {path} | "
                f"{status_code} | {duration_ms:.1f}ms"
            )


app.add_middleware(LogAndMetricsMiddleware)

# 安全中间件：API Key 认证 + 限流（公开路径放行，未配置 API_KEY 时不启用认证）
from core.middleware import SecurityMiddleware
app.add_middleware(SecurityMiddleware)


# ============================================================
# 统一业务异常处理
# ============================================================

@app.exception_handler(AppError)
async def app_error_handler(
    request: Request,
    exc: AppError,
) -> JSONResponse:
    """
    统一处理项目中的业务异常。

    例如：
    - 大模型 API Key 未配置
    - 大模型调用超时
    - 大模型鉴权失败
    - 大模型请求限流
    - 大模型服务连接失败

    以前：
        llm_service.py 会把错误作为普通字符串返回。

    现在：
        llm_service.py 抛出 AppError 的子类，
        由这个处理器统一转换成 HTTP 错误响应。
    """

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": "error",
            "error": {
                "type": exc.code,
                "message": exc.message,
                "path": request.url.path,
            },
        },
    )


# ============================================================
# 基础接口
# ============================================================

@app.get("/")
def index():
    """
    项目首页：已构建前端则返回 Vue SPA（frontend/dist），
    未构建时返回后端欢迎信息。
    """

    spa_index = os.path.join(
        "frontend", "dist", "index.html"
    )

    if os.path.exists(spa_index):
        # HTML 禁用缓存：确保发布新版本后刷新立即生效
        # （带 hash 的 JS/CSS 仍可长期缓存）
        return FileResponse(
            spa_index,
            headers={
                "Cache-Control": (
                    "no-cache, no-store, must-revalidate"
                ),
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return {
        "code": 200,
        "message": "success",
        "data": (
            "欢迎使用 AI 应用后端服务"
            "（前端未构建：cd frontend; npm run build）"
        ),
    }


@app.get("/health")
def health_check():
    """
    基础健康检查。
    """

    return {
        "code": 200,
        "message": "success",
        "data": {
            "status": "healthy",
            "service": "ai-app-backend",
            "version": "1.2.0",
        },
    }


@app.get("/metrics")
def get_metrics():
    """
    性能监控指标。

    返回：
    - 运行时间
    - 总请求数
    - 成功/失败数
    - 最慢的接口 Top 10
    """

    summary = metrics.get_summary()

    return {
        "code": 200,
        "message": "success",
        "data": summary,
    }


# ============================================================
# 注册业务路由
# ============================================================

app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(history.router)
app.include_router(rag.router)
app.include_router(agent.router)


# ============================================================
# 静态文件服务
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


@app.get("/chat")
def chat_page():
    # 旧 /chat 页面已废弃，重定向到 Vue SPA 首页
    return RedirectResponse(url="/")


# ============================================================
# 生产环境：Vue 3 SPA（frontend/dist）单端口服务
#
# 构建方式：cd frontend; npm run build
# 该挂载放在最后：API 路由（/agent、/files、/upload、
# /history、/metrics、/docs、/static、/chat 等）已在前面
# 注册并优先匹配；其余路径交由静态文件，访问 / 返回
# index.html，实现只用 8000 一个端口即可访问完整前端。
# ============================================================

_SPA_DIST = os.path.join("frontend", "dist")

if os.path.isdir(_SPA_DIST):
    app.mount(
        "/",
        StaticFiles(directory=_SPA_DIST, html=True),
        name="spa",
    )
