from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.exceptions import AppError
from database import Base, engine
from routers import chat, history, rag, upload


# 创建当前项目中尚不存在的数据库表
#
# 注意：
# create_all() 只能创建新表，
# 不能自动修改已有表的字段。
Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="AI App Backend",
    description="AI 应用开发后端服务",
    version="1.1.0",
)


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
    项目首页接口。
    """

    return {
        "code": 200,
        "message": "success",
        "data": "欢迎使用 AI 应用后端服务",
    }


@app.get("/health")
def health_check():
    """
    基础健康检查。

    当前只检查 FastAPI 应用进程是否能够正常响应。
    后续可以增加数据库、向量库和模型配置检查。
    """

    return {
        "code": 200,
        "message": "success",
        "data": {
            "status": "healthy",
            "service": "ai-app-backend",
            "version": "1.1.0",
        },
    }


# ============================================================
# 注册业务路由
# ============================================================

app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(history.router)
app.include_router(rag.router)