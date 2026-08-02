from fastapi import FastAPI

from database import Base, engine
from routers import chat, upload, history, rag

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="AI App Backend",
    description="AI 应用开发后端服务",
    version="1.0.0"
)


@app.get("/")
def index():
    return {
        "code": 200,
        "message": "success",
        "data": "欢迎使用 AI 应用后端服务"
    }


@app.get("/health")
def health_check():
    return {
        "code": 200,
        "message": "success",
        "data": "service is running"
    }


app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(history.router)
app.include_router(rag.router)