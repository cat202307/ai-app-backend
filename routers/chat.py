import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from core.exceptions import AppError, LLMServiceError
from database import SessionLocal, get_db
from models import ChatHistory
from schemas import ChatDetailRequest, ChatRequest
from services.llm_service import (
    call_llm,
    call_llm_detail,
    stream_llm,
)
from services.prompt_service import build_chat_system_message


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


# ============================================================
# 上下文构建
# ============================================================

def build_chat_messages(
    req: ChatRequest,
    db: Session,
) -> list[dict[str, str]]:
    """
    构造发送给大模型的 messages。

    流程：
    1. 添加统一的 System Message
    2. 根据 session_id 查询最近聊天记录
    3. 按照旧到新的顺序拼接历史对话
    4. 添加用户当前问题

    当前策略：
        最多读取最近 5 条 ChatHistory。

    一条 ChatHistory 包含：
        - 一条 user 消息
        - 一条 assistant 消息

    所以最多会向模型加入 10 条历史消息。
    """

    messages: list[dict[str, str]] = [
        build_chat_system_message()
    ]

    if req.session_id:
        history_records = (
            db.query(ChatHistory)
            .filter(
                ChatHistory.session_id == req.session_id
            )
            .order_by(
                ChatHistory.created_at.desc()
            )
            .limit(5)
            .all()
        )

        # 数据库查询结果是最新记录在前。
        # 模型上下文需要按照旧记录到新记录排列。
        history_records = list(
            reversed(history_records)
        )

        for item in history_records:
            messages.append({
                "role": "user",
                "content": item.user_message,
            })

            messages.append({
                "role": "assistant",
                "content": item.ai_answer,
            })

    messages.append({
        "role": "user",
        "content": req.message,
    })

    return messages


# ============================================================
# 数据库保存
# ============================================================

def save_chat_history(
    db: Session,
    user_message: str,
    ai_answer: str,
    session_id: str | None,
) -> ChatHistory:
    """
    保存一条完整聊天记录。

    只有模型成功返回有效回答后才调用本函数。

    如果数据库提交失败：
    - 执行 rollback
    - 继续抛出异常
    """

    history = ChatHistory(
        user_message=user_message,
        ai_answer=ai_answer,
        session_id=session_id,
    )

    try:
        db.add(history)
        db.commit()
        db.refresh(history)

        return history

    except Exception:
        db.rollback()
        raise


# ============================================================
# SSE 格式化
# ============================================================

def format_sse(data: dict[str, Any]) -> str:
    """
    把 Python 字典转换成 SSE 格式。

    SSE 每条消息格式：

        data: JSON字符串\\n\\n

    ensure_ascii=False：
        保证中文不会转换为 Unicode 转义字符。
    """

    json_data = json.dumps(
        data,
        ensure_ascii=False,
        default=str,
    )

    return f"data: {json_data}\n\n"


# ============================================================
# 普通聊天接口
# ============================================================

@router.post("")
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    普通大模型聊天接口。

    返回：
        只返回模型正文和聊天记录信息。

    兼容原来的接口结构。
    """

    messages = build_chat_messages(
        req=req,
        db=db,
    )

    # 如果模型调用失败，call_llm() 会抛出 AppError。
    # 代码不会继续执行，因此错误不会写入数据库。
    answer = call_llm(messages)

    history = save_chat_history(
        db=db,
        user_message=req.message,
        ai_answer=answer,
        session_id=req.session_id,
    )

    return {
        "code": 200,
        "message": "success",
        "data": {
            "id": history.id,
            "question": history.user_message,
            "answer": history.ai_answer,
            "session_id": history.session_id,
            "created_at": history.created_at,
        },
    }


# ============================================================
# 详细聊天接口
# ============================================================

@router.post("/detail")
def chat_detail(
    req: ChatDetailRequest,
    db: Session = Depends(get_db),
):
    """
    详细大模型聊天接口。

    与普通 /chat 的区别：

    /chat：
        主要供普通业务使用，只返回回答正文。

    /chat/detail：
        用于学习、调试和性能分析，
        额外返回模型、Token、停止原因和调用耗时。
    """

    messages = build_chat_messages(
        req=req,
        db=db,
    )

    result = call_llm_detail(
        messages=messages,
        temperature=req.temperature,
    )

    history = save_chat_history(
        db=db,
        user_message=req.message,
        ai_answer=result.content,
        session_id=req.session_id,
    )

    return {
        "code": 200,
        "message": "success",
        "data": {
            "id": history.id,
            "question": history.user_message,
            "answer": result.content,
            "session_id": history.session_id,
            "created_at": history.created_at,

            # 模型调用信息
            "model": result.model,
            "finish_reason": result.finish_reason,
            "usage": {
                "prompt_tokens": (
                    result.usage.prompt_tokens
                ),
                "completion_tokens": (
                    result.usage.completion_tokens
                ),
                "total_tokens": (
                    result.usage.total_tokens
                ),
            },
            "latency_ms": result.latency_ms,
            "temperature": req.temperature,
        },
    }


# ============================================================
# 流式聊天接口
# ============================================================

@router.post("/stream")
def chat_stream(
    req: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    SSE 流式聊天接口。

    事件类型：

    meta:
        告诉客户端流式响应已经开始。

    delta:
        模型生成的一小段文本。

    done:
        模型生成和聊天记录保存已经完成。

    error:
        模型或数据库发生异常。

    注意：
        StreamingResponse 一旦开始返回内容，
        HTTP 状态通常已经是 200。

        因此流式过程中的错误不能再改为 4xx/5xx，
        需要通过 type=error 的 SSE 事件通知客户端。
    """

    # 在开始流式响应前查询历史消息，
    # 避免在生成过程中反复使用请求 Session。
    messages = build_chat_messages(
        req=req,
        db=db,
    )

    def generate():
        answer_parts: list[str] = []

        try:
            yield format_sse({
                "type": "meta",
                "message": "stream started",
                "session_id": req.session_id,
            })

            for content in stream_llm(messages):
                answer_parts.append(content)

                yield format_sse({
                    "type": "delta",
                    "content": content,
                })

            full_answer = "".join(
                answer_parts
            ).strip()

            if not full_answer:
                raise LLMServiceError(
                    message="大模型没有返回有效的流式内容。"
                )

            # StreamingResponse 生命周期可能较长。
            # 这里单独创建数据库 Session 保存记录，
            # 不继续占用请求依赖中的 db。
            stream_db = SessionLocal()

            try:
                history = save_chat_history(
                    db=stream_db,
                    user_message=req.message,
                    ai_answer=full_answer,
                    session_id=req.session_id,
                )

                history_id = history.id

            finally:
                stream_db.close()

            yield format_sse({
                "type": "done",
                "message": "stream finished",
                "history_id": history_id,
            })

        except AppError as exc:
            # 大模型业务异常。
            # 不保存失败或不完整的回答。
            yield format_sse({
                "type": "error",
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            })

        except Exception:
            # 未知异常不直接将堆栈或内部路径返回给客户端。
            yield format_sse({
                "type": "error",
                "error": {
                    "code": "STREAM_INTERNAL_ERROR",
                    "message": "流式聊天处理失败。",
                },
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )