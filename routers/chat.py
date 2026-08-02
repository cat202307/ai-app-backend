import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from database import get_db
from models import ChatHistory
from schemas import ChatRequest
from services.llm_service import call_llm, stream_llm


router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)


def build_chat_messages(req: ChatRequest, db: Session) -> list[dict[str, str]]:
    """
    构造大模型 messages。

    作用：
        普通聊天接口和流式聊天接口都可以复用这段逻辑。

    流程：
        1. 添加 system 角色设定
        2. 根据 session_id 查询历史记录
        3. 拼接历史对话
        4. 添加当前用户问题
    """

    messages = [
        {
            "role": "system",
            "content": "你是一个 AI 应用开发助手，回答要清晰、准确、适合初学者理解。"
        }
    ]

    if req.session_id:
        history_records = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == req.session_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(5)
            .all()
        )

        # 数据库查出来是最新在前
        # 发送给大模型时要改成旧记录在前
        history_records = list(reversed(history_records))

        for item in history_records:
            messages.append({
                "role": "user",
                "content": item.user_message
            })

            messages.append({
                "role": "assistant",
                "content": item.ai_answer
            })

    messages.append({
        "role": "user",
        "content": req.message
    })

    return messages


def format_sse(data: dict) -> str:
    """
    把 Python 字典转换成 SSE 格式。

    SSE 格式要求：
        data: xxx\n\n

    ensure_ascii=False：
        保证中文不会被转成 Unicode 编码。
    """

    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """
    普通大模型聊天接口。

    特点：
        一次性返回完整回答。
    """

    messages = build_chat_messages(req, db)

    answer = call_llm(messages)

    history = ChatHistory(
        user_message=req.message,
        ai_answer=answer,
        session_id=req.session_id
    )

    db.add(history)
    db.commit()
    db.refresh(history)

    return {
        "code": 200,
        "message": "success",
        "data": {
            "id": history.id,
            "question": history.user_message,
            "answer": history.ai_answer,
            "session_id": history.session_id,
            "created_at": history.created_at
        }
    }


@router.post("/stream")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """
    流式大模型聊天接口。

    特点：
        大模型生成一点，接口返回一点。

    返回格式：
        使用 Server-Sent Events，简称 SSE。

    每次返回一段 JSON：
        type=meta   表示开始信息
        type=delta  表示模型生成的一小段内容
        type=done   表示完成
        type=error  表示异常
    """

    messages = build_chat_messages(req, db)

    def generate():
        # 用于收集完整回答
        # 因为流式输出时是一段一段返回，
        # 最后保存聊天记录时需要完整 answer
        answer_parts = []

        try:
            # 先返回一个 meta 事件，告诉客户端开始生成
            yield format_sse({
                "type": "meta",
                "message": "stream started",
                "session_id": req.session_id
            })

            # 调用流式大模型
            for content in stream_llm(messages):
                answer_parts.append(content)

                # 每生成一段，就返回一段
                yield format_sse({
                    "type": "delta",
                    "content": content
                })

            # 拼接完整回答
            full_answer = "".join(answer_parts)

            # 保存聊天记录
            history = ChatHistory(
                user_message=req.message,
                ai_answer=full_answer,
                session_id=req.session_id
            )

            db.add(history)
            db.commit()
            db.refresh(history)

            # 返回完成事件
            yield format_sse({
                "type": "done",
                "message": "stream finished",
                "history_id": history.id
            })

        except Exception as e:
            db.rollback()

            yield format_sse({
                "type": "error",
                "message": str(e)
            })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )