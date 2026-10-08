import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from database import get_db
from models import ChatHistory


router = APIRouter(
    prefix="/history",
    tags=["history"]
)


@router.get("")
def get_history(
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    offset = (page - 1) * size

    records = (
        db.query(ChatHistory)
        .order_by(ChatHistory.created_at.desc())
        .offset(offset)
        .limit(size)
        .all()
    )

    data = []

    for item in records:
        data.append({
            "id": item.id,
            "user_message": item.user_message,
            "ai_answer": item.ai_answer,
            "session_id": item.session_id,
            "created_at": item.created_at,
            "tool_name": item.tool_name,
            "tool_call_count": item.tool_call_count,
            "duration_ms": item.duration_ms,
            "citations": (
                json.loads(item.citations_json)
                if item.citations_json else []
            ),
        })

    return {
        "code": 200,
        "message": "success",
        "data": data
    }


@router.delete("/{history_id}")
def delete_history(
    history_id: int,
    db: Session = Depends(get_db)
):
    record = (
        db.query(ChatHistory)
        .filter(ChatHistory.id == history_id)
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail="聊天记录不存在"
        )

    db.delete(record)
    db.commit()

    return {
        "code": 200,
        "message": "delete success",
        "data": {
            "id": history_id
        }
    }


@router.get("/session/{session_id}")
def get_history_by_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    # 如果是 default，同时兼容 session_id 为 null 和 'default' 的历史数据
    if session_id == 'default':
        records = (
            db.query(ChatHistory)
            .filter(
                or_(
                    ChatHistory.session_id == 'default',
                    ChatHistory.session_id.is_(None),
                )
            )
            .order_by(ChatHistory.created_at.asc())
            .all()
        )
    else:
        records = (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.created_at.asc())
            .all()
        )

    data = []

    for item in records:
        data.append({
            "id": item.id,
            "user_message": item.user_message,
            "ai_answer": item.ai_answer,
            "session_id": item.session_id,
            "created_at": item.created_at,
            "tool_name": item.tool_name,
            "tool_call_count": item.tool_call_count,
            "duration_ms": item.duration_ms,
            "citations": (
                json.loads(item.citations_json)
                if item.citations_json else []
            ),
        })

    return {
        "code": 200,
        "message": "success",
        "data": data
    }


@router.delete("/session/{session_id}")
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
):
    """
    删除指定会话的全部聊天记录。

    session_id 为 default 时，同时清理
    session_id='default' 和 NULL 的历史遗留数据。
    """

    if session_id == 'default':
        query = db.query(ChatHistory).filter(
            or_(
                ChatHistory.session_id == 'default',
                ChatHistory.session_id.is_(None),
            )
        )
    else:
        query = db.query(ChatHistory).filter(
            ChatHistory.session_id == session_id
        )

    deleted = query.delete(synchronize_session=False)
    db.commit()

    return {
        "code": 200,
        "message": "deleted",
        "data": {
            "session_id": session_id,
            "deleted": deleted,
        },
    }
