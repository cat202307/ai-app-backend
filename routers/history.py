from fastapi import APIRouter, Depends, HTTPException
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
            "created_at": item.created_at
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
            "created_at": item.created_at
        })

    return {
        "code": 200,
        "message": "success",
        "data": data
    }