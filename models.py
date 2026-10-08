from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Float, Text

from database import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(String, nullable=False)
    ai_answer = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    tool_name = Column(String, nullable=True)
    tool_call_count = Column(Integer, default=0)
    duration_ms = Column(Integer, default=0)
    citations_json = Column(Text, nullable=True)


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

class RagQueryLog(Base):
    """
    RAG 问答日志表。

    用来记录每一次 RAG 问答的检索情况和回答结果，
    方便后续分析 RAG 效果。
    """

    __tablename__ = "rag_query_logs"

    # 主键 ID
    id = Column(Integer, primary_key=True, index=True)

    # 上传文件 ID
    file_id = Column(Integer, nullable=False)

    # 文件名
    filename = Column(String, nullable=False)

    # 用户问题
    question = Column(String, nullable=False)

    # AI 回答
    answer = Column(String, nullable=False)

    # 最相似 chunk 的 distance
    best_distance = Column(Float, nullable=True)

    # 当前设置的最大 distance 阈值
    max_distance = Column(Float, nullable=True)

    # 检索状态：matched / not_matched
    retrieval_status = Column(String, nullable=False)

    # 创建时间
    created_at = Column(DateTime, default=datetime.now)