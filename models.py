from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, ForeignKey

from database import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(String, nullable=False)
    ai_answer = Column(String, nullable=False)
    # 按会话聚合查询（加载历史）是高频路径，建索引
    session_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
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
    # 文件列表按时间倒序展示，建索引
    created_at = Column(DateTime, default=datetime.now, index=True)


class RagQueryLog(Base):
    """
    RAG 问答日志表。

    用来记录每一次 RAG 问答的检索情况和回答结果，
    方便后续分析 RAG 效果。
    """

    __tablename__ = "rag_query_logs"

    # 主键 ID
    id = Column(Integer, primary_key=True, index=True)

    # 上传文件 ID（外键关联 uploaded_files，删除文件时连带清理日志）。
    # 全局检索（跨多文档）时为 None。
    file_id = Column(
        Integer,
        ForeignKey("uploaded_files.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # 文件名（全局检索时存 "全局" 占位）
    filename = Column(String, nullable=False)

    # 用户问题
    question = Column(String, nullable=False)

    # AI 回答
    answer = Column(String, nullable=False)

    # 最相似 chunk 的 distance（基础向量检索路径）
    best_distance = Column(Float, nullable=True)

    # 当前设置的最大 distance 阈值
    max_distance = Column(Float, nullable=True)

    # ---- Hybrid + Rerank 路径的检索信息 ----
    # top1 rerank 分（阈值迭代依据）
    best_rerank_score = Column(Float, nullable=True)

    # rerank 后全部 top 片段的分数（JSON 字符串）
    rerank_scores_json = Column(Text, nullable=True)

    # 实际使用的查询数（原始问题 + 改写查询）
    query_count = Column(Integer, nullable=True)

    # 端到端检索耗时（毫秒）
    retrieval_latency_ms = Column(Integer, nullable=True)

    # 检索模式：advanced_rag / overview_opening / basic_vector ...
    retrieval_mode = Column(String, nullable=True)

    # 检索状态：matched / not_matched
    retrieval_status = Column(String, nullable=False)

    # 创建时间
    created_at = Column(DateTime, default=datetime.now, index=True)
