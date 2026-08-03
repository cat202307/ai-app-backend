from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    普通聊天和流式聊天请求模型。
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="用户输入的问题",
    )

    session_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="会话 ID；相同 ID 用于保持多轮对话",
    )


class ChatDetailRequest(ChatRequest):
    """
    详细聊天接口请求模型。

    在普通 ChatRequest 基础上增加 temperature，
    用于测试不同生成随机性。
    """

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description=(
            "模型生成随机性。"
            "数值越低回答通常越稳定，"
            "数值越高回答通常越多样。"
        ),
    )


class RagAskRequest(BaseModel):
    """
    RAG 知识库问答请求模型。
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="用户问题",
    )

    file_id: int = Field(
        ...,
        gt=0,
        description="上传文件 ID",
    )