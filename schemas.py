#请求参数模型
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入的问题")
    session_id: Optional[str] = Field(default=None, description="会话ID")

class RagAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    file_id: int = Field(..., description="上传文件ID")