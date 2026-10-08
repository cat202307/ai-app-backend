from typing import Any, cast

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from core.config import settings
from agents.base_tool import BaseTool
from models import UploadedFile
from services.vector_service import (
    search_similar_chunks,
)


class SearchKnowledgeBaseInput(BaseModel):
    """
    知识库检索工具输入参数。
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="需要在文档知识库中检索的问题或关键词",
    )

    file_id: int = Field(
        ...,
        gt=0,
        description="需要检索的文档数据库ID",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="返回最相关的文本片段数量",
    )


class SearchKnowledgeBaseTool(BaseTool):
    """
    使用现有 ChromaDB 向量检索能力搜索文档内容。
    """

    name = "search_knowledge_base"

    description = (
        "根据用户问题，在指定文档的向量知识库中搜索"
        "最相关的文本片段。"
        "当用户询问某个文档里面写了什么、"
        "是否提到某项内容、文档具体内容、"
        "技术经历或知识点时，应使用此工具。"
        "这个工具用于查询文档内容，"
        "不是查询文件名称或上传时间。"
    )

    input_model = SearchKnowledgeBaseInput

    def execute(
        self,
        arguments: BaseModel,
        db: Session,
    ) -> dict[str, Any]:

        params = cast(
            SearchKnowledgeBaseInput,
            arguments,
        )

        # ----------------------------------------------------
        # 1. 检查文件是否存在
        # ----------------------------------------------------

        document = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.id == params.file_id
            )
            .first()
        )

        if document is None:
            return {
                "success": False,
                "message": "指定文档不存在。",
                "file_id": params.file_id,
                "results": [],
            }

        # ----------------------------------------------------
        # 2. 调用你已有的向量检索函数
        # ----------------------------------------------------

        results = search_similar_chunks(
            question=params.query,
            file_id=params.file_id,
            top_k=params.top_k,
        )

        # ============================================================
        # 判断检索结果是否真正相关
        # ============================================================

        if not results:
            return {
                "success": False,
                "message": (
                    "没有检索到相关内容，"
                    "请确认文档已经建立向量索引。"
                ),
                "file_id": params.file_id,
                "filename": document.filename,
                "query": params.query,
                "retrieval_status": "not_matched",
                "results": [],
            }

        # 最相似结果
        best_distance = results[0].get(
            "distance"
        )

        # 如果最好的结果都超过距离阈值，
        # 则认为知识库没有可靠答案
        if (
                best_distance is not None
                and best_distance
                > settings.rag_max_distance
        ):
            return {
                "success": False,
                "message": (
                    "知识库中没有找到足够相关的内容。"
                ),
                "file_id": params.file_id,
                "filename": document.filename,
                "query": params.query,
                "retrieval_status": "not_matched",
                "best_distance": best_distance,
                "max_distance": (
                    settings.rag_max_distance
                ),
                "results": [],
            }

        # ----------------------------------------------------
        # 3. 整理安全返回结果
        # ----------------------------------------------------

        safe_results = []

        for index, item in enumerate(results):
            safe_results.append({
                "rank": index + 1,
                "chunk_id": item.get("chunk_id"),
                "content": item.get("content", "")[:500],
                "distance": item.get("distance"),
            })

        return {
            "success": True,
            "message": "知识库检索成功。",
            "file_id": params.file_id,
            "filename": document.filename,
            "query": params.query,

            "retrieval_status": "matched",

            "best_distance": (
                safe_results[0]["distance"]
                if safe_results
                else None
            ),

            "max_distance": (
                settings.rag_max_distance
            ),

            "results": safe_results,
        }