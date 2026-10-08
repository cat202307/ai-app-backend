from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy.orm import Session

from agents.base_tool import BaseTool
from models import UploadedFile


class ListAllDocumentsInput(BaseModel):
    """
    list_all_documents 工具的输入参数。
    不需要参数。
    """

    pass


class ListAllDocumentsTool(BaseTool):
    """
    列出系统中所有已上传的文档。
    """

    name = "list_all_documents"

    description = (
        "列出系统中所有已上传的文档，"
        "包括文档 ID、文件名、类型、上传时间。"
        "仅当用户明确询问系统里有什么文件、"
        "所有文件列表、文档总数时使用。"
        "不要把它当作回答文档内容类问题"
        "（询问技术点、具体内容、是什么、用了什么）"
        "的前置步骤：这类问题应直接调用"
        "ask_advanced_rag，无需先列出文档。"
    )

    input_model = ListAllDocumentsInput

    def execute(
        self,
        arguments: BaseModel,
        db: Session,
    ) -> dict[str, Any]:
        """
        查询所有 UploadedFile 记录。
        """

        documents = (
            db.query(UploadedFile)
            .order_by(UploadedFile.id)
            .all()
        )

        if not documents:
            return {
                "success": True,
                "message": "系统中还没有上传任何文档。",
                "total_count": 0,
                "documents": [],
            }

        document_list = []

        for doc in documents:
            created_at = getattr(
                doc,
                "created_at",
                None,
            )

            document_list.append({
                "id": doc.id,
                "filename": getattr(
                    doc,
                    "filename",
                    None,
                ),
                "file_type": getattr(
                    doc,
                    "file_type",
                    None,
                ),
                "filepath": getattr(
                    doc,
                    "filepath",
                    None,
                ),
                "created_at": (
                    created_at.isoformat()
                    if created_at is not None
                    else None
                ),
            })

        return {
            "success": True,
            "message": (
                f"系统中共有 {len(documents)} 个文档。"
            ),
            "total_count": len(documents),
            "documents": document_list,
        }
