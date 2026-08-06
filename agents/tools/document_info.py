from typing import Any, cast

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.base_tool import BaseTool
from models import UploadedFile


class GetDocumentInfoInput(BaseModel):
    """
    get_document_info 工具的输入参数。
    """

    document_id: int = Field(
        ...,
        gt=0,
        description="需要查询的文档数据库 ID",
    )


class GetDocumentInfoTool(BaseTool):
    """
    根据文档 ID 查询 UploadedFile 表中的真实文档信息。
    """

    name = "get_document_info"

    description = (
        "根据文档ID查询系统中真实存在的文档信息，"
        "包括文档名称、文件类型、保存路径和上传时间。"
        "当用户询问某个文档的名称、格式、路径、"
        "上传时间或是否存在时，应调用此工具。"
        "不要根据常识猜测文档信息。"
    )

    input_model = GetDocumentInfoInput

    def execute(
        self,
        arguments: BaseModel,
        db: Session,
    ) -> dict[str, Any]:
        """
        查询 SQLite 数据库中的 UploadedFile。
        """

        params = cast(
            GetDocumentInfoInput,
            arguments,
        )

        document = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.id == params.document_id
            )
            .first()
        )

        if document is None:
            # 文档不存在是正常业务结果，
            # 不一定属于系统异常。
            return {
                "success": False,
                "document_id": params.document_id,
                "message": "没有找到对应文档。",
                "document": None,
            }

        created_at = getattr(
            document,
            "created_at",
            None,
        )

        return {
            "success": True,
            "document_id": document.id,
            "message": "文档查询成功。",
            "document": {
                "id": document.id,
                "filename": getattr(
                    document,
                    "filename",
                    None,
                ),
                "file_type": getattr(
                    document,
                    "file_type",
                    None,
                ),
                "filepath": getattr(
                    document,
                    "filepath",
                    None,
                ),
                "created_at": (
                    created_at.isoformat()
                    if created_at is not None
                    else None
                ),
            },
        }