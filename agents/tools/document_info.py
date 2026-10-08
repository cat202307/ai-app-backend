from typing import Any, Optional, cast

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.base_tool import BaseTool
from models import UploadedFile


class GetDocumentInfoInput(BaseModel):
    """
    get_document_info 工具的输入参数。

    支持两种查询方式：
    1. 按 document_id 查询
    2. 按 filename 查询
    """

    document_id: Optional[int] = Field(
        default=None,
        description="需要查询的文档数据库 ID（如果知道的话）",
    )

    filename: Optional[str] = Field(
        default=None,
        max_length=255,
        description="需要查询的文档文件名（如果知道文件名的话）",
    )


class GetDocumentInfoTool(BaseTool):
    """
    查询系统中文档的真实信息。

    支持按 document_id 或 filename 查询：
    - 知道 ID → 用 document_id
    - 知道文件名 → 用 filename
    """

    name = "get_document_info"

    description = (
        "查询系统中文档的真实信息，"
        "包括文档名称、文件类型、保存路径和上传时间。"
        "支持两种查询方式："
        "1. 按 document_id 查询（知道文档 ID 时）"
        "2. 按 filename 查询（知道文件名时）"
        "当用户询问某个文档的名称、格式、路径、"
        "上传时间、是否存在，或者用户提到了文件名时，"
        "应使用此工具。"
        "如果用户提到了文件名，请用 filename 参数查询，"
        "不要凭空猜测 document_id。"
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

        # 至少要传一个查询条件
        if (
            params.document_id is None
            and params.filename is None
        ):
            return {
                "success": False,
                "message": (
                    "必须提供 document_id 或 filename "
                    "其中一个查询条件。"
                ),
                "document": None,
            }

        # ----------------------------------------------------
        # 1. 按 document_id 查询
        # ----------------------------------------------------

        if params.document_id is not None:
            document = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.id == params.document_id
                )
                .first()
            )

            if document is None:
                return {
                    "success": False,
                    "document_id": params.document_id,
                    "message": "没有找到对应文档。",
                    "document": None,
                    "available_files": (
                        self._list_available_files(db)
                    ),
                }

            return self._format_result(document)

        # ----------------------------------------------------
        # 2. 按 filename 查询
        # ----------------------------------------------------

        if params.filename is not None:
            # 模糊匹配，不区分大小写
            document = (
                db.query(UploadedFile)
                .filter(
                    UploadedFile.filename.ilike(
                        f"%{params.filename}%"
                    )
                )
                .first()
            )

            if document is None:
                return {
                    "success": False,
                    "filename": params.filename,
                    "message": (
                        f"没有找到包含 "
                        f"'{params.filename}' 的文件。"
                    ),
                    "document": None,
                    "available_files": (
                        self._list_available_files(db)
                    ),
                }

            return self._format_result(document)

    def _list_available_files(
        self,
        db: Session,
    ) -> list[dict[str, Any]]:
        """
        返回当前知识库中所有可用文档，
        用于在用户查询的文档不存在时引导。
        """

        rows = (
            db.query(UploadedFile)
            .order_by(UploadedFile.id)
            .all()
        )
        return [
            {
                "id": row.id,
                "filename": getattr(
                    row, "filename", None
                ),
                "file_type": getattr(
                    row, "file_type", None
                ),
            }
            for row in rows
        ]

    def _format_result(
        self,
        document: UploadedFile,
    ) -> dict[str, Any]:
        """
        把 ORM 对象格式化成返回结果。
        """

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
