import os
import re
from typing import Any, cast

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.base_tool import BaseTool
from core.config import settings
from models import UploadedFile
from services.advanced_rag_answer_service import (
    _DOCUMENT_WORDS,
    is_overview_question,
    retrieve_with_advanced_rag,
)


def _explicit_document_id(
    question: str,
    documents: list[UploadedFile],
) -> int | None:
    """
    判断问题是否明确点名某个具体文档。

    依据：问题中出现「文档N / 文件N / file N」，
    或出现某个文件名的关键词（如 test02、胡家浩）。
    命中返回该文档 ID；否则返回 None（视为全局概述）。
    """

    lowered = question.lower()

    number_match = re.search(
        r"(?:文档|文件|file)\s*([0-9]+)", lowered
    )
    if number_match:
        number = int(number_match.group(1))
        for document in documents:
            if document.id == number:
                return document.id

    for document in documents:
        stem = os.path.splitext(document.filename)[0].lower()

        if len(stem) >= 4 and stem in lowered:
            return document.id

        tokens = re.split(
            r"[\s\-_·.,，。、()（）]+", stem
        )
        for token in tokens:
            if len(token) >= 3 and token in lowered:
                return document.id

    return None


def _is_document_level_ask(
    question: str,
    document: UploadedFile,
) -> bool:
    """
    判断是否为「文件名 + 文档级概述短语」结构，如
    「test02.txt主要是什么？」。

    移除文件名后，剩余部分应是纯中文概述短语
    （主要是什么 / 内容 / 讲了什么）；若还含其他
    拉丁字母概念词（如「test02.txt里RRF是什么」），
    则不属于文档级概述。
    """

    stem = os.path.splitext(document.filename)[0]
    remainder = question
    for part in (document.filename, stem):
        if part:
            remainder = remainder.replace(part, "")

    cleaned = re.sub(
        r"[\s\W_？?。.,，、]+", "", remainder
    )

    if re.search(r"[A-Za-z0-9]", cleaned):
        return False

    markers = (
        "主要是什么", "主要是啥", "主要内容", "讲了什么",
        "讲什么", "概述", "摘要", "简介", "内容",
        "是什么", "讲的是", "主要讲", "讲的是什么",
    )
    return any(marker in cleaned for marker in markers)


class AskAdvancedRagInput(BaseModel):
    """
    ask_advanced_rag 工具的输入参数。
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="用户的问题，用于在知识库中检索相关原文片段",
    )

    file_id: int | None = Field(
        None,
        gt=0,
        description=(
            "需要查询的文档数据库 ID；"
            "不指定（留空）时在全部文档中检索"
        ),
    )


class AskAdvancedRagTool(BaseTool):
    """
    高级 RAG 检索工具：只负责检索，不负责写答案。

    - 指定 file_id：仅在该文档内检索
    - 不指定 file_id：跨全部文档检索（用户未点名文档的通用问题）

    流程：
        Query Rewrite → Hybrid Search → RRF
        → Rerank → Threshold → 返回带来源编号的原文片段

    Agent 拿到片段后，由最终流式 LLM 一次性合成带 [n] 引用的回答，
    避免工具内与 Agent 重复生成。
    """

    name = "ask_advanced_rag"

    description = (
        "在知识库中高级检索与问题相关的原文片段。"
        "工具会完成查询改写、混合检索、重排序和阈值过滤，"
        "返回带来源编号（citation_id）的片段 citations，"
        "但不会替你写最终回答：你必须基于这些片段亲自组织回答。"
        "如果用户明确点名了某个文档，传入该文档的 file_id；"
        "如果用户没有点名具体文档（通用问题），"
        "file_id 留空，系统会自动检索全部文档。"
        "凡是需要基于知识库内容回答的问题，"
        "包括解释技术点、用了什么方法/算法/模型/参数、"
        "如何实现、某部分内容是什么，都应使用此工具。"
    )

    input_model = AskAdvancedRagInput

    def execute(
        self,
        arguments: BaseModel,
        db: Session,
    ) -> dict[str, Any]:
        """
        执行 Advanced RAG 检索（单文档 / 全局），返回原文片段。
        """

        params = cast(AskAdvancedRagInput, arguments)

        # ----------------------------------------------------
        # 0. 概述类问题：未明确点名某个文档时按全局概述处理，
        # 跨所有文档逐篇概述，避免「该论文」被误限定到第一个文档
        # ----------------------------------------------------

        all_documents = db.query(UploadedFile).all()
        explicit_id = _explicit_document_id(
            params.question, all_documents
        )
        force_overview = False

        if is_overview_question(params.question):
            if explicit_id is not None:
                # 明确点名某文档的概述：限定该文档
                params.file_id = explicit_id
            elif any(
                word in params.question for word in _DOCUMENT_WORDS
            ):
                # 文档指代 + 概述词：全局概述
                params.file_id = None
        elif explicit_id is not None:
            named_document = next(
                doc for doc in all_documents
                if doc.id == explicit_id
            )
            if _is_document_level_ask(
                params.question, named_document
            ):
                # 「文件名 + 主要是什么」未被文本启发识别，
                # 强制对该文档走开头 Chunk 概述
                params.file_id = explicit_id
                force_overview = True

        # ----------------------------------------------------
        # 1. 指定文档时校验文件是否存在
        # ----------------------------------------------------

        document = None

        if params.file_id is not None:
            document = (
                db.query(UploadedFile)
                .filter(UploadedFile.id == params.file_id)
                .first()
            )

            if document is None:
                return {
                    "success": False,
                    "retrieval_status": "not_matched",
                    "file_id": params.file_id,
                    "citations": [],
                    "message": "指定文档不存在。",
                    "note": (
                        "请如实告知用户指定的文档不存在，"
                        "不要编造内容。"
                    ),
                }

        filename = (
            document.filename if document else None
        )

        # ----------------------------------------------------
        # 2. 调用 Advanced RAG 检索（不生成答案）
        # ----------------------------------------------------

        try:
            retrieval = retrieve_with_advanced_rag(
                question=params.question,
                file_id=params.file_id,
                filename=filename,
                rewrite_count=settings.rag_query_rewrite_count,
                force_overview=force_overview,
            )
        except Exception as exc:
            return {
                "success": False,
                "retrieval_status": "error",
                "file_id": params.file_id,
                "citations": [],
                "message": (
                    "Advanced RAG 检索失败："
                    f"{type(exc).__name__}: {exc}"
                ),
                "note": (
                    "请告知用户检索过程出错，"
                    "可建议稍后重试。"
                ),
            }

        # ----------------------------------------------------
        # 3. 未检索到足够相关的片段
        # ----------------------------------------------------

        if retrieval["retrieval_status"] == "not_matched":
            return {
                "success": False,
                "retrieval_status": "not_matched",
                "file_id": params.file_id,
                "filename": filename,
                "question": params.question,
                "citations": [],
                "best_rerank_score": retrieval.get(
                    "best_rerank_score"
                ),
                "rerank_threshold": retrieval.get(
                    "rerank_threshold"
                ),
                "message": retrieval.get("message"),
                "note": (
                    "没有检索到足够相关的原文片段。"
                    "请如实告知用户知识库中缺少相关信息，"
                    "不要编造。"
                ),
            }

        # ----------------------------------------------------
        # 4. 命中：返回原文片段，最终回答由 Agent 流式合成
        # ----------------------------------------------------

        global_overview = (
            params.file_id is None
            and is_overview_question(params.question)
            and len(all_documents) > 1
        )

        if global_overview:
            matched_note = (
                f"知识库中共有 {len(all_documents)} 个文档，"
                "citations 已包含每个文档开头的原文片段。"
                "请逐个文档分别概述其主要内容"
                "（例如：文档《文件名A》主要讲……；"
                "文档《文件名B》主要讲……），不要只概述其中某一个；"
                "在引用处标注 [编号]，片段之外的信息不要编造，"
                "也不要再调用检索工具。"
            )
        else:
            matched_note = (
                "citations 中每条 content 是检索到的原文片段，"
                "citation_id 是引用编号。请直接基于这些片段，"
                "用清晰简洁的中文组织最终回答（会被流式输出），"
                "在引用结论处标注 [编号]；片段之外的信息不要编造，"
                "也不要再调用检索工具。"
            )

        return {
            "success": True,
            "retrieval_status": "matched",
            "file_id": params.file_id,
            "filename": filename,
            "question": params.question,
            "best_rerank_score": retrieval.get(
                "best_rerank_score"
            ),
            "citations": retrieval["citations"],
            "note": matched_note,
        }
