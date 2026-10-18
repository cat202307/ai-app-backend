import functools
import os
import re
import time
from typing import Any

from core.config import settings
from core.metrics import metrics


def _record_retrieval_run(mode: str):
    """
    装饰器：统一记录一次生产检索运行的命中状态、查询数、
    耗时与 top1 rerank 分，避免在每个 return 处散落埋点。

    被装饰函数返回的结果字典里通常包含：
    retrieval_status / best_rerank_score / queries /
    reranked_results / retrieval_mode。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            result = func(*args, **kwargs)
            try:
                if isinstance(result, dict):
                    reranked = result.get("reranked_results") or []
                    best_score = result.get("best_rerank_score")
                    if best_score is None and reranked:
                        best_score = reranked[0].get(
                            "rerank_score"
                        )
                    queries = result.get("queries") or []
                    metrics.record_retrieval_run(
                        mode=result.get("retrieval_mode", mode),
                        status=result.get(
                            "retrieval_status", "unknown"
                        ),
                        query_count=len(queries),
                        result_count=len(reranked),
                        best_rerank_score=best_score,
                        latency_ms=(
                            time.perf_counter() - started
                        )
                        * 1000.0,
                    )
            except Exception:
                # 指标埋点失败不影响主流程
                pass
            return result

        return wrapper

    return decorator

from services.advanced_rag_service import (
    hybrid_rerank_retrieve,
    _should_skip_rewrite,
)

from services.vector_service import (
    get_opening_chunks,
    get_parent_content,
)

from services.citation_service import (
    build_citations,
    build_citation_context,
    filter_used_citations,
)

from services.llm_service import (
    call_llm,
)

from services.prompt_service import (
    build_citation_rag_messages,
)


@_record_retrieval_run(mode="advanced_rag")
def answer_with_advanced_rag(
    question: str,
    file_id: int | None = None,
    filename: str | None = None,
    rewrite_count: int = 0,
    per_retriever_top_k: int | None = None,
    fusion_top_k: int | None = None,
    final_top_k: int | None = None,
) -> dict[str, Any]:
    """
    完整 Advanced RAG 问答流程。

    流程：

        Question
        用户问题

            ↓

        Hybrid Retrieval
        混合检索

            ↓

        RRF Fusion
        排名融合

            ↓

        Rerank
        重排序

            ↓

        Threshold Filter
        阈值过滤

            ↓

        Citation
        引用来源

            ↓

        LLM Answer
        最终回答
    """

    # 检索参数统一来自配置（生产/评测共用同一画像），
    # 显式传入时以传入值为准。
    per_retriever_top_k = (
        per_retriever_top_k
        if per_retriever_top_k is not None
        else settings.rag_per_retriever_top_k
    )
    fusion_top_k = (
        fusion_top_k
        if fusion_top_k is not None
        else settings.rag_fusion_top_k
    )
    final_top_k = (
        final_top_k
        if final_top_k is not None
        else settings.rag_final_top_k
    )

    # ============================================================
    # 1. Hybrid + Rerank
    # ============================================================

    retrieval_result = (
        hybrid_rerank_retrieve(
            question=question,
            file_id=file_id,
            rewrite_count=rewrite_count,
            per_retriever_top_k=(
                per_retriever_top_k
            ),
            fusion_top_k=fusion_top_k,
            final_top_k=final_top_k,
        )
    )

    reranked_results = retrieval_result.get(
        "reranked_results",
        [],
    )

    # ============================================================
    # 2. Rerank Threshold
    #    重排序阈值过滤
    # ============================================================

    trusted_results = []

    for item in reranked_results:

        rerank_score = item.get(
            "rerank_score"
        )

        # 没有分数，不作为可靠结果
        if rerank_score is None:
            continue

        # 分数不足
        if (
            rerank_score
            < settings.rag_rerank_min_score
        ):
            continue

        trusted_results.append(
            item
        )

    # ============================================================
    # 3. 没有可信 Chunk
    # ============================================================

    if not trusted_results:

        best_rerank_score = None

        if reranked_results:

            best_rerank_score = (
                reranked_results[0].get(
                    "rerank_score"
                )
            )

        return {
            "answer": (
                "文档中没有找到足够的信息"
                "回答该问题。"
            ),

            "retrieval_status": (
                "not_matched"
            ),

            "queries": retrieval_result.get(
                "queries",
                [],
            ),

            "rerank_threshold": (
                settings.rag_rerank_min_score
            ),

            "best_rerank_score": (
                best_rerank_score
            ),

            "citations": [],

            # 调试阶段保留
            "reranked_results": (
                reranked_results
            ),
        }

    # ============================================================
    # 4. 命中子块提升为父块，构造 Citation
    # ============================================================

    parent_results = promote_children_to_parents(
        trusted_results
    )

    citations = build_citations(
        reranked_results=parent_results,
        file_id=file_id,
        filename=filename,
    )

    # ============================================================
    # 5. 构造 Citation Context
    # ============================================================

    context = build_citation_context(
        citations
    )

    # ============================================================
    # 6. 构造最终 RAG Prompt
    # ============================================================

    messages = (
        build_citation_rag_messages(
            question=question,
            context=context,
        )
    )

    # ============================================================
    # 7. 最终 LLM Answer
    # ============================================================

    answer = call_llm(
        messages=messages,
        temperature=0.0,
    )

    # ============================================================
    # 8. Citation Pruning
    #
    # 只返回最终 Answer 真正使用的 Citation。
    # ============================================================

    used_citations = filter_used_citations(
        answer=answer,
        citations=citations,
    )

    best_rerank_score = (
        parent_results[0].get(
            "rerank_score"
        )
    )

    return {
        "answer": answer,

        "retrieval_status": "matched",

        "queries": retrieval_result.get(
            "queries",
            [],
        ),

        "rerank_threshold": (
            settings.rag_rerank_min_score
        ),

        "best_rerank_score": (
            best_rerank_score
        ),

        "citations": used_citations,

        # 调试阶段保留
        "reranked_results": (
            reranked_results
        ),
    }


_DOCUMENT_WORDS = (
    "论文", "文档", "文章", "这篇", "该", "本篇", "文件",
)

_OVERVIEW_KEYWORDS = (
    "主要内容", "讲了什么", "讲的什么", "讲什么", "概述",
    "概括", "摘要", "简介", "内容概要", "大致内容", "总体内容",
    "研究内容", "核心内容", "写了什么", "主要讲", "讲的是",
    "是关于什么", "内容是什么", "介绍一下这", "总结一下这",
    "主要是什么", "主要是啥", "讲的是什么",
)

# 全局概述（跨所有文档）时，每个文档取多少个开头父块。
# 大文档（如论文）若取 final_top_k=8 块，会在上下文中占绝对
# 主导，LLM 容易只概述大文档、遗漏小文档；限制每文档块数可
# 平衡上下文，配合系统 Prompt 的多文档全覆盖规则。
# 可用环境变量 GLOBAL_OVERVIEW_OPENING_CHUNKS 调整。
GLOBAL_OVERVIEW_OPENING_CHUNKS = int(
    os.getenv("GLOBAL_OVERVIEW_OPENING_CHUNKS", "3")
)


def is_overview_question(question: str) -> bool:
    """
    判断是否为针对整篇文档的概述类问题。

    这类问题（“该论文主要内容是？”“这篇文档讲了什么？”）
    不指向某个具体 Chunk，Cross-Encoder 给分接近基线，
    需要走文档开头 Chunk，而不是 rerank 阈值门控。
    """

    text = (question or "").strip()

    has_document_word = any(
        word in text for word in _DOCUMENT_WORDS
    )
    has_overview_keyword = any(
        keyword in text for keyword in _OVERVIEW_KEYWORDS
    )

    # 同时含文档指代与概述词：整篇概述
    if has_document_word and has_overview_keyword:
        return True

    # 极短的纯概述提问（如“主要内容？”“讲了什么”）；
    # 含拉丁字母 / 数字的多为具体概念追问（如“AdamW是什么”），不在此列
    if (
        has_overview_keyword
        and len(text) <= 12
        and not re.search(r"[A-Za-z0-9]", text)
    ):
        return True

    return False


def promote_children_to_parents(
    trusted_children: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    把通过阈值的命中子块提升为父块（Small-to-Big）。

    - 多个子块属于同一父块时父块去重，保留其中最高 rerank 分；
    - 父块内容通过 get_parent_content 实时聚合，提供完整上下文；
    - 结果按父块最高分降序排列。
    """

    parents_map: dict[str, dict[str, Any]] = {}

    for child in trusted_children:
        parent_id = child.get("parent_id")

        # 兜底：缺少 parent_id（异常数据），直接以子块作为一个单元
        if not parent_id:
            key = child.get("chunk_id") or str(id(child))
            parents_map[key] = {
                "chunk_id": child.get("chunk_id"),
                "content": child.get("content", ""),
                "file_id": child.get("file_id"),
                "filename": child.get("filename"),
                "rerank_score": child.get("rerank_score"),
            }
            continue

        score = child.get("rerank_score")

        if parent_id not in parents_map:
            parents_map[parent_id] = {
                "chunk_id": parent_id,
                "content": get_parent_content(parent_id),
                "file_id": child.get("file_id"),
                "filename": child.get("filename"),
                "rerank_score": score,
            }
        elif score is not None and (
            parents_map[parent_id]["rerank_score"] is None
            or score > parents_map[parent_id]["rerank_score"]
        ):
            parents_map[parent_id]["rerank_score"] = score

    parent_results = list(parents_map.values())
    parent_results.sort(
        key=lambda item: item["rerank_score"] or -1,
        reverse=True,
    )
    return parent_results


@_record_retrieval_run(mode="agent_retrieval")
def retrieve_with_advanced_rag(
    question: str,
    file_id: int | None = None,
    filename: str | None = None,
    rewrite_count: int = 0,
    per_retriever_top_k: int | None = None,
    fusion_top_k: int | None = None,
    final_top_k: int | None = None,
    force_overview: bool = False,
) -> dict[str, Any]:
    """
    只做 Advanced RAG 的「检索部分」，不调用 LLM 生成答案。

    供 Agent 使用：工具返回相关原文片段（citations，含 content），
    最终回答交给 Agent 的流式 LLM 一次性合成，避免重复生成。

    返回：
        matched:
            retrieval_status="matched"，citations（含 content / 编号）、
            context（拼好的文本）、best_rerank_score
        not_matched:
            retrieval_status="not_matched"，citations=[]、
            best_rerank_score、rerank_threshold、message
    """

    # 检索参数统一来自配置（生产/评测共用同一画像），
    # 显式传入时以传入值为准。
    per_retriever_top_k = (
        per_retriever_top_k
        if per_retriever_top_k is not None
        else settings.rag_per_retriever_top_k
    )
    fusion_top_k = (
        fusion_top_k
        if fusion_top_k is not None
        else settings.rag_fusion_top_k
    )
    final_top_k = (
        final_top_k
        if final_top_k is not None
        else settings.rag_final_top_k
    )

    # ----------------------------------------------------
    # 概述类问题：直接取文档开头 Chunk（摘要 / 引言），
    # 绕过对概述问题不友好的 rerank 阈值门控
    # ----------------------------------------------------

    if force_overview or is_overview_question(question):
        # 全局概述时每个文档只取前 N 个开头父块，避免大文档
        # 片段垄断上下文；单文档概述仍取 final_top_k 个。
        opening_per_file = (
            GLOBAL_OVERVIEW_OPENING_CHUNKS
            if file_id is None
            else final_top_k
        )
        opening_results = get_opening_chunks(
            file_id=file_id,
            chunks_per_file=opening_per_file,
        )

        if opening_results:
            opening_citations = build_citations(
                reranked_results=opening_results,
                file_id=(
                    file_id if file_id is not None else 0
                ),
                filename=(
                    filename if filename is not None else ""
                ),
            )
            return {
                "retrieval_status": "matched",
                "citations": opening_citations,
                "context": build_citation_context(
                    opening_citations
                ),
                "best_rerank_score": None,
                "rerank_threshold": (
                    settings.rag_rerank_min_score
                ),
                "queries": [question],
                "retrieval_mode": "overview_opening",
            }

    # ----------------------------------------------------
    # 全局检索时，对比 / 指代 / 宽泛问题需要查询改写来
    # 拆分对象、扩展召回。门控会对含技术锚点的明确问题
    # 自动跳过 LLM，因此这里统一保证改写数量，不增加
    # 明确问题的 LLM 开销。
    # ----------------------------------------------------
    if file_id is None and not _should_skip_rewrite(question):
        rewrite_count = max(
            rewrite_count,
            settings.rag_global_rewrite_count,
        )

    retrieval_result = hybrid_rerank_retrieve(
        question=question,
        file_id=file_id,
        rewrite_count=rewrite_count,
        per_retriever_top_k=per_retriever_top_k,
        fusion_top_k=fusion_top_k,
        final_top_k=final_top_k,
    )

    reranked_results = retrieval_result.get(
        "reranked_results", []
    )

    # 阈值过滤，保留可信子块（子块主题聚焦，rerank 打分精准）
    trusted_children = [
        item
        for item in reranked_results
        if item.get("rerank_score") is not None
        and item.get("rerank_score")
        >= settings.rag_rerank_min_score
    ]

    # ----------------------------------------------------
    # 文件覆盖回填（仅全局检索）：
    # 对比 / 多文档问题中，小文档片段的 rerank 分数可能
    # 卡在阈值附近、被整体过滤。对 RRF 候选中出现、但
    # trusted 中缺失的每个文件，回填其 rerank 最高分片段
    # （限阈值 10 分以内，避免引入完全无关内容）。
    # ----------------------------------------------------

    if file_id is None:
        trusted_fids = {
            item.get("file_id")
            for item in trusted_children
        }
        floor = settings.rag_rerank_min_score - 10.0
        best_by_file: dict[Any, dict[str, Any]] = {}

        for item in reranked_results:
            fid = item.get("file_id")
            if fid in trusted_fids:
                continue
            score = item.get("rerank_score")
            if score is None or score < floor:
                continue
            cur = best_by_file.get(fid)
            if cur is None or score > cur.get(
                "rerank_score", -1
            ):
                best_by_file[fid] = item

        if best_by_file:
            trusted_children = sorted(
                trusted_children
                + list(best_by_file.values()),
                key=lambda x: x.get("rerank_score") or 0,
                reverse=True,
            )

    if not trusted_children:
        best_rerank_score = (
            reranked_results[0].get("rerank_score")
            if reranked_results
            else None
        )
        return {
            "retrieval_status": "not_matched",
            "citations": [],
            "best_rerank_score": best_rerank_score,
            "rerank_threshold": settings.rag_rerank_min_score,
            "queries": retrieval_result.get("queries", []),
            "message": (
                "知识库中没有检索到足够相关的内容"
                f"（最高相关分 {best_rerank_score}，"
                f"阈值 {settings.rag_rerank_min_score}）。"
            ),
        }

    # 命中子块提升为父块：父块去重、聚合完整上下文，
    # citations 以父块为单元交给 LLM
    with metrics.stage_timer("parent_promotion"):
        parent_results = promote_children_to_parents(
            trusted_children
        )

    citations = build_citations(
        reranked_results=parent_results,
        file_id=file_id,
        filename=filename,
    )
    context = build_citation_context(citations)

    return {
        "retrieval_status": "matched",
        "citations": citations,
        "context": context,
        "best_rerank_score": parent_results[0].get(
            "rerank_score"
        ),
        "rerank_threshold": settings.rag_rerank_min_score,
        "queries": retrieval_result.get("queries", []),
    }