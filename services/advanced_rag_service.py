import json
import os
import re
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from core.metrics import metrics
from services.llm_service import call_llm

from services.prompt_service import (
    build_query_rewrite_messages,
)

from services.vector_service import (
    search_similar_chunks,
    keyword_search_chunks,
)


# ============================================================
# Cross-Encoder 重排模型（本地、离线加载）
# ============================================================

# bge-reranker-v2-m3：多语言交叉编码器，
# 直接对 (问题, chunk) 配对打分，替代原先的 LLM 打分。
reranker_model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)


# ============================================================
# RRF 配置
# ============================================================

RRF_K = int(os.getenv("RRF_K", "60"))

# 关键词检索在 RRF 融合中的权重。
# 向量检索通常有 1+N 组多查询结果，关键词检索只有 1 组，
# 给予额外权重以平衡字面匹配的贡献，避免被向量结果稀释。
RRF_KEYWORD_WEIGHT = float(
    os.getenv("RRF_KEYWORD_WEIGHT", "5.0")
)


# ============================================================
# Query Rewrite 门控与缓存
# ============================================================

# 技术锚点：拉丁/数字 token（AdamW、lr、RRF、K8s、V3）
# 或明确技术内容词。命中说明原问题已含可被 embedding /
# cross-encoder 精准匹配的锚点，LLM 改写的边际收益低。
_REWRITE_ANCHOR_PATTERN = re.compile(
    r"[A-Za-z0-9]{2,}"
    r"|优化器|损失|学习率|模型|算法|网络|结构|流程|参数|实现|采用"
    r"|调度|资源|容器|梯度|权重|函数|任务|框架|架构|特征|训练"
    r"|公式|变量|矩阵|分类|回归|聚类|模块|组件"
)

# 指代 / 上下文依赖 marker：命中说明问题依赖上文，
# 改写有助于消解，不应跳过。
_REWRITE_DEP_PATTERN = re.compile(
    r"它|他|她|这|那|其|该|这个|那个|这种|那些|这些|上面|前面|刚才"
)

# 对比 / 区别类问题：需要分别检索多个对象，
# 改写可拆分出针对每个对象的查询，不应跳过。
_REWRITE_COMPARISON_PATTERN = re.compile(
    r"区别|差别|差异|对比|相比|比较|不同|异同|哪个好"
)

# 改写结果缓存：key = 归一化原问题，value = queries
_rewrite_cache: dict[str, list[str]] = {}


def _should_skip_rewrite(question: str) -> bool:
    """
    判断该问题是否可以跳过 LLM 改写、直接用原问题检索。

    保守策略：仅当问题「无指代依赖」且「含技术锚点」时跳过；
    宽泛、口语、无锚点的问题仍保留改写以扩展召回。
    """

    q = question.strip()
    if not q:
        return True

    if _REWRITE_DEP_PATTERN.search(q):
        return False

    # 对比类问题强制改写：拆分出针对每个对象的查询
    if _REWRITE_COMPARISON_PATTERN.search(q):
        return False

    if _REWRITE_ANCHOR_PATTERN.search(q):
        return True

    return False


def clean_rewrite_query(
    text: str,
) -> str:
    """
    清理模型返回的 Query。

    即使 Prompt 要求模型不要输出序号，
    实际模型仍有可能返回：

        1. fastAI 的用途是什么？
        2、fastAI 可以做什么？
        - fastAI 主要应用在哪里？

    所以这里统一清理前面的序号和列表符号。
    """

    text = text.strip()

    if not text:
        return ""

    # 删除：
    #
    # 1. xxx
    # 1、xxx
    # 1) xxx
    # - xxx
    # * xxx
    text = re.sub(
        r"^\s*(?:[-*]|\d+[.、)])\s*",
        "",
        text,
    )

    return text.strip()


def rewrite_queries(
    question: str,
    rewrite_count: int = 2,
) -> list[str]:
    """
    使用 LLM 对用户问题进行 Query Rewrite。

    参数：
        question:
            用户原始问题。

        rewrite_count:
            额外生成几个改写 Query。

    返回示例：

        [
            "fastAI 是什么？",
            "fastAI 的主要用途是什么？",
            "fastAI 是什么类型的工具或框架？"
        ]

    注意：
        原始问题始终保留在第一个位置。
    """

    # ============================================================
    # 1. 清理原始问题
    # ============================================================

    question = question.strip()

    if not question:
        return []

    # 不需要 Rewrite
    if rewrite_count <= 0:
        return [question]

    # 明确、含技术锚点的问题跳过 LLM 改写，省一次往返
    if _should_skip_rewrite(question):
        return [question]

    # 命中缓存直接返回（相同问题不重复调 LLM）
    cache_key = question
    cached = _rewrite_cache.get(cache_key)
    if cached is not None:
        return list(cached)

    # ============================================================
    # 2. 构造 Query Rewrite Prompt
    # ============================================================

    messages = build_query_rewrite_messages(
        question=question,
        rewrite_count=rewrite_count,
    )

    # ============================================================
    # 3. 调用大模型
    # ============================================================

    # Query Rewrite 不需要创造性，
    # 所以 temperature 使用 0。
    result = call_llm(
        messages=messages,
        temperature=0.0,
    )

    # ============================================================
    # 4. 解析模型结果
    # ============================================================

    rewritten_queries = []

    for line in result.splitlines():

        query = clean_rewrite_query(line)

        if not query:
            continue

        # 不重复添加原始问题
        if query == question:
            continue

        # 不允许重复 Query
        if query in rewritten_queries:
            continue

        rewritten_queries.append(query)

        # 达到指定数量就停止
        if len(rewritten_queries) >= rewrite_count:
            break

    # ============================================================
    # 5. 原始 Query + Rewrite Query
    # ============================================================

    final_queries = [
        question,
        *rewritten_queries,
    ]

    _rewrite_cache[cache_key] = list(final_queries)

    return final_queries

def multi_query_retrieve(
    question: str,
    file_id: int | None = None,
    rewrite_count: int = 2,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Multi-Query Retrieval。

    流程：

        用户原始问题
            ↓
        Query Rewrite
            ↓
        Original Query
        Rewrite Query 1
        Rewrite Query 2
            ↓
        每个 Query 分别执行向量检索
            ↓
        返回每一组检索结果

    当前阶段：
        只负责多 Query 检索。

    暂时不负责：
        - 去重
        - RRF
        - Rerank
        - 最终答案生成
    """

    # ============================================================
    # 1. Query Rewrite
    # ============================================================

    queries = rewrite_queries(
        question=question,
        rewrite_count=rewrite_count,
    )

    # 最终保存每个 Query 的检索结果
    retrieval_results = []

    # ============================================================
    # 2. 每个 Query 分别执行 Vector Search
    # ============================================================

    for query in queries:

        results = search_similar_chunks(
            question=query,
            file_id=file_id,
            top_k=top_k,
        )

        retrieval_results.append({
            "query": query,
            "results": results,
        })

    # ============================================================
    # 3. 返回完整检索过程
    # ============================================================

    return {
        "original_question": question,
        "queries": queries,
        "query_count": len(queries),
        "retrieval_results": retrieval_results,
    }

def reciprocal_rank_fusion(
    retrieval_results: list[dict[str, Any]],
    final_top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    RRF（Reciprocal Rank Fusion，倒数排名融合）

    作用：
        将多个 Query（查询）
        + 多个 Retriever（检索器）
        的结果融合成一个最终排名。

    当前支持：
        - vector（向量检索）
        - keyword（关键词检索）
    """

    fused_results: dict[
        str,
        dict[str, Any]
    ] = {}

    # ============================================================
    # 1. 遍历所有检索结果组
    # ============================================================

    for retrieval_group in retrieval_results:

        query = retrieval_group.get(
            "query",
            ""
        )

        retriever = retrieval_group.get(
            "retriever",
            "vector"
        )

        results = retrieval_group.get(
            "results",
            []
        )

        # ========================================================
        # 2. 遍历当前组中的排名结果
        # ========================================================

        for rank, item in enumerate(
            results,
            start=1,
        ):

            chunk_id = item.get(
                "chunk_id"
            )

            content = item.get(
                "content",
                ""
            )

            # Vector Search 才有 distance
            distance = item.get(
                "distance"
            )

            # Keyword Search 才有 keyword_score
            keyword_score = item.get(
                "keyword_score"
            )

            # key = 这个 Chunk 的唯一身份
            key = (
                str(chunk_id)
                if chunk_id
                else content
            )

            # ====================================================
            # 3. 第一次遇到这个 Chunk
            # ====================================================

            if key not in fused_results:

                fused_results[key] = {
                    "chunk_id": chunk_id,
                    "content": content,

                    # 文件归属（全局检索时各 chunk 可能来自不同文件）
                    "file_id": item.get("file_id"),
                    "filename": item.get("filename"),

                    # 父子块：子块所属父块，rerank 后用于提升父块
                    "parent_id": item.get("parent_id"),
                    "parent_index": item.get("parent_index"),

                    # RRF 融合得分
                    "rrf_score": 0.0,

                    # 最好的向量距离
                    "best_distance": None,

                    # 最好的关键词得分
                    "best_keyword_score": None,

                    # 命中的 Query
                    "matched_queries": [],

                    # 命中的 Retriever
                    "matched_retrievers": [],

                    # 保存每一条检索路径
                    "retrieval_paths": [],
                }

            fused_item = fused_results[key]

            # ====================================================
            # 4. 计算 RRF Score
            # ====================================================

            rrf_score = (
                1.0 / (RRF_K + rank)
            )

            # 关键词检索加权：弥补只有 1 组结果的劣势
            if retriever == "keyword":
                rrf_score *= RRF_KEYWORD_WEIGHT

            fused_item["rrf_score"] += (
                rrf_score
            )

            # ====================================================
            # 5. 保存 Query
            # ====================================================

            if (
                query
                not in fused_item[
                    "matched_queries"
                ]
            ):
                fused_item[
                    "matched_queries"
                ].append(
                    query
                )

            # ====================================================
            # 6. 保存 Retriever
            # ====================================================

            if (
                retriever
                not in fused_item[
                    "matched_retrievers"
                ]
            ):
                fused_item[
                    "matched_retrievers"
                ].append(
                    retriever
                )

            # ====================================================
            # 7. 保存完整检索路径
            # ====================================================

            fused_item[
                "retrieval_paths"
            ].append({
                "query": query,
                "retriever": retriever,
                "rank": rank,
                "distance": distance,
                "keyword_score": keyword_score,
            })

            # ====================================================
            # 8. 更新最好的 distance
            # distance 越小越好
            # ====================================================

            if distance is not None:

                old_distance = fused_item[
                    "best_distance"
                ]

                if (
                    old_distance is None
                    or distance < old_distance
                ):
                    fused_item[
                        "best_distance"
                    ] = distance

            # ====================================================
            # 9. 更新最好的 keyword_score
            # keyword_score 越大越好
            # ====================================================

            if keyword_score is not None:

                old_keyword_score = fused_item[
                    "best_keyword_score"
                ]

                if (
                    old_keyword_score is None
                    or keyword_score
                    > old_keyword_score
                ):
                    fused_item[
                        "best_keyword_score"
                    ] = keyword_score

    # ============================================================
    # 10. 按 RRF Score 排序
    # ============================================================

    sorted_results = sorted(
        fused_results.values(),
        key=lambda item: item[
            "rrf_score"
        ],
        reverse=True,
    )

    return sorted_results[:final_top_k]


def diversify_by_file(
    fused_results: list[dict[str, Any]],
    final_top_k: int,
    per_file_limit: int = 3,
) -> list[dict[str, Any]]:
    """
    按文件分组轮转合并，保证每个文档都有片段进入最终候选。

    解决跨文档对比 / 概述问题中，小文档（如 test02）被
    大文档（如论文）的片段完全挤出 top_k 的问题。

    策略：
        1. 按 file_id 分组，每组保留 per_file_limit 条；
        2. 各组按 RRF 分数排序；
        3. 轮转（round-robin）合并，取 final_top_k。
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []

    for item in fused_results:
        key = str(
            item.get("file_id")
            or item.get("filename")
            or "unknown"
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        if len(groups[key]) < per_file_limit:
            groups[key].append(item)

    result: list[dict[str, Any]] = []
    max_depth = (
        max((len(g) for g in groups.values()), default=0)
    )

    for depth in range(max_depth):
        for key in order:
            if depth < len(groups[key]):
                result.append(groups[key][depth])
                if len(result) >= final_top_k:
                    return result

    return result


def advanced_retrieve(
    question: str,
    file_id: int | None = None,
    rewrite_count: int = 2,
    per_query_top_k: int = 3,
    final_top_k: int = 5,
) -> dict[str, Any]:
    """
    Advanced RAG 当前阶段完整 Retriever。

    流程：

        Question
            ↓
        Query Rewrite
            ↓
        Multi-Query Retrieval
            ↓
        RRF
            ↓
        Final Top K
    """

    # ============================================================
    # 1. Multi-Query Retrieval
    # ============================================================

    multi_query_result = (
        multi_query_retrieve(
            question=question,
            file_id=file_id,
            rewrite_count=rewrite_count,
            top_k=per_query_top_k,
        )
    )

    # ============================================================
    # 2. Hybrid Search：向量检索 + 关键词检索，RRF 融合
    # ============================================================

    # 关键词检索：弥补纯向量检索对字面匹配（如"实验结果"）
    # 召回不足的问题，与向量结果互补。
    keyword_results = keyword_search_chunks(
        question=question,
        file_id=file_id,
        top_k=max(per_query_top_k * 2, 8),
    )

    all_retrieval_results = (
        multi_query_result["retrieval_results"]
        + [{
            "query": question,
            "retriever": "keyword",
            "results": keyword_results,
        }]
    )

    # 全局检索时扩大 RRF 候选池，再按文件多样性轮转，
    # 保证每个文档（尤其小文档）都有片段进入最终候选。
    if file_id is None:
        rrf_pool_size = max(final_top_k * 3, 15)
        fused_results = reciprocal_rank_fusion(
            retrieval_results=all_retrieval_results,
            final_top_k=rrf_pool_size,
        )
        fused_results = diversify_by_file(
            fused_results,
            final_top_k=final_top_k,
        )
    else:
        fused_results = reciprocal_rank_fusion(
            retrieval_results=all_retrieval_results,
            final_top_k=final_top_k,
        )

    # ============================================================
    # 3. 返回
    # ============================================================

    return {
        "original_question": (
            multi_query_result[
                "original_question"
            ]
        ),

        "queries": (
            multi_query_result[
                "queries"
            ]
        ),

        "query_count": (
            multi_query_result[
                "query_count"
            ]
        ),

        # 原始 Multi-Query 检索过程
        "retrieval_results": (
            multi_query_result[
                "retrieval_results"
            ]
        ),

        # RRF 融合结果
        "fused_results": (
            fused_results
        ),
    }

def hybrid_retrieve(
    question: str,
    file_id: int | None = None,
    rewrite_count: int = 2,
    per_retriever_top_k: int = 3,
    final_top_k: int = 5,
) -> dict[str, Any]:
    """
    Hybrid Search（混合检索）

    Hybrid
        = 混合

    Search
        = 搜索 / 检索

    流程：

        用户问题
            ↓
        Query Rewrite（查询改写）
            ↓
        多个 Query（查询）
            ↓
        每个 Query 同时做：
            - Vector Search（向量检索）
            - Keyword Search（关键词检索）
            ↓
        RRF Fusion（排名融合）
            ↓
        Final Top K（最终前 K 个结果）
    """

    # ============================================================
    # 1. Query Rewrite
    # ============================================================

    with metrics.stage_timer("query_rewrite"):
        queries = rewrite_queries(
            question=question,
            rewrite_count=rewrite_count,
        )

    retrieval_results = []

    # ============================================================
    # 2. 对每个 Query 分别执行两种检索
    # ============================================================

    for query in queries:

        # --------------------------------------------------------
        # 2.1 Vector Search
        # --------------------------------------------------------

        with metrics.stage_timer("vector_search"):
            vector_results = (
                search_similar_chunks(
                    question=query,
                    file_id=file_id,
                    top_k=per_retriever_top_k,
                )
            )

        retrieval_results.append({
            "query": query,
            "retriever": "vector",
            "results": vector_results,
        })

        # --------------------------------------------------------
        # 2.2 Keyword Search
        # --------------------------------------------------------

        with metrics.stage_timer("keyword_search"):
            keyword_results = (
                keyword_search_chunks(
                    question=query,
                    file_id=file_id,
                    top_k=per_retriever_top_k,
                )
            )

        retrieval_results.append({
            "query": query,
            "retriever": "keyword",
            "results": keyword_results,
        })

    # ============================================================
    # 3. RRF Fusion
    # ============================================================

    with metrics.stage_timer("rrf_fusion"):
        # 全局检索时扩大 RRF 候选池，再按文件多样性轮转，
        # 保证每个文档（尤其小文档）都有片段进入候选。
        if file_id is None:
            rrf_pool_size = max(final_top_k * 3, 15)
            fused_results = reciprocal_rank_fusion(
                retrieval_results=retrieval_results,
                final_top_k=rrf_pool_size,
            )
            fused_results = diversify_by_file(
                fused_results,
                final_top_k=final_top_k,
            )
        else:
            fused_results = reciprocal_rank_fusion(
                retrieval_results=retrieval_results,
                final_top_k=final_top_k,
            )

    # ============================================================
    # 4. 返回结果
    # ============================================================

    return {
        "original_question": question,
        "queries": queries,
        "query_count": len(queries),

        # 当前有 2 种 Retriever
        "retriever_count": 2,

        # 例如：
        # 3 个 Query × 2 个 Retriever = 6
        "retrieval_group_count": len(
            retrieval_results
        ),

        # 原始检索过程
        "retrieval_results": retrieval_results,

        # 融合后的最终结果
        "fused_results": fused_results,
    }

def parse_rerank_json(
    text: str,
) -> list[dict[str, Any]]:
    """
    解析 Reranker 返回的 JSON。

    parse
        = 解析

    JSON
        = JavaScript Object Notation
        = 一种常用结构化数据格式
    """

    text = text.strip()

    # 删除可能出现的 Markdown JSON 代码块
    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    try:
        data = json.loads(
            text
        )

    except json.JSONDecodeError:
        return []

    if not isinstance(
        data,
        list,
    ):
        return []

    return [
        item
        for item in data
        if isinstance(item, dict)
    ]

def rerank_chunks(
    question: str,
    candidates: list[dict[str, Any]],
    final_top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Rerank（重排序）。

    使用本地 Cross-Encoder（bge-reranker-v2-m3）
    对 (问题, chunk) 配对做精细相关性打分，
    替代原先调用 DeepSeek 打分。

    输入：
        Hybrid Search + RRF 得到的一批候选 Chunk。

    输出：
        按 Cross-Encoder 相关性重新排序后的 Chunk，
        含 rerank_score(0-100)、rerank_reason、rerank_rank。
    """

    if not candidates:
        return []

    # ============================================================
    # 1. 构造 (问题, chunk 内容) 配对
    # ============================================================

    pairs = [
        (
            question,
            item.get("content", ""),
        )
        for item in candidates
    ]

    # ============================================================
    # 2. Cross-Encoder 打分（输出 raw logits）
    # ============================================================

    raw_scores = reranker_model.predict(pairs)

    # ============================================================
    # 3. sigmoid 归一化到 0-1，再映射到 0-100
    # ============================================================

    probs = 1.0 / (
        1.0 + np.exp(
            -np.asarray(raw_scores, dtype=float)
        )
    )

    score_hundred = probs * 100.0

    # ============================================================
    # 4. 组装候选并附加分数
    # ============================================================

    scored_candidates = []

    for index, candidate in enumerate(
        candidates
    ):

        item = dict(candidate)
        score = float(score_hundred[index])

        item["rerank_score"] = round(score, 2)
        item["rerank_reason"] = (
            f"Cross-Encoder 相关性 {score:.1f}%"
        )

        scored_candidates.append(item)

    # ============================================================
    # 5. 按相关性从高到低排序
    # ============================================================

    scored_candidates.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    # ============================================================
    # 6. 附加最终排名，取 Top K
    # ============================================================

    final_results = []

    for rank, item in enumerate(
        scored_candidates[:final_top_k],
        start=1,
    ):

        item["rerank_rank"] = rank
        final_results.append(item)

    return final_results

def hybrid_rerank_retrieve(
    question: str,
    file_id: int | None = None,
    rewrite_count: int = 2,
    per_retriever_top_k: int = 5,
    fusion_top_k: int = 6,
    final_top_k: int = 3,
) -> dict[str, Any]:
    """
    Hybrid + Rerank 完整检索流程。

    Pipeline
        = 流程 / 处理流水线

    流程：

        Query Rewrite
            ↓
        Vector + Keyword
            ↓
        RRF
            ↓
        Candidate Chunks
            ↓
        Rerank
            ↓
        Final Top K
    """

    # ============================================================
    # 1. Hybrid Retrieval
    # ============================================================

    hybrid_result = (
        hybrid_retrieve(
            question=question,
            file_id=file_id,
            rewrite_count=rewrite_count,
            per_retriever_top_k=(
                per_retriever_top_k
            ),
            final_top_k=(
                fusion_top_k
            ),
        )
    )

    candidates = hybrid_result[
        "fused_results"
    ]

    # ============================================================
    # 2. Rerank
    # ============================================================

    with metrics.stage_timer("rerank"):
        reranked_results = rerank_chunks(
            question=question,
            candidates=candidates,
            final_top_k=final_top_k,
        )

    # ============================================================
    # 3. 返回
    # ============================================================

    return {
        "original_question": question,

        "queries": hybrid_result[
            "queries"
        ],

        # RRF 得到的候选结果
        "fused_results": candidates,

        # Rerank 后最终结果
        "reranked_results": (
            reranked_results
        ),
    }