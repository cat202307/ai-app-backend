import json
import re

from pathlib import Path
from typing import Any
from core.config import settings
from statistics import mean, pstdev
from database import SessionLocal
from models import UploadedFile
from services.vector_service import (
    search_similar_chunks,
)

from services.advanced_rag_service import (
    advanced_retrieve,
    hybrid_retrieve,
    hybrid_rerank_retrieve,
)

from services.advanced_rag_answer_service import (
    answer_with_advanced_rag,
)

from services.citation_service import (
    extract_citation_ids,
)

def load_eval_dataset(
    dataset_path: str,
) -> list[dict[str, Any]]:
    """
    加载 RAG Evaluation Dataset。

    load
        = 加载

    dataset
        = 数据集
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"评测数据集不存在：{dataset_path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(
            file
        )

    if not isinstance(data, list):
        raise ValueError(
            "RAG 评测数据集必须是 JSON 数组。"
        )

    return data

def calculate_retrieval_metrics(
    ranked_chunk_ids: list[str],
    relevant_chunk_ids: list[str],
) -> dict[str, Any]:
    """
    计算单条问题的检索指标。

    ranked_chunk_ids：
        系统实际返回的 Chunk 排名。

    relevant_chunk_ids：
        标准答案中的相关 Chunk。
    """

    relevant_set = set(
        relevant_chunk_ids
    )

    # ============================================================
    # 1. 没有标准相关 Chunk
    #    属于负向问题，不计算传统 Retrieval 指标
    # ============================================================

    if not relevant_set:
        return {
            "hit": None,
            "recall": None,
            "reciprocal_rank": None,
        }

    # ============================================================
    # 2. Hit@K
    #
    # Top K 中只要出现一个正确 Chunk，就算 Hit。
    # ============================================================

    retrieved_relevant = [
        chunk_id
        for chunk_id in ranked_chunk_ids
        if chunk_id in relevant_set
    ]

    hit = (
        len(retrieved_relevant) > 0
    )

    # ============================================================
    # 3. Recall@K
    #
    # 找回来的正确 Chunk 数量
    # -------------------------
    # 标准相关 Chunk 总数量
    # ============================================================

    recall = (
        len(set(retrieved_relevant))
        / len(relevant_set)
    )

    # ============================================================
    # 4. Reciprocal Rank
    #
    # reciprocal
    # = 倒数
    #
    # 第一个正确结果排名第 1：
    # 1 / 1 = 1
    #
    # 排名第 2：
    # 1 / 2 = 0.5
    # ============================================================

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        ranked_chunk_ids,
        start=1,
    ):

        if chunk_id in relevant_set:

            reciprocal_rank = (
                1.0 / rank
            )

            break

    return {
        "hit": hit,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
    }

def evaluate_case(
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    评测一条 RAG Case（测试用例）。
    """

    question = case["question"]
    file_id = case["file_id"]

    expected_answerable = case[
        "expected_answerable"
    ]

    relevant_chunk_ids = case.get(
        "relevant_chunk_ids",
        [],
    )

    # ============================================================
    # 1. 跑 Hybrid + Rerank
    # ============================================================

    # 召回侧参数（改写数 / 每路召回 / RRF 融合）与生产共用同一份
    # 配置，保证评测结论反映生产行为；final_top_k 固定为评测指标
    # 截断（Hit@3 / Recall@3 / MRR@3），它是生产 final_top_k 的
    # 子集，相当于更严格的口径。
    result = hybrid_rerank_retrieve(
        question=question,
        file_id=file_id,
        rewrite_count=settings.rag_global_rewrite_count,
        per_retriever_top_k=settings.rag_per_retriever_top_k,
        fusion_top_k=settings.rag_fusion_top_k,
        final_top_k=3,
    )

    reranked_results = result.get(
        "reranked_results",
        [],
    )

    ranked_chunk_ids = [
        item.get("chunk_id")
        for item in reranked_results
        if item.get("chunk_id")
    ]

    # ============================================================
    # 2. Retrieval Metrics
    # ============================================================

    metrics = calculate_retrieval_metrics(
        ranked_chunk_ids=ranked_chunk_ids,
        relevant_chunk_ids=relevant_chunk_ids,
    )

    # ============================================================
    # 3. Rerank Top1 Accuracy
    # ============================================================

    top1_chunk_id = (
        ranked_chunk_ids[0]
        if ranked_chunk_ids
        else None
    )

    rerank_top1_correct = None

    if relevant_chunk_ids:

        rerank_top1_correct = (
            top1_chunk_id
            in set(relevant_chunk_ids)
        )

    # ============================================================
    # 4. 根据 Rerank Threshold 判断是否“可回答”
    # ============================================================

    best_rerank_score = None

    if reranked_results:

        best_rerank_score = (
            reranked_results[0].get(
                "rerank_score"
            )
        )

    predicted_answerable = (
            best_rerank_score is not None
            and best_rerank_score
            >= settings.rag_rerank_min_score
    )
    # predicted
    # = 系统预测的

    answerability_correct = (
        predicted_answerable
        == expected_answerable
    )

    return {
        "case_id": case["case_id"],
        "question": question,

        "expected_answerable": (
            expected_answerable
        ),

        "predicted_answerable": (
            predicted_answerable
        ),

        "answerability_correct": (
            answerability_correct
        ),

        "relevant_chunk_ids": (
            relevant_chunk_ids
        ),

        "ranked_chunk_ids": (
            ranked_chunk_ids
        ),

        "hit": metrics["hit"],
        "recall": metrics["recall"],
        "reciprocal_rank": (
            metrics["reciprocal_rank"]
        ),

        "rerank_top1_correct": (
            rerank_top1_correct
        ),

        "best_rerank_score": (
            best_rerank_score
        ),
    }

def evaluate_dataset(
    dataset_path: str,
) -> dict[str, Any]:
    """
    运行整个 RAG Evaluation Dataset。
    """

    dataset = load_eval_dataset(
        dataset_path
    )

    case_results = []

    for case in dataset:

        result = evaluate_case(
            case
        )

        case_results.append(
            result
        )

    # ============================================================
    # 统计 Answerability Accuracy
    # ============================================================

    total_count = len(
        case_results
    )

    answerability_correct_count = sum(
        1
        for item in case_results
        if item[
            "answerability_correct"
        ]
    )

    answerability_accuracy = (
        answerability_correct_count
        / total_count
        if total_count
        else 0.0
    )

    # ============================================================
    # 只统计正向问题的 Retrieval Metric
    # ============================================================

    positive_results = [
        item
        for item in case_results
        if item["expected_answerable"]
    ]

    hit_values = [
        1.0 if item["hit"] else 0.0
        for item in positive_results
    ]

    recall_values = [
        item["recall"]
        for item in positive_results
        if item["recall"] is not None
    ]

    rr_values = [
        item["reciprocal_rank"]
        for item in positive_results
        if item[
            "reciprocal_rank"
        ] is not None
    ]

    top1_values = [
        1.0
        if item[
            "rerank_top1_correct"
        ]
        else 0.0
        for item in positive_results
    ]

    def average(
        values: list[float],
    ) -> float:

        if not values:
            return 0.0

        return sum(values) / len(values)

    return {
        "summary": {
            "total_cases": total_count,

            "answerability_accuracy": (
                answerability_accuracy
            ),

            "hit_rate": average(
                hit_values
            ),

            "mean_recall": average(
                recall_values
            ),

            # MRR
            # Mean Reciprocal Rank
            "mrr": average(
                rr_values
            ),

            "rerank_top1_accuracy": (
                average(
                    top1_values
                )
            ),
        },

        "cases": case_results,
    }

def extract_ranked_chunk_ids(
    results: list[dict[str, Any]],
) -> list[str]:
    """
    从检索结果中提取有顺序的 Chunk ID。

    ranked
        = 已排序的

    chunk_ids
        = 文本片段 ID

    例如：

        [
            {
                "chunk_id": "file_5_chunk_1"
            },
            {
                "chunk_id": "file_5_chunk_0"
            }
        ]

    转换成：

        [
            "file_5_chunk_1",
            "file_5_chunk_0"
        ]
    """

    return [
        item["chunk_id"]
        for item in results
        if item.get("chunk_id")
    ]

def run_retrieval_variant(
    variant: str,
    question: str,
    file_id: int,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    执行不同版本的 RAG Retrieval。

    variant
        = 不同方案 / 变体

    当前比较：

        basic_vector
            基础向量检索

        multi_query
            多查询 + Vector + RRF

        hybrid
            Vector + Keyword + RRF

        hybrid_rerank
            Hybrid + RRF + Rerank
    """

    # ============================================================
    # 1. Basic Vector
    #    基础向量检索
    # ============================================================

    if variant == "basic_vector":

        return search_similar_chunks(
            question=question,
            file_id=file_id,
            top_k=top_k,
        )

    # ============================================================
    # 2. Multi Query
    #    Query Rewrite + Vector + RRF
    # ============================================================

    if variant == "multi_query":

        result = advanced_retrieve(
            question=question,
            file_id=file_id,
            rewrite_count=2,
            per_query_top_k=top_k,
            final_top_k=top_k,
        )

        return result.get(
            "fused_results",
            [],
        )

    # ============================================================
    # 3. Hybrid
    #    Vector + Keyword + RRF
    # ============================================================

    if variant == "hybrid":

        result = hybrid_retrieve(
            question=question,
            file_id=file_id,
            rewrite_count=2,
            per_retriever_top_k=top_k,
            final_top_k=top_k,
        )

        return result.get(
            "fused_results",
            [],
        )

    # ============================================================
    # 4. Hybrid + Rerank
    # ============================================================

    if variant == "hybrid_rerank":

        # 旗舰管线：召回侧参数与生产共用同一份配置，
        # 只把最终截断换成评测的 top_k（指标口径）。
        result = hybrid_rerank_retrieve(
            question=question,
            file_id=file_id,
            rewrite_count=settings.rag_global_rewrite_count,
            per_retriever_top_k=settings.rag_per_retriever_top_k,
            fusion_top_k=settings.rag_fusion_top_k,
            final_top_k=top_k,
        )

        return result.get(
            "reranked_results",
            [],
        )

    raise ValueError(
        f"未知 Retrieval Variant：{variant}"
    )

def compare_retrieval_variants(
    dataset_path: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    比较不同 Retrieval Pipeline 的效果。

    comparison
        = 对比

    pipeline
        = 流水线 / 处理流程
    """

    dataset = load_eval_dataset(
        dataset_path
    )

    # ============================================================
    # 只比较正向问题
    #
    # 因为传统 Hit / Recall / MRR
    # 需要存在标准 relevant_chunk_ids。
    # ============================================================

    positive_cases = [
        case
        for case in dataset
        if case.get(
            "expected_answerable"
        )
    ]

    variants = [
        "basic_vector",
        "multi_query",
        "hybrid",
        "hybrid_rerank",
    ]

    all_results = {}

    # ============================================================
    # 每一种 Variant 分别评测
    # ============================================================

    for variant in variants:

        case_results = []

        for case in positive_cases:

            results = run_retrieval_variant(
                variant=variant,
                question=case["question"],
                file_id=case["file_id"],
                top_k=top_k,
            )

            ranked_chunk_ids = (
                extract_ranked_chunk_ids(
                    results
                )
            )

            metrics = (
                calculate_retrieval_metrics(
                    ranked_chunk_ids=(
                        ranked_chunk_ids
                    ),
                    relevant_chunk_ids=(
                        case[
                            "relevant_chunk_ids"
                        ]
                    ),
                )
            )

            relevant_set = set(
                case[
                    "relevant_chunk_ids"
                ]
            )

            top1_correct = (
                bool(ranked_chunk_ids)
                and ranked_chunk_ids[0]
                in relevant_set
            )

            case_results.append({
                "case_id": case[
                    "case_id"
                ],

                "question": case[
                    "question"
                ],

                "ranked_chunk_ids": (
                    ranked_chunk_ids
                ),

                "hit": metrics[
                    "hit"
                ],

                "recall": metrics[
                    "recall"
                ],

                "reciprocal_rank": (
                    metrics[
                        "reciprocal_rank"
                    ]
                ),

                "top1_correct": (
                    top1_correct
                ),
            })

        # ========================================================
        # 汇总当前 Variant
        # ========================================================

        hit_values = [
            1.0 if item["hit"] else 0.0
            for item in case_results
        ]

        recall_values = [
            item["recall"]
            for item in case_results
        ]

        rr_values = [
            item["reciprocal_rank"]
            for item in case_results
        ]

        top1_values = [
            1.0
            if item["top1_correct"]
            else 0.0
            for item in case_results
        ]

        def avg(
            values: list[float],
        ) -> float:

            if not values:
                return 0.0

            return (
                sum(values)
                / len(values)
            )

        all_results[variant] = {
            "summary": {
                "case_count": len(
                    case_results
                ),

                f"hit_at_{top_k}": avg(
                    hit_values
                ),

                f"recall_at_{top_k}": avg(
                    recall_values
                ),

                "mrr": avg(
                    rr_values
                ),

                "top1_accuracy": avg(
                    top1_values
                ),
            },

            "cases": case_results,
        }

    return {
        "top_k": top_k,
        "variants": all_results,
    }

def calculate_answerability_metrics(
    case_results: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """
    根据指定 Threshold（阈值）
    计算 Answerability（可回答性）指标。

    Answerability：
        系统判断当前文档是否具有足够证据
        回答用户问题。

    threshold：
        Rerank Score（重排序分数）的最低可信阈值。
    """

    # ============================================================
    # 1. 二分类统计
    # ============================================================

    # TP = True Positive
    # 真阳性：
    # 本来可以回答，并且系统也判断可以回答。
    true_positive = 0

    # TN = True Negative
    # 真阴性：
    # 本来不能回答，并且系统也正确拒答。
    true_negative = 0

    # FP = False Positive
    # 假阳性：
    # 本来不能回答，但系统错误认为可以回答。
    false_positive = 0

    # FN = False Negative
    # 假阴性：
    # 本来可以回答，但系统错误拒答。
    false_negative = 0

    case_predictions = []

    # ============================================================
    # 2. 遍历所有 Case
    # ============================================================

    for item in case_results:

        expected = bool(
            item["expected_answerable"]
        )

        best_score = item.get(
            "best_rerank_score"
        )

        # predicted
        # = 系统预测的
        predicted = (
            best_score is not None
            and best_score >= threshold
        )

        # --------------------------------------------------------
        # TP
        # --------------------------------------------------------

        if expected and predicted:
            true_positive += 1

        # --------------------------------------------------------
        # TN
        # --------------------------------------------------------

        elif (
            not expected
            and not predicted
        ):
            true_negative += 1

        # --------------------------------------------------------
        # FP
        # --------------------------------------------------------

        elif (
            not expected
            and predicted
        ):
            false_positive += 1

        # --------------------------------------------------------
        # FN
        # --------------------------------------------------------

        elif (
            expected
            and not predicted
        ):
            false_negative += 1

        case_predictions.append({
            "case_id": item[
                "case_id"
            ],

            "expected_answerable": (
                expected
            ),

            "predicted_answerable": (
                predicted
            ),

            "best_rerank_score": (
                best_score
            ),

            "correct": (
                expected == predicted
            ),
        })

    # ============================================================
    # 3. 基础数量
    # ============================================================

    total = len(
        case_results
    )

    # ============================================================
    # 4. Accuracy（准确率）
    # ============================================================

    accuracy = (
        (
            true_positive
            + true_negative
        )
        / total
        if total
        else 0.0
    )

    # ============================================================
    # 5. Precision（精确率）
    #
    # 系统判断“可以回答”的问题中，
    # 有多少真的可以回答。
    # ============================================================

    precision_denominator = (
        true_positive
        + false_positive
    )

    precision = (
        true_positive
        / precision_denominator
        if precision_denominator
        else 0.0
    )

    # ============================================================
    # 6. Recall（召回率）
    #
    # 真正可以回答的问题中，
    # 系统成功识别了多少。
    # ============================================================

    recall_denominator = (
        true_positive
        + false_negative
    )

    recall = (
        true_positive
        / recall_denominator
        if recall_denominator
        else 0.0
    )

    # ============================================================
    # 7. F1 Score
    #
    # Precision 和 Recall 的调和平均值。
    # ============================================================

    f1 = 0.0

    if (
        precision
        + recall
        > 0
    ):
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    return {
        "threshold": threshold,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "true_positive": (
            true_positive
        ),

        "true_negative": (
            true_negative
        ),

        "false_positive": (
            false_positive
        ),

        "false_negative": (
            false_negative
        ),

        "cases": case_predictions,
    }

def evaluate_thresholds(
    dataset_path: str,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """
    Threshold Sweep（阈值扫描）。

    重要设计：

        整个 Dataset 只运行一次 Rerank。

    然后使用同一批：
        best_rerank_score

    模拟不同 Threshold 的效果。

    避免：
        - 重复请求 LLM
        - 浪费 Token
        - 不同轮模型评分波动
    """

    # ============================================================
    # 1. 默认需要扫描的 Threshold
    # ============================================================

    if thresholds is None:

        thresholds = [
            20,
            30,
            40,
            50,
            55,
            60,
            70,
            80,
            85,
            90,
        ]

    # ============================================================
    # 2. RAG Evaluation 只运行一次
    # ============================================================

    evaluation_result = (
        evaluate_dataset(
            dataset_path
        )
    )

    case_results = evaluation_result[
        "cases"
    ]

    # ============================================================
    # 3. 分析正向 / 负向 Score 分布
    # ============================================================

    positive_scores = []

    negative_scores = []

    for item in case_results:

        score = item.get(
            "best_rerank_score"
        )

        if score is None:
            continue

        if item[
            "expected_answerable"
        ]:
            positive_scores.append(
                float(score)
            )

        else:
            negative_scores.append(
                float(score)
            )

    min_positive_score = (
        min(positive_scores)
        if positive_scores
        else None
    )

    max_negative_score = (
        max(negative_scores)
        if negative_scores
        else None
    )

    # ============================================================
    # 4. 扫描所有 Threshold
    # ============================================================

    threshold_results = []

    for threshold in thresholds:

        metrics = (
            calculate_answerability_metrics(
                case_results=case_results,
                threshold=threshold,
            )
        )

        threshold_results.append(
            metrics
        )

    # ============================================================
    # 5. 找出最高 Accuracy
    # ============================================================

    best_accuracy = max(
        (
            item["accuracy"]
            for item in threshold_results
        ),
        default=0.0,
    )

    # 可能多个 Threshold 并列最好
    best_thresholds = [
        item["threshold"]
        for item in threshold_results
        if item[
            "accuracy"
        ] == best_accuracy
    ]

    return {
        # 原始 Evaluation 结果
        "base_summary": (
            evaluation_result[
                "summary"
            ]
        ),

        # Score Distribution
        # = 分数分布
        "score_distribution": {
            "positive_scores": (
                sorted(
                    positive_scores
                )
            ),

            "negative_scores": (
                sorted(
                    negative_scores
                )
            ),

            # 最低正向分
            "min_positive_score": (
                min_positive_score
            ),

            # 最高负向分
            "max_negative_score": (
                max_negative_score
            ),

            # 两组之间的间隔
            "score_margin": (
                (
                    min_positive_score
                    - max_negative_score
                )
                if (
                    min_positive_score
                    is not None
                    and max_negative_score
                    is not None
                )
                else None
            ),
        },

        "best_accuracy": (
            best_accuracy
        ),

        "best_thresholds": (
            best_thresholds
        ),

        "threshold_results": (
            threshold_results
        ),
    }

def evaluate_rerank_stability(
    dataset_path: str,
    repeat_count: int = 2,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """
    Rerank Stability Evaluation
    重排序稳定性评测。

    目标：

        同一个 Evaluation Dataset
        重复运行多次。

        观察：

        1. 每个 Case 的 Rerank Score 波动
        2. 不同 Threshold 在多轮实验中的稳定程度
        3. 找到更加稳健的 Threshold

    repeat_count：
        重复运行次数。

    建议学习阶段先使用 2，
    避免过多 LLM Token 消耗。
    """

    if repeat_count < 1:
        raise ValueError(
            "repeat_count 必须大于等于 1。"
        )

    if thresholds is None:

        thresholds = [
            40,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
            85,
            90,
        ]

    # ============================================================
    # 1. 保存每一轮完整 Evaluation
    # ============================================================

    evaluation_runs = []

    for run_index in range(
        repeat_count
    ):

        result = evaluate_dataset(
            dataset_path
        )

        evaluation_runs.append(
            result
        )

    # ============================================================
    # 2. 收集每个 Case 多轮 Rerank Score
    # ============================================================

    case_score_map = {}

    for run_index, run in enumerate(
        evaluation_runs,
        start=1,
    ):

        for case in run["cases"]:

            case_id = case[
                "case_id"
            ]

            if (
                case_id
                not in case_score_map
            ):

                case_score_map[
                    case_id
                ] = {
                    "question": case[
                        "question"
                    ],

                    "expected_answerable": (
                        case[
                            "expected_answerable"
                        ]
                    ),

                    "scores": [],
                }

            score = case.get(
                "best_rerank_score"
            )

            if score is not None:

                case_score_map[
                    case_id
                ][
                    "scores"
                ].append(
                    float(score)
                )

    # ============================================================
    # 3. 计算每个 Case 的 Score Stability
    # ============================================================

    case_stability = []

    for case_id, info in (
        case_score_map.items()
    ):

        scores = info[
            "scores"
        ]

        average_score = (
            mean(scores)
            if scores
            else None
        )

        min_score = (
            min(scores)
            if scores
            else None
        )

        max_score = (
            max(scores)
            if scores
            else None
        )

        score_range = None

        if (
            min_score is not None
            and max_score is not None
        ):
            score_range = (
                max_score
                - min_score
            )

        score_std = (
            pstdev(scores)
            if len(scores) >= 2
            else 0.0
        )

        case_stability.append({
            "case_id": case_id,

            "question": info[
                "question"
            ],

            "expected_answerable": (
                info[
                    "expected_answerable"
                ]
            ),

            "scores": scores,

            "average_score": (
                average_score
            ),

            "min_score": (
                min_score
            ),

            "max_score": (
                max_score
            ),

            "score_range": (
                score_range
            ),

            "score_std": (
                score_std
            ),
        })

    # ============================================================
    # 4. 找 Worst-Case 分数边界
    #
    # Positive：
    # 多轮实验里正向样本出现过的最低分
    #
    # Negative：
    # 多轮实验里负向样本出现过的最高分
    # ============================================================

    positive_min_scores = []

    negative_max_scores = []

    for item in case_stability:

        if item[
            "expected_answerable"
        ]:

            if (
                item["min_score"]
                is not None
            ):

                positive_min_scores.append(
                    item[
                        "min_score"
                    ]
                )

        else:

            if (
                item["max_score"]
                is not None
            ):

                negative_max_scores.append(
                    item[
                        "max_score"
                    ]
                )

    worst_positive_score = (
        min(
            positive_min_scores
        )
        if positive_min_scores
        else None
    )

    worst_negative_score = (
        max(
            negative_max_scores
        )
        if negative_max_scores
        else None
    )

    robust_margin = None

    if (
        worst_positive_score
        is not None
        and worst_negative_score
        is not None
    ):

        robust_margin = (
            worst_positive_score
            - worst_negative_score
        )

    # ============================================================
    # 5. 对每一个 Threshold
    #    计算多轮表现
    # ============================================================

    threshold_stability = []

    for threshold in thresholds:

        run_metrics = []

        for run in evaluation_runs:

            metrics = (
                calculate_answerability_metrics(
                    case_results=run[
                        "cases"
                    ],
                    threshold=threshold,
                )
            )

            run_metrics.append(
                metrics
            )

        accuracies = [
            item["accuracy"]
            for item in run_metrics
        ]

        precisions = [
            item["precision"]
            for item in run_metrics
        ]

        recalls = [
            item["recall"]
            for item in run_metrics
        ]

        f1_values = [
            item["f1"]
            for item in run_metrics
        ]

        total_fp = sum(
            item[
                "false_positive"
            ]
            for item in run_metrics
        )

        total_fn = sum(
            item[
                "false_negative"
            ]
            for item in run_metrics
        )

        threshold_stability.append({
            "threshold": (
                threshold
            ),

            "average_accuracy": (
                mean(accuracies)
            ),

            # 最差一轮 Accuracy
            "min_accuracy": (
                min(accuracies)
            ),

            # 最好一轮 Accuracy
            "max_accuracy": (
                max(accuracies)
            ),

            "average_precision": (
                mean(precisions)
            ),

            "average_recall": (
                mean(recalls)
            ),

            "average_f1": (
                mean(f1_values)
            ),

            "total_false_positive": (
                total_fp
            ),

            "total_false_negative": (
                total_fn
            ),
        })

    # ============================================================
    # 6. 找稳定性最好的 Threshold
    #
    # 优先：
    #
    # 1. min_accuracy 越高越好
    # 2. average_accuracy 越高越好
    # 3. F1 越高越好
    # ============================================================

    sorted_thresholds = sorted(
        threshold_stability,
        key=lambda item: (
            item[
                "min_accuracy"
            ],
            item[
                "average_accuracy"
            ],
            item[
                "average_f1"
            ],
        ),
        reverse=True,
    )

    best_threshold = None

    if sorted_thresholds:

        best_threshold = (
            sorted_thresholds[0][
                "threshold"
            ]
        )

    return {
        "repeat_count": (
            repeat_count
        ),

        "worst_case_scores": {
            "worst_positive_score": (
                worst_positive_score
            ),

            "worst_negative_score": (
                worst_negative_score
            ),

            "robust_margin": (
                robust_margin
            ),
        },

        "best_threshold": (
            best_threshold
        ),

        "case_stability": (
            case_stability
        ),

        "threshold_stability": (
            threshold_stability
        ),
    }


def get_filename_by_file_id(
    file_id: int,
) -> str:
    """
    根据 file_id 查询数据库中的文件名。
    """

    db = SessionLocal()

    try:

        uploaded_file = (
            db.query(UploadedFile)
            .filter(
                UploadedFile.id
                == file_id
            )
            .first()
        )

        if uploaded_file is None:
            raise ValueError(
                f"文件不存在：file_id={file_id}"
            )

        return uploaded_file.filename

    finally:
        db.close()

def evaluate_citation_case(
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    评测单条 Citation Case。
    """

    question = case[
        "question"
    ]

    file_id = case[
        "file_id"
    ]

    expected_answerable = case[
        "expected_answerable"
    ]

    relevant_chunk_ids = set(
        case.get(
            "relevant_chunk_ids",
            [],
        )
    )

    # ============================================================
    # 1. 查询文件名
    # ============================================================

    filename = get_filename_by_file_id(
        file_id
    )

    # ============================================================
    # 2. 跑完整 Advanced RAG
    # ============================================================

    result = answer_with_advanced_rag(
        question=question,
        file_id=file_id,
        filename=filename,
    )

    answer = result.get(
        "answer",
        "",
    )

    citations = result.get(
        "citations",
        [],
    )

    # ============================================================
    # 3. 提取回答中的 [1][2]
    # ============================================================

    answer_citation_ids = (
        extract_citation_ids(
            answer
        )
    )

    returned_citation_ids = {
        int(item["citation_id"])
        for item in citations
        if item.get(
            "citation_id"
        ) is not None
    }

    # ============================================================
    # 4. Citation Presence
    #
    # 正向问题：
    #     应该存在 citation。
    #
    # 负向问题：
    #     应该不存在 citation。
    # ============================================================

    if expected_answerable:

        citation_presence_correct = (
            len(citations) > 0
        )

    else:

        citation_presence_correct = (
            len(citations) == 0
        )

    # ============================================================
    # 5. Reference Validity
    #
    # 回答中出现的 [1][2]
    # 必须真的存在于 citations。
    # ============================================================

    invalid_reference_ids = [
        citation_id
        for citation_id
        in answer_citation_ids
        if citation_id
        not in returned_citation_ids
    ]

    citation_reference_valid = (
        len(
            invalid_reference_ids
        ) == 0
    )

    # 正向问题至少应该在正文中使用一个引用
    if (
        expected_answerable
        and not answer_citation_ids
    ):
        citation_reference_valid = False

    # ============================================================
    # 6. Ground Truth Chunk Accuracy
    #
    # ground truth
    # = 人工标注的标准答案
    # ============================================================

    returned_chunk_ids = [
        item.get(
            "chunk_id"
        )
        for item in citations
        if item.get(
            "chunk_id"
        )
    ]

    relevant_citation_hit = None

    if expected_answerable:

        relevant_citation_hit = any(
            chunk_id
            in relevant_chunk_ids
            for chunk_id
            in returned_chunk_ids
        )

    # ============================================================
    # 7. 是否所有返回 Citation
    #    都是标准相关 Chunk
    # ============================================================

    all_citations_relevant = None

    if expected_answerable:

        all_citations_relevant = (
            bool(returned_chunk_ids)
            and all(
                chunk_id
                in relevant_chunk_ids
                for chunk_id
                in returned_chunk_ids
            )
        )

    return {
        "case_id": case[
            "case_id"
        ],

        "question": question,

        "expected_answerable": (
            expected_answerable
        ),

        "retrieval_status": result.get(
            "retrieval_status"
        ),

        "answer": answer,

        "answer_citation_ids": (
            answer_citation_ids
        ),

        "returned_citation_ids": (
            sorted(
                returned_citation_ids
            )
        ),

        "returned_chunk_ids": (
            returned_chunk_ids
        ),

        "relevant_chunk_ids": list(
            relevant_chunk_ids
        ),

        "citation_presence_correct": (
            citation_presence_correct
        ),

        "citation_reference_valid": (
            citation_reference_valid
        ),

        "invalid_reference_ids": (
            invalid_reference_ids
        ),

        "relevant_citation_hit": (
            relevant_citation_hit
        ),

        "all_citations_relevant": (
            all_citations_relevant
        ),
    }

def evaluate_citations(
    dataset_path: str,
) -> dict[str, Any]:
    """
    对整个 Evaluation Dataset
    运行 Citation Evaluation。
    """

    dataset = load_eval_dataset(
        dataset_path
    )

    case_results = []

    for case in dataset:

        result = evaluate_citation_case(
            case
        )

        case_results.append(
            result
        )

    total_cases = len(
        case_results
    )

    # ============================================================
    # Citation Presence Accuracy
    # ============================================================

    presence_correct_count = sum(
        1
        for item in case_results
        if item[
            "citation_presence_correct"
        ]
    )

    citation_presence_accuracy = (
        presence_correct_count
        / total_cases
        if total_cases
        else 0.0
    )

    # ============================================================
    # Citation Reference Validity
    # ============================================================

    valid_reference_count = sum(
        1
        for item in case_results
        if item[
            "citation_reference_valid"
        ]
    )

    citation_reference_accuracy = (
        valid_reference_count
        / total_cases
        if total_cases
        else 0.0
    )

    # ============================================================
    # 正向问题
    # ============================================================

    positive_cases = [
        item
        for item in case_results
        if item[
            "expected_answerable"
        ]
    ]

    citation_hit_values = [
        1.0
        if item[
            "relevant_citation_hit"
        ]
        else 0.0
        for item in positive_cases
    ]

    all_relevant_values = [
        1.0
        if item[
            "all_citations_relevant"
        ]
        else 0.0
        for item in positive_cases
    ]

    def avg(
        values: list[float],
    ) -> float:

        if not values:
            return 0.0

        return (
            sum(values)
            / len(values)
        )

    return {
        "summary": {
            "total_cases": (
                total_cases
            ),

            "positive_cases": len(
                positive_cases
            ),

            # 是否该有引用时有引用，
            # 该拒答时没有引用
            "citation_presence_accuracy": (
                citation_presence_accuracy
            ),

            # 回答中的 [n]
            # 是否都有真实 Citation
            "citation_reference_accuracy": (
                citation_reference_accuracy
            ),

            # 正向问题是否至少引用了
            # 一个标准相关 Chunk
            "relevant_citation_hit_rate": (
                avg(
                    citation_hit_values
                )
            ),

            # 返回的 Citation
            # 是否全部属于标准相关 Chunk
            "all_citations_relevant_rate": (
                avg(
                    all_relevant_values
                )
            ),
        },

        "cases": case_results,
    }