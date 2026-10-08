# -*- coding: utf-8 -*-
"""
检索质量评测脚本

对四个检索阶段分别计算 Recall@k / Precision@k / MRR / nDCG@k：
    1. vector                向量检索（MiniLM）
    2. keyword               关键词检索（jieba + BM25）
    3. rrf_fusion            向量 + 关键词 RRF 融合
    4. cross_encoder_rerank  RRF + Cross-Encoder 重排

运行（在项目根）：
    python evaluation/retrieval_eval.py

结果保存：
    evaluation/results/retrieval_evaluation.json
"""
import json
import math
import os
import sys
import time

# 让脚本可直接运行时找到项目根
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from services.vector_service import (  # noqa: E402
    search_similar_chunks,
    keyword_search_chunks,
)
from services.advanced_rag_service import (  # noqa: E402
    reciprocal_rank_fusion,
    rerank_chunks,
)

DATASET_PATH = os.path.join(BASE_DIR, "retrieval_dataset.json")
RESULT_PATH = os.path.join(BASE_DIR, "results", "retrieval_evaluation.json")

STAGES = [
    "vector",
    "keyword",
    "rrf_fusion",
    "cross_encoder_rerank",
]
CANDIDATE_POOL = 10
EVAL_KS = [3, 5]


def chunk_index_of(chunk_id):
    """file_{file_id}_chunk_{index} -> chunk index"""
    try:
        return int(str(chunk_id).split("_chunk_")[-1])
    except (ValueError, IndexError):
        return None


def recall_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def precision_at_k(ranked, gold, k):
    top = ranked[:k]
    if not top:
        return 0.0
    return len(set(top) & set(gold)) / len(top)


def mrr_at_k(ranked, gold, k):
    for i, cid in enumerate(ranked[:k]):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, gold, k):
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, cid in enumerate(ranked[:k])
        if cid in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def run_stages(question):
    vec = search_similar_chunks(
        question, file_id=None, top_k=CANDIDATE_POOL
    )
    kw = keyword_search_chunks(
        question, file_id=None, top_k=CANDIDATE_POOL
    )

    groups = [
        {"query": question, "retriever": "vector", "results": vec},
        {"query": question, "retriever": "keyword", "results": kw},
    ]
    fused = reciprocal_rank_fusion(
        groups, final_top_k=CANDIDATE_POOL
    )
    reranked = rerank_chunks(
        question, fused, final_top_k=CANDIDATE_POOL
    )

    def to_indices(items):
        return [
            idx
            for idx in (
                chunk_index_of(x.get("chunk_id")) for x in items
            )
            if idx is not None
        ]

    return {
        "vector": to_indices(vec),
        "keyword": to_indices(kw),
        "rrf_fusion": to_indices(fused),
        "cross_encoder_rerank": to_indices(reranked),
    }


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    raw = {
        stage: {
            k: {"recall": [], "precision": [], "mrr": [], "ndcg": []}
            for k in EVAL_KS
        }
        for stage in STAGES
    }
    per_question = []

    t0 = time.time()
    for item in dataset:
        question = item["question"]
        gold = set(item["gold_chunks"])
        ranked_by_stage = run_stages(question)

        qrec = {
            "id": item["id"],
            "question": question,
            "gold": sorted(gold),
            "stages": {},
        }
        for stage in STAGES:
            ranked = ranked_by_stage[stage]
            qrec["stages"][stage] = ranked
            for k in EVAL_KS:
                raw[stage][k]["recall"].append(
                    recall_at_k(ranked, gold, k)
                )
                raw[stage][k]["precision"].append(
                    precision_at_k(ranked, gold, k)
                )
                raw[stage][k]["mrr"].append(
                    mrr_at_k(ranked, gold, k)
                )
                raw[stage][k]["ndcg"].append(
                    ndcg_at_k(ranked, gold, k)
                )
        per_question.append(qrec)

    def avg(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {}
    for stage in STAGES:
        summary[stage] = {}
        for k in EVAL_KS:
            bucket = raw[stage][k]
            summary[stage]["@%d" % k] = {
                "recall": avg(bucket["recall"]),
                "precision": avg(bucket["precision"]),
                "mrr": avg(bucket["mrr"]),
                "ndcg": avg(bucket["ndcg"]),
            }

    result = {
        "num_questions": len(dataset),
        "elapsed_seconds": round(time.time() - t0, 2),
        "summary": summary,
        "per_question": per_question,
    }

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("questions=%d  elapsed=%.1fs" % (
        len(dataset), result["elapsed_seconds"]
    ))
    for k in EVAL_KS:
        print("\n===== @%d =====" % k)
        print("%-24s %8s %9s %7s %7s" % (
            "stage", "recall", "precision", "mrr", "ndcg"
        ))
        for stage in STAGES:
            m = summary[stage]["@%d" % k]
            print("%-24s %8.3f %9.3f %7.3f %7.3f" % (
                stage,
                m["recall"],
                m["precision"],
                m["mrr"],
                m["ndcg"],
            ))

    print("\nsaved:", RESULT_PATH)


if __name__ == "__main__":
    main()
