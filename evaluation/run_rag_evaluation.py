import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import settings

from services.rag_evaluation_service import (
    evaluate_dataset,
    compare_retrieval_variants,
    evaluate_rerank_stability,
    evaluate_citations,
)


# ============================================================
# 路径配置
# ============================================================

EVALUATION_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    EVALUATION_DIR
    / "rag_eval_dataset.json"
)

RESULTS_DIR = (
    EVALUATION_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# JSON 保存
# ============================================================

def save_json_result(
    filename: str,
    data: dict[str, Any],
) -> Path:
    """
    把 Evaluation Result 保存成 JSON。

    snapshot
        = 快照

    这样评测结果不会只存在 Terminal 中。
    """

    output_path = (
        RESULTS_DIR
        / filename
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return output_path


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    """
    运行完整 RAG Evaluation Pipeline。

    包括：

    1. Base Evaluation
    2. Retrieval Variant Comparison
    3. Rerank Stability
    4. Citation Evaluation
    """

    dataset_path = str(
        DATASET_PATH
    )

    print(
        "\n"
        "========================================"
    )
    print(
        "RAG Evaluation 开始"
    )
    print(
        "========================================"
    )

    # ========================================================
    # 1. Base Evaluation
    # ========================================================

    print(
        "\n[1/4] Base Evaluation..."
    )

    base_result = (
        evaluate_dataset(
            dataset_path
        )
    )

    base_path = save_json_result(
        "base_evaluation.json",
        base_result,
    )

    print(
        f"已保存：{base_path}"
    )

    # ========================================================
    # 2. Retrieval Comparison
    # ========================================================

    print(
        "\n[2/4] Retrieval Comparison..."
    )

    retrieval_result = (
        compare_retrieval_variants(
            dataset_path,
            top_k=3,
        )
    )

    retrieval_path = save_json_result(
        "retrieval_comparison.json",
        retrieval_result,
    )

    print(
        f"已保存：{retrieval_path}"
    )

    # ========================================================
    # 3. Rerank Stability
    # ========================================================

    print(
        "\n[3/4] Rerank Stability..."
    )

    stability_result = (
        evaluate_rerank_stability(
            dataset_path,
            repeat_count=2,
        )
    )

    stability_path = save_json_result(
        "rerank_stability.json",
        stability_result,
    )

    print(
        f"已保存：{stability_path}"
    )

    # ========================================================
    # 4. Citation Evaluation
    # ========================================================

    print(
        "\n[4/4] Citation Evaluation..."
    )

    citation_result = (
        evaluate_citations(
            dataset_path
        )
    )

    citation_path = save_json_result(
        "citation_evaluation.json",
        citation_result,
    )

    print(
        f"已保存：{citation_path}"
    )

    # ========================================================
    # 5. 保存本次实验元信息
    # ========================================================

    metadata = {
        "generated_at": (
            datetime.now().isoformat()
        ),

        "dataset_path": (
            str(DATASET_PATH)
        ),

        "rerank_threshold": (
            settings.rag_rerank_min_score
        ),

        "retrieval_top_k": 3,

        "stability_repeat_count": 2,
    }

    metadata_path = save_json_result(
        "evaluation_metadata.json",
        metadata,
    )

    print(
        f"已保存：{metadata_path}"
    )

    print(
        "\n"
        "========================================"
    )
    print(
        "RAG Evaluation 全部完成"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()