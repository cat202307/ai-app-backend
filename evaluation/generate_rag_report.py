import json
from pathlib import Path
from typing import Any


EVALUATION_DIR = (
    Path(__file__).resolve().parent
)

RESULTS_DIR = (
    EVALUATION_DIR
    / "results"
)


def load_json(
    filename: str,
) -> dict[str, Any]:
    """
    读取 Evaluation Snapshot。
    """

    path = (
        RESULTS_DIR
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"评测结果不存在：{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def percent(
    value: float | None,
) -> str:
    """
    例如：

        1.0
        ->
        100.00%
    """

    if value is None:
        return "-"

    return (
        f"{value * 100:.2f}%"
    )


def main() -> None:

    # ============================================================
    # 1. 加载评测结果
    # ============================================================

    base = load_json(
        "base_evaluation.json"
    )

    retrieval = load_json(
        "retrieval_comparison.json"
    )

    stability = load_json(
        "rerank_stability.json"
    )

    citation = load_json(
        "citation_evaluation.json"
    )

    metadata = load_json(
        "evaluation_metadata.json"
    )

    # ============================================================
    # 2. 提取基础数据
    # ============================================================

    base_summary = (
        base["summary"]
    )

    variants = (
        retrieval["variants"]
    )

    worst_scores = (
        stability[
            "worst_case_scores"
        ]
    )

    citation_summary = (
        citation["summary"]
    )

    top_k = retrieval[
        "top_k"
    ]

    # ============================================================
    # 3. 构造 Markdown
    # ============================================================

    lines = []

    lines.append(
        "# RAG Evaluation Report"
    )

    lines.append("")

    lines.append(
        "## 1. Evaluation Configuration"
    )

    lines.append("")

    lines.append(
        f"- Dataset Cases："
        f"{base_summary['total_cases']}"
    )

    lines.append(
        f"- Retrieval Top K：{top_k}"
    )

    lines.append(
        f"- Rerank Threshold："
        f"{metadata['rerank_threshold']}"
    )

    lines.append(
        f"- Stability Repeat Count："
        f"{metadata['stability_repeat_count']}"
    )

    lines.append(
        f"- Generated At："
        f"{metadata['generated_at']}"
    )

    # ============================================================
    # 4. Base Evaluation
    # ============================================================

    lines.append("")

    lines.append(
        "## 2. Base Evaluation"
    )

    lines.append("")

    lines.append(
        "| Metric | Result |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        "| Answerability Accuracy | "
        f"{percent(base_summary['answerability_accuracy'])} |"
    )

    lines.append(
        "| Hit Rate | "
        f"{percent(base_summary['hit_rate'])} |"
    )

    lines.append(
        "| Mean Recall | "
        f"{percent(base_summary['mean_recall'])} |"
    )

    lines.append(
        "| MRR | "
        f"{base_summary['mrr']:.4f} |"
    )

    lines.append(
        "| Rerank Top1 Accuracy | "
        f"{percent(base_summary['rerank_top1_accuracy'])} |"
    )

    # ============================================================
    # 5. Retrieval Comparison
    # ============================================================

    lines.append("")

    lines.append(
        "## 3. Retrieval Variant Comparison"
    )

    lines.append("")

    lines.append(
        "| Variant | "
        f"Hit@{top_k} | "
        f"Recall@{top_k} | "
        "MRR | "
        "Top1 Accuracy |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    variant_order = [
        "basic_vector",
        "multi_query",
        "hybrid",
        "hybrid_rerank",
    ]

    for variant_name in variant_order:

        summary = (
            variants[
                variant_name
            ][
                "summary"
            ]
        )

        hit_key = (
            f"hit_at_{top_k}"
        )

        recall_key = (
            f"recall_at_{top_k}"
        )

        lines.append(
            f"| {variant_name} "
            f"| {percent(summary[hit_key])} "
            f"| {percent(summary[recall_key])} "
            f"| {summary['mrr']:.4f} "
            f"| {percent(summary['top1_accuracy'])} |"
        )

    # ============================================================
    # 6. Rerank Stability
    # ============================================================

    lines.append("")

    lines.append(
        "## 4. Rerank Stability"
    )

    lines.append("")

    lines.append(
        "| Metric | Result |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        "| Worst Positive Score | "
        f"{worst_scores['worst_positive_score']} |"
    )

    lines.append(
        "| Worst Negative Score | "
        f"{worst_scores['worst_negative_score']} |"
    )

    lines.append(
        "| Robust Margin | "
        f"{worst_scores['robust_margin']} |"
    )

    lines.append(
        "| Configured Threshold | "
        f"{metadata['rerank_threshold']} |"
    )

    # ============================================================
    # 7. Citation Evaluation
    # ============================================================

    lines.append("")

    lines.append(
        "## 5. Citation Evaluation"
    )

    lines.append("")

    lines.append(
        "| Metric | Result |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        "| Citation Presence Accuracy | "
        f"{percent(citation_summary['citation_presence_accuracy'])} |"
    )

    lines.append(
        "| Citation Reference Accuracy | "
        f"{percent(citation_summary['citation_reference_accuracy'])} |"
    )

    lines.append(
        "| Relevant Citation Hit Rate | "
        f"{percent(citation_summary['relevant_citation_hit_rate'])} |"
    )

    lines.append(
        "| All Citations Relevant Rate | "
        f"{percent(citation_summary['all_citations_relevant_rate'])} |"
    )

    # ============================================================
    # 8. 项目结论
    # ============================================================

    lines.append("")

    lines.append(
        "## 6. Conclusion"
    )

    lines.append("")

    lines.append(
        "本项目构建了完整的 Advanced RAG Pipeline，"
        "包括 Query Rewrite、Multi-Query Retrieval、"
        "Vector Search、Keyword Search、Hybrid Search、"
        "RRF、Rerank、Answerability Threshold、"
        "Citation 和 Citation Pruning。"
    )

    lines.append("")

    lines.append(
        "通过 Evaluation Dataset、Hard Cases、"
        "Retrieval Variant Comparison、Threshold Sweep、"
        "Rerank Stability 和 Citation Evaluation，"
        "对 RAG 系统进行了量化评测和稳定性验证。"
    )

    # ============================================================
    # 9. 保存 Markdown
    # ============================================================

    output_path = (
        RESULTS_DIR
        / "rag_evaluation_report.md"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )

    print(
        f"Evaluation Report 已生成："
        f"{output_path}"
    )


if __name__ == "__main__":
    main()