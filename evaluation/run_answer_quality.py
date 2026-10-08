# -*- coding: utf-8 -*-
"""
端到端答案质量评测脚本（Answer Quality Evaluation）。

流程：
1. 复用 retrieval_dataset.json 的问题集；
2. 通过 HTTP SSE 调用已运行的后端，收集每条最终答案 answer
   与检索到的参考片段 contexts；
3. 调用 answer_quality_service（LLM 裁判）评估
   Faithfulness（忠实度）与 Answer Relevancy（相关性）；
4. 汇总指标，输出 JSON 与 Markdown 报告。

运行（在项目根）：
    python -m evaluation.run_answer_quality
或：
    python evaluation/run_answer_quality.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.answer_quality_service import (  # noqa: E402
    judge_answer_quality,
)


EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "retrieval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://localhost:8000"
STREAM_URL = f"{BASE_URL}/agent/graph-run-stream"


# ============================================================
# 通过 SSE 收集答案与上下文
# ============================================================

def collect_answer(question: str) -> dict[str, Any]:
    """调用流式接口，收集答案、参考片段、工具步骤。"""

    body = {"message": question, "max_steps": 5}

    answer_parts: list[str] = []
    contexts: list[str] = []
    tools: list[str] = []
    error_message: str | None = None

    with requests.post(
        STREAM_URL,
        json=body,
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()

        for raw in response.iter_lines():
            if not raw:
                continue

            line = raw.decode("utf-8")
            if not line.startswith("data:"):
                continue

            data = json.loads(line[5:].strip())
            kind = data.get("type")

            if kind == "answer_chunk":
                answer_parts.append(data.get("content", ""))

            elif kind == "citations":
                for citation in data.get("citations", []):
                    content = citation.get("content", "")
                    if content:
                        contexts.append(content)

            elif kind == "step":
                tools.append(
                    data.get("step", {}).get("tool_name")
                )

            elif kind == "error":
                error_message = data.get("message")

    return {
        "answer": "".join(answer_parts),
        "contexts": contexts,
        "tools": tools,
        "error": error_message,
    }


# ============================================================
# 汇总
# ============================================================

def _mean(values: list[Any]) -> float | None:
    numbers = [v for v in values if v is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


# ============================================================
# 报告
# ============================================================

def build_markdown_report(
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    """生成 Markdown 评测报告。"""

    lines: list[str] = []

    lines.append("# 端到端答案质量评测报告（Answer Quality）")
    lines.append("")
    lines.append("## 1. 评测配置")
    lines.append("")
    lines.append(f"- 评测用例数：{summary['total_cases']}")
    lines.append(
        f"- 评测指标：Faithfulness（忠实度）、"
        "Answer Relevancy（答案相关性）"
    )
    lines.append("- 裁判模型：DeepSeek（LLM-as-a-Judge，关闭 thinking）")
    lines.append(f"- 生成时间：{summary['generated_at']}")
    lines.append("")

    lines.append("## 2. 总体结果")
    lines.append("")
    lines.append("| 指标 | 结果 |")
    lines.append("|---|---:|")
    lines.append(
        f"| 平均忠实度（Faithfulness） | "
        f"{summary['avg_faithfulness_pct']} |"
    )
    lines.append(
        f"| 平均答案相关性（Relevancy） | "
        f"{summary['avg_relevancy_pct']} |"
    )
    lines.append(
        f"| 整体陈述支撑率（{summary['supported_claims']}/"
        f"{summary['total_claims']}） | "
        f"{summary['overall_faithfulness_pct']} |"
    )
    lines.append(
        f"| 答案全部可被支撑的用例占比 | "
        f"{summary['perfect_faithfulness_pct']} |"
    )
    lines.append("")

    lines.append("## 3. 逐条明细")
    lines.append("")
    lines.append(
        "| ID | 问题 | 忠实度 | 相关性 | "
        "支撑陈述 | 未支撑陈述数 |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|")

    for case in cases:
        faith = (
            f"{case['faithfulness'] * 100:.1f}%"
            if case["faithfulness"] is not None
            else "N/A"
        )
        rel = (
            f"{case['relevancy_score'] * 100:.1f}%"
            if case["relevancy_score"] is not None
            else "N/A"
        )
        unsupported = (
            case["total_claims"] - case["supported_claims"]
        )
        lines.append(
            f"| {case['id']} | {case['question']} | "
            f"{faith} | {rel} | "
            f"{case['supported_claims']}/{case['total_claims']} | "
            f"{unsupported} |"
        )

    lines.append("")

    # 列出所有未被支撑的陈述（潜在幻觉点）
    flagged: list[str] = []
    for case in cases:
        for claim in case.get("unsupported_claims", []):
            flagged.append(
                f"- [用例 {case['id']}] {claim}"
            )

    lines.append("## 4. 未被上下文支撑的陈述（潜在幻觉点）")
    lines.append("")
    if flagged:
        lines.extend(flagged)
    else:
        lines.append("无：所有答案陈述均能由检索上下文支撑。")
    lines.append("")

    lines.append("## 5. 结论")
    lines.append("")
    lines.append(
        "- Faithfulness 衡量答案是否忠实于检索上下文（越低幻觉越多）；"
        "Answer Relevancy 衡量答案是否切题。"
    )
    lines.append(
        "- 本报告与检索层评测（Recall@k / MRR）互补，"
        "共同构成「检索 → 生成」的完整质量视图。"
    )
    lines.append("")

    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    print(
        "\n========================================"
        "\n端到端答案质量评测开始"
        f"\n用例数：{len(dataset)}"
        "\n========================================"
    )

    cases: list[dict[str, Any]] = []

    for index, case in enumerate(dataset):
        question = case["question"]
        print(
            f"\n[{index + 1}/{len(dataset)}] {question}"
        )

        collected = collect_answer(question)

        judged = judge_answer_quality(
            question=question,
            answer=collected["answer"],
            contexts=collected["contexts"],
        )

        unsupported_claims = [
            claim["claim"]
            for claim in judged.get("claims", [])
            if not claim["supported"]
        ]

        case_result = {
            "id": case["id"],
            "question": question,
            "answer": collected["answer"],
            "context_count": len(collected["contexts"]),
            "tools": collected["tools"],
            "faithfulness": judged.get("faithfulness"),
            "relevancy_score": judged.get("relevancy_score"),
            "total_claims": judged.get("total_claims", 0),
            "supported_claims": judged.get(
                "supported_claims", 0
            ),
            "relevancy_reason": judged.get(
                "relevancy_reason", ""
            ),
            "unsupported_claims": unsupported_claims,
        }
        cases.append(case_result)

        faith = judged.get("faithfulness")
        rel = judged.get("relevancy_score")
        print(
            "  faithfulness=%s relevancy=%s claims=%d/%d"
            % (
                f"{faith:.2f}" if faith is not None else "N/A",
                f"{rel:.2f}" if rel is not None else "N/A",
                judged.get("supported_claims", 0),
                judged.get("total_claims", 0),
            )
        )

    # ---------------- 汇总 ----------------

    avg_faithfulness = _mean(
        [c["faithfulness"] for c in cases]
    )
    avg_relevancy = _mean(
        [c["relevancy_score"] for c in cases]
    )

    total_claims = sum(c["total_claims"] for c in cases)
    supported_claims = sum(
        c["supported_claims"] for c in cases
    )
    overall_faithfulness = (
        supported_claims / total_claims
        if total_claims > 0
        else None
    )

    perfect_count = sum(
        1
        for c in cases
        if c["faithfulness"] is not None
        and c["faithfulness"] >= 0.999
    )

    def _pct(value: float | None) -> str:
        return f"{value * 100:.2f}%" if value is not None else "N/A"

    generated_at = datetime.now().isoformat()

    summary = {
        "total_cases": len(cases),
        "avg_faithfulness": _round(avg_faithfulness),
        "avg_relevancy": _round(avg_relevancy),
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "overall_faithfulness": _round(
            overall_faithfulness
        ),
        "perfect_faithfulness_cases": perfect_count,
        "avg_faithfulness_pct": _pct(avg_faithfulness),
        "avg_relevancy_pct": _pct(avg_relevancy),
        "overall_faithfulness_pct": _pct(
            overall_faithfulness
        ),
        "perfect_faithfulness_pct": _pct(
            perfect_count / len(cases) if cases else None
        ),
        "generated_at": generated_at,
    }

    output = {
        "summary": summary,
        "cases": cases,
    }

    # ---------------- 保存 JSON ----------------

    json_path = RESULTS_DIR / "answer_quality.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ---------------- 保存 Markdown ----------------

    report = build_markdown_report(summary, cases)
    report_path = RESULTS_DIR / "answer_quality_report.md"
    report_path.write_text(
        report, encoding="utf-8"
    )

    print(
        "\n========================================"
        "\n端到端答案质量评测结果"
        "\n========================================"
    )
    print(f"平均忠实度：{summary['avg_faithfulness_pct']}")
    print(f"平均相关性：{summary['avg_relevancy_pct']}")
    print(
        "整体陈述支撑率："
        f"{summary['overall_faithfulness_pct']}"
        f"（{supported_claims}/{total_claims}）"
    )
    print(f"\nJSON：{json_path}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
