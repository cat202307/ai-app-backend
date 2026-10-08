# -*- coding: utf-8 -*-
"""
拒答压力测试（Abstention / Hallucination Stress Test）。

对一组"文档中确定没有答案"的问题，验证系统是否：
- 不编造信息；
- 恰当地说明无法从所提供的文档中回答。

指标：拒答正确率（Abstention Accuracy）。

运行（在项目根）：
    python -m evaluation.run_abstention_test
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.answer_quality_service import (  # noqa: E402
    judge_abstention,
)


EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "abstention_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STREAM_URL = (
    "http://localhost:8000/agent/graph-run-stream"
)


def collect(question: str) -> dict[str, Any]:
    """SSE 收集答案、上下文、工具。"""

    body = {"message": question, "max_steps": 5}
    answer_parts: list[str] = []
    contexts: list[str] = []
    tools: list[str] = []

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
                    if citation.get("content"):
                        contexts.append(citation["content"])
            elif kind == "step":
                tools.append(
                    data.get("step", {}).get("tool_name")
                )

    return {
        "answer": "".join(answer_parts),
        "contexts": contexts,
        "tools": tools,
    }


def build_report(
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    lines: list[str] = []

    lines.append("# 拒答压力测试报告（Abstention Test）")
    lines.append("")
    lines.append("## 1. 配置")
    lines.append("")
    lines.append(f"- 文档外问题数：{summary['total_cases']}")
    lines.append(
        "- 判定方式：LLM 裁判判断系统是否未编造、"
        "并恰当说明无法从文档回答"
    )
    lines.append(f"- 生成时间：{summary['generated_at']}")
    lines.append("")

    lines.append("## 2. 总体结果")
    lines.append("")
    lines.append("| 指标 | 结果 |")
    lines.append("|---|---:|")
    lines.append(
        f"| 拒答正确率（Abstention Accuracy） | "
        f"{summary['abstention_accuracy_pct']} |"
    )
    lines.append(
        f"| 正确拒答数 | {summary['correct_count']}/"
        f"{summary['total_cases']} |"
    )
    lines.append("")

    lines.append("## 3. 逐条结果")
    lines.append("")
    lines.append(
        "| ID | 问题 | 是否正确拒答 | 检索片段数 | 系统回答（节选） |"
    )
    lines.append("|---:|---|:---:|---:|---|")

    for case in cases:
        result = case["abstained_correctly"]
        if result is True:
            mark = "是"
        elif result is False:
            mark = "否"
        else:
            mark = "N/A"
        answer_preview = case["answer"][:80].replace(
            "\n", " "
        )
        lines.append(
            f"| {case['id']} | {case['question']} | "
            f"{mark} | {case['context_count']} | "
            f"{answer_preview} |"
        )

    lines.append("")

    fabrications = [
        f"- [用例 {c['id']}] {c['fabricated_info']}"
        for c in cases
        if c.get("fabricated_info")
    ]

    lines.append("## 4. 系统编造的信息（应为空）")
    lines.append("")
    if fabrications:
        lines.extend(fabrications)
    else:
        lines.append("无：系统对文档外问题均未编造信息。")
    lines.append("")

    lines.append("## 5. 结论")
    lines.append("")
    lines.append(
        "该测试与「文档内答案质量评测」互补：前者检验无答案时不"
        "乱答（拒答），后者检验有答案时忠实、切题。"
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    with DATASET_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        dataset = json.load(file)

    print(
        "\n========================================"
        "\n拒答压力测试开始"
        f"\n问题数：{len(dataset)}"
        "\n========================================"
    )

    cases: list[dict[str, Any]] = []

    for index, item in enumerate(dataset):
        question = item["question"]
        print(
            f"\n[{index + 1}/{len(dataset)}] {question}"
        )

        collected = collect(question)

        judged = judge_abstention(
            question=question,
            answer=collected["answer"],
            contexts=collected["contexts"],
        )

        case_result = {
            "id": item["id"],
            "question": question,
            "answer": collected["answer"],
            "context_count": len(collected["contexts"]),
            "tools": collected["tools"],
            "abstained_correctly": judged.get(
                "abstained_correctly"
            ),
            "fabricated_info": judged.get(
                "fabricated_info", ""
            ),
            "reason": judged.get("reason", ""),
        }
        cases.append(case_result)

        print(
            "  abstained_correctly=%s"
            % judged.get("abstained_correctly")
        )

    correct_count = sum(
        1
        for c in cases
        if c["abstained_correctly"] is True
    )
    total = len(cases)
    accuracy = correct_count / total if total else None

    generated_at = datetime.now().isoformat()

    def _pct(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"

    summary = {
        "total_cases": total,
        "correct_count": correct_count,
        "abstention_accuracy": accuracy,
        "abstention_accuracy_pct": _pct(accuracy),
        "generated_at": generated_at,
    }

    output = {
        "summary": summary,
        "cases": cases,
    }

    json_path = RESULTS_DIR / "abstention_test.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    report = build_report(summary, cases)
    report_path = RESULTS_DIR / "abstention_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(
        "\n========================================"
        "\n拒答压力测试结果"
        "\n========================================"
    )
    print(
        f"拒答正确率：{summary['abstention_accuracy_pct']}"
        f"（{correct_count}/{total}）"
    )
    print(f"JSON：{json_path}")
    print(f"报告：{report_path}")


if __name__ == "__main__":
    main()
