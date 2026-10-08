"""
Agent Evaluation 评测脚本。

评测指标：
1. 工具选择准确率（Tool Selection Accuracy）
2. 任务完成率（Task Completion Rate）
3. 平均步数、平均工具调用数
4. 平均 Token 消耗、平均耗时
5. 逐 case 详细结果
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from database import SessionLocal
from agents.graph_agent import run_agent_graph


# ============================================================
# 路径配置
# ============================================================

EVALUATION_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    EVALUATION_DIR
    / "agent_eval_dataset.json"
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
    """把评测结果保存成 JSON。"""

    output_path = RESULTS_DIR / filename

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
# 单 case 评测
# ============================================================

def evaluate_single_case(
    case: dict[str, Any],
    db,
) -> dict[str, Any]:
    """
    评测单个 Agent 用例。
    """

    case_id = case["case_id"]
    user_message = case["user_message"]
    expected_tools = set(case["expected_tools"])
    expected_completed = case["expected_completed"]

    try:
        result = run_agent_graph(
            user_message=user_message,
            db=db,
            max_steps=5,
        )

        # 实际调用的工具集合
        actual_tools = set(
            trace["tool_name"]
            for trace in result.get("traces", [])
        )

        # 工具选择是否匹配
        tools_match = (
            actual_tools == expected_tools
        )

        # 任务是否完成
        completed = result.get("completed", False)
        completion_match = (
            completed == expected_completed
        )

        return {
            "case_id": case_id,
            "user_message": user_message,
            "expected_tools": sorted(expected_tools),
            "actual_tools": sorted(actual_tools),
            "tools_match": tools_match,
            "completed": completed,
            "completion_match": completion_match,
            "steps": result.get("steps", 0),
            "tool_call_count": result.get("tool_call_count", 0),
            "total_tokens": result.get(
                "usage", {}
            ).get("total_tokens", 0),
            "total_duration_ms": result.get(
                "timing", {}
            ).get("total_duration_ms", 0),
            "answer_preview": (
                result.get("answer", "")[:200]
            ),
            "error": None,
        }

    except Exception as exc:
        return {
            "case_id": case_id,
            "user_message": user_message,
            "expected_tools": sorted(expected_tools),
            "actual_tools": [],
            "tools_match": False,
            "completed": False,
            "completion_match": False,
            "steps": 0,
            "tool_call_count": 0,
            "total_tokens": 0,
            "total_duration_ms": 0,
            "answer_preview": "",
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


# ============================================================
# 汇总统计
# ============================================================

def summarize_results(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    汇总所有 case 的评测指标。
    """

    total = len(case_results)

    # 工具选择准确的数量
    tools_correct = sum(
        1 for r in case_results if r["tools_match"]
    )

    # 任务完成且符合预期的数量
    completion_correct = sum(
        1 for r in case_results if r["completion_match"]
    )

    # 成功完成的数量
    completed_count = sum(
        1 for r in case_results if r["completed"]
    )

    # 报错的数量
    error_count = sum(
        1 for r in case_results if r["error"]
    )

    # 平均值
    avg_steps = (
        sum(r["steps"] for r in case_results) / total
        if total > 0 else 0
    )

    avg_tool_calls = (
        sum(r["tool_call_count"] for r in case_results) / total
        if total > 0 else 0
    )

    avg_tokens = (
        sum(r["total_tokens"] for r in case_results) / total
        if total > 0 else 0
    )

    avg_duration_ms = (
        sum(r["total_duration_ms"] for r in case_results) / total
        if total > 0 else 0
    )

    return {
        "total_cases": total,
        "tool_selection_accuracy": (
            tools_correct / total
            if total > 0 else 0
        ),
        "completion_accuracy": (
            completion_correct / total
            if total > 0 else 0
        ),
        "completed_count": completed_count,
        "error_count": error_count,
        "avg_steps": round(avg_steps, 2),
        "avg_tool_calls": round(avg_tool_calls, 2),
        "avg_tokens": round(avg_tokens, 1),
        "avg_duration_ms": round(avg_duration_ms, 1),
        "tools_correct": tools_correct,
        "completion_correct": completion_correct,
    }


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    """
    运行 Agent Evaluation。
    """

    print(
        "\n"
        "========================================"
    )
    print(
        "Agent Evaluation 开始"
    )
    print(
        "========================================"
    )

    # 加载数据集
    with DATASET_PATH.open(
        "r", encoding="utf-8"
    ) as f:
        dataset = json.load(f)

    print(
        f"加载数据集：{len(dataset)} 条"
    )

    # 创建数据库 session
    db = SessionLocal()

    try:
        # 逐 case 评测
        case_results = []

        for i, case in enumerate(dataset):
            print(
                f"\n[{i+1}/{len(dataset)}] "
                f"评测 {case['case_id']}..."
            )
            print(
                f"  问题：{case['user_message'][:50]}"
            )

            result = evaluate_single_case(
                case=case,
                db=db,
            )

            case_results.append(result)

            status = "✅" if result["tools_match"] else "❌"
            print(
                f"  {status} 工具匹配：{result['tools_match']}"
            )
            print(
                f"  步数：{result['steps']}, "
                f"工具数：{result['tool_call_count']}"
            )
            if result["error"]:
                print(
                    f"  ⚠️ 错误：{result['error'][:100]}"
                )

        # 汇总统计
        summary = summarize_results(case_results)

        # 保存结果
        output = {
            "summary": summary,
            "cases": case_results,
            "generated_at": datetime.now().isoformat(),
        }

        output_path = save_json_result(
            "agent_evaluation.json",
            output,
        )

        # 打印汇总
        print(
            "\n"
            "========================================"
        )
        print(
            "Agent Evaluation 结果汇总"
        )
        print(
            "========================================"
        )
        print(
            f"总用例数：{summary['total_cases']}"
        )
        print(
            f"工具选择准确率："
            f"{summary['tool_selection_accuracy']:.1%}"
            f" ({summary['tools_correct']}/{summary['total_cases']})"
        )
        print(
            f"完成率："
            f"{summary['completion_accuracy']:.1%}"
            f" ({summary['completion_correct']}/{summary['total_cases']})"
        )
        print(
            f"平均步数：{summary['avg_steps']}"
        )
        print(
            f"平均工具调用数：{summary['avg_tool_calls']}"
        )
        print(
            f"平均 Token：{summary['avg_tokens']}"
        )
        print(
            f"平均耗时：{summary['avg_duration_ms']} ms"
        )
        print(
            f"\n结果已保存：{output_path}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
