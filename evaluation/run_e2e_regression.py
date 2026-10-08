# -*- coding: utf-8 -*-
"""
端到端回归测试（通过 HTTP /agent/graph-run）。

覆盖：
  1. 具体技术点（父子块修复）：优化器、损失函数
  2. 全局 / 单文档概述
  3. 多轮追问指代消解
  4. 无关问题稳定拒答
  5. 文件元信息（验证纠偏不误伤真正的列表请求）

运行：
  python evaluation/run_e2e_regression.py
"""
import json
import os
import sys
import time
import urllib.request

BASE_URL = os.environ.get(
    "E2E_BASE_URL", "http://localhost:8000"
)

# 让控制台在 Windows 下正确输出中文
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def graph_run(message, history=None, max_steps=5):
    payload = {
        "message": message,
        "max_steps": max_steps,
        "history": history or [],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/agent/graph-run",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["data"]


# (用例名, 问题, history, 判定函数 answer->bool, 期望说明)
def contains(*words):
    def check(answer):
        return all(w in answer for w in words)
    return check


def is_abstention(answer):
    return "没有找到" in answer


def not_abstention(answer):
    return "没有找到" not in answer and len(answer) > 30


CASES = [
    (
        "技术点-优化器",
        "论文用了什么优化器？",
        None,
        contains("AdamW"),
        "答 AdamW",
    ),
    (
        "技术点-损失函数",
        "论文的损失函数是什么？",
        None,
        contains("MSE"),
        "答 MSE",
    ),
    (
        "单文档概述",
        "这篇论文主要讲什么？",
        None,
        not_abstention,
        "给出概述、非拒答",
    ),
    (
        "跨文档对比",
        "论文和test02有什么区别",
        None,
        lambda answer: (
            "没有找到" not in answer
            and "test02" in answer
            and (
                "调度" in answer
                or "Kubernetes" in answer
            )
            and (
                "RAG" in answer
                or "检索" in answer
            )
        ),
        "同时覆盖两个文档并对比",
    ),
    (
        "多轮追问-学习率",
        "学习率设置为多少？",
        [
            {"role": "user", "content": "论文用了什么优化器？"},
            {
                "role": "assistant",
                "content": "论文使用 AdamW 优化器，初始学习率 lr=1e-3。",
            },
        ],
        contains("1e-3"),
        "指代消解后答 1e-3",
    ),
    (
        "多轮追问-用途",
        "有什么用",
        [
            {"role": "user", "content": "test02文档主要讲了哪些内容？"},
            {
                "role": "assistant",
                "content": (
                    "本文主要介绍 FastAPI（Python Web 框架）、"
                    "RAG（检索增强生成）与 Hybrid Search（混合检索）。"
                ),
            },
        ],
        lambda answer: (
            "没有找到" not in answer
            and len(answer) > 30
            and (
                "API" in answer
                or "接口" in answer
                or "检索" in answer
            )
        ),
        "指代具体对象后答用途、非拒答",
    ),
    (
        "多轮追问-对象用途",
        "它有什么用",
        [
            {"role": "user", "content": "论文用了什么优化器？"},
            {
                "role": "assistant",
                "content": (
                    "论文采用 AdamW 优化器，初始学习率 1e-3，"
                    "权重衰减 1e-4。"
                ),
            },
        ],
        lambda answer: (
            "没有找到" not in answer
            and "无法回答" not in answer
            and any(
                k in answer
                for k in ("优化", "参数", "损失", "训练")
            )
        ),
        "依据对象角色答用途、非拒答",
    ),
    (
        "无关问题拒答",
        "今天天气怎么样？",
        None,
        is_abstention,
        "稳定拒答",
    ),
    (
        "无上下文-省略追问引导",
        "有什么用",
        None,
        lambda answer: (
            ("缺少上下文" in answer or "具体指的是什么" in answer)
            and "没有找到" not in answer
        ),
        "无历史时引导补充对象、非拒答",
    ),
    (
        "元信息-文件列表",
        "系统里有哪些文件？",
        None,
        contains("test02"),
        "列出文件、不被纠偏误伤",
    ),
]


def main():
    results = []
    passed = 0

    for name, question, history, checker, expect in CASES:
        started = time.time()
        try:
            data = graph_run(question, history=history)
            answer = data.get("answer") or ""
            tools = [
                t.get("tool_name")
                for t in data.get("traces", [])
            ]
            ok = checker(answer)
            error = None
        except Exception as exc:  # noqa: BLE001
            answer = ""
            tools = []
            ok = False
            error = "%s: %s" % (
                type(exc).__name__,
                exc,
            )

        elapsed = round(time.time() - started, 1)
        if ok:
            passed += 1

        results.append(
            {
                "name": name,
                "question": question,
                "expect": expect,
                "tools": tools,
                "answer": answer,
                "passed": ok,
                "elapsed_s": elapsed,
                "error": error,
            }
        )

        mark = "PASS" if ok else "FAIL"
        print(
            "[%s] %s | tools=%s | %ss"
            % (mark, name, ",".join(tools) or "-", elapsed)
        )
        if error:
            print("       error: " + error)

    print(
        "\n==== %d / %d passed ====" % (passed, len(CASES))
    )

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, "e2e_regression.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": len(CASES),
                "passed": passed,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("saved:", out_path)

    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
