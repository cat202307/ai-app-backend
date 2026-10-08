# -*- coding: utf-8 -*-
"""
Answer Quality Evaluation Service
端到端答案质量评测服务（LLM-as-a-Judge）。

用 DeepSeek 作为裁判，对 RAG / Agent 生成的最终答案评估两项指标：

1. Faithfulness（忠实度）
   把答案拆成原子事实陈述，逐条判断是否被检索上下文支撑，
   衡量模型有没有编造（幻觉）。

2. Answer Relevancy（答案相关性）
   答案是否切中并完整回应用户问题，衡量有没有答非所问或冗余。

注意：裁判只依据给定上下文，不使用自身外部知识。
"""

import json
from typing import Any

from core.config import settings
from services.llm_service import get_llm_client


# ============================================================
# Prompt 构建
# ============================================================

JUDGE_SYSTEM_PROMPT = """你是一名严格、客观的 RAG 系统评测专家。
你将收到三部分内容：用户问题、系统给出的答案、系统检索到的若干参考上下文片段。

你只能依据给定的参考上下文做判断，不要使用你自己的外部知识。

请完成两项评估：

一、忠实度（Faithfulness）
1. 把"系统答案"拆解成若干条最小的、可独立验证的事实陈述（claims）。
2. 逐条判断该陈述能否由参考上下文直接推出或明确支持：
   - 只有上下文明确包含该信息，才标记 supported=true；
   - 上下文未提及、需要外部知识、或与上下文矛盾的，标记 supported=false。
3. faithfulness = supported 为 true 的陈述数 / 总陈述数。

二、答案相关性（Answer Relevancy）
评估答案是否切中用户问题、是否完整回应、有没有无关冗余或答非所问。
给出 0 到 1 的分数：1 表示完全切题且完整，0 表示完全无关。

只输出一个 JSON 对象，不要输出任何解释性文字，也不要使用 Markdown 代码块。
格式必须严格如下：
{
  "claims": [
    {"claim": "陈述内容", "supported": true}
  ],
  "faithfulness": 0.0,
  "relevancy_score": 0.0,
  "relevancy_reason": "简短说明"
}"""


def build_judge_messages(
    question: str,
    answer: str,
    contexts: list[str],
) -> list[dict[str, str]]:
    """构建裁判消息。"""

    numbered_context = "\n\n".join(
        f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
    )

    if not numbered_context:
        numbered_context = "（系统未检索到任何参考上下文）"

    user_content = (
        f"【用户问题】\n{question}\n\n"
        f"【系统答案】\n{answer}\n\n"
        f"【参考上下文】\n{numbered_context}"
    )

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ============================================================
# JSON 解析
# ============================================================

def _parse_judge_json(text: str) -> dict[str, Any] | None:
    """从裁判输出中提取 JSON 对象。"""

    if not text:
        return None

    cleaned = text.strip()

    # 兼容模型偶尔包裹 ```json ... ```
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                cleaned = candidate
                break

    # 直接截取最外层花括号
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    snippet = cleaned[start : end + 1]

    try:
        return json.loads(snippet)
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_result(
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """归一化裁判结果，并用 claims 现算 faithfulness。"""

    raw_claims = parsed.get("claims", [])
    claims: list[dict[str, Any]] = []

    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim_text = item.get("claim")
        if not claim_text:
            continue
        claims.append(
            {
                "claim": str(claim_text),
                "supported": bool(item.get("supported")),
            }
        )

    total_claims = len(claims)
    supported_claims = sum(1 for c in claims if c["supported"])

    if total_claims > 0:
        faithfulness = supported_claims / total_claims
    else:
        faithfulness = None

    def _to_unit_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number < 0:
            return 0.0
        if number > 1:
            return 1.0
        return number

    relevancy_score = _to_unit_float(
        parsed.get("relevancy_score")
    )

    return {
        "faithfulness": faithfulness,
        "relevancy_score": relevancy_score,
        "claims": claims,
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "relevancy_reason": str(
            parsed.get("relevancy_reason", "")
        ),
    }


# ============================================================
# 裁判调用
# ============================================================

def judge_answer_quality(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    """
    评估单条答案质量。

    返回字段：
        faithfulness, relevancy_score, claims,
        total_claims, supported_claims, relevancy_reason
    解析失败时相关评分为 None，并带 error / raw。
    """

    client = get_llm_client()
    messages = build_judge_messages(
        question=question,
        answer=answer,
        contexts=contexts,
    )

    parsed: dict[str, Any] | None = None
    raw_text = ""

    # 最多尝试两次（第二次强调只输出 JSON）
    for attempt in range(2):
        call_messages = messages
        if attempt == 1:
            call_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "你的上一次回复无法被解析。"
                        "请只输出一个合法的 JSON 对象，"
                        "不要输出任何其他文字或代码块。"
                    ),
                }
            ]

        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=call_messages,
            temperature=0.0,
            timeout=settings.llm_timeout,
            extra_body={
                "thinking": {"type": "disabled"}
            },
        )

        raw_text = response.choices[0].message.content or ""
        parsed = _parse_judge_json(raw_text)

        if parsed is not None:
            break

    if parsed is None:
        return {
            "faithfulness": None,
            "relevancy_score": None,
            "claims": [],
            "total_claims": 0,
            "supported_claims": 0,
            "relevancy_reason": "",
            "error": "judge_output_parse_failed",
            "raw": raw_text[:1000],
        }

    return _normalize_result(parsed)


# ============================================================
# 拒答判定（文档中没有答案时，系统是否正确拒答）
# ============================================================

ABSTENTION_SYSTEM_PROMPT = """你是一名严格、客观的 RAG 系统评测专家。
用户向一个"文档问答系统"提问，而这个问题可能在系统文档中没有答案。
你将看到：用户问题、系统检索到的参考上下文（可能为空或不相关）、系统给出的回答。

请判断系统是否"正确拒答"：
- abstained_correctly=true：系统没有编造文档中不存在的具体信息，并恰当地说明无法从所提供的文档中找到或回答该问题。
- abstained_correctly=false：系统凭空编造了文档中没有的信息，或给出了与文档无关的具体答案。

只输出一个 JSON 对象，不要输出任何其他文字或代码块，格式如下：
{
  "abstained_correctly": true,
  "fabricated_info": "若系统编造了信息，在此列出；没有则为空字符串",
  "reason": "简短说明"
}"""


def build_abstention_messages(
    question: str,
    answer: str,
    contexts: list[str],
) -> list[dict[str, str]]:
    """构建拒答判定消息。"""

    numbered = "\n\n".join(
        f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
    )
    if not numbered:
        numbered = "（系统未检索到任何上下文）"

    user_content = (
        f"【用户问题】\n{question}\n\n"
        f"【检索上下文】\n{numbered}\n\n"
        f"【系统回答】\n{answer}"
    )

    return [
        {"role": "system", "content": ABSTENTION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def judge_abstention(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    """判断系统在"无答案"问题上是否正确拒答。"""

    client = get_llm_client()

    parsed: dict[str, Any] | None = None
    raw_text = ""

    for attempt in range(2):
        messages = build_abstention_messages(
            question=question,
            answer=answer,
            contexts=contexts,
        )
        if attempt == 1:
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "请只输出一个合法 JSON 对象，"
                        "不要输出其他文字或代码块。"
                    ),
                }
            ]

        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=0.0,
            timeout=settings.llm_timeout,
            extra_body={
                "thinking": {"type": "disabled"}
            },
        )

        raw_text = (
            response.choices[0].message.content or ""
        )
        parsed = _parse_judge_json(raw_text)

        if parsed is not None:
            break

    if parsed is None:
        return {
            "abstained_correctly": None,
            "fabricated_info": "",
            "reason": "",
            "error": "judge_output_parse_failed",
            "raw": raw_text[:1000],
        }

    return {
        "abstained_correctly": bool(
            parsed.get("abstained_correctly")
        ),
        "fabricated_info": str(
            parsed.get("fabricated_info", "")
        ),
        "reason": str(parsed.get("reason", "")),
    }
