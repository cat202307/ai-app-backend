"""
多轮追问指代消解服务。

对极短/含指代词且无明确主体的问题，结合对话历史
用 LLM 改写成独立完整问题再检索；无历史时引导用户补充。
"""

import logging
import re

from services.advanced_rag_answer_service import is_overview_question
from services.llm_service import rewrite_standalone_question

logger = logging.getLogger(__name__)


# 需要上下文感知的工具（这些工具的 question 参数可能是承前省略的追问）
CONTEXT_AWARE_TOOLS = ("ask_advanced_rag",)

# 指代词/省略标记
CONTEXT_MARKERS = (
    "有什么用", "有什么作用", "作用", "它", "他", "她",
    "这", "那", "其", "该", "这个", "那个", "这种",
    "那种", "这些", "那些", "上面", "前面", "刚才",
    "呢", "怎么做", "怎么实现", "为什么",
    "啥", "然后",
)

# 中等长度（7-14 字）问题：只有以这些指代词 / 省略词开头，
# 才视为依赖上下文；避免「今天天气怎么样」「为什么用X」这类
# 自带主语的完整问题被误判为省略追问。
_START_MARKERS = (
    "它", "他", "她", "这", "那", "其", "该",
    "这个", "那个", "这种", "那种", "这些", "那些",
    "上面", "前面", "刚才", "然后", "有什么",
    "怎么做", "怎么实现", "怎么", "为什么", "啥",
)


def needs_standalone_question(question: str) -> bool:
    """
    判断当前问题是否疑似依赖对话上下文（省略主语 /
    含指代词的短问），需要结合历史改写为独立完整
    问题后再检索。
    """

    text = (question or "").strip()
    if not text:
        return False

    # 概述类问题走专门的概述路由
    if is_overview_question(text):
        return False

    # 含连续拉丁字母 / 数字实义主体，视为自包含
    if re.search(r"[A-Za-z0-9]{2,}", text):
        return False

    # 极短问：仅当含明确指代 / 依赖 marker 时才视为
    # 依赖上下文（如「有什么用」「它呢」）；若短问本身
    # 是具体技术名词（如「学习率设置」），可直接检索，
    # 不必改写——改写加的「论文中的……是什么」模板反而
    # 会稀释 rerank 匹配、导致误拒。
    if len(text) <= 6:
        return any(
            marker in text for marker in CONTEXT_MARKERS
        )

    # 中等长度（7-14 字）：只有以指代词 / 省略词开头才视为
    # 依赖上下文（如「这个怎么做」「它呢」）；自带主语的完整
    # 问题（如「今天天气怎么样」「为什么用AdamW」）直接检索。
    if len(text) <= 14 and any(
        text.startswith(m) for m in _START_MARKERS
    ):
        return True

    return False


def build_history_text(
    messages: list,
    current_question: str = "",
    max_turns: int = 6,
) -> str:
    """
    从 messages 中提取最近的用户 / 助手纯文本对话作为历史，
    排除当前待改写的用户问题。

    同时支持 dict 格式和 LangGraph message 对象。
    """

    pairs: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
        else:
            role = (
                getattr(message, "type", None)
                or getattr(message, "role", None)
            )
            content = getattr(message, "content", "")
            tool_calls = getattr(message, "tool_calls", None)

        if role not in ("user", "assistant", "human", "ai"):
            continue
        if not content or tool_calls:
            continue
        speaker = "用户" if role in ("user", "human") else "助手"
        pairs.append(f"{speaker}：{str(content).strip()}")

    # 排除当前待改写的用户问题。
    # 必须先确认 current_question 非空：空字符串 in 任意
    # 字符串恒为 True，否则会在未传当前问题时误删最后一条。
    current = current_question.strip()
    if (
        pairs
        and current
        and pairs[-1].startswith("用户")
        and current in pairs[-1]
    ):
        pairs = pairs[:-1]

    return "\n".join(pairs[-max_turns:])


def resolve_query(question: str, messages: list) -> tuple[str, bool]:
    """
    解析检索问题：必要时结合历史改写为独立完整问题。

    Returns:
        (resolved_question, is_missing_context)
        - is_missing_context=True 时，resolved_question 是引导用户补充的提示，
          调用方应直接返回给用户，不再执行检索。
    """
    if not needs_standalone_question(question):
        return question, False

    history_text = build_history_text(
        messages=messages,
        current_question=question,
    )
    if not history_text:
        hint = (
            f"用户问题缺少上下文（如省略了主语的追问"
            f"'{question}'），且当前对话没有历史可参照。"
            f"请直接回复用户：'你的问题缺少上下文，"
            f"请问你具体指的是什么？'，不要编造答案，"
            f"也不要继续调用工具。"
        )
        return hint, True

    try:
        rewritten = rewrite_standalone_question(
            history_text=history_text,
            question=question,
        )
        if rewritten and rewritten != question:
            logger.info(f"追问改写: '{question}' -> '{rewritten}'")
            return rewritten, False
    except Exception as e:
        logger.warning(f"追问改写失败，使用原问题: {e}")

    return question, False
