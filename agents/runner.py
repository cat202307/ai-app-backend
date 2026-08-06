import json
from typing import Any

from sqlalchemy.orm import Session

from agents.executor import execute_tool_call
from agents.registry import tool_registry
from core.exceptions import LLMServiceError
from services.llm_service import (
    ToolLLMResult,
    call_llm_with_tools,
)
from services.prompt_service import (
    build_tool_calling_messages,
)


def usage_to_dict(
    result: ToolLLMResult,
) -> dict[str, int]:
    """
    将模型 Token 信息转换为字典。
    """

    return {
        "prompt_tokens": (
            result.usage.prompt_tokens
        ),
        "completion_tokens": (
            result.usage.completion_tokens
        ),
        "total_tokens": (
            result.usage.total_tokens
        ),
    }


def run_single_tool_call(
    user_message: str,
    db: Session,
    temperature: float = 0.0,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    """
    执行单轮 Tool Calling。

    当前版本最多支持：

    第一次模型调用
    → 执行模型请求的工具
    → 第二次模型调用
    → 输出最终答案

    后续 Agent Loop 会把它升级成：

    模型
    → 工具
    → 模型
    → 工具
    → ...
    → 最终答案
    """

    messages: list[dict[str, Any]] = (
        build_tool_calling_messages(
            user_message=user_message
        )
    )

    tools = tool_registry.get_api_schemas()

    # 第一次调用：
    # 让模型判断是否需要工具。
    first_result = call_llm_with_tools(
        messages=messages,
        tools=tools,
        temperature=temperature,
        tool_choice=tool_choice,
    )

    first_message = first_result.message
    tool_calls = first_message.tool_calls

    # 模型认为不需要工具，直接回答。
    if not tool_calls:
        answer = first_message.content

        if not answer:
            raise LLMServiceError(
                message=(
                    "模型既没有返回回答，"
                    "也没有请求调用工具。"
                )
            )

        return {
            "answer": answer,
            "tool_used": False,
            "tool_traces": [],
            "model": first_result.model,
            "finish_reason": (
                first_result.finish_reason
            ),
            "usage": usage_to_dict(
                first_result
            ),
            "latency_ms": (
                first_result.latency_ms
            ),
        }

    # 保留模型完整 assistant message。
    #
    # 这里不能只保存 content，
    # 因为其中还包含 tool_calls。
    messages.append(
        first_message.model_dump(
            exclude_none=True
        )
    )

    tool_traces: list[dict[str, Any]] = []

    # 执行模型请求的所有工具。
    for tool_call in tool_calls:
        execution = execute_tool_call(
            tool_call=tool_call,
            db=db,
        )

        tool_traces.append({
            "tool_call_id": (
                execution.tool_call_id
            ),
            "tool_name": (
                execution.tool_name
            ),
            "arguments": (
                execution.arguments
            ),
            "result": execution.result,
            "duration_ms": (
                execution.duration_ms
            ),
        })

        # 将真实工具结果返回给模型。
        messages.append({
            "role": "tool",
            "tool_call_id": (
                execution.tool_call_id
            ),
            "content": json.dumps(
                execution.result,
                ensure_ascii=False,
                default=str,
            ),
        })

    # 第二次模型调用：
    # 根据工具结果生成最终自然语言答案。
    second_result = call_llm_with_tools(
        messages=messages,
        tools=tools,
        temperature=temperature,
        tool_choice="auto",
    )

    # 单轮学习版本暂时不支持第二次继续调用工具。
    if second_result.message.tool_calls:
        raise LLMServiceError(
            message=(
                "模型在第二轮仍请求调用工具。"
                "当前单轮版本暂不支持，"
                "后续将在 Agent Loop 中处理。"
            )
        )

    final_answer = second_result.message.content

    if not final_answer:
        raise LLMServiceError(
            message="模型没有生成最终回答。"
        )

    return {
        "answer": final_answer,
        "tool_used": True,
        "tool_traces": tool_traces,
        "model": second_result.model,
        "finish_reason": (
            second_result.finish_reason
        ),
        "usage": {
            "prompt_tokens": (
                first_result.usage.prompt_tokens
                + second_result.usage.prompt_tokens
            ),
            "completion_tokens": (
                first_result.usage.completion_tokens
                + second_result.usage.completion_tokens
            ),
            "total_tokens": (
                first_result.usage.total_tokens
                + second_result.usage.total_tokens
            ),
        },
        "latency_ms": (
            first_result.latency_ms
            + second_result.latency_ms
        ),
    }