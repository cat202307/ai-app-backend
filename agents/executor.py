import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from agents.registry import (
    ToolRegistry,
    tool_registry,
)
from core.exceptions import (
    ToolArgumentsError,
    ToolExecutionError,
)


@dataclass
class ToolExecutionResult:
    """
    一次工具执行的完整结果。
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_ms: int


def execute_tool_call(
    tool_call: Any,
    db: Session,
    registry: ToolRegistry = tool_registry,
) -> ToolExecutionResult:
    """
    校验并执行一次模型生成的工具调用。

    流程：

    模型参数字符串
    → json.loads()
    → 工具白名单检查
    → Pydantic 参数校验
    → 执行真实 Python 函数
    → 返回结构化执行结果
    """

    tool_name = tool_call.function.name
    raw_arguments = tool_call.function.arguments

    try:
        parsed_arguments = json.loads(
            raw_arguments or "{}"
        )

    except json.JSONDecodeError as exc:
        raise ToolArgumentsError(
            message=(
                f"工具 {tool_name} 的参数不是合法 JSON："
                f"{raw_arguments}"
            )
        ) from exc

    if not isinstance(parsed_arguments, dict):
        raise ToolArgumentsError(
            message=(
                f"工具 {tool_name} 的参数必须是 JSON 对象。"
            )
        )

    tool = registry.get(tool_name)

    try:
        validated_arguments = (
            tool.input_model.model_validate(
                parsed_arguments
            )
        )

    except ValidationError as exc:
        raise ToolArgumentsError(
            message=(
                f"工具 {tool_name} 参数校验失败："
                f"{exc}"
            )
        ) from exc

    start_time = time.perf_counter()

    try:
        result = tool.execute(
            arguments=validated_arguments,
            db=db,
        )

    except Exception as exc:
        raise ToolExecutionError(
            message=(
                f"工具 {tool_name} 执行失败："
                f"{type(exc).__name__}"
            )
        ) from exc

    duration_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    return ToolExecutionResult(
        tool_call_id=tool_call.id,
        tool_name=tool_name,
        arguments=parsed_arguments,
        result=result,
        duration_ms=duration_ms,
    )