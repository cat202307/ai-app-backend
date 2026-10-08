"""
基于 LangGraph 的多工具 Agent 实现。

将原 agents/loop.py 中手写的 Multi-Tool Agent Loop
迁移为 LangGraph StateGraph Workflow。

Graph 结构：

    START
      ↓
    agent_node  ──(有 tool_calls)──→  tool_node
      ↑                                    │
      └────────(无 tool_calls)────────────┘
                    ↓
                   END

节点职责：
- agent_node：调用 LLM with tools，判断是否需要工具
- tool_node：执行本轮所有工具调用，结果写回 messages

保留能力：
- Max Steps Guard
- 重复工具调用检测
- Trace 执行轨迹
- Token 统计与耗时统计
- 与 run_agent_loop 完全一致的返回格式
"""

import json
import re
import time
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from agents.executor import (
    ToolExecutionResult,
    execute_tool_call,
)
from agents.registry import tool_registry
from core.config import settings
from core.exceptions import (
    AgentMaxStepsError,
    AgentRepeatedToolCallError,
    LLMServiceError,
)
from services.llm_service import (
    ToolLLMResult,
    call_llm_with_tools,
    stream_llm_with_tools,
)
from services.prompt_service import build_agent_messages
from services.contextual_query_service import (
    CONTEXT_AWARE_TOOLS,
    resolve_query,
)


# ============================================================
# Graph State
# ============================================================

class AgentState(TypedDict):
    """
    LangGraph 中在节点之间传递的状态。

    注意：
    - db (Session) 不放在 state 中，通过闭包传递给节点。
    - list 类型字段在节点返回时会被整体替换，
      因此节点函数必须返回完整更新后的 list。
    """

    # 完整对话上下文（system + user + assistant + tool ...）
    messages: list[dict[str, Any]]

    # 工具执行轨迹
    traces: list[dict[str, Any]]

    # 累计 Token 使用
    usage: dict[str, int]

    # 累计 LLM 调用耗时（毫秒）
    llm_latency_ms: int

    # 当前步骤数（从 1 开始）
    step: int

    # 最大步骤数
    max_steps: int

    # 已执行过的工具调用签名（用于重复检测）
    called_signatures: list[str]

    # 最终回答（为 None 表示尚未完成）
    answer: str | None

    # 整个任务开始时间（perf_counter 浮点值）
    start_time: float


# ============================================================
# 工具选择纠偏（确定性）
# ============================================================

# 明确的文件列表 / 元信息请求：这些才应该用 list_all_documents
_FILE_LIST_RE = re.compile(
    r"(有?哪?些?\s*(文件|文档)|文件列表|文档列表|"
    r"所有\s*(文件|文档)|一共.*(多少|几个).*(文件|文档)|"
    r"(多少|几个).*(文件|文档)|列出.*(文件|文档)|"
    r"文档总数|文件总数)"
)

# 内容 / 技术点类问题的指示词
_CONTENT_CUE_RE = re.compile(
    r"(用了?什么|用的什么|是什么|是怎样|怎么样|如何|"
    r"为什么|介绍|解释|原理|内容|讲了|讲什么|方法|"
    r"算法|模型|参数|优化器|损失|流程|结构|功能|作用|"
    r"有什么用|区别|关系|实现|采用)"
)


def _is_content_seeking(question: str) -> bool:
    """
    判断用户问题是否在询问文档内容 / 技术点。

    先排除明确的文件列表请求，再看是否含内容类指示词。
    用于纠正模型把内容类问题误路由到 list_all_documents。
    """

    text = question or ""
    if _FILE_LIST_RE.search(text):
        return False
    return bool(_CONTENT_CUE_RE.search(text))


def _latest_user_question(messages: list[dict[str, Any]]) -> str:
    """取消息列表中最后一条 user 消息内容。"""

    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "") or ""
    return ""


# ============================================================
# 辅助函数（复用 loop.py 中的逻辑）
# ============================================================

def _add_usage(
    total_usage: dict[str, int],
    result: ToolLLMResult,
) -> None:
    """将一次模型调用的 Token 累加到总统计中。"""

    total_usage["prompt_tokens"] += result.usage.prompt_tokens
    total_usage["completion_tokens"] += result.usage.completion_tokens
    total_usage["total_tokens"] += result.usage.total_tokens


def _build_tool_signature(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """构建稳定的工具调用签名，用于重复调用检测。"""

    arguments_json = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"{tool_name}:{arguments_json}"


def _serialize_tool_result(
    result: dict[str, Any],
) -> str:
    """将工具结果转成 JSON 字符串，并控制最大长度。"""

    text = json.dumps(
        result,
        ensure_ascii=False,
        default=str,
    )

    max_chars = settings.agent_tool_result_max_chars

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n...[工具结果过长，已截断]"


# ============================================================
# 节点函数
# ============================================================

def _make_agent_node(
    db: Session,
    on_content: Callable[[str], None] | None = None,
):
    """
    工厂函数：创建 agent_node，通过闭包持有 db 和 on_content。

    agent_node 职责：
    1. 调用 LLM with tools
    2. 累加 Token / 耗时
    3. step + 1
    4. 如果没有 tool_calls → 设置 answer（任务完成）
    5. 如果有 tool_calls → 把 assistant message 写入 messages
    """

    tools = tool_registry.get_api_schemas()

    def agent_node(state: AgentState) -> dict[str, Any]:
        # 步骤数 +1
        new_step = state["step"] + 1

        # 调用模型（流式，正文 token 实时经 on_content 推送）
        # 第一轮（尚未调用任何工具）强制先调用工具，
        # 避免模型跳过检索、仅凭上下文直接作答而丢失参考来源；
        # 拿到工具结果后的后续轮恢复 auto，让模型自由生成最终答案。
        forced_tool_choice = (
            "auto" if state.get("traces") else "required"
        )

        llm_result = stream_llm_with_tools(
            messages=state["messages"],
            tools=tools,
            temperature=0.0,
            tool_choice=forced_tool_choice,
            on_content=on_content,
        )

        # 累加 Token
        new_usage = dict(state["usage"])
        _add_usage(new_usage, llm_result)

        # 累加 LLM 耗时
        new_llm_latency_ms = (
            state["llm_latency_ms"] + llm_result.latency_ms
        )

        assistant_message = llm_result.message
        tool_calls = assistant_message.tool_calls or []

        # 复制 messages（因为 list 是替换语义）
        new_messages = list(state["messages"])

        if not tool_calls:
            # 模型没有调用工具 → 任务完成
            final_answer = (assistant_message.content or "").strip()

            if not final_answer:
                raise LLMServiceError(
                    message=(
                        "Agent 没有调用工具，"
                        "也没有生成最终回答。"
                    )
                )

            return {
                "messages": new_messages,
                "traces": state["traces"],
                "usage": new_usage,
                "llm_latency_ms": new_llm_latency_ms,
                "step": new_step,
                "max_steps": state["max_steps"],
                "called_signatures": state["called_signatures"],
                "answer": final_answer,
                "start_time": state["start_time"],
            }

        # 有 tool_calls → 保存 assistant message（含 tool_calls）
        new_messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        return {
            "messages": new_messages,
            "traces": state["traces"],
            "usage": new_usage,
            "llm_latency_ms": new_llm_latency_ms,
            "step": new_step,
            "max_steps": state["max_steps"],
            "called_signatures": state["called_signatures"],
            "answer": None,
            "start_time": state["start_time"],
        }

    return agent_node


def _make_tool_node(
    db: Session,
    on_step: Callable[[dict[str, Any]], None] | None = None,
):
    """
    工厂函数：创建 tool_node，通过闭包持有 db 和 on_step。

    tool_node 职责：
    1. 从 messages 中取出最后一条 assistant message 的 tool_calls
    2. 对每个 tool_call：
       - 解析参数
       - 检测重复调用
       - 执行工具
       - 保存 Trace
       - 将工具结果写入 messages
    """

    def tool_node(state: AgentState) -> dict[str, Any]:
        new_messages = list(state["messages"])
        new_traces = list(state["traces"])
        new_called_signatures = list(state["called_signatures"])

        # 取出最后一条 assistant message（其中包含 tool_calls）
        # messages 中最后一条就是 agent_node 刚写入的 assistant message
        last_assistant = new_messages[-1]
        tool_calls = last_assistant.get("tool_calls", []) or []

        current_step = state["step"]

        for tool_call in tool_calls:
            # 解析工具名和参数
            tool_name = tool_call["function"]["name"]
            raw_arguments = tool_call["function"].get("arguments") or "{}"

            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"_raw": raw_arguments}

            # 多轮追问的指代 / 省略消解：
            # 检索工具收到「有什么用」这类依赖上下文的短问时，
            # 结合对话历史改写为完整问题再检索；无历史时标记 missing_context
            missing_context_note = None
            if (
                tool_name in CONTEXT_AWARE_TOOLS
                and isinstance(parsed_arguments, dict)
                and isinstance(parsed_arguments.get("question"), str)
            ):
                original_question = parsed_arguments["question"]
                resolved_q, is_missing = resolve_query(
                    original_question, new_messages
                )
                if is_missing:
                    missing_context_note = resolved_q
                elif resolved_q != original_question:
                    parsed_arguments["question"] = resolved_q
                    tool_call["function"]["arguments"] = json.dumps(
                        parsed_arguments, ensure_ascii=False
                    )

            # 工具选择纠偏：内容 / 技术点类问题被模型误路由到
            # list_all_documents 时，确定性地重路由为 ask_advanced_rag，
            # 避免模型列出文档后反问、不进入检索。
            if tool_name == "list_all_documents":
                asked_question = _latest_user_question(
                    new_messages
                )
                if _is_content_seeking(asked_question):
                    tool_name = "ask_advanced_rag"
                    parsed_arguments = {
                        "question": asked_question
                    }
                    tool_call["function"]["name"] = (
                        "ask_advanced_rag"
                    )
                    tool_call["function"]["arguments"] = (
                        json.dumps(
                            parsed_arguments,
                            ensure_ascii=False,
                        )
                    )

            # 重复调用检测
            signature = _build_tool_signature(
                tool_name=tool_name,
                arguments=parsed_arguments,
            )

            if signature in new_called_signatures:
                raise AgentRepeatedToolCallError(
                    tool_name=tool_name,
                )

            new_called_signatures.append(signature)

            # 执行工具（需要把 dict 转回 OpenAI 兼容的 tool_call 对象）
            # execute_tool_call 接收的是 OpenAI SDK 的 ChatCompletionMessageToolCall
            # 这里我们构造一个简易的兼容对象
            if missing_context_note:
                # 省略式追问但无对话历史可参照：不检索，直接返回提示
                execution = ToolExecutionResult(
                    tool_call_id=tool_call.get("id", ""),
                    tool_name=tool_name,
                    arguments=parsed_arguments,
                    result={
                        "matched": False,
                        "citations": [],
                        "note": missing_context_note,
                        "retrieval_status": "missing_context",
                    },
                    duration_ms=0,
                )
            else:
                execution = execute_tool_call(
                    tool_call=_ToolCallCompat(tool_call),
                    db=db,
                )

            # 保存 Trace
            new_traces.append({
                "step": current_step,
                "tool_call_id": execution.tool_call_id,
                "tool_name": execution.tool_name,
                "arguments": execution.arguments,
                "result": execution.result,
                "duration_ms": execution.duration_ms,
            })

            # 实时把这一步工具调用推送出去（用于 SSE）
            if on_step is not None:
                on_step(new_traces[-1])

            # 将工具结果写入 messages
            new_messages.append({
                "role": "tool",
                "tool_call_id": execution.tool_call_id,
                "content": _serialize_tool_result(execution.result),
            })

        return {
            "messages": new_messages,
            "traces": new_traces,
            "usage": state["usage"],
            "llm_latency_ms": state["llm_latency_ms"],
            "step": current_step,
            "max_steps": state["max_steps"],
            "called_signatures": new_called_signatures,
            "answer": None,
            "start_time": state["start_time"],
        }

    return tool_node


class _ToolCallCompat:
    """
    将 LangGraph state 中保存的 dict 形式 tool_call
    包装成 execute_tool_call 期望的对象格式。

    execute_tool_call 访问：
    - tool_call.function.name
    - tool_call.function.arguments
    - tool_call.id
    """

    def __init__(self, tool_call_dict: dict[str, Any]):
        self.id = tool_call_dict.get("id", "")
        self.function = _FunctionCompat(
            tool_call_dict.get("function", {})
        )


class _FunctionCompat:
    def __init__(self, function_dict: dict[str, Any]):
        self.name = function_dict.get("name", "")
        self.arguments = function_dict.get("arguments") or "{}"


# ============================================================
# 条件边
# ============================================================

def _should_continue(state: AgentState) -> str:
    """
    条件边：判断 agent_node 之后应该走哪条路。

    - answer 不为 None → 任务完成 → END
    - step >= max_steps → 达到最大步数 → 抛出异常
    - 否则 → 继续执行工具 → "tools"
    """

    if state["answer"] is not None:
        return "end"

    if state["step"] >= state["max_steps"]:
        raise AgentMaxStepsError(max_steps=state["max_steps"])

    return "continue"


# ============================================================
# 构建 Graph
# ============================================================

def _build_graph(
    db: Session,
    on_content: Callable[[str], None] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
):
    """
    构建并编译 LangGraph Agent Workflow。

    结构：
        START → agent_node
        agent_node --(continue)--→ tool_node → agent_node
        agent_node --(end)------→ END
    """

    graph = StateGraph(AgentState)

    graph.add_node(
        "agent",
        _make_agent_node(db, on_content),
    )
    graph.add_node(
        "tools",
        _make_tool_node(db, on_step),
    )

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================
# 对外入口
# ============================================================

def run_agent_graph(
    user_message: str,
    db: Session,
    max_steps: int | None = None,
    history: list[dict[str, str]] | None = None,
    on_token: Callable[[str], None] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    基于 LangGraph 的多步骤 Agent 执行入口。

    返回格式与 agents/loop.run_agent_loop 完全一致，
    便于新旧实现对比和无缝替换。

    参数：
        user_message: 用户任务描述
        db: SQLAlchemy Session
        max_steps: 最大执行步骤，为空时使用系统默认

    返回：
        {
            "answer": str,
            "completed": bool,
            "steps": int,
            "tool_call_count": int,
            "traces": list,
            "usage": dict,
            "timing": {
                "llm_latency_ms": int,
                "total_duration_ms": int,
            },
        }
    """

    actual_max_steps = (
        max_steps
        if max_steps is not None
        else settings.agent_max_steps
    )

    # 构建初始状态
    initial_state: AgentState = {
        "messages": build_agent_messages(user_message=user_message, history=history),
        "traces": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "llm_latency_ms": 0,
        "step": 0,
        "max_steps": actual_max_steps,
        "called_signatures": [],
        "answer": None,
        "start_time": time.perf_counter(),
    }

    # 构建并运行 Graph
    app = _build_graph(
        db,
        on_content=on_token,
        on_step=on_step,
    )
    final_state = app.invoke(initial_state)

    # 计算总耗时
    total_duration_ms = int(
        (time.perf_counter() - final_state["start_time"]) * 1000
    )

    return {
        "answer": final_state["answer"],
        "completed": True,
        "steps": final_state["step"],
        "tool_call_count": len(final_state["traces"]),
        "traces": final_state["traces"],
        "usage": final_state["usage"],
        "timing": {
            "llm_latency_ms": final_state["llm_latency_ms"],
            "total_duration_ms": total_duration_ms,
        },
    }
