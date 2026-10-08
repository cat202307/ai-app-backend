import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterator, cast

from core.logging_config import get_logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from core.config import settings
from core.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
)

# ============================================================
# 创建大模型客户端
# ============================================================

# OpenAI SDK 同时支持 OpenAI 兼容接口。
# 当前 base_url 指向 DeepSeek，因此实际调用的是 DeepSeek。
# client = OpenAI(
#     api_key=settings.deepseek_api_key,
#     base_url=settings.deepseek_base_url,
# )


# ============================================================
# 模型调用结果数据结构
# ============================================================

@dataclass
class LLMUsage:
    """
    一次大模型调用的 Token 使用情况。

    prompt_tokens:
        输入给模型的 Token 数量。

    completion_tokens:
        模型生成回答使用的 Token 数量。

    total_tokens:
        输入 Token 与输出 Token 的总和。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResult:
    """
    一次普通大模型调用的完整结果。

    content:
        模型生成的回答正文。

    model:
        实际使用的模型名称。

    finish_reason:
        模型停止生成的原因，例如 stop、length。

    usage:
        Token 使用情况。

    latency_ms:
        本次模型调用耗时，单位为毫秒。
    """

    content: str
    model: str
    finish_reason: str | None
    usage: LLMUsage
    latency_ms: int


@dataclass
class ToolLLMResult:
    """
    支持 Tool Calling 的模型调用结果。

    message 中可能包含：

    - content
    - tool_calls
    - reasoning_content
    """

    message: Any
    model: str
    finish_reason: str | None
    usage: LLMUsage
    latency_ms: int

# ============================================================
# 公共校验函数
# ============================================================

def validate_llm_configuration() -> None:
    """
    检查大模型配置是否完整。

    如果没有配置 API Key，直接抛出业务异常，
    不允许把配置错误当作普通模型回答返回。
    """

    if not settings.deepseek_api_key:
        raise LLMConfigurationError(
            message=(
                "未配置 DEEPSEEK_API_KEY，"
                "请先在项目根目录的 .env 文件中配置。"
            )
        )

    if not settings.deepseek_model:
        raise LLMConfigurationError(
            message="未配置 DEEPSEEK_MODEL。"
        )

    if not settings.deepseek_base_url:
        raise LLMConfigurationError(
            message="未配置 DEEPSEEK_BASE_URL。"
        )


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """
    延迟创建大模型客户端。

    为什么不在模块顶部直接创建：
    当 API Key 为空时，新版 OpenAI SDK 会在创建客户端时
    直接抛出 Missing credentials，导致 FastAPI 无法启动。

    现在先由 validate_llm_configuration() 检查配置，
    检查通过后才创建客户端。
    """

    validate_llm_configuration()

    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )

def build_usage(response_usage: Any) -> LLMUsage:
    """
    将 SDK 返回的 usage 对象转换为项目自己的 LLMUsage。

    部分模型或特殊响应可能不返回 usage，
    因此需要兼容 None。
    """

    if response_usage is None:
        result_usage = LLMUsage()
    else:
        result_usage = LLMUsage(
            prompt_tokens=getattr(
                response_usage,
                "prompt_tokens",
                0,
            ) or 0,
            completion_tokens=getattr(
                response_usage,
                "completion_tokens",
                0,
            ) or 0,
            total_tokens=getattr(
                response_usage,
                "total_tokens",
                0,
            ) or 0,
        )

    # 上报全局 token / 成本统计（覆盖所有 LLM 调用）
    try:
        from core.metrics import metrics

        metrics.record_llm_usage(
            prompt_tokens=result_usage.prompt_tokens,
            completion_tokens=result_usage.completion_tokens,
        )
    except Exception:
        pass

    return result_usage


# ============================================================
# 普通模型调用：返回完整结果
# ============================================================

def _is_retryable_llm_error(exc: BaseException) -> bool:
    """
    判断哪些 LLM 异常值得重试（临时性故障）。

    可重试：超时、限流(429)、连接错误、5xx 服务端错误。
    不可重试：鉴权失败(401)、其他 4xx 客户端错误。
    """
    if isinstance(exc, (APITimeoutError, RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception(_is_retryable_llm_error),
    reraise=True,
)
def _create_chat_completion(client, model, messages, timeout, **kwargs):
    """
    带重试的 chat.completions.create 包装。

    对超时 / 限流 / 连接错误 / 5xx 自动指数退避重试最多 3 次，
    重试耗尽后抛出原始异常，由上层转为业务异常。
    流式调用不使用此包装。
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
        **kwargs,
    )


def call_llm_detail(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
) -> LLMResult:
    """
    普通大模型调用，返回完整调用结果。

    参数：
        messages:
            模型消息列表，例如：

            [
                {
                    "role": "system",
                    "content": "你是一名 AI 助手。"
                },
                {
                    "role": "user",
                    "content": "什么是 RAG？"
                }
            ]

        temperature:
            控制模型回答随机性。

            较低：
                回答更稳定、更确定。

            较高：
                回答更多样、更有创造性。

    返回：
        LLMResult，包含：

        - 回答正文
        - 模型名称
        - finish_reason
        - Token 使用量
        - 调用耗时

    异常：
        不会返回“大模型调用失败”字符串，
        而是抛出明确的业务异常。
    """

    validate_llm_configuration()

    logger = get_logger(__name__)

    if not messages:
        raise LLMServiceError(
            message="模型消息列表不能为空。"
        )

    client = get_llm_client()

    start_time = time.perf_counter()

    try:
        response = _create_chat_completion(
            client=client,
            model=cast(Any, settings.deepseek_model),
            messages=cast(Any, messages),
            temperature=temperature,
            stream=False,
            timeout=settings.llm_timeout,
        )

    except APITimeoutError as exc:
        raise LLMTimeoutError(
            message=(
                f"大模型在 {settings.llm_timeout} 秒内"
                "没有返回结果。"
            )
        ) from exc

    except AuthenticationError as exc:
        raise LLMAuthenticationError(
            message=(
                "大模型 API 鉴权失败，"
                "请检查 DEEPSEEK_API_KEY 是否正确或有效。"
            )
        ) from exc

    except RateLimitError as exc:
        raise LLMRateLimitError(
            message=(
                "大模型请求过于频繁，"
                "或者当前账户额度不足，请稍后重试。"
            )
        ) from exc

    except APIConnectionError as exc:
        raise LLMServiceError(
            message=(
                "无法连接大模型服务，"
                "请检查网络、代理和 DEEPSEEK_BASE_URL。"
            )
        ) from exc

    except APIStatusError as exc:
        raise LLMServiceError(
            message=(
                "大模型服务返回异常状态，"
                f"HTTP 状态码：{exc.status_code}。"
            )
        ) from exc

    except Exception as exc:
        raise LLMServiceError(
            message=(
                "大模型调用发生未知异常，"
                f"异常类型：{type(exc).__name__}。"
            )
        ) from exc

    latency_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    if not response.choices:
        raise LLMServiceError(
            message="大模型响应中没有 choices 数据。"
        )

    choice = response.choices[0]
    answer = choice.message.content

    if not answer or not answer.strip():
        raise LLMServiceError(
            message="大模型没有返回有效的文本内容。"
        )

    result = LLMResult(
        content=answer,
        model=(
            response.model
            or settings.deepseek_model
        ),
        finish_reason=choice.finish_reason,
        usage=build_usage(response.usage),
        latency_ms=latency_ms,
    )

    logger.info(
        f"LLM call | model={result.model} | "
        f"latency={latency_ms}ms | "
        f"tokens={result.usage.total_tokens}"
    )

    return result


# ============================================================
# 普通模型调用：兼容旧项目
# ============================================================

def call_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
) -> str:
    """
    普通大模型调用，只返回回答正文。

    这个函数用于兼容当前已有接口：

    - POST /chat
    - POST /rag/ask
    - POST /rag/search

    内部实际调用 call_llm_detail()，
    因此同样具备完整异常处理能力。
    """

    result = call_llm_detail(
        messages=messages,
        temperature=temperature,
    )

    return result.content


def rewrite_standalone_question(
    history_text: str,
    question: str,
) -> str:
    """
    结合对话历史，把含指代 / 省略的追问改写为
    独立完整问题；问题本身完整时原样返回。

    只返回改写后的问题文本。
    """

    from services.prompt_service import (
        build_standalone_question_messages,
    )

    messages = build_standalone_question_messages(
        history_text=history_text,
        question=question,
    )
    rewritten = call_llm(messages=messages, temperature=0.0)
    rewritten = (rewritten or "").strip()
    rewritten = rewritten.strip("\"'“”‘’ ")

    return rewritten or question


# ============================================================
# 流式模型调用
# ============================================================

def stream_llm(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
) -> Iterator[str]:
    """
    流式调用大模型。

    特点：
        模型生成一小段内容，就立即 yield 返回一小段。

    用于：
        - POST /chat/stream
        - POST /rag/search/stream

    重要说明：
        调用失败时不再 yield 错误文本。

        旧版本会把：
            “大模型流式调用失败……”

        当作正常模型回答返回并保存。

        当前版本会直接抛出业务异常，
        由 Router 或流式生成器负责转换成 error 事件。
    """

    validate_llm_configuration()

    logger = get_logger(__name__)

    if not messages:
        raise LLMServiceError(
            message="模型消息列表不能为空。"
        )

    client = get_llm_client()

    try:
        response = client.chat.completions.create(
            model=cast(
                Any,
                settings.deepseek_model,
            ),
            messages=cast(
                Any,
                messages,
            ),
            temperature=temperature,
            stream=True,
            timeout=settings.llm_stream_timeout,
        )

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            content = getattr(
                delta,
                "content",
                None,
            )

            if content:
                yield content

    except APITimeoutError as exc:
        raise LLMTimeoutError(
            message=(
                f"大模型流式响应在 "
                f"{settings.llm_stream_timeout} 秒内超时。"
            )
        ) from exc

    except AuthenticationError as exc:
        raise LLMAuthenticationError(
            message=(
                "大模型 API 鉴权失败，"
                "请检查 DEEPSEEK_API_KEY。"
            )
        ) from exc

    except RateLimitError as exc:
        raise LLMRateLimitError(
            message=(
                "大模型请求过于频繁，"
                "或者当前账户额度不足，请稍后重试。"
            )
        ) from exc

    except APIConnectionError as exc:
        raise LLMServiceError(
            message=(
                "无法连接大模型服务，"
                "请检查网络、代理和接口地址。"
            )
        ) from exc

    except APIStatusError as exc:
        raise LLMServiceError(
            message=(
                "大模型流式服务返回异常状态，"
                f"HTTP 状态码：{exc.status_code}。"
            )
        ) from exc

    except Exception as exc:
        raise LLMServiceError(
            message=(
                "大模型流式调用发生未知异常，"
                f"异常类型：{type(exc).__name__}。"
            )
        ) from exc

def call_llm_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float = 0.0,
    tool_choice: str = "auto",
) -> ToolLLMResult:
    """
    调用支持 Tool Calling 的模型接口。

    tool_choice:

    auto:
        模型自行判断是直接回答还是调用工具。

    required:
        模型必须调用至少一个工具。

    本阶段显式关闭思考模式，
    先集中学习 Tool Calling 基础流程。
    """

    validate_llm_configuration()

    logger = get_logger(__name__)

    if not messages:
        raise LLMServiceError(
            message="模型消息列表不能为空。"
        )

    if not tools:
        raise LLMServiceError(
            message="工具列表不能为空。"
        )

    client = get_llm_client()
    start_time = time.perf_counter()

    try:
        response = _create_chat_completion(
            client=client,
            model=cast(Any, settings.deepseek_model),
            messages=cast(Any, messages),
            tools=cast(Any, tools),
            tool_choice=cast(Any, tool_choice),
            temperature=temperature,
            stream=False,
            timeout=settings.llm_timeout,
            extra_body={
                "thinking": {"type": "disabled"}
            },
        )

    except APITimeoutError as exc:
        raise LLMTimeoutError() from exc

    except AuthenticationError as exc:
        raise LLMAuthenticationError() from exc

    except RateLimitError as exc:
        raise LLMRateLimitError() from exc

    except APIConnectionError as exc:
        raise LLMServiceError(
            message="无法连接大模型服务。"
        ) from exc

    except APIStatusError as exc:
        raise LLMServiceError(
            message=(
                "大模型工具调用接口返回异常，"
                f"HTTP 状态码：{exc.status_code}。"
            )
        ) from exc

    except Exception as exc:
        raise LLMServiceError(
            message=(
                "大模型工具调用发生未知异常，"
                f"异常类型：{type(exc).__name__}。"
            )
        ) from exc

    latency_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    if not response.choices:
        raise LLMServiceError(
            message="大模型没有返回 choices。"
        )

    choice = response.choices[0]

    return ToolLLMResult(
        message=choice.message,
        model=(
            response.model
            or settings.deepseek_model
        ),
        finish_reason=choice.finish_reason,
        usage=build_usage(response.usage),
        latency_ms=latency_ms,
    )


def stream_llm_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float = 0.0,
    tool_choice: str = "auto",
    on_content: Callable[[str], None] | None = None,
) -> ToolLLMResult:
    """
    流式调用支持 Tool Calling 的模型。

    与 call_llm_with_tools 的区别：
        stream=True，模型每产出一段正文就通过 on_content 回调实时推送，
        供 SSE 真流式使用；流式返回的 tool_calls 增量片段会被拼接还原。
    返回值结构与 call_llm_with_tools 一致。
    """

    validate_llm_configuration()

    logger = get_logger(__name__)

    if not messages:
        raise LLMServiceError(
            message="模型消息列表不能为空。"
        )

    if not tools:
        raise LLMServiceError(
            message="工具列表不能为空。"
        )

    client = get_llm_client()
    start_time = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=cast(Any, settings.deepseek_model),
            messages=cast(Any, messages),
            tools=cast(Any, tools),
            tool_choice=cast(Any, tool_choice),
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            timeout=settings.llm_stream_timeout,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )

        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, str]] = {}
        finish_reason = None
        model_name = settings.deepseek_model
        response_usage = None

        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                response_usage = chunk.usage

            if getattr(chunk, "model", None):
                model_name = chunk.model or model_name

            choices = getattr(chunk, "choices", None)
            if not choices:
                # include_usage 时最后一个 chunk 只有 usage、没有 choices
                continue

            choice0 = choices[0]
            if choice0.finish_reason:
                finish_reason = choice0.finish_reason

            delta = choice0.delta

            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                if on_content is not None:
                    on_content(delta.content)

            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = (
                        tc.index
                        if tc.index is not None
                        else 0
                    )
                    slot = tool_acc.setdefault(
                        idx,
                        {"id": "", "name": "", "arguments": ""},
                    )

                    if getattr(tc, "id", None):
                        slot["id"] = tc.id

                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            slot["arguments"] += fn.arguments

    except APITimeoutError as exc:
        raise LLMTimeoutError() from exc

    except AuthenticationError as exc:
        raise LLMAuthenticationError() from exc

    except RateLimitError as exc:
        raise LLMRateLimitError() from exc

    except APIConnectionError as exc:
        raise LLMServiceError(
            message="无法连接大模型服务。"
        ) from exc

    except APIStatusError as exc:
        raise LLMServiceError(
            message=(
                "大模型流式工具调用接口返回异常，"
                f"HTTP 状态码：{exc.status_code}。"
            )
        ) from exc

    except LLMServiceError:
        raise

    except Exception as exc:
        raise LLMServiceError(
            message=(
                "大模型流式调用发生未知异常，"
                f"异常类型：{type(exc).__name__}。"
            )
        ) from exc

    latency_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    from openai.types.chat import (
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    )

    full_content = "".join(content_parts)

    tool_calls = []
    for idx in sorted(tool_acc):
        slot = tool_acc[idx]
        tool_calls.append(
            ChatCompletionMessageToolCall(
                id=slot["id"] or ("call_%d" % idx),
                type="function",
                function={
                    "name": slot["name"],
                    "arguments": slot["arguments"],
                },
            )
        )

    message = ChatCompletionMessage(
        role="assistant",
        content=full_content or None,
        tool_calls=tool_calls or None,
    )

    logger.info(
        "stream tool call finished | latency=%dms | tool_calls=%d",
        latency_ms,
        len(tool_calls),
    )

    return ToolLLMResult(
        message=message,
        model=model_name or settings.deepseek_model,
        finish_reason=finish_reason,
        usage=build_usage(response_usage),
        latency_ms=latency_ms,
    )