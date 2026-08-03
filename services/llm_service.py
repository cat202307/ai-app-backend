import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator, cast

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
        return LLMUsage()

    return LLMUsage(
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


# ============================================================
# 普通模型调用：返回完整结果
# ============================================================

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

    if not messages:
        raise LLMServiceError(
            message="模型消息列表不能为空。"
        )

    client = get_llm_client()

    start_time = time.perf_counter()

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

    return LLMResult(
        content=answer,
        model=(
            response.model
            or settings.deepseek_model
        ),
        finish_reason=choice.finish_reason,
        usage=build_usage(response.usage),
        latency_ms=latency_ms,
    )


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