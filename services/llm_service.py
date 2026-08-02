import os
from typing import Any, Iterator, cast

from dotenv import load_dotenv
from openai import OpenAI


# 加载 .env 文件
load_dotenv()


# DeepSeek API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# DeepSeek OpenAI 兼容接口地址
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 模型名称
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# 创建 OpenAI SDK 客户端
# 虽然这里类名是 OpenAI，但 base_url 指向 DeepSeek
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


def call_llm(messages: list[dict[str, str]]) -> str:
    """
    普通大模型调用。

    特点：
        等模型完整生成后，一次性返回完整 answer。

    用于：
        POST /chat
        POST /rag/search
    """

    if not DEEPSEEK_API_KEY:
        return "未配置 DEEPSEEK_API_KEY，请先在 .env 文件中配置 API Key。"

    try:
        response = client.chat.completions.create(
            model=cast(Any, DEEPSEEK_MODEL),
            messages=cast(Any, messages),
            temperature=0.7,
            stream=False,
            timeout=30.0
        )

        answer = response.choices[0].message.content

        if answer is None:
            return "大模型没有返回有效内容。"

        return answer

    except Exception as e:
        return f"大模型调用失败：{str(e)}"


def stream_llm(messages: list[dict[str, str]]) -> Iterator[str]:
    """
    流式大模型调用。

    特点：
        模型生成一点，就 yield 返回一点。

    用于：
        POST /chat/stream
        POST /rag/search/stream
    """

    if not DEEPSEEK_API_KEY:
        yield "未配置 DEEPSEEK_API_KEY，请先在 .env 文件中配置 API Key。"
        return

    try:
        response = client.chat.completions.create(
            model=cast(Any, DEEPSEEK_MODEL),
            messages=cast(Any, messages),
            temperature=0.7,
            stream=True,
            timeout=60.0
        )

        # response 是一个可迭代对象
        # 每次循环拿到模型生成的一小段内容
        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 流式返回时，内容一般在 delta.content 中
            content = getattr(delta, "content", None)

            if content:
                yield content

    except Exception as e:
        yield f"大模型流式调用失败：{str(e)}"