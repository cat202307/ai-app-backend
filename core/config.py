import os

from dotenv import load_dotenv


# 加载项目根目录中的 .env 文件
load_dotenv()


class Settings:
    """
    项目统一配置类。

    所有模型、RAG、上传和 Agent 相关配置，
    后续都从这里统一读取，避免散落在不同文件中。
    """

    # DeepSeek API Key
    deepseek_api_key: str = os.getenv(
        "DEEPSEEK_API_KEY",
        "",
    )

    # DeepSeek OpenAI 兼容接口地址
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    )

    # 默认使用的模型名称
    deepseek_model: str = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-chat",
    )

    # 普通大模型调用超时时间，单位为秒
    llm_timeout: float = float(
        os.getenv(
            "LLM_TIMEOUT",
            "30",
        )
    )

    # 流式模型调用超时时间
    llm_stream_timeout: float = float(
        os.getenv(
            "LLM_STREAM_TIMEOUT",
            "60",
        )
    )

    # 当前 RAG 检索距离阈值
    # 暂时保留 17.0，后面通过评测集重新校准
    rag_max_distance: float = float(
        os.getenv(
            "RAG_MAX_DISTANCE",
            "17.0",
        )
    )


# 创建全局配置对象
settings = Settings()