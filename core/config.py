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
        "deepseek-v4-pro",
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

    # LLM 单价（人民币 / 百万 token），用于 /metrics
    # 的成本估算。默认值为 DeepSeek 经典价的参考值，
    # 请按所用模型（deepseek_model）的实际单价在 .env 覆盖。
    llm_price_per_million_input: float = float(
        os.getenv(
            "LLM_PRICE_PER_MILLION_INPUT",
            "2.0",
        )
    )
    llm_price_per_million_output: float = float(
        os.getenv(
            "LLM_PRICE_PER_MILLION_OUTPUT",
            "8.0",
        )
    )

    # RAG 向量检索距离阈值（越大越宽松）
    rag_max_distance: float = float(
        os.getenv(
            "RAG_MAX_DISTANCE",
            "19.0",
        )
    )

    # ============================================================
    # Advanced RAG 配置
    # ============================================================

    # Rerank 最低可信分数（0-100，sigmoid 归一化后）。
    # 低于此分数的片段视为未匹配，触发拒答。
    # 经评测校准：相关问题 top 67-72.5，概述/无关约 50，
    # 阈值 50：bge-reranker-v2-m3 对相关片段通常给 60+，
    # 边缘相关（如实验结果的定性描述）给 50-55，降到 50 可召回。
    rag_rerank_min_score: float = float(
        os.getenv(
            "RAG_RERANK_MIN_SCORE",
            "50.0",
        )
    )

    # 查询改写（Query Rewrite）生成的改写数量。
    # 用 LLM 把原始问题扩展为多个近义表述，提升召回：
    # 解决用户用泛称（如「优化器」）、文档用具体名
    # （如「AdamW」）导致的词面不匹配。
    # 设为 0 可关闭；默认 1（额外一次 LLM 调用）。
    rag_query_rewrite_count: int = int(
        os.getenv(
            "RAG_QUERY_REWRITE_COUNT",
            "1",
        )
    )

    # 全局检索（跨多文档）命中改写门控时，保证的改写数量。
    # 生产与评测共用，作为唯一事实来源，避免参数漂移。
    rag_global_rewrite_count: int = int(
        os.getenv("RAG_GLOBAL_REWRITE_COUNT", "2")
    )

    # ---- Hybrid + Rerank 检索管线统一参数（生产 / 评测共用）----
    # 每个 Query 在每个 Retriever（向量 / 关键词）上各召回多少。
    rag_per_retriever_top_k: int = int(
        os.getenv("RAG_PER_RETRIEVER_TOP_K", "5")
    )
    # RRF 融合后保留多少候选送进 Rerank。
    rag_fusion_top_k: int = int(
        os.getenv("RAG_FUSION_TOP_K", "6")
    )
    # Rerank 后最终保留多少片段进上下文。
    rag_final_top_k: int = int(
        os.getenv("RAG_FINAL_TOP_K", "8")
    )

    # 父子块切分参数（Parent-Child / Small-to-Big）。
    # 子块按段落边界切、主题聚焦，负责精准检索；
    # 父块由若干子块聚合，命中后提供完整上下文。
    rag_child_chunk_size: int = int(
        os.getenv("RAG_CHILD_CHUNK_SIZE", "200")
    )

    # 每个父块包含的子块数量（父块 ≈ 4×200=800 字）
    rag_children_per_parent: int = int(
        os.getenv("RAG_CHILDREN_PER_PARENT", "4")
    )

    # 每次聊天最多读取多少条历史记录
    #
    # 一条 ChatHistory 包含：
    # - 一次 user 提问
    # - 一次 assistant 回答
    chat_history_limit: int = int(
        os.getenv(
            "CHAT_HISTORY_LIMIT",
            "5",
        )
    )

    # 历史聊天内容允许使用的最大字符数
    #
    # 这里只是简单字符预算，不等同于精确 Token 数。
    # 后续可以接入模型对应的 tokenizer。
    chat_history_max_chars: int = int(
        os.getenv(
            "CHAT_HISTORY_MAX_CHARS",
            "12000",
        )
    )

    # ============================================================
    # Agent 配置
    # ============================================================

    # Agent 最大执行步骤。
    #
    # 防止模型不断调用工具进入死循环。
    agent_max_steps: int = int(
        os.getenv(
            "AGENT_MAX_STEPS",
            "5",
        )
    )

    # 单次工具结果允许放回模型上下文的最大字符数。
    #
    # 知识库工具可能返回较长 Chunk，
    # 必须限制长度，避免 Context 无限膨胀。
    agent_tool_result_max_chars: int = int(
        os.getenv(
            "AGENT_TOOL_RESULT_MAX_CHARS",
            "6000",
        )
    )

    # ============================================================
    # 进程内缓存配置（LRU + TTL）
    # ============================================================

    # Embedding 向量缓存：容量上限与过期秒数（0=不过期）。
    # 防止随语料/查询增多内存只增不减。
    embedding_cache_maxsize: int = int(
        os.getenv("EMBEDDING_CACHE_MAXSIZE", "4096")
    )
    embedding_cache_ttl: int = int(
        os.getenv("EMBEDDING_CACHE_TTL", "3600")
    )

    # 查询改写（LLM Rewrite）结果缓存：相同问题不重复调 LLM。
    rewrite_cache_maxsize: int = int(
        os.getenv("REWRITE_CACHE_MAXSIZE", "512")
    )
    rewrite_cache_ttl: int = int(
        os.getenv("REWRITE_CACHE_TTL", "86400")
    )

    # ============================================================
    # 数据库配置
    # ============================================================

    # SQLAlchemy 数据库连接地址。
    # 默认单机 SQLite；生产可通过环境变量切换到 Postgres 等。
    database_url: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./ai_app.db",
    )

    # ============================================================
    # 服务 / 部署配置
    # ============================================================

    # CORS 允许的来源（逗号分隔）。
    # 开发环境默认允许 localhost:3000（Vue dev server）。
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if origin.strip()
    ]

    # ============================================================
    # 安全配置
    # ============================================================

    # API 访问密钥。
    # 为空时不启用认证（开发模式）；
    # 生产环境应设置为强随机字符串，前端通过 X-API-Key header 传入。
    api_key: str = os.getenv("API_KEY", "")

    # 限流：每个 IP 每分钟对 LLM 接口的最大请求数。
    rate_limit_per_minute: int = int(
        os.getenv("RATE_LIMIT_PER_MINUTE", "20")
    )

    # ============================================================
    # 文件上传配置
    # ============================================================

    # 单个上传文件大小上限（MB）。
    # 上传内容会读入内存，必须限制大小防止超大文件导致 OOM。
    upload_max_mb: int = int(
        os.getenv("UPLOAD_MAX_MB", "50")
    )


# 创建全局配置对象
settings = Settings()