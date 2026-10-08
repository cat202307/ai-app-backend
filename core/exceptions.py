class AppError(Exception):
    """
    项目统一业务异常基类。

    code:
        程序内部错误码，例如 LLM_TIMEOUT。

    message:
        返回给客户端的错误说明。

    status_code:
        对应的 HTTP 状态码。
    """

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code

        super().__init__(message)


class LLMConfigurationError(AppError):
    """
    大模型配置错误。

    例如：
    - 没有配置 API Key
    - 配置项缺失
    """

    def __init__(
        self,
        message: str = "未配置大模型 API Key。",
    ):
        super().__init__(
            code="LLM_CONFIGURATION_ERROR",
            message=message,
            status_code=500,
        )


class LLMTimeoutError(AppError):
    """
    大模型响应超时。
    """

    def __init__(
        self,
        message: str = "大模型响应超时，请稍后重试。",
    ):
        super().__init__(
            code="LLM_TIMEOUT",
            message=message,
            status_code=504,
        )


class LLMAuthenticationError(AppError):
    """
    大模型鉴权失败。

    一般表示 API Key 错误、失效或没有权限。
    """

    def __init__(
        self,
        message: str = "大模型 API 鉴权失败，请检查 API Key。",
    ):
        super().__init__(
            code="LLM_AUTHENTICATION_ERROR",
            message=message,
            status_code=502,
        )


class LLMRateLimitError(AppError):
    """
    大模型请求被限流。
    """

    def __init__(
        self,
        message: str = "大模型请求过于频繁，请稍后重试。",
    ):
        super().__init__(
            code="LLM_RATE_LIMIT_ERROR",
            message=message,
            status_code=429,
        )


class LLMServiceError(AppError):
    """
    其他大模型服务错误。
    """

    def __init__(
        self,
        message: str = "大模型服务调用失败。",
    ):
        super().__init__(
            code="LLM_SERVICE_ERROR",
            message=message,
            status_code=502,
        )

class ToolNotFoundError(AppError):
    """
    模型请求调用的工具不存在。
    """

    def __init__(
        self,
        tool_name: str,
    ):
        super().__init__(
            code="TOOL_NOT_FOUND",
            message=f"工具不存在或未注册：{tool_name}",
            status_code=400,
        )


class ToolArgumentsError(AppError):
    """
    工具参数无法解析或没有通过 Schema 校验。
    """

    def __init__(
        self,
        message: str = "工具参数不合法。",
    ):
        super().__init__(
            code="TOOL_ARGUMENTS_ERROR",
            message=message,
            status_code=400,
        )


class ToolExecutionError(AppError):
    """
    工具真实执行过程中发生异常。
    """

    def __init__(
        self,
        message: str = "工具执行失败。",
    ):
        super().__init__(
            code="TOOL_EXECUTION_ERROR",
            message=message,
            status_code=500,
        )

class AgentMaxStepsError(AppError):
    """
    Agent 超过最大执行步骤。
    """

    def __init__(
        self,
        max_steps: int,
    ):
        super().__init__(
            code="AGENT_MAX_STEPS_EXCEEDED",
            message=(
                f"Agent 已达到最大执行步骤 "
                f"{max_steps}，任务仍未完成。"
            ),
            status_code=500,
        )


class AgentRepeatedToolCallError(AppError):
    """
    Agent 重复调用完全相同的工具和参数。
    """

    def __init__(
        self,
        tool_name: str,
    ):
        super().__init__(
            code="AGENT_REPEATED_TOOL_CALL",
            message=(
                f"检测到 Agent 重复调用工具："
                f"{tool_name}。"
            ),
            status_code=500,
        )