from typing import Any

from agents.base_tool import BaseTool
from agents.tools.document_info import (
    GetDocumentInfoTool,
)
from agents.tools.knowledge_search import (
    SearchKnowledgeBaseTool,
)
from agents.tools.advanced_rag_search import (
    AskAdvancedRagTool,
)
from agents.tools.list_documents import (
    ListAllDocumentsTool,
)
from core.exceptions import ToolNotFoundError


class ToolRegistry:
    """
    工具注册中心。

    主要职责：

    1. 注册允许模型调用的工具。
    2. 根据工具名找到真实 Python 工具。
    3. 生成发送给模型的 tools Schema。
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(
        self,
        tool: BaseTool,
    ) -> None:
        """
        注册工具。
        """

        if not tool.name:
            raise ValueError("工具名称不能为空。")

        self._tools[tool.name] = tool

    def get(
        self,
        tool_name: str,
    ) -> BaseTool:
        """
        根据名称获取工具。
        """

        tool = self._tools.get(tool_name)

        if tool is None:
            raise ToolNotFoundError(tool_name)

        return tool

    def get_api_schemas(
        self,
    ) -> list[dict[str, Any]]:
        """
        获取发送给模型的所有工具 Schema。
        """

        return [
            tool.to_api_schema()
            for tool in self._tools.values()
        ]

    def get_tool_names(
        self,
    ) -> list[str]:
        """
        返回当前所有工具名称。
        """

        return list(self._tools.keys())


# 创建项目默认工具注册中心
tool_registry = ToolRegistry()

# 注册当前第一个工具
tool_registry.register(
    GetDocumentInfoTool()
)

tool_registry.register(
    SearchKnowledgeBaseTool()
)

tool_registry.register(
    AskAdvancedRagTool()
)

tool_registry.register(
    ListAllDocumentsTool()
)