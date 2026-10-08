"""
Agent 工具测试。

测试各个工具的输入验证、Schema 生成和基本功能。
"""

import pytest

from agents.registry import tool_registry
from agents.tools.document_info import (
    GetDocumentInfoInput,
    GetDocumentInfoTool,
)
from agents.tools.list_documents import (
    ListAllDocumentsInput,
    ListAllDocumentsTool,
)
from agents.tools.knowledge_search import (
    SearchKnowledgeBaseInput,
    SearchKnowledgeBaseTool,
)
from agents.base_tool import BaseTool


class TestToolRegistry:
    """
    工具注册中心测试。
    """

    def test_registry_has_tools(self):
        """
        测试注册中心至少有 4 个工具。
        """
        tools = tool_registry.get_tool_names()
        assert len(tools) >= 4
        assert "get_document_info" in tools
        assert "list_all_documents" in tools
        assert "search_knowledge_base" in tools
        assert "ask_advanced_rag" in tools

    def test_get_api_schemas(self):
        """
        测试所有工具都能生成 API Schema。
        """
        schemas = tool_registry.get_api_schemas()
        assert len(schemas) >= 4

        for schema in schemas:
            assert "type" in schema
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_get_tool_by_name(self):
        """
        测试按名称获取工具。
        """
        tool = tool_registry.get("get_document_info")
        assert tool.name == "get_document_info"

    def test_get_nonexistent_tool_raises(self):
        """
        测试获取不存在的工具会报错。
        """
        with pytest.raises(Exception):
            tool_registry.get("nonexistent_tool")


class TestGetDocumentInfoTool:
    """
    get_document_info 工具测试。
    """

    def test_tool_name(self):
        tool = GetDocumentInfoTool()
        assert tool.name == "get_document_info"

    def test_input_model_fields(self):
        """
        测试输入模型包含 document_id 和 filename 字段。
        """
        fields = GetDocumentInfoInput.model_fields
        assert "document_id" in fields
        assert "filename" in fields

    def test_input_optional(self):
        """
        测试两个参数都是可选的。
        """
        # 不传参数也应该能创建
        inp = GetDocumentInfoInput()
        assert inp.document_id is None
        assert inp.filename is None

    def test_input_with_id(self):
        """
        测试只传 document_id。
        """
        inp = GetDocumentInfoInput(document_id=1)
        assert inp.document_id == 1
        assert inp.filename is None

    def test_input_with_filename(self):
        """
        测试只传 filename。
        """
        inp = GetDocumentInfoInput(filename="test.txt")
        assert inp.filename == "test.txt"
        assert inp.document_id is None


class TestListAllDocumentsTool:
    """
    list_all_documents 工具测试。
    """

    def test_tool_name(self):
        tool = ListAllDocumentsTool()
        assert tool.name == "list_all_documents"

    def test_input_model_empty(self):
        """
        测试输入模型没有必填参数。
        """
        inp = ListAllDocumentsInput()
        # 不应该有任何必填字段
        assert True


class TestSearchKnowledgeBaseTool:
    """
    search_knowledge_base 工具测试。
    """

    def test_tool_name(self):
        tool = SearchKnowledgeBaseTool()
        assert tool.name == "search_knowledge_base"

    def test_input_model_fields(self):
        """
        测试输入模型包含必要字段。
        """
        fields = SearchKnowledgeBaseInput.model_fields
        assert "query" in fields
        assert "file_id" in fields
