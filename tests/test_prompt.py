"""
Prompt 构建测试。

测试各个 Prompt 构建函数是否正确输出。
"""

from services.prompt_service import (
    build_chat_system_message,
    build_rag_messages,
    build_agent_messages,
    build_tool_calling_messages,
    build_query_rewrite_messages,
    build_rerank_messages,
    build_citation_rag_messages,
)


class TestChatPrompt:
    """
    普通聊天 Prompt 测试。
    """

    def test_chat_system_message(self):
        """
        测试普通聊天 system message 构建。
        """
        msg = build_chat_system_message()
        assert msg["role"] == "system"
        assert "AI" in msg["content"] or "助手" in msg["content"]


class TestRagPrompt:
    """
    RAG 相关 Prompt 测试。
    """

    def test_rag_messages(self):
        """
        测试基础 RAG Prompt 构建。
        """
        context = "FastAPI 是一个现代 Web 框架。"
        question = "FastAPI 是什么？"

        messages = build_rag_messages(context, question)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "FastAPI" in messages[1]["content"]

    def test_query_rewrite_messages(self):
        """
        测试 Query Rewrite Prompt 构建。
        """
        question = "FastAPI 怎么用？"
        messages = build_query_rewrite_messages(question, rewrite_count=2)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Query Rewrite" in messages[0]["content"] or "改写" in messages[0]["content"]

    def test_rerank_messages(self):
        """
        测试 Rerank Prompt 构建。
        """
        question = "FastAPI 是什么？"
        candidates_text = "[1] FastAPI 是 Web 框架\n[2] 今天天气很好"

        messages = build_rerank_messages(question, candidates_text)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Rerank" in messages[0]["content"] or "排序" in messages[0]["content"]

    def test_citation_rag_messages(self):
        """
        测试带引用的 RAG Prompt 构建。
        """
        context = "[1] FastAPI 是一个现代 Web 框架。"
        question = "FastAPI 是什么？"

        messages = build_citation_rag_messages(question, context)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "引用" in messages[0]["content"] or "citation" in messages[0]["content"]


class TestAgentPrompt:
    """
    Agent Prompt 测试。
    """

    def test_tool_calling_messages(self):
        """
        测试单轮 Tool Calling Prompt。
        """
        user_msg = "系统里有哪些文件？"
        messages = build_tool_calling_messages(user_msg)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == user_msg

    def test_agent_messages_contains_all_tools(self):
        """
        测试 Agent system prompt 包含所有工具说明。
        """
        user_msg = "查一下文件内容"
        messages = build_agent_messages(user_msg)

        system_content = messages[0]["content"]

        # 应该包含所有工具的名称
        assert "list_all_documents" in system_content
        assert "get_document_info" in system_content
        assert "ask_advanced_rag" in system_content
        assert "search_knowledge_base" in system_content

    def test_agent_messages_contains_rules(self):
        """
        测试 Agent system prompt 包含关键规则。
        """
        user_msg = "测试"
        messages = build_agent_messages(user_msg)

        system_content = messages[0]["content"]

        # 应该包含关键规则
        assert "工具" in system_content
        assert "不要" in system_content or "禁止" in system_content
