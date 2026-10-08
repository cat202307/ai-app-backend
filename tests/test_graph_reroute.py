"""
Agent 工具选择纠偏测试（agents.graph_agent 辅助函数）。

纯逻辑：
  - _is_content_seeking：内容 / 技术点问题 True，
    文件列表请求 False
  - _latest_user_question：取最后一条 user 消息
"""

import pytest

from agents.graph_agent import (
    _is_content_seeking,
    _latest_user_question,
)


class TestIsContentSeeking:
    """内容类问题识别。"""

    @pytest.mark.parametrize(
        "question",
        [
            "论文用了什么优化器？",
            "损失函数是什么？",
            "模型如何训练？",
            "介绍一下算法流程",
            "采用了什么参数？",
            "为什么使用梯度裁剪？",
        ],
    )
    def test_content_questions_true(self, question):
        assert _is_content_seeking(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "有哪些文件？",
            "系统里有哪些文档？",
            "列出所有文件",
            "一共多少个文件？",
            "文档总数",
        ],
    )
    def test_file_list_questions_false(self, question):
        assert _is_content_seeking(question) is False

    @pytest.mark.parametrize(
        "question",
        ["", "你好", "谢谢"],
    )
    def test_neutral_questions_false(self, question):
        assert _is_content_seeking(question) is False


class TestLatestUserQuestion:
    """最后一条 user 消息提取。"""

    def test_returns_last_user(self):
        messages = [
            {"role": "user", "content": "第一个问题"},
            {"role": "assistant", "content": "第一个回答"},
            {"role": "user", "content": "第二个问题"},
        ]
        assert (
            _latest_user_question(messages) == "第二个问题"
        )

    def test_ignores_trailing_assistant(self):
        messages = [
            {"role": "user", "content": "唯一问题"},
            {"role": "assistant", "content": "回答"},
        ]
        assert (
            _latest_user_question(messages) == "唯一问题"
        )

    def test_no_user_returns_empty(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "回答"},
        ]
        assert _latest_user_question(messages) == ""

    def test_empty_list(self):
        assert _latest_user_question([]) == ""
