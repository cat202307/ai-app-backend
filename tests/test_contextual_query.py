"""
追问改写判定测试（services.contextual_query_service）。

纯逻辑 + monkeypatch（不真实调用 LLM）：
  - needs_standalone_question：短技术名词视为自包含，
    仅含明确指代 / 依赖词才改写
  - resolve_query：自包含原样返回、无历史标记 missing、
    有历史时返回改写结果
  - build_history_text：提取纯文本、跳过工具消息、排除当前问题
"""

import pytest

from services import contextual_query_service as cq
from services.contextual_query_service import (
    build_history_text,
    needs_standalone_question,
    resolve_query,
)


class TestNeedsStandalone:
    """是否需要结合历史改写。"""

    @pytest.mark.parametrize(
        "question",
        [
            # 短但含具体技术名词：本次修复，不再改写
            "学习率设置",
            "优化器参数",
            "损失函数",
        ],
    )
    def test_short_technical_noun_self_contained(
        self, question
    ):
        assert needs_standalone_question(question) is False

    @pytest.mark.parametrize(
        "question",
        [
            "有什么用",
            "它呢",
            "怎么做",
            "这个怎么实现",
        ],
    )
    def test_dependency_markers_need_rewrite(
        self, question
    ):
        assert needs_standalone_question(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "AdamW 优化器",
            "lr 设置为 1e-3",
            "论文用了什么优化器？",
        ],
    )
    def test_latin_or_full_question_self_contained(
        self, question
    ):
        assert needs_standalone_question(question) is False

    def test_overview_routed_elsewhere(self):
        # 概述类即使含「这」也不在此改写
        assert (
            needs_standalone_question("这篇论文主要讲什么")
            is False
        )

    def test_empty(self):
        assert needs_standalone_question("") is False


class TestResolveQuery:
    """resolve_query 行为。"""

    def test_self_contained_returned_as_is(self):
        resolved, missing = resolve_query(
            "学习率设置", messages=[]
        )
        assert resolved == "学习率设置"
        assert missing is False

    def test_dependent_without_history_is_missing(self):
        resolved, missing = resolve_query(
            "有什么用", messages=[]
        )
        assert missing is True
        assert "缺少上下文" in resolved

    def test_dependent_with_history_rewritten(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            cq,
            "rewrite_standalone_question",
            lambda history_text, question:
                "AdamW 优化器有什么作用？",
        )
        messages = [
            {"role": "user", "content": "论文用了什么优化器？"},
            {"role": "assistant", "content": "使用 AdamW。"},
            {"role": "user", "content": "有什么用"},
        ]
        resolved, missing = resolve_query(
            "有什么用", messages
        )
        assert missing is False
        assert resolved == "AdamW 优化器有什么作用？"

    def test_rewrite_failure_falls_back(
        self, monkeypatch
    ):
        def _raise(history_text, question):
            raise RuntimeError("llm unavailable")

        monkeypatch.setattr(
            cq, "rewrite_standalone_question", _raise
        )
        messages = [
            {"role": "user", "content": "优化器"},
            {"role": "assistant", "content": "AdamW"},
            {"role": "user", "content": "它呢"},
        ]
        resolved, missing = resolve_query(
            "它呢", messages
        )
        # 改写失败回退原问题，不标记 missing
        assert resolved == "它呢"
        assert missing is False


class TestBuildHistoryText:
    """历史文本提取。"""

    def test_basic_pairs(self):
        messages = [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "回答一"},
        ]
        text = build_history_text(messages)
        assert text == "用户：问题一\n助手：回答一"

    def test_excludes_current_question(self):
        messages = [
            {"role": "user", "content": "问题一"},
            {"role": "assistant", "content": "回答一"},
            {"role": "user", "content": "它有什么用"},
        ]
        text = build_history_text(
            messages, current_question="它有什么用"
        )
        assert "它有什么用" not in text
        assert "问题一" in text

    def test_skips_tool_and_system_messages(self):
        messages = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "一个问题"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1"}],
            },
            {"role": "tool", "content": "工具返回"},
        ]
        text = build_history_text(messages)
        assert text == "用户：一个问题"

    def test_respects_max_turns(self):
        messages = []
        for i in range(10):
            messages.append(
                {"role": "user", "content": "问题%d" % i}
            )
            messages.append(
                {"role": "assistant", "content": "回答%d" % i}
            )
        text = build_history_text(messages, max_turns=4)
        lines = text.split("\n")
        assert len(lines) == 4
        # 保留最近的对话
        assert "问题9" in text
        assert "问题0" not in text
