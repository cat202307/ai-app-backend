"""
Query Rewrite 门控与缓存测试。

  - _should_skip_rewrite：技术锚点跳过、指代/宽泛保留
  - rewrite_queries：跳过路径不调 LLM；缓存命中不重复调 LLM
"""

import pytest

from services import advanced_rag_service as ars


class TestShouldSkip:
    def test_latin_token_skips(self):
        assert ars._should_skip_rewrite("RRF怎么实现？") is True
        assert ars._should_skip_rewrite("AdamW的lr是多少") is True

    def test_content_word_skips(self):
        assert ars._should_skip_rewrite(
            "论文用了什么优化器？"
        ) is True
        assert ars._should_skip_rewrite("学习率设置为多少？") is True
        assert ars._should_skip_rewrite("损失函数有哪些") is True

    def test_dependency_marker_keeps(self):
        assert ars._should_skip_rewrite("它有什么用？") is False
        assert ars._should_skip_rewrite("这个怎么做") is False

    def test_comparison_question_keeps(self):
        # 对比类问题即使含拉丁 token（test02）也强制改写，
        # 以便拆分出针对每个文档的查询
        assert ars._should_skip_rewrite(
            "论文和test02有什么区别"
        ) is False
        assert ars._should_skip_rewrite(
            "两种方法的差异是什么"
        ) is False
        assert ars._should_skip_rewrite(
            "A和B相比哪个好"
        ) is False

    def test_listing_question_still_skips(self):
        # “哪些”是列举而非对比，含技术锚点时仍跳过改写
        assert ars._should_skip_rewrite(
            "损失函数有哪些"
        ) is True

    def test_broad_no_anchor_keeps(self):
        # 无技术锚点、无指代：宽泛问题保留改写以扩展召回
        assert ars._should_skip_rewrite(
            "帮我梳理一下整体背景"
        ) is False

    def test_empty_skips(self):
        assert ars._should_skip_rewrite("") is True


class TestRewriteQueries:
    def test_skip_path_does_not_call_llm(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("不应调用 LLM")

        monkeypatch.setattr(ars, "call_llm", boom)
        result = ars.rewrite_queries(
            "RRF怎么实现？", rewrite_count=1
        )
        assert result == ["RRF怎么实现？"]

    def test_cache_avoids_second_llm_call(self, monkeypatch):
        calls = {"n": 0}

        def fake_llm(messages, **kwargs):
            calls["n"] += 1
            return "该主题的另一种问法"

        monkeypatch.setattr(ars, "call_llm", fake_llm)

        # 纯中文、无技术锚点、无指代 marker 的问题：
        # 不会跳过改写，用于验证第二次走缓存。
        q = "帮我梳理一下整体背景与来龙去脉"

        ars._rewrite_cache.pop(q, None)
        first = ars.rewrite_queries(q, rewrite_count=1)
        second = ars.rewrite_queries(q, rewrite_count=1)

        assert calls["n"] == 1  # 第二次走缓存
        assert first == second
        assert first[0] == q
