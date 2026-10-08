"""
Metrics 收集器测试（core.metrics）。

用局部 MetricsCollector 实例，避免污染全局单例：
  - 检索阶段计时（record_retrieval_stage / stage_timer）
  - LLM token 与成本累计（record_llm_usage）
  - get_summary 结构
"""

import pytest

from core.metrics import MetricsCollector, metrics


class TestRetrievalStages:
    def test_record_stage_accumulates(self):
        mc = MetricsCollector()
        mc.record_retrieval_stage("rerank", 100.0)
        mc.record_retrieval_stage("rerank", 50.0)

        summary = mc.get_summary()
        stages = {
            s["stage"]: s for s in summary["retrieval_stages"]
        }
        rerank = stages["rerank"]
        assert rerank["count"] == 2
        assert rerank["total_ms"] == 150.0
        assert rerank["max_ms"] == 100.0
        assert rerank["avg_ms"] == 75.0

    def test_stage_timer_records(self):
        mc = MetricsCollector()
        with mc.stage_timer("vector_search"):
            pass  # 极快也应记录一次

        stages = {
            s["stage"]: s
            for s in mc.get_summary()["retrieval_stages"]
        }
        assert stages["vector_search"]["count"] == 1


class TestLLMUsage:
    def test_tokens_accumulate(self):
        mc = MetricsCollector()
        mc.record_llm_usage(
            prompt_tokens=100, completion_tokens=50
        )
        mc.record_llm_usage(
            prompt_tokens=200, completion_tokens=80
        )

        usage = mc.get_summary()["llm_usage"]
        assert usage["calls"] == 2
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 130
        assert usage["total_tokens"] == 430

    def test_cost_estimated_from_settings(self):
        # 默认单价 input 2 元 / M、output 8 元 / M
        mc = MetricsCollector()
        mc.record_llm_usage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        )
        usage = mc.get_summary()["llm_usage"]
        assert usage["estimated_cost"] == pytest.approx(
            10.0, abs=0.01
        )

    def test_zero_usage_no_cost(self):
        mc = MetricsCollector()
        mc.record_llm_usage(0, 0)
        usage = mc.get_summary()["llm_usage"]
        assert usage["estimated_cost"] == 0
        assert usage["total_tokens"] == 0


class TestSummaryShape:
    def test_summary_contains_new_sections(self):
        mc = MetricsCollector()
        summary = mc.get_summary()
        assert "retrieval_stages" in summary
        assert "llm_usage" in summary

    def test_global_singleton_exists(self):
        assert metrics is not None
        summary = metrics.get_summary()
        assert "llm_usage" in summary
